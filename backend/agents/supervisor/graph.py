# -*- coding: utf-8 -*-
"""
Supervisor 智能体图定义
========================
多智能体协调器，带对话记忆（checkpointer），是整个系统的总调度入口。

核心职责：
1. 接收用户消息，分析意图
2. 决定调用哪个子智能体（Knowledge / Search / Email）
3. 收集子智能体返回结果，继续决策或生成最终回复
4. 支持多轮对话记忆，重启不丢失

多智能体交互模式：
    Supervisor 采用 "Agent-as-Tool" 模式 ——
    每个子智能体被包装成一个 LangChain @tool，
    Supervisor LLM 通过 Function Calling 自主决定调用哪个子智能体。
    这是一种标准的 LangGraph Supervisor + Specialized Agents 架构。
"""

import ssl
import httpx
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from .state import SupervisorState
from .nodes import should_continue
from .services.coordinator import create_supervisor_tools, get_supervisor_system_prompt
from app.core.config import settings, SUPPORTED_MODELS


def create_supervisor_agent():
    """
    创建并编译 Supervisor 多智能体。

    ------------------------------------------------------------
    架构概览
    ------------------------------------------------------------
                    ┌─────────────────────┐
                    │   Supervisor Agent  │  ← 总调度 LLM
                    │  (call_supervisor_model)
                    └────────┬────────────┘
                             │ 绑定以下工具（子智能体）
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ Knowledge  │  │   Email    │  │  Search    │
     │ 知识库问答  │  │  邮件发送   │  │  联网搜索   │
     └────────────┘  └────────────┘  └────────────┘

    子智能体被包装成 LangChain @tool，Supervisor
    通过 Function Calling 决定调用哪一个。

    """
    print("\n[Graph] 开始构建 Supervisor 多智能体系统")

    # ================================================================
    # 1. SSL / HTTP 客户端配置
    # ================================================================
    # 如果配置关闭了 SSL 验证（内网环境或调试场景），
    # 全局关闭 Python 的 SSL 证书校验，避免 HTTPS 请求报错。
    if not settings.ssl_verify:
        ssl._create_default_https_context = ssl._create_unverified_context

    # 创建 HTTP 客户端，供所有 ChatOpenAI 实例复用。
    # 复用连接池可以避免反复建立 TCP 连接，提升性能。
    http_client = httpx.Client(
        verify=settings.ssl_verify,
        timeout=settings.timeout
    )

    # ================================================================
    # 2. 创建默认模型（仅用于构建工具 schema）
    # ================================================================
    # 这里创建模型只是为了能在注册工具时拿到模型的能力描述，
    # 实际请求时会根据前端传入的 model 参数动态切换模型。
    default_model = ChatOpenAI(
        model=settings.default_model,           # 默认模型，如 qwen3.7-plus
        temperature=settings.temperature,       # 控制生成随机性
        base_url=settings.dashscope_base_url,   # DashScope API 地址
        api_key=settings.dashscope_api_key,     # API Key
        streaming=False,                        # Supervisor 不使用流式
        timeout=settings.timeout,               # 请求超时
        max_retries=settings.max_retries,       # 重试次数
        http_client=http_client
    )

    # ================================================================
    # 3. 构建子智能体工具列表 & ToolNode
    # ================================================================
    # create_supervisor_tools 会创建以下工具（每个工具包装一个子 Agent）：
    #   - knowledge_agent      → 本地知识库问答
    #   - email_agent          → 邮件操作
    #   - search_agent         → 联网搜索
    # ToolNode 是 LangGraph 的预置节点，负责接收 LLM 的 tool_call，
    # 执行对应的 Python 函数，并返回 ToolMessage。
    supervisor_tools = create_supervisor_tools(default_model)
    tool_node = ToolNode(supervisor_tools)

    print(f"[Graph] 已创建 {len(supervisor_tools)} 个子智能体工具")

    # ================================================================
    # 4. Supervisor 核心节点：call_supervisor_model
    # ================================================================
    def call_supervisor_model(state: SupervisorState, config) -> dict:
        """
        Supervisor 主推理节点。

        这个函数是 LangGraph 图的核心节点 "supervisor"。
        每次执行时：
        1. 从 config 中读取当前请求使用的模型名称和参数
        2. 动态创建 ChatOpenAI 实例
        3. 将子智能体工具绑定到模型（Function Calling）
        4. 自动注入 System Prompt（只在首轮对话时）
        5. 调用 LLM 推理，返回 AI 消息（可能包含 tool_calls）

        参数
        ----
        state : SupervisorState
            当前图状态，包含 messages（对话历史）
        config : dict
            LangGraph 运行时配置，包含：
            - configurable.model : 用户选择的模型名
            - configurable.temperature : 温度参数

        返回
        ----
        dict : {"messages": [AIMessage]}
            模型返回的消息，可能包含 tool_calls（需要调工具）
            或只是普通回复文本（对话结束）
        """
        from langchain_core.messages import SystemMessage

        # ------------------------------------------------------------
        # 4.1 读取运行时配置中的模型名称
        # ------------------------------------------------------------
        # 优先使用前端传入的模型；若未传入，回退到 .env 的默认模型。
        # 这样实现了"前端右上角切换模型 → 后端真实覆盖"。
        model_name = config.get("configurable", {}).get("model", settings.default_model)

        # ------------------------------------------------------------
        # 4.2 模型白名单校验
        # ------------------------------------------------------------
        # 防止使用未注册的模型（可能导致 API 报错或计费异常）。
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"模型 '{model_name}' 不在支持列表中。"
                f"可用模型: {list(SUPPORTED_MODELS.keys())}"
            )

        print(f"\n[Supervisor] 当前使用模型: {model_name}")

        try:
            # --------------------------------------------------------
            # 4.3 为本次请求创建模型实例
            # --------------------------------------------------------
            # 每次请求创建一个新的 ChatOpenAI 实例，
            # 确保温度和模型名与当前请求一致，不与全局默认模型混淆。
            model = ChatOpenAI(
                model=model_name,
                temperature=config.get("configurable", {}).get("temperature", settings.temperature),
                base_url=settings.dashscope_base_url,
                api_key=settings.dashscope_api_key,
                streaming=False,
                timeout=settings.timeout,
                max_retries=settings.max_retries,
                http_client=http_client
            )

            # --------------------------------------------------------
            # 4.4 绑定工具（子智能体）到模型
            # --------------------------------------------------------
            # bind_tools() 将子智能体工具的 schema（名称、描述、参数）
            # 注入到 LLM 的请求上下文中。LLM 看到用户问题后，
            # 如果判断需要调用某工具，会在 AIMessage 中返回 tool_calls。
            model_with_tools = model.bind_tools(supervisor_tools)

            # --------------------------------------------------------
            # 4.5 获取当前对话历史
            # --------------------------------------------------------
            messages = state["messages"]
            print(f"[Supervisor] 当前对话共 {len(messages)} 条消息")
            for i, msg in enumerate(messages):
                print(f"[Supervisor] 消息 {i+1}: {msg.type} - {str(msg.content)[:50]}...")

            # --------------------------------------------------------
            # 4.6 注入 System Prompt
            # --------------------------------------------------------
            # 只在首轮对话（messages 只有 1 条用户消息）时才注入，
            # 避免多轮对话中重复添加 System Prompt。
            if len(messages) == 1 or not any(isinstance(msg, SystemMessage) for msg in messages):
                system_prompt = get_supervisor_system_prompt()
                messages = [SystemMessage(content=system_prompt)] + messages

            print("[Supervisor] 正在分析用户请求...")

            # --------------------------------------------------------
            # 4.7 调用 LLM 推理
            # --------------------------------------------------------
            # invoke() 是同步调用。
            # 返回的 AIMessage 中：
            #   - 如果 response.tool_calls 非空 → LangGraph 会路由到 tools 节点
            #   - 如果 response.tool_calls 为空 → LangGraph 会路由到 END
            response = model_with_tools.invoke(messages)

            print("[Supervisor] 推理完成")

            # --------------------------------------------------------
            # 4.8 返回结果（更新图状态）
            # --------------------------------------------------------
            # 返回 dict 格式，LangGraph 自动将 AIMessage
            # 追加到 state["messages"] 中。
            return {"messages": [response]}

        except Exception as e:
            error_msg = str(e)
            print(f"[Supervisor] 错误: {error_msg}")
            raise Exception(f"Supervisor 智能体执行失败: {error_msg}")

    # ================================================================
    # 5. 构建 LangGraph 图
    # ================================================================

    # ----------------------------------------------------------------
    # 5.1 创建 StateGraph 构建器
    # ----------------------------------------------------------------
    # 泛型参数 SupervisorState 定义了图共享状态的类型（包含 messages 列表）。
    builder = StateGraph(SupervisorState)

    # ----------------------------------------------------------------
    # 5.2 添加节点
    # ----------------------------------------------------------------
    # "supervisor" 节点：核心推理节点，调用 LLM 决定下一步动作。
    builder.add_node("supervisor", call_supervisor_model)
    # "tools" 节点：工具执行节点（LangGraph 预置 ToolNode），
    # 接收 AIMessage.tool_calls，执行对应的 Python 函数。
    builder.add_node("tools", tool_node)

    # ----------------------------------------------------------------
    # 5.3 添加边
    # ----------------------------------------------------------------
    # START → supervisor : 图启动后直接进入 supervisor 节点。
    builder.add_edge(START, "supervisor")

    # supervisor → 条件路由 : 根据 AIMessage 中是否有 tool_calls，
    # 决定是进入 tools 节点还是结束对话。
    builder.add_conditional_edges(
        "supervisor",           # 从 supervisor 节点出发
        should_continue,        # 路由函数：检查最后一条消息是否有 tool_calls
        {
            "tools": "tools",   # 有 tool_calls → 执行工具
            "__end__": END      # 无 tool_calls → 对话结束，返回最终回复
        }
    )

    # tools → supervisor : 工具执行完毕后，无条件回到 supervisor，
    # 让 LLM 根据工具返回的 ToolMessage 继续推理。
    builder.add_edge("tools", "supervisor")

    # ----------------------------------------------------------------
    # 5.4 添加记忆检查点（Memory Checkpointer）
    # ----------------------------------------------------------------
    # MemorySaver 在内存中保存对话状态（thread_id 隔离多会话）。
    # 重启进程后数据会丢失。
    # 如需持久化，可替换为 AsyncPostgresSaver。
    memory = MemorySaver()

    # ----------------------------------------------------------------
    # 5.5 编译图
    # ----------------------------------------------------------------
    # compile() 将图定义编译为可执行对象。
    # checkpointer 参数使图支持多轮对话记忆和状态持久化。
    graph = builder.compile(checkpointer=memory)

    print("[Graph] Supervisor 智能体创建成功")
    print("[Graph] 工作流: supervisor → should_continue → tools/end")

    return graph
