# -*- coding: utf-8 -*-
"""
Supervisor 协调器服务
====================
管理子智能体的注册、创建和工具包装，是整个多智能体系统的"中枢神经"。

核心职责：
1. 维护子智能体注册表（SUB_AGENTS_REGISTRY）
2. 将每个子智能体包装成 LangChain @tool，供 Supervisor LLM 通过 Function Calling 调用
3. 为 Supervisor 生成 System Prompt，告知 LLM 有哪些子智能体可用
4. 处理同步/异步桥接（工具调用是同步的，但 Knowledge Agent 内部是异步的）
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

from agents.specialized.email import create_email_agent, EMAIL_AGENT_INFO
from agents.specialized.search import create_search_agent, SEARCH_AGENT_INFO
from app.core.prompts import SUPERVISOR_SYSTEM_PROMPT
from app.core.config import settings


# ================================================================
# 子智能体元信息定义
# ================================================================
# Knowledge Agent 是特殊的：它不通过 LangGraph 子图调用，
# 而是直接复用已有的 knowledge_service（RAG 流水线）。
# Email / Search Agent 则各自有独立的 LangGraph 子图和工具列表。
KNOWLEDGE_AGENT_INFO = {
    "name": "knowledge_agent",
    "display_name": "知识库智能体",
    "description": "专门处理本地知识库、文档问答、PDF/文档内容检索、知识库引用溯源等任务",
    "capabilities": [
        "知识库问答",
        "文档内容检索",
        "PDF 问答",
        "多文档综合回答",
        "引用来源返回"
    ],
    "keywords": ["知识库", "文档", "PDF", "资料", "文件", "根据文档", "knowledge", "document"]
}


# ================================================================
# 子智能体注册表
# ================================================================
# 每个子智能体的注册信息只保留 info：
#   - info : 元信息字典（名称、描述、能力、关键词），用于生成 System Prompt
#
# 新增子智能体的步骤：
#   1. 在 specialized/ 目录下创建 Agent 模块
#   2. 定义 X_AGENT_INFO 元信息
#   3. 导出 create_x_agent 创建函数
#   4. 在这里注册到 SUB_AGENTS_REGISTRY
#   5. 在 create_supervisor_tools() 中添加包装工具
SUB_AGENTS_REGISTRY = {
    "knowledge_agent": {
        "info": KNOWLEDGE_AGENT_INFO
    },
    "email_agent": {
        "info": EMAIL_AGENT_INFO
    },
    "search_agent": {
        "info": SEARCH_AGENT_INFO
    }
}


def _run_async_from_tool(coro):
    """
    在同步工具函数中安全地执行异步协程。

    背景：
        LangChain @tool 装饰的函数是同步的，但 Knowledge Agent
        的 invoke_knowledge_qa 是异步的。这个函数负责桥接同步/异步。

    处理逻辑：
        - 如果当前线程没有运行中的事件循环 → 直接用 asyncio.run() 执行
        - 如果已经在事件循环中（Nest 场景）→ 用线程池执行 asyncio.run()

    参数
    ----
    coro : coroutine
        要执行的异步协程

    返回
    ----
    Any : 协程的返回值
    """
    try:
        # 尝试获取当前线程的事件循环
        asyncio.get_running_loop()
    except RuntimeError:
        # 没有运行中的事件循环，安全直接执行
        return asyncio.run(coro)

    # 已有事件循环在运行（嵌套调用），用线程池隔离执行
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


def create_supervisor_tools(model):
    """
    创建 Supervisor 可调用的子智能体工具列表。

    这是多智能体系统的核心编排函数：将每个子智能体包装成一个
    LangChain @tool，使 Supervisor LLM 能通过 Function Calling
    自主选择调用哪个子智能体。

    参数
    ----
    model : ChatOpenAI
        默认 LLM 模型实例，传给需要它的子智能体（Email/Search）

    返回
    ----
    list[Callable] : 供 Supervisor 使用的工具函数列表，当前包含：
        - call_knowledge_agent : 本地知识库问答
        - call_email_agent     : 邮件操作
        - call_search_agent    : 联网搜索
    """
    tools = []

    # ================================================================
    # 工具 1: Knowledge Agent — 本地知识库问答
    # ================================================================
    # Knowledge Agent 比较特殊：它不创建独立的 LangGraph 子图，
    # 而是直接复用已有的 knowledge_service（RAG 流水线）。
    # 这样做的好处是可以复用成熟的 RAG 逻辑（混合检索、Rerank、图谱），
    # 缺点是与 Email/Search 的调用方式不一致。
    @tool(
        KNOWLEDGE_AGENT_INFO["name"],
        description=KNOWLEDGE_AGENT_INFO["description"]
    )
    def call_knowledge_agent(query: str, collection: str = "") -> str:
        """
        调用知识库智能体，基于本地文档和知识库回答问题。

        参数
        ----
        query : str
            用户的问题或查询内容
        collection : str, 可选
            指定知识库名称；留空则使用默认行为（跨库检索）

        返回
        ----
        str : 包含答案和来源引用的文本
        """
        print(f"\n[Supervisor] 委派给 Knowledge Agent: {query}")

        # 导入知识库服务（延迟导入避免循环依赖）
        from app.services.knowledge_service import invoke_knowledge_qa

        # 执行 RAG 问答（内部走完整的 Knowledge Agent 流水线）
        result = _run_async_from_tool(invoke_knowledge_qa(
            query=query,
            model_name=settings.default_model,
            session_id=f"supervisor_knowledge_{uuid.uuid4().hex}",
            collection=collection or None,
        ))

        # 提取答案和来源
        answer = result.get("answer", "")
        sources = result.get("sources") or []

        # 如果有来源，拼接到答案末尾（最多展示 5 条）
        if sources:
            source_lines = []
            for i, source in enumerate(sources[:5], 1):
                file_name = source.get("file_name") or source.get("source") or "unknown"
                chunk_index = source.get("chunk_index")
                suffix = f" chunk={chunk_index}" if chunk_index is not None else ""
                source_lines.append(f"{i}. {file_name}{suffix}")
            answer = f"{answer}\n\n来源：\n" + "\n".join(source_lines)

        print(f"[Supervisor] Knowledge Agent 完成: {answer[:100]}...")
        return answer

    tools.append(call_knowledge_agent)

    # ================================================================
    # 工具 2: Email Agent — 邮件操作（真实 SMTP 发信）
    # ================================================================
    # Email Agent 有独立的 LangGraph 子图，内部绑定了 send_email 等工具。
    # Supervisor 通过 invoke 子图来触发邮件操作，
    # Email Agent 内部的 LLM 再通过 Function Calling 决定调用哪个邮件工具。
    email_agent = create_email_agent(model)

    @tool(
        EMAIL_AGENT_INFO["name"],
        description=EMAIL_AGENT_INFO["description"]
    )
    def call_email_agent(query: str) -> str:
        """
        调用邮件智能体处理邮件相关任务。

        支持的操作：
            - 发送邮件（使用系统统一 SMTP 邮箱发件）
            - 用户只需提供收件人、主题、正文

        参数
        ----
        query : str
            邮件任务的描述，例如：
            "把这份总结发送给 xxx@163.com，主题为项目摘要"

        返回
        ----
        str : 邮件操作的结果描述
        """
        print(f"\n[Supervisor] 委派给 Email Agent: {query}")

        # invoke 子图：将用户请求作为 HumanMessage 传入 Email Agent
        result = email_agent.invoke({
            "messages": [HumanMessage(content=query)]
        })

        # 提取子智能体的最终回复（最后一条 AI 消息）
        final_message = result["messages"][-1].content
        print(f"[Supervisor] Email Agent 完成: {final_message[:100]}...")

        return final_message

    tools.append(call_email_agent)

    # ================================================================
    # 工具 3: Search Agent — 联网搜索（Tavily Search API）
    # ================================================================
    # Search Agent 有独立的 LangGraph 子图，内部绑定了 search_web 工具。
    # Supervisor 通过 invoke 子图来触发搜索，
    # Search Agent 内部的 LLM 再通过 Function Calling 决定调用搜索工具。
    search_agent = create_search_agent(model)

    @tool(
        SEARCH_AGENT_INFO["name"],
        description=SEARCH_AGENT_INFO["description"]
    )
    def call_search_agent(query: str) -> str:
        """
        调用搜索智能体处理联网搜索和信息检索任务。

        支持的操作：
            - 联网搜索最新信息、新闻、天气等
            - 使用 Tavily Search API 获取实时数据

        参数
        ----
        query : str
            搜索查询，例如：
            "帮我搜索今天AI行业的重要新闻"

        返回
        ----
        str : 搜索结果摘要
        """
        print(f"\n[Supervisor] 委派给 Search Agent: {query}")

        # invoke 子图：将用户请求作为 HumanMessage 传入 Search Agent
        result = search_agent.invoke({
            "messages": [HumanMessage(content=query)]
        })

        # 提取子智能体的最终回复（最后一条 AI 消息）
        final_message = result["messages"][-1].content
        print(f"[Supervisor] Search Agent 完成: {final_message[:100]}...")

        return final_message

    tools.append(call_search_agent)

    return tools


def format_agents_info() -> str:
    """
    格式化子智能体信息，用于注入 Supervisor 的 System Prompt。

    从 SUB_AGENTS_REGISTRY 中读取每个子智能体的元信息，
    生成结构化的文本描述，帮助 Supervisor LLM 理解：
        - 有哪些子智能体可用
        - 每个子智能体能做什么
        - 何时应该调用哪个子智能体

    返回
    ----
    str : 格式化的子智能体信息文本

    示例输出：
        Available Specialized Agents:

        **知识库智能体** (knowledge_agent)
          Description: 专门处理本地知识库、文档问答...
          Capabilities:
            - 知识库问答
            - 文档内容检索
            ...
          Keywords: 知识库, 文档, PDF, ...
    """
    info = "Available Specialized Agents:\n\n"

    for agent_data in SUB_AGENTS_REGISTRY.values():
        agent_info = agent_data["info"]
        info += f"**{agent_info['display_name']}** ({agent_info['name']})\n"
        info += f"  Description: {agent_info['description']}\n"
        info += f"  Capabilities:\n"
        for cap in agent_info['capabilities']:
            info += f"    - {cap}\n"
        info += f"  Keywords: {', '.join(agent_info['keywords'])}\n\n"

    return info


def get_supervisor_system_prompt() -> str:
    """
    获取 Supervisor 的 System Prompt。

    从 prompts.py 中读取模板，注入子智能体信息，
    生成完整的 System Prompt 文本。

    注入流程：
        SUPERVISOR_SYSTEM_PROMPT 模板中有一个 {agents_info} 占位符
        → format_agents_info() 生成子智能体描述
        → .format(agents_info=...) 替换占位符
        → 返回完整 Prompt

    返回
    ----
    str : 注入子智能体信息后的完整 System Prompt
    """
    agents_info = format_agents_info()
    return SUPERVISOR_SYSTEM_PROMPT.format(agents_info=agents_info)
