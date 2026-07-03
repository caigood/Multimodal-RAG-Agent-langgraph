# -*- coding: utf-8 -*-
"""
Email Agent 图定义

Email Agent 是一个专门处理邮件任务的子智能体。
它本身也是一个 LangGraph 图，内部仍然采用 ReAct / Tool Calling 模式：

用户/上级 Supervisor 的邮件任务
    ↓
agent 节点调用大模型理解任务
    ↓
如果需要真实发送邮件，则生成 send_email 工具调用
    ↓
tools 节点执行 SMTP 发信工具
    ↓
工具结果返回 agent 节点，由大模型生成最终回复

注意：
这里的 Email Agent 并不是简单的函数调用，
而是“子智能体 + 工具节点”的独立小图。
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from .state import EmailAgentState
from .nodes import call_email_model, should_continue
from .services.email_service import get_email_tools


def create_email_agent(model):
    """
    创建并编译 Email Agent 图。

    Args:
        model: 上层传入的大模型实例。Email Agent 会在自己的 agent 节点中，
            将该模型与邮件工具绑定，让模型可以自主决定是否调用 send_email。

    Returns:
        编译后的 Email Agent 图对象，可被 Supervisor 当作子智能体调用。
    """
    print("\n[Graph] Building Email Agent")

    # 获取 Email Agent 可使用的工具列表。
    # 当前只有 send_email 一个真实工具。
    # 这里是在创建图时获取一次工具列表，后续节点复用，避免每轮调用重复创建。
    email_tools = get_email_tools()

    # 定义 agent 节点的执行函数。
    # 每次进入 agent 节点时，都会调用 call_email_model：
    # 1. 将模型与 email_tools 绑定；
    # 2. 让大模型读取当前 messages；
    # 3. 判断是直接回复，还是生成 tool_calls 调用 send_email。
    def model_caller(state: EmailAgentState):
        return call_email_model(state, model, email_tools)

    # 创建 Email Agent 的状态图。
    # EmailAgentState 中主要维护 messages 列表，
    # 用于保存用户任务、AI 工具调用请求、工具执行结果和最终回复。
    builder = StateGraph(EmailAgentState)

    # agent 节点：负责调用大模型进行推理和工具调用决策。
    builder.add_node("agent", model_caller)

    # tools 节点：LangGraph 内置 ToolNode。
    # 它会读取上一条 AIMessage 中的 tool_calls，
    # 根据工具名匹配 email_tools 中的真实 Python 工具函数并执行。
    builder.add_node("tools", ToolNode(email_tools))

    # 图入口：每次 Email Agent 被调用时，先进入 agent 节点。
    builder.add_edge(START, "agent")

    # 条件路由：agent 节点执行后，检查最新 AIMessage 是否包含 tool_calls。
    # - 如果有 tool_calls，进入 tools 节点执行邮件工具；
    # - 如果没有 tool_calls，说明模型已经直接给出最终回答，图执行结束。
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "__end__": END}
    )

    # 工具执行完成后，需要回到 agent 节点。
    # 这样大模型可以读取 send_email 的执行结果，
    # 再组织成用户能看懂的最终回复。
    builder.add_edge("tools", "agent")

    # 编译图，返回可 invoke 的 Email Agent。
    graph = builder.compile()

    print("[Graph] Email Agent created successfully")

    return graph