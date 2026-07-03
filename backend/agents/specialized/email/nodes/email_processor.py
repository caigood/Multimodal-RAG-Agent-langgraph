# -*- coding: utf-8 -*-
"""
Email Agent 处理节点

这里定义 Email Agent 图中的核心节点逻辑：
1. call_email_model：调用大模型，让模型决定是否调用邮件工具；
2. should_continue：检查模型返回中是否包含 tool_calls，并决定下一步路由。

这个文件只负责“模型推理”和“路由判断”，
真正的 SMTP 发信逻辑在 services/email_service.py 中。
"""

from langchain_core.messages import AIMessage
from ..state import EmailAgentState


def call_email_model(state: EmailAgentState, model, email_tools):
    """
    调用 Email Agent 的大模型节点。

    执行流程：
    1. 将邮件工具列表绑定到大模型；
    2. 把当前 messages 交给大模型推理；
    3. 大模型可以选择直接回复，也可以生成 send_email 的 tool_call；
    4. 将本次大模型返回的 AIMessage 追加回图状态。

    Args:
        state: 当前 Email Agent 状态，主要包含 messages 消息列表。
        model: 上层传入的大模型实例。
        email_tools: Email Agent 可用的工具列表，当前主要是 send_email。

    Returns:
        状态增量，包含本次大模型返回的新消息。
    """
    # 将工具绑定到模型后，模型才知道自己可以调用哪些邮件工具。
    # 这里不是强制调用工具，而是把工具 schema 提供给模型，让模型自主判断。
    model_with_tools = model.bind_tools(email_tools)

    # 调用大模型。
    # 如果模型认为需要发邮件，会返回带 tool_calls 的 AIMessage；
    # 如果模型认为不需要工具，会直接返回普通文本回答。
    response = model_with_tools.invoke(state["messages"])

    # LangGraph 会把这里返回的消息合并到 state["messages"] 后面。
    return {"messages": [response]}


def should_continue(state: EmailAgentState):
    """
    Email Agent 的条件路由函数。

    agent 节点调用大模型后，LangGraph 会调用该函数判断下一步：
    - 如果最新 AIMessage 中有 tool_calls，则进入 tools 节点执行 send_email；
    - 如果没有 tool_calls，则说明模型已经完成回复，结束 Email Agent 图。

    Args:
        state: 当前 Email Agent 状态。

    Returns:
        "tools": 进入工具节点；
        "__end__": 结束当前 Email Agent 图。
    """
    last_message = state["messages"][-1]

    # 只有大模型返回的 AIMessage 才可能包含 tool_calls。
    # tool_calls 不为空时，说明模型请求调用邮件工具。
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # 没有工具调用请求，则结束 Email Agent 流程。
    return "__end__"