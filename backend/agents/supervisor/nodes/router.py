# -*- coding: utf-8 -*-
"""
Supervisor 路由节点

这个模块只负责一件事：
根据 Supervisor 主智能体最新一次大模型返回结果，判断下一步应该：
1. 进入 tools 节点执行工具/子智能体；
2. 还是直接结束当前图执行。

注意：
这里不负责真正执行工具，也不负责匹配具体工具函数。
真正的工具执行由 LangGraph 的 ToolNode 完成。
"""

from typing import Literal
from langchain_core.messages import AIMessage
from ..state import SupervisorState


def should_continue(state: SupervisorState) -> Literal["tools", "__end__"]:
    """
    条件边路由函数：判断 Supervisor 节点执行后，下一步走向哪里。

    LangGraph 会在 supervisor 节点执行完成后调用该函数。
    此时 state["messages"] 的最后一条消息通常是主智能体调用大模型后的 AIMessage。

    如果这条 AIMessage 中包含 tool_calls，说明大模型希望调用某个工具，
    例如 knowledge_agent / search_agent / email_agent，
    那么这里返回 "tools"，让 LangGraph 路由到 ToolNode。

    如果没有 tool_calls，说明主智能体已经给出了最终回答，
    那么这里返回 "__end__"，结束当前图执行。

    Args:
        state: 当前 Supervisor 图状态，包含对话消息列表等信息。

    Returns:
        "tools": 继续进入工具节点，由 ToolNode 执行具体工具调用。
        "__end__": 当前轮对话结束，不再调用工具。
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 主智能体调用大模型后，如果模型返回了 tool_calls，
    # 说明它不是要直接回答用户，而是请求调用一个或多个工具/子智能体。
    # 定消息类型且  tool_call 这个参数的内容不为null  防御性编程，其实这里必然是AIMessage
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # 如果没有工具调用请求，则认为主智能体已经完成回答，结束流程。
    return "__end__"
