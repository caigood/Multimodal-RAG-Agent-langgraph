# -*- coding: utf-8 -*-
"""
Supervisor Coordinator Service
Manages sub-agents and provides coordination logic
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


# Registry of available sub-agents
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


SUB_AGENTS_REGISTRY = {
    "knowledge_agent": {
        "info": KNOWLEDGE_AGENT_INFO,
        "creator": None
    },
    "email_agent": {
        "info": EMAIL_AGENT_INFO,
        "creator": create_email_agent
    },
    "search_agent": {
        "info": SEARCH_AGENT_INFO,
        "creator": create_search_agent
    }
}


def _run_async_from_tool(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


def create_supervisor_tools(model):
    """
    Create tools that wrap sub-agents for the supervisor
    
    Args:
        model: LLM model instance to pass to sub-agents
        
    Returns:
        List of tool functions
    """
    tools = []

    @tool(
        KNOWLEDGE_AGENT_INFO["name"],
        description=KNOWLEDGE_AGENT_INFO["description"]
    )
    def call_knowledge_agent(query: str, collection: str = "") -> str:
        """
        Call the knowledge agent to answer questions from local knowledge bases and documents.

        Args:
            query: The document or knowledge-base question.
            collection: Optional knowledge base name. Leave empty to use the default behavior.

        Returns:
            Answer from knowledge agent with source information when available.
        """
        print(f"\n[Supervisor] Delegating to Knowledge Agent: {query}")

        from app.services.knowledge_service import invoke_knowledge_qa

        result = _run_async_from_tool(invoke_knowledge_qa(
            query=query,
            model_name=settings.default_model,
            session_id=f"supervisor_knowledge_{uuid.uuid4().hex}",
            collection=collection or None,
        ))

        answer = result.get("answer", "")
        sources = result.get("sources") or []
        if sources:
            source_lines = []
            for i, source in enumerate(sources[:5], 1):
                file_name = source.get("file_name") or source.get("source") or "unknown"
                chunk_index = source.get("chunk_index")
                suffix = f" chunk={chunk_index}" if chunk_index is not None else ""
                source_lines.append(f"{i}. {file_name}{suffix}")
            answer = f"{answer}\n\n来源：\n" + "\n".join(source_lines)

        print(f"[Supervisor] Knowledge Agent completed: {answer[:100]}...")
        return answer

    tools.append(call_knowledge_agent)
    
    # Create email agent tool
    email_agent = create_email_agent(model)
    
    @tool(
        EMAIL_AGENT_INFO["name"],
        description=EMAIL_AGENT_INFO["description"]
    )
    def call_email_agent(query: str) -> str:
        """
        Call the email agent to handle email-related tasks
        
        Args:
            query: The email-related task or question
            
        Returns:
            Result from email agent
        """
        print(f"\n[Supervisor] Delegating to Email Agent: {query}")
        
        result = email_agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        # Extract final response
        final_message = result["messages"][-1].content
        print(f"[Supervisor] Email Agent completed: {final_message[:100]}...")
        
        return final_message
    
    tools.append(call_email_agent)
    
    # Create search agent tool
    search_agent = create_search_agent(model)
    
    @tool(
        SEARCH_AGENT_INFO["name"],
        description=SEARCH_AGENT_INFO["description"]
    )
    def call_search_agent(query: str) -> str:
        """
        Call the search agent to handle search and information retrieval tasks
        
        Args:
            query: The search query or information request
            
        Returns:
            Result from search agent
        """
        print(f"\n[Supervisor] Delegating to Search Agent: {query}")
        
        result = search_agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        # Extract final response
        final_message = result["messages"][-1].content
        print(f"[Supervisor] Search Agent completed: {final_message[:100]}...")
        
        return final_message
    
    tools.append(call_search_agent)
    
    return tools


def format_agents_info() -> str:
    """
    Format sub-agents information for supervisor's system prompt
    
    Returns:
        Formatted string describing available agents
    """
    info = "Available Specialized Agents:\n\n"
    
    for agent_name, agent_data in SUB_AGENTS_REGISTRY.items():
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
    Get system prompt for supervisor agent

    Returns:
        System prompt string
    """
    agents_info = format_agents_info()
    return SUPERVISOR_SYSTEM_PROMPT.format(agents_info=agents_info)