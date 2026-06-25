# -*- coding: utf-8 -*-
"""
Search Service
Web search and information retrieval tools (Tavily)
"""

import os
import requests
from langchain_core.tools import tool

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_API_URL = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")


def _call_tavily(query: str, max_results: int = 5) -> dict:
    """Call Tavily search API"""
    if not TAVILY_API_KEY:
        return {
            "error": "TAVILY_API_KEY 未配置，请在 .env 文件中设置",
            "results": []
        }

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False,
    }

    try:
        resp = requests.post(TAVILY_API_URL, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": f"搜索请求失败: {e}", "results": []}


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for real-time information using Tavily

    Args:
        query: Search query string
        max_results: Maximum number of results to return (1-10)

    Returns:
        Search results with titles, URLs, and content summaries
    """
    print(f"[Search Agent] Tavily searching: {query}")

    data = _call_tavily(query, max_results)

    if "error" in data:
        return f"搜索失败: {data['error']}"

    answer = data.get("answer", "")
    results = data.get("results", [])

    lines = [f"搜索: {query}\n"]
    if answer:
        lines.append(f"【总结】{answer}\n")

    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "无标题")
        url = r.get("url", "")
        content = r.get("content", "")[:300]
        lines.append(f"{i}. {title}\n   链接: {url}\n   摘要: {content}...\n")

    return "\n".join(lines)


def get_search_tools():
    """
    Get list of search tools

    Returns:
        List of search tool functions
    """
    return [search_web]


# Agent metadata for supervisor
SEARCH_AGENT_INFO = {
    "name": "search_agent",
    "display_name": "搜索智能体",
    "description": "使用 Tavily 处理网络搜索、新闻、天气、实时信息等外部信息检索任务",
    "capabilities": [
        "网络搜索",
        "新闻搜索",
        "天气搜索",
        "实时信息检索"
    ],
    "keywords": ["搜索", "search", "查询", "天气", "新闻", "最新", "实时", "信息", "weather", "news"]
}
