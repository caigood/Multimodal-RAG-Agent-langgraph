# -*- coding: utf-8 -*-
"""Analyze the current query for conversational rewriting and retrieval scope."""

import json
import logging
import re
from typing import Any, Dict, List

from dashscope import Generation

from ..state import KnowledgeAgentState
from app.core.config import settings
from app.core.prompts import KNOWLEDGE_QUERY_ANALYSIS_SYSTEM

logger = logging.getLogger(__name__)

_MAX_QUERY_LENGTH = 2000
_REFERENCE_PATTERN = re.compile(
    r"(?:它|这个|那个|上述|前面|后者|其中|这种|该内容|该文件|该文档|这个文档|那个文档|这份文件)"
)
_ELLIPTICAL_PATTERN = re.compile(
    r"^(?:为什么|怎么做|怎么办|有什么风险|有何风险|还有呢|区别呢|有什么区别|然后呢|接下来呢)[？?。！!]*$"
)
_EXPLICIT_SINGLE_DOC_PATTERN = re.compile(
    r"(?:这个文档|该文档|本文|这个文件|该文件|这份文件|文件名|"
    r"《[^》]{1,100}》|[\w\-\u4e00-\u9fff]{1,100}\.(?:pdf|docx?|xlsx?|pptx?|txt|md))",
    re.IGNORECASE,
)
_MULTI_DOC_PATTERN = re.compile(
    r"(?:比较|对比|综合|汇总|多个文档|多份文档|各文档|所有文档|全部文档|跨文档|多个来源|知识库)"
)


def _recent_history(state: KnowledgeAgentState) -> List[Any]:
    messages = state.get("messages") or []
    original_query = state["original_query"].strip()
    last = messages[-1] if messages else None
    is_current_human = (
        getattr(last, "type", "") == "human"
        and str(getattr(last, "content", "") or "").strip() == original_query
    )
    history = messages[:-1] if is_current_human else messages
    memory_turns = max(0, state["config"].memory_turns)
    recent = history[-(2 * memory_turns):] if memory_turns else []
    return [message for message in recent if str(getattr(message, "content", "") or "").strip()]


def _requires_llm(query: str, history: List[Any]) -> bool:
    if not history:
        return False
    return bool(_REFERENCE_PATTERN.search(query) or _ELLIPTICAL_PATTERN.fullmatch(query.strip()))


def _local_query_type(query: str, force_multi_doc: Any) -> str:
    if force_multi_doc is True:
        return "multi_doc"
    if _MULTI_DOC_PATTERN.search(query):
        return "multi_doc"
    if _EXPLICIT_SINGLE_DOC_PATTERN.search(query):
        return "single_doc"
    return "multi_doc"


def _history_messages(history: List[Any]) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    for message in history:
        message_type = getattr(message, "type", "")
        role = "user" if message_type == "human" else "assistant" if message_type == "ai" else ""
        content = getattr(message, "content", "")
        if role and isinstance(content, str) and content.strip():
            result.append({"role": role, "content": content})
    return result


def _extract_json_object(content: str) -> Dict[str, Any]:
    cleaned = content.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(cleaned)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                data, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    raise ValueError("模型输出中没有有效 JSON 对象")


def _parse_analysis(content: str, original_query: str, force_multi_doc: Any) -> Dict[str, Any]:
    data = _extract_json_object(content)
    expected = {"needs_rewrite", "standalone_query", "search_query", "query_type"}
    if set(data) != expected:
        raise ValueError("模型输出字段不符合约定")
    if not isinstance(data["needs_rewrite"], bool):
        raise ValueError("needs_rewrite 必须是布尔值")
    for field in ("standalone_query", "search_query"):
        value = data[field]
        if not isinstance(value, str) or not (2 <= len(value.strip()) <= _MAX_QUERY_LENGTH):
            raise ValueError(f"{field} 必须是有效字符串")
        data[field] = value.strip()
    if data["query_type"] not in {"single_doc", "multi_doc"}:
        raise ValueError("query_type 枚举值非法")
    if force_multi_doc is True:
        data["query_type"] = "multi_doc"
    data["needs_rewrite"] = data["standalone_query"] != original_query
    return data


def _fallback(original_query: str, force_multi_doc: Any, warning: str) -> Dict[str, Any]:
    return {
        "needs_rewrite": False,
        "standalone_query": original_query,
        "search_query": original_query,
        "query_type": _local_query_type(original_query, force_multi_doc),
        "all_warnings": [warning],
    }


def analyze_query(state: KnowledgeAgentState) -> Dict[str, Any]:
    """Produce standalone/search queries and retrieval scope with at most one LLM call."""
    original_query = state["original_query"].strip()
    force_multi_doc = state["config"].force_multi_doc
    history = _recent_history(state)

    if not _requires_llm(original_query, history):
        query_type = _local_query_type(original_query, force_multi_doc)
        logger.info("[AnalyzeQuery] 本地快速通道: query_type=%s", query_type)
        return {
            "needs_rewrite": False,
            "standalone_query": original_query,
            "search_query": original_query,
            "query_type": query_type,
        }

    messages = [{"role": "system", "content": KNOWLEDGE_QUERY_ANALYSIS_SYSTEM}]
    messages.extend(_history_messages(history))
    messages.append({"role": "user", "content": f"当前问题：{original_query}"})
    try:
        response = Generation.call(
            api_key=settings.dashscope_api_key,
            model=settings.llm_clean_model,
            messages=messages,
            result_format="message",
        )
        if response.status_code != 200:
            raise RuntimeError(f"DashScope status {response.status_code}")
        content = response.output.choices[0].message.get("content", "")
        result = _parse_analysis(content, original_query, force_multi_doc)
        logger.info(
            "[AnalyzeQuery] LLM 分析完成: rewrite=%s, query_type=%s",
            result["needs_rewrite"], result["query_type"],
        )
        return result
    except Exception as exc:
        logger.warning("[AnalyzeQuery] 分析失败，使用安全降级: %s", exc)
        return _fallback(original_query, force_multi_doc, f"问题分析失败，已使用原始问题和本地范围判断: {exc}")
