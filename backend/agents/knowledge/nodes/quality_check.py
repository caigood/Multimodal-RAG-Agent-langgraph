# -*- coding: utf-8 -*-
"""检索质量门控、查询重写与统一降级节点。"""

import logging
from typing import Any, Dict, List

from dashscope import Generation
from langchain_core.messages import AIMessage

from ..state import KnowledgeAgentState
from app.core.config import settings

logger = logging.getLogger(__name__)


def _value(chunk: Any, key: str, default=None):
    return chunk.get(key, default) if isinstance(chunk, dict) else getattr(chunk, key, default)


def _deduplicated_evidence(state: KnowledgeAgentState) -> List[Any]:
    evidence, seen = [], set()
    vector_chunks = state.get("reranked_chunks") or state.get("merged_chunks") or []
    for chunk in vector_chunks + (state.get("kg_graph_chunks") or []):
        content = str(_value(chunk, "content", "") or "").strip()
        identity = str(_value(chunk, "chunk_id", None) or _value(chunk, "id", None) or content)
        if content and identity not in seen:
            seen.add(identity)
            evidence.append(chunk)
    return evidence


def check_retrieval_quality(state: KnowledgeAgentState) -> Dict[str, Any]:
    """检查证据是否足够；只有 rerank 分数存在时才提供可选相关度。"""
    vector_chunks = state.get("reranked_chunks") or state.get("merged_chunks") or []
    evidence = _deduplicated_evidence(state)
    config = state["config"]
    issues: List[str] = []
    rerank_scores = [
        float(_value(chunk, "rerank_score"))
        for chunk in vector_chunks
        if _value(chunk, "rerank_score") is not None
    ]

    score = None
    if not evidence:
        passed = False
        issues.append("未检索到可用证据（向量与图谱均为空）")
    elif not getattr(config, "retrieval_quality_enabled", True):
        passed = True
    elif rerank_scores:
        score = sum(rerank_scores[:3]) / min(len(rerank_scores), 3)
        passed = score >= config.retrieval_quality_threshold
        if not passed:
            issues.append(f"向量 rerank 相关性 {score:.3f} 低于阈值 {config.retrieval_quality_threshold:.3f}")
    else:
        # RRF 与图谱结果没有统一的 0-1 相关度；存在证据即可生成，但不伪造分数。
        passed = True

    reason = "; ".join(issues) or None
    return {
        "retrieval_quality_passed": passed,
        "retrieval_quality_score": score,
        "retrieval_quality_issues": issues,
        "retrieval_retry_reason": reason,
    }


def rewrite_retrieval_query(state: KnowledgeAgentState) -> Dict[str, Any]:
    """按失败原因生成二次召回查询，并清空上一轮检索和输出。"""
    original = state["original_query"]
    current = state.get("search_query") or original
    reason = state.get("retrieval_retry_reason") or "; ".join(state.get("retrieval_quality_issues") or [])
    fallback_query = f"{current} {original}" if current != original else f"{original} 相关事实 关键内容"
    try:
        response = Generation.call(
            api_key=settings.dashscope_api_key,
            model=settings.llm_clean_model,
            messages=[
                {"role": "system", "content": "你是检索查询优化器。只输出一条更适合知识库召回的查询，不回答问题。"},
                {"role": "user", "content": f"原问题：{original}\n当前查询：{current}\n失败原因：{reason}\n请扩展同义词并保留核心实体。"},
            ],
            result_format="message",
        )
        rewritten = response.output.choices[0].message.get("content", "").strip() if response.status_code == 200 else ""
    except Exception as exc:
        logger.warning("二次检索查询改写失败，使用安全扩展: %s", exc)
        rewritten = ""
    rewritten = rewritten if len(rewritten) >= 2 else fallback_query
    return {
        "search_query": rewritten,
        "retrieval_retry_count": state.get("retrieval_retry_count", 0) + 1,
        "merged_chunks": [], "filtered_chunks": [], "reranked_chunks": [], "kg_graph_chunks": [],
        "retrieval_quality_passed": False,
        "retrieval_quality_score": None, "retrieval_quality_issues": [],
        "retrieval_retry_reason": None,
        "answer": "", "sources": [], "confidence": None, "image_map": None,
    }


def fallback_answer(state: KnowledgeAgentState) -> Dict[str, Any]:
    """检索质量失败时返回统一提示。"""
    message = state["config"].fallback_message or getattr(settings, "fallback_message", "抱歉，我无法找到相关信息。")
    return {
        "answer": message,
        "confidence": None,
        "sources": [],
        "image_map": None,
        "messages": [AIMessage(content=message)],
    }
