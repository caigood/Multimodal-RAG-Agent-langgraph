# -*- coding: utf-8 -*-
"""
Select / Rerank Node（single_doc 和 multi_doc 路径共用）

无 rerank（rerank_enabled=False）：
  按 RRF score 降序取 llm_context_top_k，与原逻辑完全一致。

有 rerank（rerank_enabled=True，非多模态知识库）：
  调用系统配置的 Rerank 模型对候选 chunks 重排，
  single_doc 取 single_doc_rerank_top_k，multi_doc 取 multi_doc_rerank_top_k。
  rerank 失败时自动降级为原始 score 排序。

数据来源按 query_type 确定：
  multi_doc 路径使用 filtered_chunks（包括明确的空列表）。
  single_doc 路径始终使用 merged_chunks。
"""

from typing import Dict, Any
from dataclasses import replace

from ..state import KnowledgeAgentState


def _score(chunk) -> float:
    if isinstance(chunk, dict):
        return chunk.get("rerank_score") or chunk.get("score", 0.0) or 0.0
    return getattr(chunk, "rerank_score", None) or getattr(chunk, "score", 0.0) or 0.0


def select_top_k_chunks(state: KnowledgeAgentState) -> Dict[str, Any]:
    """
    统一的截断 / rerank 节点，single_doc 和 multi_doc 路径共用。
    """
    config = state["config"]
    is_multi_doc = state.get("query_type") == "multi_doc"
    if is_multi_doc:
        candidates = state.get("filtered_chunks")
        if candidates is None:
            candidates = []
    else:
        candidates = state.get("merged_chunks") or []

    is_multimodal = getattr(config, "kb_type", "standard") == "multimodal"
    rerank_enabled = getattr(config, "rerank_enabled", False) and not is_multimodal

    print(f"\n[SelectTopK] candidates={len(candidates)}, rerank={rerank_enabled}, multi_doc={is_multi_doc}")

    try:
        if rerank_enabled and candidates:
            # ── Rerank 路径 ──────────────────────────────────────────────────
            top_k = (
                getattr(config, "multi_doc_rerank_top_k", 10)
                if is_multi_doc
                else getattr(config, "single_doc_rerank_top_k", 5)
            )
            query = state.get("search_query", "")
            from app.core.config import settings
            from app.services.rerank_service import get_rerank_service
            top_chunks = get_rerank_service().rerank(
                query=query,
                chunks=candidates,
                top_n=top_k,
            )
            method = f"rerank({settings.rerank_model})"
        else:
            # ── 原始 score 排序路径 ──────────────────────────────────────────
            top_k = getattr(config, "llm_context_top_k", 10)
            top_chunks = sorted(candidates, key=_score, reverse=True)[:top_k]
            method = "score_sort"

        print(f"[SelectTopK] method={method}, selected={len(top_chunks)}")

        metrics = replace(state["metrics"], chunks_after_rerank=len(top_chunks))

        return {
            "reranked_chunks": top_chunks,
            "merged_chunks": top_chunks,
            "metrics": metrics,
        }

    except Exception as e:
        print(f"[SelectTopK] Error: {e}")
        return {"all_errors": [f"select_top_k failed: {e}"]}
