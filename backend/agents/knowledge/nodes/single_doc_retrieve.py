# -*- coding: utf-8 -*-
"""
Single Document Retrieve Node - 单文档查询
针对特定文档进行检索，返回 single_doc_top_k 条结果（默认 20）
"""

import logging
from dataclasses import replace
from datetime import datetime

from ..state import KnowledgeAgentState, RetrievalStrategy
from ..services.retrieval import get_retrieval_service

logger = logging.getLogger(__name__)


def single_doc_retrieve(state: KnowledgeAgentState) -> KnowledgeAgentState:
    """
    单文档查询节点
    
    功能:
    - 针对特定文档进行检索
    - 使用queryContent参数指定文档
    - 返回 single_doc_top_k 条结果（默认 20）
    - 根据retrieval_strategy使用相应的检索方式
    
    Args:
        state: 当前agent状态
        
    Returns:
        更新后的状态,包含检索到的chunks
    """
    start_time = datetime.now()
    
    try:
        query = state["search_query"]
        retrieval_strategy = state.get("retrieval_strategy", RetrievalStrategy.HYBRID)
        _cfg = state.get("config")
        collection = _cfg.collection if _cfg else None

        # 从 RAGConfig 读检索参数
        rrf_k        = _cfg.rrf_k           if _cfg else 60
        top_k        = _cfg.single_doc_top_k if _cfg else 20
        keyword_filter = _cfg.keyword_filter if _cfg else None

        # 多模态知识库走专用检索节点
        if getattr(_cfg, "kb_type", "standard") == "multimodal":
            from .multimodal_retrieve import multimodal_retrieve
            return multimodal_retrieve(state)

        logger.info(f"[SingleDocRetrieve] 开始单文档检索: {query}")
        logger.info(f"[SingleDocRetrieve] 检索策略: {retrieval_strategy.value}, collection={collection}, top_k={top_k}")
        
        # 获取检索服务
        retrieval_service = get_retrieval_service()
        
        # 根据策略选择检索方式
        if retrieval_strategy == RetrievalStrategy.KEYWORD_ONLY:
            logger.info("[SingleDocRetrieve] 使用关键词检索")
            chunks = retrieval_service.keyword_search(
                query=query,
                top_k=top_k,
                collection=collection,
                keyword_filter=keyword_filter,
                rrf_k=rrf_k,
            )
        elif retrieval_strategy == RetrievalStrategy.HYBRID:
            logger.info("[SingleDocRetrieve] 使用混合检索")
            chunks = retrieval_service.hybrid_search(
                query=query,
                top_k=top_k,
                collection=collection,
                rrf_k=rrf_k,
            )
        else:
            logger.warning(f"[SingleDocRetrieve] 未知策略 {retrieval_strategy}, 使用混合检索")
            chunks = retrieval_service.hybrid_search(
                query=query,
                top_k=top_k,
                collection=collection,
                rrf_k=rrf_k,
            )
        
        duration = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(f"[SingleDocRetrieve] 检索完成 ({duration:.0f}ms): 找到 {len(chunks)} 个结果")
        
        metrics = replace(
            state["metrics"],
            retrieval_duration_ms=duration,
            total_chunks_retrieved=len(chunks),
        )

        return {
            "merged_chunks": chunks,
            "metrics": metrics,
        }

    except Exception as e:
        logger.error(f"[SingleDocRetrieve] 检索失败: {e}", exc_info=True)
        return {
            "merged_chunks": [],
            "all_errors": [f"单文档检索失败: {e}"],
        }
