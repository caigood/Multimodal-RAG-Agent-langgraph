# -*- coding: utf-8 -*-
"""
Knowledge Base QA Agent State Definition
只保留当前 LangGraph 工作流实际读写的状态字段。
"""

from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional, Annotated
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import operator

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from app.core.config import settings


class RetrievalStrategy(str, Enum):
    """检索策略"""
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"


@dataclass
class RAGConfig:
    """RAG pipeline configuration"""
    model: str = settings.default_model

    # Retrieval settings
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    vector_score_threshold: float = 0.0
    llm_context_top_k: int = 10

    # Rerank 配置
    rerank_enabled: bool = False
    single_doc_rerank_top_k: int = 5
    multi_doc_rerank_top_k: int = 10
    rrf_k: int = 60

    # Multi-doc retrieval
    multi_doc_top_k: int = 20
    multi_doc_group_size: int = 3
    strict_group_size: bool = False

    # Single-doc retrieval
    single_doc_top_k: int = 20

    # User overrides
    force_multi_doc: Optional[bool] = None
    keyword_filter: Optional[str] = None

    # 多模态
    kb_type: str = "standard"
    query_image_url: Optional[str] = None
    image_vector_dim: int = 1024

    # Memory settings
    memory_turns: int = 2

    # Knowledge graph（Neo4j）
    kg_enabled: bool = True
    kg_graph_id: Optional[str] = None
    kg_top_k: int = 5
    kg_timeout_seconds: float = 2.0

    # 检索质量门控
    retrieval_quality_enabled: bool = True
    retrieval_quality_threshold: float = 0.35
    max_retrieval_retries: int = 1
    fallback_message: str = "抱歉，我无法找到相关信息。"

    # Knowledge base
    collection: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance and usage metrics"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
    retrieval_duration_ms: float = 0.0
    filter_duration_ms: float = 0.0
    rerank_duration_ms: float = 0.0
    generation_duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    total_chunks_retrieved: int = 0
    chunks_after_filter: int = 0
    chunks_after_rerank: int = 0
    confidence_score: float = 0.0
    estimated_cost: float = 0.0


class KnowledgeAgentState(TypedDict):
    """Knowledge Agent 当前工作流实际使用的状态。"""

    # Conversation Memory
    messages: Annotated[List[BaseMessage], add_messages]

    # Input / query analysis
    original_query: str
    needs_rewrite: bool
    standalone_query: str
    search_query: str

    # Configuration
    config: RAGConfig

    # Query Processing
    query_type: Optional[str]
    retrieval_strategy: Optional[RetrievalStrategy]

    # Knowledge graph retrieval
    kg_graph_chunks: List[Dict[str, Any]]

    # Retrieval / filtering / rerank
    merged_chunks: List[Dict[str, Any]]
    filtered_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]

    # Generation output
    sources: List[Dict[str, Any]]
    answer: str
    confidence: Optional[float]
    image_map: Optional[Dict[str, str]]

    # Retrieval Quality Control / Retry
    retrieval_quality_passed: bool
    retrieval_quality_score: Optional[float]
    retrieval_quality_issues: List[str]
    retrieval_retry_reason: Optional[str]
    retrieval_retry_count: int

    # Monitoring
    metrics: PerformanceMetrics

    # Accumulated Data (with Reducers)
    all_errors: Annotated[List[str], operator.add]
    all_warnings: Annotated[List[str], operator.add]


def create_initial_state(
    query: str,
    config: Optional[RAGConfig] = None,
    messages: Optional[List[BaseMessage]] = None,
) -> KnowledgeAgentState:
    """创建工作流初始状态；仅注入查询、配置和可选普通对话历史，其余流水线字段归零。"""
    from langchain_core.messages import HumanMessage

    return {
        "messages": messages or [HumanMessage(content=query)],
        "original_query": query,
        "needs_rewrite": False,
        "standalone_query": query,
        "search_query": query,
        "config": config or RAGConfig(vector_score_threshold=settings.vector_score_threshold),
        "query_type": None,
        "retrieval_strategy": None,
        "kg_graph_chunks": [],
        "merged_chunks": [],
        "filtered_chunks": [],
        "reranked_chunks": [],
        "sources": [],
        "answer": "",
        "confidence": None,
        "image_map": None,
        "retrieval_quality_passed": False,
        "retrieval_quality_score": None,
        "retrieval_quality_issues": [],
        "retrieval_retry_reason": None,
        "retrieval_retry_count": 0,
        "metrics": PerformanceMetrics(start_time=datetime.now()),
        "all_errors": [],
        "all_warnings": [],
    }
