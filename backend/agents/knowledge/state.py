# -*- coding: utf-8 -*-
"""
Knowledge Base QA Agent State Definition
只保留当前 LangGraph 工作流实际读写的状态字段。
"""

from typing_extensions import TypedDict, NotRequired
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


class AnswerQuality(str, Enum):
    """答案质量等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RAGConfig:
    """RAG pipeline configuration"""
    model: str = settings.default_model

    # Retrieval settings
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    vector_score_threshold: float = 0.0
    llm_context_top_k: int = 10

    # Qwen3-Rerank 配置
    rerank_enabled: bool = False
    rerank_model_name: str = "qwen3-rerank"
    single_doc_rerank_top_k: int = 5
    multi_doc_rerank_top_k: int = 10
    ranker: str = "RRF"
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

    # Quality control
    min_confidence_threshold: float = 0.6
    enable_fallback: bool = True
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
    answer_quality: Optional[AnswerQuality] = None
    confidence_score: float = 0.0
    estimated_cost: float = 0.0


class KnowledgeAgentState(TypedDict):
    """Knowledge Agent 当前工作流实际使用的状态。"""

    # Conversation Memory
    messages: Annotated[List[BaseMessage], add_messages]

    # Input
    query: str
    original_query: str
    rewritten_query: Optional[str]

    # Configuration
    config: RAGConfig

    # Query Processing
    query_type: Optional[str]
    retrieval_strategy: Optional[RetrievalStrategy]
    retrieval_strategy_reason: Optional[str]

    # Knowledge graph retrieval
    kg_graph_chunks: List[Dict[str, Any]]

    # Retrieval / filtering / rerank
    merged_chunks: List[Dict[str, Any]]
    retrieval_strategy_used: Optional[RetrievalStrategy]
    total_candidates: int
    filtered_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]

    # Generation output
    context: str
    sources: List[Dict[str, Any]]
    answer: str
    confidence: float
    image_map: Optional[Dict[str, str]]
    tools_used: NotRequired[List[str]]

    # SSE 流式：在 interrupt_before generate 之后由服务层写入
    precomputed_answer: NotRequired[Optional[str]]

    # Quality Control
    answer_quality: Optional[AnswerQuality]
    quality_passed: bool
    quality_issues: List[str]
    used_fallback: bool
    fallback_reason: Optional[str]

    # Monitoring
    metrics: PerformanceMetrics

    # Error Handling
    error: Optional[str]
    error_stage: Optional[str]

    # Accumulated Data (with Reducers)
    all_errors: Annotated[List[str], operator.add]
    all_warnings: Annotated[List[str], operator.add]
    processing_log: Annotated[List[Dict[str, Any]], operator.add]


def create_initial_state(
    query: str,
    user_id: str,
    session_id: str,
    config: Optional[RAGConfig] = None,
    messages: Optional[List[BaseMessage]] = None,
) -> KnowledgeAgentState:
    """Create initial state for RAG pipeline."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": messages or [HumanMessage(content=query)],
        "query": query,
        "original_query": query,
        "rewritten_query": None,
        "config": config or RAGConfig(vector_score_threshold=settings.vector_score_threshold),
        "query_type": None,
        "retrieval_strategy": None,
        "retrieval_strategy_reason": None,
        "kg_graph_chunks": [],
        "merged_chunks": [],
        "retrieval_strategy_used": None,
        "total_candidates": 0,
        "filtered_chunks": [],
        "reranked_chunks": [],
        "context": "",
        "sources": [],
        "answer": "",
        "confidence": 0.0,
        "image_map": None,
        "answer_quality": None,
        "quality_passed": False,
        "quality_issues": [],
        "used_fallback": False,
        "fallback_reason": None,
        "metrics": PerformanceMetrics(start_time=datetime.now()),
        "error": None,
        "error_stage": None,
        "all_errors": [],
        "all_warnings": [],
        "processing_log": [],
    }
