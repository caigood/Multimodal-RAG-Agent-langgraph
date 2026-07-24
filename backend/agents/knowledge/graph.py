# -*- coding: utf-8 -*-
"""
整体架构::

    START
      ↓
    analyze_query
      ↓
    determine_retrieval_strategy
      ↓
    graph_retrieve
      ├──────────────────────────────┐
      ↓                              ↓
    kg_enabled = true              kg_enabled = false
      ↓                              ↓
    查询 Neo4j                     跳过图谱查询
      ↓                              ↓
    写入 kg_graph_chunks           清空 kg_graph_chunks
      └──────────────────────────────┘
                      ↓
              route_by_query_type
      ┌──────────────────────────────┴──────────────────────────────┐
      ↓                                                             ↓
    query_type = single_doc                                       query_type = multi_doc
      ↓                                                             ↓
    single_doc_retrieve                                           multi_doc_retrieve
      ↓                                                             ↓
    select_top_k_chunks                                           filter_chunks
                                                                    ↓
                                                                  select_top_k_chunks
      └──────────────────────────────┬──────────────────────────────┘
                                     ↓
                             retrieval_quality
      ┌──────────────────────────────┼────────────────────────────────────┐
      ↓                              ↓                                    ↓
    检索通过                   检索失败且仍有重试预算              检索失败且重试预算耗尽
      ↓                              ↓                                    ↓
    generate_answer                rewrite_retrieval_query               fallback_answer
      ↓                              ↓                                    ↓
    finalize_metrics              清空上一轮检索和生成结果               finalize_metrics
      ↓                              ↓                                    ↓
    END                           仅更新 search_query                    END
                                     ↓
                                 graph_retrieve
                                     ↓
                         重新执行图谱开关判断
                                     ↓
                         route_by_query_type
                                     ↓
                     重新执行检索、筛选和 retrieval_quality

状态不变量：
- graph_retrieve 是固定流程节点，但只有 kg_enabled=true 时才查询 Neo4j；关闭时立即跳过。
- 检索失败最多按 config.max_retrieval_retries 重试（当前配置范围 0..2）。
- 二次检索从 rewrite_retrieval_query 明确回到 graph_retrieve，随后重新执行完整检索链路。
- 二次检索会清空上一轮检索结果和生成结果。
- fallback_answer 是检索质量失败后的唯一用户出口。
- 图谱证据与向量证据去重后共同参与检索门控。
- rerank 分数只评价向量检索结果。
- 首次和二次图谱检索都严格服从 config.kg_enabled；关闭后不会因重试自动开启。
"""

from typing import List, Literal, Optional

from langgraph.graph import END, START, StateGraph

from .state import KnowledgeAgentState
from .nodes import (
    analyze_query,
    check_retrieval_quality,
    determine_retrieval_strategy,
    fallback_answer,
    filter_chunks,
    finalize_metrics,
    generate_answer,
    graph_retrieve,
    multi_doc_retrieve,
    rewrite_retrieval_query,
    select_top_k_chunks,
    single_doc_retrieve,
)


def route_by_query_type(state: KnowledgeAgentState) -> Literal["single_doc_retrieve", "multi_doc_retrieve"]:
    """按分类节点写入的 query_type 选择单文档或多文档检索；不消耗任何重试预算。"""
    return "single_doc_retrieve" if state.get("query_type") == "single_doc" else "multi_doc_retrieve"


def route_retrieval_quality(state: KnowledgeAgentState) -> Literal["generate_answer", "rewrite_retrieval_query", "fallback_answer"]:
    """检索门控通过则生成；失败且检索预算未尽则改写重检，否则进入唯一 fallback 出口。"""
    if state.get("retrieval_quality_passed"):
        return "generate_answer"
    config = state["config"]
    if state.get("retrieval_retry_count", 0) < config.max_retrieval_retries:
        return "rewrite_retrieval_query"
    return "fallback_answer"


def create_knowledge_agent(checkpointer=None, interrupt_before: Optional[List[str]] = None):
    """装配并编译检索门控图；所有成功/失败路径最终经 finalize_metrics 到 END。"""
    builder = StateGraph(KnowledgeAgentState)
    for name, node in (
        ("analyze_query", analyze_query),
        ("determine_retrieval_strategy", determine_retrieval_strategy),
        ("graph_retrieve", graph_retrieve),
        ("single_doc_retrieve", single_doc_retrieve),
        ("multi_doc_retrieve", multi_doc_retrieve),
        ("filter_chunks", filter_chunks),
        ("select_top_k_chunks", select_top_k_chunks),
        ("retrieval_quality", check_retrieval_quality),
        ("rewrite_retrieval_query", rewrite_retrieval_query),
        ("generate_answer", generate_answer),
        ("fallback_answer", fallback_answer),
        ("finalize_metrics", finalize_metrics),
    ):
        builder.add_node(name, node)

    builder.add_edge(START, "analyze_query")
    builder.add_edge("analyze_query", "determine_retrieval_strategy")
    builder.add_edge("determine_retrieval_strategy", "graph_retrieve")
    builder.add_conditional_edges("graph_retrieve", route_by_query_type)
    builder.add_edge("single_doc_retrieve", "select_top_k_chunks")
    builder.add_edge("multi_doc_retrieve", "filter_chunks")
    builder.add_edge("filter_chunks", "select_top_k_chunks")
    builder.add_edge("select_top_k_chunks", "retrieval_quality")
    builder.add_conditional_edges("retrieval_quality", route_retrieval_quality)

    # 二次检索显式重新走 graph_retrieve、single/multi、rerank 与检索门控；
    # graph_retrieve 仍严格检查 config.kg_enabled，不会在二次检索时自动开启图谱。
    builder.add_edge("rewrite_retrieval_query", "graph_retrieve")
    builder.add_edge("generate_answer", "finalize_metrics")
    builder.add_edge("fallback_answer", "finalize_metrics")
    builder.add_edge("finalize_metrics", END)

    compile_kw = {"checkpointer": checkpointer}
    if interrupt_before:
        compile_kw["interrupt_before"] = interrupt_before
    return builder.compile(**compile_kw)
