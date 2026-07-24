# -*- coding: utf-8 -*-
"""
Knowledge RAG 业务逻辑（Knowledge Agent 调用封装）
"""
import asyncio
import json
import re
import uuid
import logging
from contextlib import suppress
from typing import Optional, AsyncIterator, Dict, Any

from app.core.config import SUPPORTED_MODELS
from app.core.exceptions import ValidationError, ExternalServiceError

logger = logging.getLogger(__name__)


def _sse(event: Optional[str], data: Dict[str, Any]) -> str:
    """单条 SSE 文本帧（UTF-8 由 StreamingResponse 编码）。"""
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n")
    return "\n".join(lines) + "\n"


def _log_agent_diagnostics(state: Dict[str, Any], request_id: str, session_id: str) -> None:
    """将工作流累积的诊断信息写入服务端日志，不加入 API 响应。"""
    context = {"request_id": request_id, "session_id": session_id}
    errors = state.get("all_errors") or []
    warnings = state.get("all_warnings") or []
    if errors:
        logger.error("Knowledge Agent 工作流错误", extra={**context, "all_errors": errors})
    if warnings:
        logger.warning("Knowledge Agent 工作流警告", extra={**context, "all_warnings": warnings})


async def cleanup_unpersisted_query_image(query_image_oss_key: Optional[str]) -> None:
    """仅清理尚未归属任何持久化消息的查询图片。"""
    if not query_image_oss_key:
        return

    def _cleanup() -> None:
        from app.db import get_conversation_repository
        if get_conversation_repository().is_query_image_referenced(query_image_oss_key):
            return
        from app.services.oss_service import get_oss_service
        get_oss_service().delete_objects([query_image_oss_key])

    try:
        await asyncio.to_thread(_cleanup)
    except Exception as e:
        logger.warning("未持久化查询图片补偿删除失败", extra={"oss_key": query_image_oss_key, "error": str(e)})


async def _persist_conversation_messages(
    session_id: str,
    query: str,
    answer_text: str,
    sources: list,
    confidence: Optional[float],
    query_image_oss_key: Optional[str],
) -> None:
    """会话存在时在线程中写入 user/assistant 消息。"""
    def _persist() -> None:
        from app.db import get_conversation_repository
        conv_repo = get_conversation_repository()
        if not conv_repo.get_session(session_id):
            return
        phs = list(set(re.findall(r"<<IMAGE:[0-9a-f]+>>", answer_text or "")))
        conv_repo.add_exchange(
            session_id=session_id,
            query=query,
            answer=answer_text or "",
            sources=sources or [],
            confidence=confidence,
            image_placeholders=phs,
            query_image_oss_key=query_image_oss_key,
        )

    persist_task = asyncio.create_task(asyncio.to_thread(_persist))
    try:
        await asyncio.shield(persist_task)
    except asyncio.CancelledError:
        # to_thread 取消后底层写入仍会继续；先等归属落库，避免外层 finally 误删 OSS 图片。
        try:
            await persist_task
        except Exception as e:
            logger.warning(f"消息持久化失败（不影响回答）: {e}")
        raise
    except Exception as e:
        logger.warning(f"消息持久化失败（不影响回答）: {e}")


async def _load_kb_retrieval(collection: Optional[str]):
    """在线程中读取知识库 retrieval_config；失败时沿用默认值。"""
    if not collection:
        return None, {}

    def _load():
        from app.db import get_kb_repository
        kb = get_kb_repository().get_by_name(collection)
        rc = dict(kb.get("retrieval_config") or {}) if kb else {}
        retries = rc.get("max_retrieval_retries")
        if retries is not None:
            try:
                rc["max_retrieval_retries"] = max(0, min(2, int(retries)))
            except (TypeError, ValueError):
                rc.pop("max_retrieval_retries", None)
        return kb, rc

    try:
        return await asyncio.to_thread(_load)
    except Exception as e:
        logger.warning(f"读取 kb retrieval_config 失败，使用默认值: {e}")
        return None, {}


def _build_rag_config(RAGConfig, model_name: str, collection: Optional[str], kb: Optional[dict], rc: dict,
                      force_multi_doc: Optional[bool], keyword_filter: Optional[str], query_image_url: Optional[str]):
    """合并 RAG 默认值、知识库检索配置与本次请求覆盖项，构造流式/非流式共用配置。"""
    defaults = RAGConfig()
    configurable = (
        "rrf_k", "multi_doc_top_k", "multi_doc_group_size", "strict_group_size",
        "single_doc_top_k", "llm_context_top_k", "memory_turns", "image_vector_dim",
        "kg_enabled", "kg_graph_id", "kg_top_k", "kg_timeout_seconds", "rerank_enabled",
        "single_doc_rerank_top_k", "multi_doc_rerank_top_k",
        "retrieval_quality_enabled", "retrieval_quality_threshold",
        "max_retrieval_retries", "fallback_message",
    )
    values = {name: rc.get(name, getattr(defaults, name)) for name in configurable}
    return RAGConfig(
        model=model_name, retrieval_strategy="hybrid", collection=collection or None,
        kb_type=kb.get("kb_type", "standard") if kb else "standard", query_image_url=query_image_url,
        force_multi_doc=force_multi_doc, keyword_filter=keyword_filter, **values,
    )


async def invoke_knowledge_qa(
    query: str,
    model_name: str,
    session_id: str,
    collection: Optional[str] = None,
    force_multi_doc: Optional[bool] = None,
    keyword_filter: Optional[str] = None,
    query_image_url: Optional[str] = None,
    query_image_oss_key: Optional[str] = None,
) -> dict:
    """调用 Knowledge Agent 执行 RAG 问答。"""
    if model_name not in SUPPORTED_MODELS:
        raise ValidationError(f"Model '{model_name}' not supported. Available: {list(SUPPORTED_MODELS.keys())}")

    from agents.knowledge import get_knowledge_agent, create_initial_state, RAGConfig
    agent = get_knowledge_agent()

    request_id = str(uuid.uuid4())

    # 读取 kb 的 retrieval_config 注入到 RAGConfig
    kb, rc = await _load_kb_retrieval(collection)

    # LangGraph 通过 checkpointer（thread_id=session_id）自动恢复历史对话，

    initial_state = create_initial_state(
        query=query,
        config=_build_rag_config(
            RAGConfig, model_name, collection, kb, rc, force_multi_doc, keyword_filter, query_image_url
        ),
    )

    config = {
        "recursion_limit": 40,
        "configurable": {
            "model": model_name,
            "session_id": session_id,
            "thread_id": session_id,  # LangGraph 用 session_id 作为 thread_id 实现对话记忆
        }
    }

    logger.info("处理 knowledge 请求", extra={"session_id": session_id, "request_id": request_id})

    try:
        result = await agent.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.exception("Knowledge Agent 调用失败", extra={"session_id": session_id, "request_id": request_id})
        raise ExternalServiceError("Knowledge Agent 调用失败") from e

    metrics = result.get("metrics", {})
    _log_agent_diagnostics(result, request_id, session_id)
    thoughts = {
        "query_analysis": {
            "needs_rewrite": bool(result.get("needs_rewrite")),
            "query_type": result.get("query_type"),
        },
        "retrieval": {
            "chunks_retrieved": metrics.total_chunks_retrieved if hasattr(metrics, "total_chunks_retrieved") else 0,
            "chunks_used": metrics.chunks_after_rerank if hasattr(metrics, "chunks_after_rerank") else 0,
        },
        "conversation_turns": 1,
    }

    return_data = {
        "request_id": request_id,
        "session_id": session_id,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"],
        "model": model_name,
        "thoughts": thoughts,
        "image_map": result.get("image_map") or None,
    }

    await _persist_conversation_messages(
        session_id,
        query,
        result.get("answer") or "",
        result.get("sources") or [],
        result.get("confidence"),
        query_image_oss_key,
    )

    return return_data


def _thoughts_from_state_values(vals: Dict[str, Any]) -> Dict[str, Any]:
    metrics = vals.get("metrics", {})
    return {
        "query_analysis": {
            "needs_rewrite": bool(vals.get("needs_rewrite")),
            "query_type": vals.get("query_type"),
        },
        "retrieval": {
            "chunks_retrieved": metrics.total_chunks_retrieved if hasattr(metrics, "total_chunks_retrieved") else 0,
            "chunks_used": metrics.chunks_after_rerank if hasattr(metrics, "chunks_after_rerank") else 0,
        },
        "conversation_turns": 1,
    }


async def stream_knowledge_qa_sse(
    query: str,
    model_name: str,
    session_id: str,
    collection: Optional[str] = None,
    force_multi_doc: Optional[bool] = None,
    keyword_filter: Optional[str] = None,
    query_image_url: Optional[str] = None,
    query_image_oss_key: Optional[str] = None,
) -> AsyncIterator[str]:
    """执行 Knowledge LangGraph，完成检索门控后发送 meta、答案分片和 done；失败发送 error。"""
    if model_name not in SUPPORTED_MODELS:
        yield _sse("error", {"message": f"Model '{model_name}' not supported. Available: {list(SUPPORTED_MODELS.keys())}"})
        return

    from agents.knowledge import get_knowledge_agent, create_initial_state, RAGConfig

    agent = get_knowledge_agent()
    request_id = str(uuid.uuid4())
    kb, rc = await _load_kb_retrieval(collection)

    initial_state = create_initial_state(
        query=query,
        config=_build_rag_config(
            RAGConfig, model_name, collection, kb, rc, force_multi_doc, keyword_filter, query_image_url
        ),
    )

    config = {
        "recursion_limit": 40,
        "configurable": {
            "model": model_name,
            "session_id": session_id,
            "thread_id": session_id,
        }
    }

    logger.info("处理 knowledge 流式请求", extra={"session_id": session_id, "request_id": request_id})

    invoke_task = None
    try:
        # 图执行期间仅发送 SSE 注释心跳保持代理连接。
        invoke_task = asyncio.create_task(agent.ainvoke(initial_state, config=config))
        while not invoke_task.done():
            yield ": keep-alive\n\n"
            try:
                await asyncio.wait_for(asyncio.shield(invoke_task), timeout=15)
            except asyncio.TimeoutError:
                continue
        vals = await invoke_task
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Knowledge 检索阶段失败")
        yield _sse("error", {"message": "Knowledge Agent 检索失败"})
        return
    finally:
        if invoke_task is not None and not invoke_task.done():
            invoke_task.cancel()
            with suppress(asyncio.CancelledError):
                await invoke_task

    final = vals
    answer_final = final.get("answer") or ""
    yield _sse(
        "meta",
        {
            "request_id": request_id,
            "session_id": session_id,
            "model": model_name,
            "thoughts": _thoughts_from_state_values(final),
            "sources": final.get("sources") or [],
            "image_map": final.get("image_map") or {},
        },
    )

    # 图完成后按文本片段模拟流式。
    chunk_size = 24
    for offset in range(0, len(answer_final), chunk_size):
        yield _sse("delta", {"text": answer_final[offset:offset + chunk_size]})

    _log_agent_diagnostics(final, request_id, session_id)

    await _persist_conversation_messages(
        session_id,
        query,
        answer_final,
        final.get("sources") or [],
        final.get("confidence"),
        query_image_oss_key,
    )

    # 在持久化完成后再发送 done，客户端收到 done 后立即断开也不会丢失图片所有权。
    yield _sse(
        "done",
        {
            "request_id": request_id,
            "session_id": session_id,
            "answer": answer_final,
            "confidence": final.get("confidence"),
            "sources": final.get("sources") or [],
            "model": model_name,
            "thoughts": _thoughts_from_state_values(final),
            "image_map": final.get("image_map") or {},
            "finish_reason": "stop",
        },
    )
