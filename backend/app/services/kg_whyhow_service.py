# -*- coding: utf-8 -*-
"""
Neo4j 知识图谱检索服务。

问答阶段使用：
1. 从问题中抽取实体/关键词
2. 在 Neo4j 中检索相关实体关系
3. 根据关系上的 chunk_id 回填 PostgreSQL 原文
4. 返回 kg_graph_chunks，交给 generate 节点融合进回答上下文
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class KGRetrievalService:
    """Neo4j 图谱检索服务，保留旧 query_graph 方法签名。"""

    def query_graph(
        self,
        graph_id: str,
        query: str,
        *,
        top_k: int = 5,
        timeout: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        graph_id 在新方案中等同于 kb_name。

        Returns:
            List[{
                "chunk_id": str,
                "content": str,
                "file_name": str,
                "chunk_index": int,
                "metadata": dict,
                "graph_score": float,
                "unified_score": float,
                "relation_types": List[str],
                "source_node_names": List[str],
                "updated_at": str,
            }]
        """
        kb_name = graph_id
        logger.info("[KGRetrieval] Neo4j query | kb=%s query=%r top_k=%d", kb_name, query[:80], top_k)

        try:
            from app.services.kg_graph_sync_service import get_kg_graph_sync_service

            kg_sync = get_kg_graph_sync_service()
            relation_rows = kg_sync.search_related_chunks(kb_name=kb_name, query=query, top_k=top_k)
        except Exception as e:
            logger.warning("[KGRetrieval] Neo4j 查询失败: %s", e)
            return []

        if not relation_rows:
            logger.info("[KGRetrieval] Neo4j no related chunks | kb=%s query=%r", kb_name, query[:40])
            return []

        relation_map: Dict[str, set] = {}
        node_map: Dict[str, set] = {}
        score_map: Dict[str, float] = {}
        for row in relation_rows:
            cid = row.get("chunk_id")
            if not cid:
                continue
            relation_map.setdefault(cid, set()).add(str(row.get("relation") or "关联"))
            node_map.setdefault(cid, set()).update([str(row.get("head") or ""), str(row.get("tail") or "")])
            score_map[cid] = max(score_map.get(cid, 0.0), float(row.get("confidence") or 0.7))

        chunk_ids = list(score_map.keys())
        enriched = self._enrich_from_pg(chunk_ids, relation_map, node_map, score_map, top_k=top_k)
        logger.info("[KGRetrieval] Neo4j chunks=%d enriched=%d", len(chunk_ids), len(enriched))
        return enriched

    def _enrich_from_pg(
        self,
        chunk_ids: List[str],
        relation_map: Dict[str, set],
        node_map: Dict[str, set],
        score_map: Dict[str, float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []

        try:
            from app.db import get_chunk_repository

            chunk_repo = get_chunk_repository()
            pg_rows = chunk_repo.get_by_ids_with_file_names(chunk_ids)
            pg_map = {r["chunk_id"]: r for r in pg_rows}
        except Exception as e:
            logger.warning("[KGRetrieval] PG 回填失败: %s", e)
            return []

        scored: List[Dict[str, Any]] = []
        for cid in chunk_ids:
            pg = pg_map.get(cid)
            if not pg:
                continue

            content = pg.get("current_content") or pg.get("content") or ""
            file_name = pg.get("file_name") or ""
            chunk_index = pg.get("chunk_index")
            updated_at = pg.get("updated_at") or _now_ts()
            graph_score = max(0.0, min(1.0, score_map.get(cid, 0.7)))
            metadata = pg.get("metadata") or {}

            scored.append({
                "chunk_id": cid,
                "content": content,
                "file_name": file_name,
                "chunk_index": chunk_index,
                "metadata": metadata,
                "graph_score": graph_score,
                "unified_score": graph_score,
                "relation_types": sorted(relation_map.get(cid, set())),
                "source_node_names": sorted(x for x in node_map.get(cid, set()) if x),
                "updated_at": str(updated_at),
                "retrieval_method": "knowledge_graph",
                "score": graph_score,
            })

        scored.sort(key=lambda x: x.get("unified_score", 0.0), reverse=True)
        return scored[:top_k]


_instance: Optional[KGRetrievalService] = None


def get_kg_retrieval_service() -> KGRetrievalService:
    global _instance
    if _instance is None:
        _instance = KGRetrievalService()
    return _instance
