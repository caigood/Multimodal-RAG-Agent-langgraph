# -*- coding: utf-8 -*-
"""
Neo4j 知识图谱同步服务。

替代旧 Knowledge Table 图谱链路：
- 文档向量化完成后，从 chunk 中抽取实体/关系
- 写入 Neo4j
- 删除文件时按 job_id 清理关系和 chunk 节点
- 前端图谱面板按 job_id 查询 triples
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from dashscope import Generation
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = [
    "概念", "组织", "人员", "系统", "模块", "产品", "流程", "制度", "文档",
    "指标", "风险", "需求", "接口", "参数", "时间", "地点", "事件",
]

DEFAULT_RELATION_TYPES = [
    "定义", "包含", "属于", "负责", "依赖", "影响", "导致", "适用于", "要求",
    "实现", "调用", "配置", "限制", "关联", "组成部分", "前置条件", "输出",
]


class KGGraphSyncService:
    """Neo4j 图谱同步客户端，保留旧服务的方法签名。"""

    def __init__(self) -> None:
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self.database = settings.neo4j_database
        self.extract_model = settings.kg_extract_model
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._ensure_schema()
        return self._driver

    def _ensure_schema(self) -> None:
        with self._driver.session(database=self.database) as session:
            session.run("CREATE CONSTRAINT kg_entity_id IF NOT EXISTS FOR (e:KGEntity) REQUIRE e.id IS UNIQUE")
            session.run("CREATE CONSTRAINT kg_chunk_id IF NOT EXISTS FOR (c:KGChunk) REQUIRE c.id IS UNIQUE")

    async def sync_chunks_to_graph(
        self,
        job_id: str,
        kb_name: str,
        file_name: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """从 chunks 抽取实体/关系并写入 Neo4j。"""
        usable_chunks = [c for c in chunks if c.get("content")]
        logger.info("[KGGraphSync] Neo4j sync start job_id=%s kb=%s chunks=%d", job_id, kb_name, len(usable_chunks))

        if not usable_chunks:
            return {"graph_id": kb_name, "triples_count": 0, "nodes_count": 0, "message": "No chunks."}

        triples: List[Dict[str, Any]] = []
        entity_keys: set[str] = set()

        for chunk in usable_chunks:
            extracted = self._extract_from_chunk(chunk.get("content", ""))
            entities = extracted.get("entities", [])
            entity_map = {
                self._norm_name(e.get("name", "")): e
                for e in entities
                if e.get("name")
            }

            for rel in extracted.get("relations", []):
                head = self._norm_name(rel.get("head", ""))
                tail = self._norm_name(rel.get("tail", ""))
                if not head or not tail or head == tail:
                    continue

                head_entity = entity_map.get(head, {"name": rel.get("head", ""), "type": rel.get("head_type") or "概念"})
                tail_entity = entity_map.get(tail, {"name": rel.get("tail", ""), "type": rel.get("tail_type") or "概念"})
                relation = self._clean_relation(rel.get("relation") or "关联")

                triple = {
                    "triple_id": f"t{uuid.uuid4().hex[:12]}",
                    "job_id": job_id,
                    "kb_name": kb_name,
                    "file_name": file_name,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "head": head_entity.get("name", "").strip(),
                    "head_type": head_entity.get("type") or rel.get("head_type") or "概念",
                    "tail": tail_entity.get("name", "").strip(),
                    "tail_type": tail_entity.get("type") or rel.get("tail_type") or "概念",
                    "relation": relation,
                    "evidence": str(rel.get("evidence") or chunk.get("content", ""))[:500],
                    "confidence": float(rel.get("confidence") or 0.7),
                }
                triples.append(triple)
                entity_keys.add(self._entity_id(kb_name, triple["head_type"], triple["head"]))
                entity_keys.add(self._entity_id(kb_name, triple["tail_type"], triple["tail"]))

        if not triples:
            logger.info("[KGGraphSync] Neo4j sync no triples job_id=%s", job_id)
            return {"graph_id": kb_name, "triples_count": 0, "nodes_count": 0, "message": "No relations extracted."}

        with self.driver.session(database=self.database) as session:
            session.execute_write(self._delete_job_tx, job_id)
            session.execute_write(self._upsert_triples_tx, triples)

        logger.info("[KGGraphSync] Neo4j sync done job_id=%s triples=%d nodes=%d", job_id, len(triples), len(entity_keys))
        return {
            "graph_id": kb_name,
            "triples_count": len(triples),
            "nodes_count": len(entity_keys),
            "message": f"Synced {len(triples)} triples to Neo4j.",
        }

    async def delete_graph_by_job(self, job_id: str) -> Dict[str, Any]:
        """根据 job_id 删除图谱关系和对应 chunk 节点。"""
        with self.driver.session(database=self.database) as session:
            deleted_count = session.execute_write(self._delete_job_tx, job_id)
        logger.info("[KGGraphSync] Neo4j deleted job_id=%s count=%d", job_id, deleted_count)
        return {"job_id": job_id, "deleted_count": deleted_count, "message": "Done."}

    async def query_graph(self, job_id: str, kb_name: str) -> Dict[str, Any]:
        """查询某个文件任务对应的 triples，供前端图谱面板使用。"""
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                """
                MATCH (h:KGEntity)-[r:KG_RELATION {job_id: $job_id, kb_name: $kb_name}]->(t:KGEntity)
                RETURN r.triple_id AS triple_id,
                       r.chunk_id AS chunk_id,
                       h.name AS head_content,
                       t.name AS tail_content,
                       h.id AS head_chunk_id,
                       t.id AS tail_chunk_id,
                       r.relation AS relation,
                       r.evidence AS evidence,
                       r.confidence AS confidence
                ORDER BY r.chunk_index ASC, r.confidence DESC
                """,
                job_id=job_id,
                kb_name=kb_name,
            )
            triples = [dict(r) for r in rows]
        return {"job_id": job_id, "kb_name": kb_name, "triples": triples, "total": len(triples)}

    def search_related_chunks(self, kb_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """根据问题中的实体/关键词，从 Neo4j 找到相关关系证据和 chunk_id。"""
        terms = self._extract_query_terms(query)
        if not terms:
            return []

        with self.driver.session(database=self.database) as session:
            rows = session.run(
                """
                MATCH (e:KGEntity {kb_name: $kb_name})
                WHERE any(term IN $terms WHERE
                    toLower(e.name) CONTAINS toLower(term) OR toLower(term) CONTAINS toLower(e.name)
                )
                MATCH (e)-[r:KG_RELATION {kb_name: $kb_name}]-(other:KGEntity)
                RETURN DISTINCT r.chunk_id AS chunk_id,
                       r.relation AS relation,
                       r.evidence AS evidence,
                       r.confidence AS confidence,
                       startNode(r).name AS head,
                       endNode(r).name AS tail
                ORDER BY r.confidence DESC
                LIMIT $limit
                """,
                kb_name=kb_name,
                terms=terms,
                limit=max(top_k * 4, 10),
            )
            return [dict(r) for r in rows]

    @staticmethod
    def _upsert_triples_tx(tx, triples: List[Dict[str, Any]]) -> None:
        for t in triples:
            head_id = KGGraphSyncService._entity_id(t["kb_name"], t["head_type"], t["head"])
            tail_id = KGGraphSyncService._entity_id(t["kb_name"], t["tail_type"], t["tail"])
            tx.run(
                """
                MERGE (h:KGEntity {id: $head_id})
                SET h.name = $head, h.type = $head_type, h.kb_name = $kb_name
                MERGE (ta:KGEntity {id: $tail_id})
                SET ta.name = $tail, ta.type = $tail_type, ta.kb_name = $kb_name
                MERGE (c:KGChunk {id: $chunk_id})
                SET c.job_id = $job_id, c.kb_name = $kb_name, c.file_name = $file_name, c.chunk_index = $chunk_index
                MERGE (h)-[r:KG_RELATION {
                    triple_id: $triple_id,
                    job_id: $job_id,
                    kb_name: $kb_name,
                    chunk_id: $chunk_id
                }]->(ta)
                SET r.relation = $relation,
                    r.evidence = $evidence,
                    r.confidence = $confidence,
                    r.file_name = $file_name,
                    r.chunk_index = $chunk_index
                MERGE (c)-[:HAS_HEAD]->(h)
                MERGE (c)-[:HAS_TAIL]->(ta)
                """,
                **t,
                head_id=head_id,
                tail_id=tail_id,
            )

    @staticmethod
    def _delete_job_tx(tx, job_id: str) -> int:
        result = tx.run(
            """
            MATCH ()-[r:KG_RELATION {job_id: $job_id}]->()
            WITH collect(r) AS rels
            FOREACH (r IN rels | DELETE r)
            WITH size(rels) AS deleted
            OPTIONAL MATCH (c:KGChunk {job_id: $job_id})
            DETACH DELETE c
            RETURN deleted
            """,
            job_id=job_id,
        )
        row = result.single()
        return int(row["deleted"] if row else 0)

    def _extract_from_chunk(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"entities": [], "relations": []}

        prompt = f"""
你是企业知识图谱抽取器。请从文本中抽取实体和关系，只输出 JSON，不要输出解释。

实体类型只能从以下列表选择：{', '.join(DEFAULT_ENTITY_TYPES)}。
关系类型只能从以下列表选择：{', '.join(DEFAULT_RELATION_TYPES)}。

要求：
1. 实体必须是文本中明确出现或明确指代的业务对象，不要抽象发挥。
2. 关系必须有原文证据 evidence。
3. 最多输出 8 个实体、8 条关系。
4. confidence 取 0 到 1。

输出格式：
{{
  "entities": [{{"name": "实体名", "type": "实体类型", "description": "简短说明"}}],
  "relations": [{{
    "head": "头实体", "head_type": "实体类型",
    "relation": "关系类型",
    "tail": "尾实体", "tail_type": "实体类型",
    "evidence": "原文证据",
    "confidence": 0.8
  }}]
}}

文本：
{text[:3500]}
""".strip()

        try:
            response = Generation.call(
                api_key=settings.dashscope_api_key,
                model=self.extract_model,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
            )
            if response.status_code != 200:
                logger.warning("[KGGraphSync] extract failed status=%s", response.status_code)
                return {"entities": [], "relations": []}
            msg = response.output.choices[0].message
            content = (msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""
            return self._parse_json(content)
        except Exception as e:
            logger.warning("[KGGraphSync] extract exception: %s", e)
            return {"entities": [], "relations": []}

    def _extract_query_terms(self, query: str) -> List[str]:
        prompt = f"""
从用户问题中抽取适合知识图谱检索的实体名、专有名词和关键词。只输出 JSON 数组字符串。
用户问题：{query}
示例：["权限模块", "角色服务"]
""".strip()
        terms: List[str] = []
        try:
            response = Generation.call(
                api_key=settings.dashscope_api_key,
                model=settings.llm_clean_model,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
            )
            if response.status_code == 200:
                msg = response.output.choices[0].message
                content = (msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""
                parsed = self._parse_json(content, default=[])
                if isinstance(parsed, list):
                    terms = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception as e:
            logger.debug("[KGGraphSync] query term extract failed: %s", e)

        if not terms:
            terms = [w for w in re.split(r"[\s,，。；;：:、？?！!()（）\[\]{}]+", query) if len(w) >= 2]
        return terms[:8]

    @staticmethod
    def _parse_json(text: str, default: Optional[Any] = None) -> Any:
        if default is None:
            default = {"entities": [], "relations": []}
        raw = (text or "").strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
        if match:
            raw = match.group(1)
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data.setdefault("entities", [])
                data.setdefault("relations", [])
            return data
        except Exception:
            return default

    @staticmethod
    def _norm_name(name: str) -> str:
        return re.sub(r"\s+", "", str(name or "").strip().lower())

    @staticmethod
    def _clean_relation(relation: str) -> str:
        relation = str(relation or "关联").strip()
        return relation if relation in DEFAULT_RELATION_TYPES else "关联"

    @staticmethod
    def _entity_id(kb_name: str, entity_type: str, name: str) -> str:
        norm = KGGraphSyncService._norm_name(name)
        return f"{kb_name}:{entity_type}:{norm}"


_instance: KGGraphSyncService | None = None


def get_kg_graph_sync_service() -> KGGraphSyncService:
    global _instance
    if _instance is None:
        _instance = KGGraphSyncService()
    return _instance
