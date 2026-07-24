# -*- coding: utf-8 -*-
"""
知识库文件仓储
表：knowledge_file
"""
import logging
from typing import Any, Dict, List, Optional
from app.db.base_repository import BaseRepository
from app.db.job_repository import JobRepository

logger = logging.getLogger(__name__)


class FileRepository(BaseRepository):

    def create(
        self,
        *,
        kb_id: str,
        file_name: str,
        oss_key: str,
        category_file_id: Optional[str] = None,
        file_size: Optional[int] = None,
        mime_type: Optional[str] = None,
        status: str = "pending",
        sync_graph: bool = False,
    ) -> Dict[str, Any]:
        graph_sync_status = "pending" if sync_graph else "disabled"
        rows = self._execute_returning(
            """
            INSERT INTO knowledge_file(
                kb_id, category_file_id, file_name, oss_key, file_size, mime_type,
                status, sync_graph, graph_sync_status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (kb_id, category_file_id, file_name, oss_key, file_size, mime_type,
             status, sync_graph, graph_sync_status),
        )
        return self._normalize(rows[0]) if rows else {}

    def get_by_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        rows = self._execute_select(
            "SELECT * FROM knowledge_file WHERE id = %s LIMIT 1", (file_id,)
        )
        return self._normalize(rows[0]) if rows else None

    def get_by_kb_and_oss_key(self, kb_id: str, oss_key: str) -> Optional[Dict[str, Any]]:
        rows = self._execute_select(
            "SELECT * FROM knowledge_file WHERE kb_id = %s AND oss_key = %s LIMIT 1",
            (kb_id, oss_key),
        )
        return self._normalize(rows[0]) if rows else None

    def get_by_kb_and_filename(self, kb_id: str, file_name: str) -> Optional[Dict[str, Any]]:
        rows = self._execute_select(
            "SELECT * FROM knowledge_file WHERE kb_id = %s AND file_name = %s ORDER BY created_at DESC LIMIT 1",
            (kb_id, file_name),
        )
        return self._normalize(rows[0]) if rows else None

    def list_by_kb(self, kb_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        rows = self._execute_select(
            """
            SELECT f.*, row_to_json(j) AS latest_job
            FROM knowledge_file f
            LEFT JOIN LATERAL (
                SELECT * FROM knowledge_job
                WHERE file_id = f.id
                ORDER BY created_at DESC
                LIMIT 1
            ) j ON TRUE
            WHERE f.kb_id = %s
            ORDER BY f.created_at DESC
            LIMIT %s
            """,
            (kb_id, limit),
        )
        return [self._normalize(r) for r in rows]

    def count_by_oss_key(self, oss_key: str) -> int:
        rows = self._execute_select(
            "SELECT COUNT(*) AS count FROM knowledge_file WHERE oss_key = %s", (oss_key,)
        )
        return int(rows[0]["count"]) if rows else 0

    def update_status(self, file_id: str, status: str, error_msg: Optional[str] = None):
        self._execute_sql(
            "UPDATE knowledge_file SET status = %s, error_msg = %s, updated_at = NOW() WHERE id = %s",
            (status, error_msg, file_id),
        )

    def delete(self, file_id: str):
        self._execute_sql("DELETE FROM knowledge_file WHERE id = %s", (file_id,))

    def delete_by_kb(self, kb_id: str):
        self._execute_sql("DELETE FROM knowledge_file WHERE kb_id = %s", (kb_id,))

    @staticmethod
    def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "kb_id": str(row["kb_id"]),
            "category_file_id": str(row["category_file_id"]) if row.get("category_file_id") else None,
            "file_name": row["file_name"],
            "oss_key": row["oss_key"],
            "file_size": row.get("file_size"),
            "mime_type": row.get("mime_type"),
            "status": row.get("status", "pending"),
            "error_msg": row.get("error_msg"),
            "sync_graph": row.get("sync_graph", False),
            "graph_sync_status": row.get("graph_sync_status", "disabled"),
            "graph_sync_error": row.get("graph_sync_error"),
            "graph_synced_at": str(row["graph_synced_at"]) if row.get("graph_synced_at") else None,
            "job": JobRepository._normalize(row["latest_job"]) if row.get("latest_job") else None,
            "created_at": str(row["created_at"]) if row.get("created_at") else None,
            "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
        }

    def update_sync_graph(self, file_id: str, sync_graph: bool) -> None:
        self._execute_sql(
            """
            UPDATE knowledge_file
            SET sync_graph = %s, graph_sync_status = %s,
                graph_sync_error = NULL, graph_synced_at = NULL, updated_at = NOW()
            WHERE id = %s
            """,
            (sync_graph, "pending" if sync_graph else "disabled", file_id),
        )

    def update_graph_sync_status(
        self, file_id: str, status: str, error: Optional[str] = None
    ) -> None:
        synced_at = "NOW()" if status == "synced" else "NULL"
        self._execute_sql(
            f"""
            UPDATE knowledge_file
            SET graph_sync_status = %s, graph_sync_error = %s,
                graph_synced_at = {synced_at}, updated_at = NOW()
            WHERE id = %s
            """,
            (status, error, file_id),
        )


_instance: Optional[FileRepository] = None


def get_file_repository() -> FileRepository:
    global _instance
    if _instance is None:
        _instance = FileRepository()
    return _instance
