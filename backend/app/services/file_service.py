# -*- coding: utf-8 -*-
"""文件列表业务逻辑"""
import logging

from app.core.exceptions import ConflictError, NotFoundError
from app.db import (
    get_category_file_repository,
    get_category_repository,
    get_chunk_image_repository,
    get_file_repository,
    get_job_repository,
    get_kb_repository,
)
from app.services.oss_service import get_oss_service

logger = logging.getLogger(__name__)
_ACTIVE_JOB_STATUSES = {"pending", "chunking", "embedding"}


def list_files(kb_name: str, limit: int = 200) -> dict:
    kb = get_kb_repository().get_by_name(kb_name)
    if not kb:
        return {"files": [], "total": 0}
    files = get_file_repository().list_by_kb(kb["id"], limit=limit)
    return {"files": files, "total": len(files)}


def _delete_unreferenced_original(oss_key: str) -> None:
    if not oss_key:
        return
    refs = (
        get_file_repository().count_by_oss_key(oss_key)
        + get_category_file_repository().count_by_oss_key(oss_key)
    )
    if refs == 0:
        try:
            get_oss_service().delete_objects([oss_key])
        except Exception as e:
            logger.warning("OSS 原文件删除失败（继续）", extra={"oss_key": oss_key, "error": str(e)})


async def delete_file(file_id: str) -> str:
    """清理文件全部任务的外部数据，再删除 PG 引用及无引用 OSS 原文件。"""
    file_repo = get_file_repository()
    file_record = file_repo.get_by_id(file_id)
    if not file_record:
        raise NotFoundError("文件记录不存在")

    jobs = get_job_repository().list_by_file(file_id)
    active = [job for job in jobs if job.get("status") in _ACTIVE_JOB_STATUSES]
    if active or file_record.get("status") in {"pending", "processing"}:
        raise ConflictError("文件正在后台处理中，请等待任务完成或失败后再删除")

    kb = get_kb_repository().get_by_id(file_record["kb_id"])
    kb_name = kb["name"] if kb else None
    for job in jobs:
        job_id = job["id"]
        if kb_name:
            try:
                from app.services.milvus_service import get_milvus_service
                get_milvus_service().delete_by_job(kb_name, job_id)
            except Exception as e:
                logger.warning("Milvus 删除失败（继续）", extra={"job_id": job_id, "error": str(e)})
        try:
            from app.services.kg_graph_sync_service import get_kg_graph_sync_service
            await get_kg_graph_sync_service().delete_graph_by_job(job_id)
        except Exception as e:
            logger.warning("知识图谱删除失败（继续）", extra={"job_id": job_id, "error": str(e)})
        try:
            oss_keys = get_chunk_image_repository().get_oss_keys_by_job(job_id)
            if oss_keys:
                get_oss_service().delete_objects(oss_keys)
        except Exception as e:
            logger.warning("切片图片删除失败（继续）", extra={"job_id": job_id, "error": str(e)})

    category_file_id = file_record.get("category_file_id")
    category_file = None
    if category_file_id:
        category_file = get_category_file_repository().get_by_id(category_file_id)

    # 先删 knowledge_file，使 category_file 外键解除；__default__ 记录只由直传文件拥有。
    file_repo.delete(file_id)
    if category_file:
        category = get_category_repository().get(category_file["category_id"])
        if category and category["name"] == "__default__":
            get_category_file_repository().delete(category_file_id)

    _delete_unreferenced_original(file_record.get("oss_key"))
    logger.info("文件已删除", extra={"file_id": file_id, "jobs": len(jobs)})
    return file_record["file_name"]


async def batch_delete_files(file_ids: list, kb_name: str) -> dict:
    deleted, failed = [], []
    for file_id in file_ids:
        try:
            file_name = await delete_file(file_id)
            deleted.append(file_name)
        except Exception as e:
            logger.error("批量删除单文件失败", extra={"file_id": file_id, "error": str(e)})
            failed.append({"file_id": file_id, "error": str(e)})
    return {"deleted": deleted, "failed": failed}
