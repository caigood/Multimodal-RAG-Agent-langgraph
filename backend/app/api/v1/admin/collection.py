# -*- coding: utf-8 -*-
"""知识库管理 API（原 collection）"""
from typing import Optional, List
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from app.db import get_kb_repository
from app.services.milvus_service import get_milvus_service
from app.core.exceptions import NotFoundError, ConflictError

router = APIRouter(prefix="/collections", tags=["admin-collections"])


class MetadataFieldConfig(BaseModel):
    key: str
    type: str = "text"
    fulltext: bool = False
    index: bool = False
    auto_inject: Optional[str] = None


class RetrievalConfig(BaseModel):
    rrf_k: int = 60
    multi_doc_top_k: int = 20
    multi_doc_group_size: int = 3
    strict_group_size: bool = False
    single_doc_top_k: int = 20
    llm_context_top_k: int = 10
    image_vector_dim: int = 1024
    # Rerank 配置
    rerank_enabled: bool = False
    single_doc_rerank_top_k: int = 5
    multi_doc_rerank_top_k: int = 10
    # 会话与知识图谱
    memory_turns: int = 2
    kg_enabled: bool = True
    kg_graph_id: Optional[str] = None
    kg_top_k: int = 5
    kg_timeout_seconds: float = 2.0
    # 检索质量门控
    retrieval_quality_enabled: bool = True
    retrieval_quality_threshold: float = 0.35
    max_retrieval_retries: int = Field(default=1, ge=0, le=2)
    fallback_message: str = "抱歉，我无法找到相关信息。"


class RetrievalConfigPatch(BaseModel):
    rrf_k: Optional[int] = None
    multi_doc_top_k: Optional[int] = None
    multi_doc_group_size: Optional[int] = None
    strict_group_size: Optional[bool] = None
    single_doc_top_k: Optional[int] = None
    llm_context_top_k: Optional[int] = None
    image_vector_dim: Optional[int] = None
    rerank_enabled: Optional[bool] = None
    single_doc_rerank_top_k: Optional[int] = None
    multi_doc_rerank_top_k: Optional[int] = None
    memory_turns: Optional[int] = None
    kg_enabled: Optional[bool] = None
    kg_graph_id: Optional[str] = None
    kg_top_k: Optional[int] = None
    kg_timeout_seconds: Optional[float] = None
    retrieval_quality_enabled: Optional[bool] = None
    retrieval_quality_threshold: Optional[float] = None
    max_retrieval_retries: Optional[int] = Field(default=None, ge=0, le=2)
    fallback_message: Optional[str] = None


class CreateKbRequest(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    # image_mode 控制文档图片解析/展示；kb_type 控制检索与 Milvus Schema。
    image_mode: bool = False
    kb_type: str = "standard"               # standard 可图文展示；multimodal 使用图文向量检索
    embedding_model: str = "text-embedding-v3"
    vector_dim: int = 1536
    metadata_fields: Optional[List[MetadataFieldConfig]] = None
    retrieval_config: Optional[RetrievalConfig] = None

    @model_validator(mode="after")
    def validate_modes(self):
        if self.kb_type not in {"standard", "multimodal"}:
            raise ValueError("kb_type 仅支持 standard 或 multimodal")
        if self.kb_type == "multimodal" and not self.image_mode:
            raise ValueError("multimodal 知识库必须启用 image_mode")
        return self


class UpdateKbRequest(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    image_mode: Optional[bool] = None
    retrieval_config: Optional[RetrievalConfigPatch] = None


@router.get("")
async def list_collections():
    kbs = get_kb_repository().list_all()
    return JSONResponse(content={"success": True, "data": {"collections": kbs, "total": len(kbs)}})


@router.post("")
async def create_collection(req: CreateKbRequest):
    kb_repo = get_kb_repository()
    if kb_repo.get_by_name(req.name):
        raise ConflictError(f"知识库「{req.name}」已存在")

    mf = [f.model_dump() for f in req.metadata_fields] if req.metadata_fields else []
    rc = req.retrieval_config.model_dump() if req.retrieval_config else {}

    # 多模态 kb：强制使用 qwen3-vl-embedding，且 dense 和 image_dense 维度必须一致
    embedding_model = req.embedding_model
    vector_dim = req.vector_dim
    if req.kb_type == "multimodal":
        embedding_model = "qwen3-vl-embedding"
        image_vector_dim = rc.get("image_vector_dim", 1024)
        vector_dim = image_vector_dim  # dense 和 image_dense 用同一维度
        rc["image_vector_dim"] = image_vector_dim

    # 在 Milvus 中创建 collection（传入 metadata_fields 以建倒排索引）
    get_milvus_service().get_or_create_collection(
        collection_name=req.name,
        dim=vector_dim,
        image_mode=req.image_mode,
        metadata_fields=mf,
        kb_type=req.kb_type,
        image_vector_dim=rc.get("image_vector_dim", 1024),
    )

    # 在 PG 中记录配置
    kb = kb_repo.create(
        name=req.name,
        display_name=req.display_name,
        description=req.description,
        image_mode=req.image_mode,
        kb_type=req.kb_type,
        embedding_model=embedding_model,
        vector_dim=vector_dim,
        metadata_fields=mf,
        retrieval_config=rc,
    )
    return JSONResponse(content={"success": True, "message": f"知识库「{req.name}」创建成功", "data": kb})


@router.get("/{kb_name}")
async def get_collection(kb_name: str):
    kb = get_kb_repository().get_by_name(kb_name)
    if not kb:
        raise NotFoundError(f"知识库「{kb_name}」不存在")
    return JSONResponse(content={"success": True, "data": kb})


@router.put("/{kb_name}")
async def update_collection(kb_name: str, req: UpdateKbRequest):
    kb_repo = get_kb_repository()
    kb = kb_repo.get_by_name(kb_name)
    if not kb:
        raise NotFoundError(f"知识库「{kb_name}」不存在")
    retrieval_config = None
    if req.image_mode is not None and req.image_mode != kb.get("image_mode"):
        from app.db import get_file_repository
        if get_file_repository().list_by_kb(kb["id"], limit=1):
            raise ConflictError("知识库已有文件，不能切换 image_mode；该变更会改变解析方式和 Milvus Schema")
        if kb.get("kb_type") == "multimodal" and not req.image_mode:
            raise ConflictError("multimodal 知识库必须保持 image_mode=true")
    if req.retrieval_config is not None:
        retrieval_config = dict(kb.get("retrieval_config") or {})
        for key, value in req.retrieval_config.model_dump(exclude_unset=True).items():
            if value is not None:
                retrieval_config[key] = value

    updated = kb_repo.update(
        kb["id"],
        display_name=req.display_name,
        description=req.description,
        image_mode=req.image_mode,
        retrieval_config=retrieval_config,
    )
    return JSONResponse(content={"success": True, "data": updated})


@router.delete("/{kb_name}")
async def delete_collection(kb_name: str):
    kb_repo = get_kb_repository()
    kb = kb_repo.get_by_name(kb_name)
    if not kb:
        raise NotFoundError(f"知识库「{kb_name}」不存在")

    # 检查是否有文件
    from app.db import get_file_repository
    files = get_file_repository().list_by_kb(kb["id"], limit=1)
    if files:
        raise ConflictError(f"知识库「{kb_name}」中还有文件，请先删除所有文件后再删除知识库")

    # 删除 Milvus collection
    try:
        get_milvus_service().delete_collection(kb_name)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Milvus collection 删除失败（继续）: {e}")

    # 删除 PG 记录（级联删除 file/job/chunk）
    kb_repo.delete(kb["id"])
    return JSONResponse(content={"success": True, "message": f"知识库「{kb_name}」已删除"})
