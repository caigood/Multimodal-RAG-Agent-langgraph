# -*- coding: utf-8 -*-
"""
Application Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _validate_env() -> None:
    """启动时校验必填配置"""
    missing = []
    for key in ["DASHSCOPE_API_KEY", "OSS_BUCKET", "PG_HOST", "PG_USER", "PG_PASSWORD", "MILVUS_HOST"]:
        if not os.getenv(key):
            missing.append(key)

    if not os.getenv("OSS_ACCESS_KEY_ID") and not os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"):
        missing.append("OSS_ACCESS_KEY_ID 或 ALIBABA_CLOUD_ACCESS_KEY_ID")
    if not os.getenv("OSS_ACCESS_KEY_SECRET") and not os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"):
        missing.append("OSS_ACCESS_KEY_SECRET 或 ALIBABA_CLOUD_ACCESS_KEY_SECRET")

    if missing:
        raise EnvironmentError(
            f"缺少必要的环境变量，请检查 .env 文件: {', '.join(missing)}"
        )


class Settings:
    # ── LLM ──────────────────────────────────────────────────────────────────
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    default_model: str = os.getenv("LLM_MODEL", "qwen3.7-plus")
    llm_clean_model: str = os.getenv("LLM_CLEAN_MODEL", "deepseek-v4-flash")
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    max_retries: int = 2
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_COMPATIBLE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # ── Milvus ────────────────────────────────────────────────────────────────
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_user: str = os.getenv("MILVUS_USER", "")
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "")

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    pg_host: str = os.getenv("PG_HOST", "localhost")
    pg_port: int = int(os.getenv("PG_PORT", "5432"))
    pg_db: str = os.getenv("PG_DB", "")
    pg_user: str = os.getenv("PG_USER", "")
    pg_password: str = os.getenv("PG_PASSWORD", "")

    # ── Neo4j 知识图谱 ────────────────────────────────────────────────────────
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "neo4j_password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    kg_extract_model: str = os.getenv("KG_EXTRACT_MODEL", os.getenv("LLM_CLEAN_MODEL", "deepseek-v4-flash"))

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))

    # ── 向量检索 ──────────────────────────────────────────────────────────────
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "10"))
    vector_score_threshold: float = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0"))

    # ── 切片 ──────────────────────────────────────────────────────────────────
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ── OSS ───────────────────────────────────────────────────────────────────
    oss_access_key_id: str = os.getenv("OSS_ACCESS_KEY_ID", os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", ""))
    oss_access_key_secret: str = os.getenv("OSS_ACCESS_KEY_SECRET", os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", ""))
    oss_region: str = os.getenv("OSS_REGION", "cn-shanghai")
    oss_endpoint: str = os.getenv("OSS_ENDPOINT", "https://oss-cn-shanghai.aliyuncs.com")
    oss_bucket: str = os.getenv("OSS_BUCKET", "")

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    api_title: str = "Knowledge Agent API"
    api_version: str = "1.0.0"
    cors_origins: list = ["*"]
    ssl_verify: bool = os.getenv("SSL_VERIFY", "false").lower() == "true"

    # ── Tavily 搜索 ───────────────────────────────────────────────────────────
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    tavily_api_url: str = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")

    # ── Email SMTP ───────────────────────────────────────────────────────────
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "465"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from_name: str = os.getenv("SMTP_FROM_NAME", "RAG知识库助手")
    smtp_use_ssl: bool = os.getenv("SMTP_USE_SSL", "true").lower() == "true"

    # ── 其他 ──────────────────────────────────────────────────────────────────
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    preload_graphs: bool = os.getenv("PRELOAD_GRAPHS", "false").lower() == "true"

    def __init__(self):
        _validate_env()


settings = Settings()


SUPPORTED_MODELS = {
    "qwen3.7-plus": {"name": "qwen3.7-plus", "description": "主力对话模型，适合复杂问答与多智能体调度", "provider": "dashscope", "max_tokens": 32000},
    "kimi-k2.6": {"name": "kimi-k2.6", "description": "Kimi 主力模型，适合通用问答与复杂推理", "provider": "dashscope", "max_tokens": 32000},
}


class SuccessMessages:
    API_READY = "Knowledge Agent API is ready"
