"""Configuration management for the AI Vehicle Monitoring System."""

from functools import lru_cache
from typing import Optional, List
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = Field(default="AI Vehicle Monitoring System", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # OpenAI configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=2048, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    
    # Database configuration
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Vector database configuration
    faiss_index_path: str = Field(default="./data/faiss_index", env="FAISS_INDEX_PATH")
    chromadb_path: str = Field(default="./data/chromadb", env="CHROMADB_PATH")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        env="EMBEDDING_MODEL"
    )
    vector_dimension: int = Field(default=384, env="VECTOR_DIMENSION")
    
    # Security configuration
    secret_key: str = Field(..., env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    
    # MCP configuration
    mcp_context_window: int = Field(default=4096, env="MCP_CONTEXT_WINDOW")
    mcp_max_context_updates: int = Field(default=100, env="MCP_MAX_CONTEXT_UPDATES")
    mcp_cache_ttl: int = Field(default=300, env="MCP_CACHE_TTL")
    
    # Performance configuration
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    vector_search_timeout: int = Field(default=5, env="VECTOR_SEARCH_TIMEOUT")
    
    # Monitoring configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # AWS configuration
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    s3_bucket_name: Optional[str] = Field(default=None, env="S3_BUCKET_NAME")
    
    # Redis configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_ttl: int = Field(default=3600, env="REDIS_TTL")
    
    # Ghana government specific
    ghana_data_protection_compliance: bool = Field(default=True, env="GHANA_DATA_PROTECTION_COMPLIANCE")
    audit_log_retention_days: int = Field(default=365, env="AUDIT_LOG_RETENTION_DAYS")
    encryption_key: str = Field(..., env="ENCRYPTION_KEY")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        env="CORS_ALLOW_METHODS"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        env="CORS_ALLOW_HEADERS"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Example configuration
        schema_extra = {
            "example": {
                "app_name": "AI Vehicle Monitoring System",
                "debug": False,
                "api_host": "0.0.0.0",
                "api_port": 8000,
                "openai_api_key": "sk-...",
                "database_url": "postgresql://user:pass@localhost/db",
                "secret_key": "your-secret-key",
                "faiss_index_path": "./data/faiss_index",
                "chromadb_path": "./data/chromadb"
            }
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
