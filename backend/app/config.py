"""Application configuration - centralized settings from environment variables."""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All configuration values are centralized here and loaded from .env file
    or environment variables. Other modules should import from here.
    """

    # -------------------------------------------------------------------------
    # Server Configuration
    # -------------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:3001,http://localhost:3002,http://localhost:3003"

    # -------------------------------------------------------------------------
    # Database Configuration
    # -------------------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./jobs.db"

    # -------------------------------------------------------------------------
    # Job Management
    # -------------------------------------------------------------------------
    job_temp_dir: str = "/tmp/summarizer-jobs"
    job_retention_hours: int = 24
    cleanup_interval_minutes: int = 60
    cleanup_max_age_hours: int = 24

    # -------------------------------------------------------------------------
    # Ollama Configuration (Local LLM)
    # -------------------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_base_url_1: Optional[str] = None
    ollama_base_url_2: Optional[str] = None

    # -------------------------------------------------------------------------
    # Vision Processing Configuration
    # -------------------------------------------------------------------------
    vision_provider: str = "ollama"
    vision_model: str = "ministral-3:latest"

    # Separate classifier/extractor configuration (optional overrides)
    vision_classifier_provider: Optional[str] = None  # Inherits from vision_provider
    vision_classifier_model: Optional[str] = None     # Inherits from provider default
    vision_extractor_provider: Optional[str] = None   # Inherits from vision_provider
    vision_extractor_model: Optional[str] = None      # Inherits from provider default

    # -------------------------------------------------------------------------
    # OpenAI Configuration
    # -------------------------------------------------------------------------
    openai_api_key: Optional[str] = None
    openai_vision_model: str = "gpt-4.1-mini"

    # -------------------------------------------------------------------------
    # Google Gemini Configuration
    # -------------------------------------------------------------------------
    gemini_api_key: Optional[str] = None
    gemini_vision_model: str = "gemini-2.5-flash"

    # -------------------------------------------------------------------------
    # CLI Executor Configuration
    # -------------------------------------------------------------------------
    # Per-stage CLI provider selection (codex or claude)
    vision_cli_provider: str = "codex"
    summarizer_cli_provider: str = "codex"

    # Codex CLI settings
    codex_cli_timeout: int = 600  # 10 minutes - should be >= stream_idle_timeout_ms
    codex_cli_sandbox_policy: str = "workspace-write"
    codex_cli_reasoning_effort: Optional[str] = None  # minimal | low | medium | high | xhigh
    codex_cli_stream_idle_timeout_ms: Optional[int] = None  # Stream idle timeout (default 300000 = 5 min)

    # Claude CLI settings
    claude_cli_model: str = "sonnet"  # opus, sonnet, haiku
    claude_cli_timeout: int = 600  # 10 minutes

    # -------------------------------------------------------------------------
    # API Retry/Timeout Configuration
    # -------------------------------------------------------------------------
    api_max_retries: int = 3
    api_retry_delay: float = 5.0
    api_request_timeout: float = 120.0

    # -------------------------------------------------------------------------
    # Summarization Configuration
    # -------------------------------------------------------------------------
    summarizer_model_tier_1: str = "ministral-3:latest"
    summarizer_model_tier_2: str = "ministral-3:latest"
    summarizer_model_tier_3: str = "ministral-3:latest"
    summarizer_quality_threshold_high: int = 90
    summarizer_quality_threshold_low: int = 85
    summarizer_max_attempts: int = 30
    openai_summarizer_model: str = "gpt-4.1-mini"

    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------
    log_level: str = "INFO"
    log_level_extraction: Optional[str] = None
    log_level_vision: Optional[str] = None
    log_level_summarization: Optional[str] = None
    log_level_cli: Optional[str] = None  # For CLI-specific log level (Codex/Claude)
    log_dir: str = "./logs"
    log_max_bytes: int = 10_485_760  # 10MB
    log_backup_count: int = 5
    log_format: str = "%(asctime)s | %(levelname)-8s | [%(job_id)s] %(name)s | %(message)s"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def ollama_url_1(self) -> str:
        """Get first Ollama URL, falling back to base URL."""
        return self.ollama_base_url_1 or self.ollama_base_url

    @property
    def ollama_url_2(self) -> str:
        """Get second Ollama URL, falling back to first URL."""
        return self.ollama_base_url_2 or self.ollama_url_1

    @property
    def effective_log_level_extraction(self) -> str:
        """Get extraction log level with fallback to global."""
        return self.log_level_extraction or self.log_level

    @property
    def effective_log_level_vision(self) -> str:
        """Get vision log level with fallback to global."""
        return self.log_level_vision or self.log_level

    @property
    def effective_log_level_summarization(self) -> str:
        """Get summarization log level with fallback to global."""
        return self.log_level_summarization or self.log_level

    @property
    def effective_log_level_cli(self) -> str:
        """Get CLI log level with fallback to global."""
        return self.log_level_cli or self.log_level

    @property
    def effective_vision_classifier_provider(self) -> str:
        """Get classifier provider with fallback to vision_provider."""
        return self.vision_classifier_provider or self.vision_provider

    @property
    def effective_vision_extractor_provider(self) -> str:
        """Get extractor provider with fallback to vision_provider."""
        return self.vision_extractor_provider or self.vision_provider

    def get_vision_model_for_provider(self, provider: str) -> str:
        """Get the default model for a vision provider."""
        if provider == "openai":
            return self.openai_vision_model
        elif provider == "gemini":
            return self.gemini_vision_model
        return self.vision_model

    @property
    def effective_vision_classifier_model(self) -> str:
        """Get classifier model with fallback chain."""
        if self.vision_classifier_model:
            return self.vision_classifier_model
        return self.get_vision_model_for_provider(self.effective_vision_classifier_provider)

    @property
    def effective_vision_extractor_model(self) -> str:
        """Get extractor model with fallback chain."""
        if self.vision_extractor_model:
            return self.vision_extractor_model
        return self.get_vision_model_for_provider(self.effective_vision_extractor_provider)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


settings = Settings()
