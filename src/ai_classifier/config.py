"""Settings, from AC_-prefixed environment variables or a .env file.

No list-typed fields. Pydantic Settings parses those as JSON, so `AC_PROVIDERS=offline,vertex`
fails with a decode error that reads like a bug in the tool rather than a malformed value.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AC_", env_file=".env", extra="ignore")

    products: Path = Path("data/products.csv")
    labels: Path = Path("data/labels.csv")
    output_dir: Path = Path("output")
    provider: str = "offline"

    bin_count: int = 10
    # 0.95 is a default, not a recommendation. The right target depends on what an error
    # costs, and `calibrate --target` exists because that is a business decision.
    target_accuracy: float = 0.95

    log_level: str = "INFO"


def load_settings() -> Settings:
    return Settings()
