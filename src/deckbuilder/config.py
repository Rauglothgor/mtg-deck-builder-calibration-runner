"""Application configuration."""

from functools import lru_cache
from os import path
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    database_url: str = Field(
        default="postgresql+psycopg://deckbuilder:deckbuilder@localhost:5432/deckbuilder"
    )
    log_level: str = Field(default="INFO")
    llm_disabled: bool = Field(default=True)
    forge_root: Path = Field(default=Path("/opt/forge"))
    forge_decks_dir: Path = Field(default=Path.home() / ".forge" / "decks" / "commander")
    forge_bundled_deck_dir: Path = Field(
        default=Path("/opt/forge/res/adventure/Realm of Legends/decks/legends")
    )
    default_seed: int = Field(default=42)

    model_config = SettingsConfigDict(env_prefix="DECKBUILDER_", env_file=".env", extra="ignore")

    @field_validator("forge_root", "forge_decks_dir", "forge_bundled_deck_dir", mode="before")
    @classmethod
    def expand_path_value(cls, value: object) -> object:
        """Expand shell-style path markers from `.env` values."""
        if isinstance(value, str):
            return Path(path.expandvars(value)).expanduser()
        if isinstance(value, Path):
            return value.expanduser()
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return validated application settings."""
    return Settings()
