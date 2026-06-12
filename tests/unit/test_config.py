from _pytest.monkeypatch import MonkeyPatch

from deckbuilder.config import get_settings


def test_forge_ai_metadata_settings_default(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("DECKBUILDER_FORGE_AI_PROFILE", raising=False)
    monkeypatch.delenv("DECKBUILDER_FORGE_BUILD_ID", raising=False)
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.forge_ai_profile == "forge-baseline"
    assert settings.forge_build_id == "unknown"


def test_forge_ai_metadata_settings_override(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("DECKBUILDER_FORGE_AI_PROFILE", "forge-daily-snapshot")
    monkeypatch.setenv(
        "DECKBUILDER_FORGE_BUILD_ID",
        "2.0.13-SNAPSHOT-06.11-2026-06-11T19:12:03Z",
    )
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.forge_ai_profile == "forge-daily-snapshot"
    assert settings.forge_build_id == "2.0.13-SNAPSHOT-06.11-2026-06-11T19:12:03Z"

    get_settings.cache_clear()
