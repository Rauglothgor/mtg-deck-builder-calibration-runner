"""Basic package smoke tests."""

from deckbuilder import __version__
from deckbuilder.config import get_settings


def test_package_version_and_settings() -> None:
    """The package imports and default settings validate."""
    assert __version__ == "0.5.0"
    assert get_settings().default_seed == 42
