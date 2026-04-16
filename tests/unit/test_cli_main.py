import pytest
import typer

from recsys_lab.cli.main import _resolve_split_cache_override


def test_resolve_split_cache_override_supports_auto_enable_disable() -> None:
    assert _resolve_split_cache_override("auto") is None
    assert _resolve_split_cache_override("enable") is True
    assert _resolve_split_cache_override("disable") is False


def test_resolve_split_cache_override_rejects_invalid_values() -> None:
    with pytest.raises(typer.BadParameter):
        _resolve_split_cache_override("invalid")
