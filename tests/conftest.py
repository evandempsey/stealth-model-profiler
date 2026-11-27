import pytest

from stealth_model_profiler import cli


@pytest.fixture
def family_index(tmp_path, monkeypatch):
    path = tmp_path / cli.FAMILY_INDEX.name
    monkeypatch.setattr(cli, "FAMILY_INDEX", path)
    return path
