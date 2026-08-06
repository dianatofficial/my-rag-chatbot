import importlib
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def preserve_index_dir():
    module = importlib.import_module("rag_engine")
    index_dir = Path(module.INDEX_DIR)
    backup = None
    if index_dir.exists():
        backup = index_dir.with_suffix(".bak")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(index_dir), str(backup))

    yield

    if backup is not None and backup.exists():
        if index_dir.exists():
            shutil.rmtree(index_dir)
        shutil.move(str(backup), str(index_dir))
    elif index_dir.exists():
        shutil.rmtree(index_dir)


def test_load_vectorstore_builds_index_when_missing(monkeypatch):
    module = importlib.reload(importlib.import_module("rag_engine"))
    index_dir = Path(module.INDEX_DIR)
    if index_dir.exists():
        shutil.rmtree(index_dir)

    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("USE_OPENAI_EMBEDDINGS", "0")

    store = module.load_vectorstore(force_build=True)
    assert store.index.ntotal > 0
