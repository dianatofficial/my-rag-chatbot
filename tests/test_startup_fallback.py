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


def test_build_rag_chain_falls_back_when_llm_is_unavailable(monkeypatch):
    module = importlib.reload(importlib.import_module("rag_engine"))

    def fail_llm(*_args, **_kwargs):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(module, "get_llm", fail_llm)

    chain = module.build_rag_chain(data_dir="data", k=2, rebuild=False)
    result = chain.invoke("چه داده‌ای در این دیتاست وجود دارد؟")

    assert result["answer"]
    assert "پاسخ جایگزین" in result["answer"] or "دیتاست" in result["answer"]


def test_embedding_dimension_for_large_model_does_not_probe_network():
    module = importlib.reload(importlib.import_module("rag_engine"))

    class DummyEmbeddings:
        model = "text-embedding-3-large"

        def embed_query(self, _text):
            raise AssertionError("embed_query should not be called for known models")

    assert module._embedding_dimension(DummyEmbeddings()) == 3072
