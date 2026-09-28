import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from resident.kb.parse import load_chunks  # noqa: E402
from resident.retrieval.bm25 import BM25Retriever  # noqa: E402


@pytest.fixture(scope="session")
def chunks():
    return load_chunks()


@pytest.fixture(scope="session")
def kb(chunks):
    return {c.chunk_id: c for c in chunks}


@pytest.fixture(scope="session")
def bm25(chunks):
    return BM25Retriever(chunks)
