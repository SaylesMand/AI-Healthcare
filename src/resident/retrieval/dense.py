from __future__ import annotations

import hashlib
import json

import numpy as np

from resident.config import KB_DIR, settings
from resident.retrieval.base import Retriever, doc_text
from resident.schemas import Chunk

# Префиксы из карточки модели; добавляем вручную и НЕ передаём prompt_name,
# чтобы префикс не оказался продублирован.
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "


def _fingerprint(chunks: list[Chunk]) -> str:
    """Считает хэш состава чанков для проверки кэша эмбеддингов."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(doc_text(c).encode())
    return h.hexdigest()[:16]


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, chunks: list[Chunk], model_name: str | None = None, device: str | None = None,
                 max_seq_length: int = 2048):
        super().__init__(chunks)
        from sentence_transformers import SentenceTransformer  # тяжёлый импорт - только здесь

        self.model_name = model_name or settings.embed_model
        self.model = SentenceTransformer(self.model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.doc_emb = self._load_or_encode()

    @property
    def _cache_paths(self):
        """Возвращает пути файлов кэша эмбеддингов для модели."""
        slug = self.model_name.replace("/", "__")
        return KB_DIR / f"emb_{slug}.npy", KB_DIR / f"emb_{slug}.json"

    def _load_or_encode(self) -> np.ndarray:
        """Загружает эмбеддинги из кэша или считает их заново."""
        emb_path, meta_path = self._cache_paths
        fp = _fingerprint(self.chunks)
        if emb_path.exists() and meta_path.exists():
            if json.loads(meta_path.read_text()).get("fingerprint") == fp:
                return np.load(emb_path)
        emb = self.model.encode(
            [DOC_PREFIX + doc_text(c) for c in self.chunks],
            batch_size=8, normalize_embeddings=True, show_progress_bar=True,
        ).astype(np.float32)
        np.save(emb_path, emb)
        meta_path.write_text(json.dumps({"fingerprint": fp, "model": self.model_name, "n": len(self.chunks)}))
        return emb

    def scores(self, query: str) -> np.ndarray:
        """Возвращает косинусное сходство запроса со всеми чанками."""
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        return self.doc_emb @ q
