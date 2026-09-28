from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.getenv("RESIDENT_ROOT", Path(__file__).resolve().parents[2]))
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
KB_DIR = DATA / "kb"
CASES_DIR = DATA / "cases"
EVAL_DIR = DATA / "eval"
LLM_CACHE_DIR = DATA / "llm_cache"

CR_ID = "154_4"
CR_TITLE = "Острый коронарный синдром без подъема сегмента ST электрокардиограммы"
CHUNKS_PATH = KB_DIR / "chunks.jsonl"
ABBR_PATH = KB_DIR / "abbreviations.json"


@dataclass(frozen=True)
class Settings:
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    # "провайдер:модель", провайдер - groq | openrouter | mistral
    llm_model: str = os.getenv("LLM_MODEL", "groq:qwen/qwen3.8-27b")  # планировщик и пересмотр плана
    auditor_model: str = os.getenv("AUDITOR_MODEL", "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free")
    llm_mode: str = os.getenv("LLM_MODE", "replay")  # live | replay
    retriever: str = os.getenv("RETRIEVER", "hybrid")  # bm25 | dense | hybrid
    embed_model: str = os.getenv("EMBED_MODEL", "deepvk/USER2-base")


settings = Settings()
