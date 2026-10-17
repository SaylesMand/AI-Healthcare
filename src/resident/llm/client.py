from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Protocol, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from resident.config import LLM_CACHE_DIR, settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class CacheMiss(LLMError):
    pass


class LLMClient(Protocol):
    def complete(self, system: str, user: str, schema: type[T]) -> T:
        """Возвращает ответ LLM, провалидированный по схеме."""


def cache_key(model: str, system: str, user: str, schema: type[BaseModel]) -> str:
    """Считает ключ кэша по модели, промптам и схеме ответа."""
    payload = json.dumps(
        {"model": model, "system": system, "user": user, "schema": schema.model_json_schema()},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class CachedLLM:
    """Клиент LLM с дисковым кэшем, валидацией ответа и ретраями."""

    def __init__(self, provider: Provider | None = None, model: str | None = None,
                 mode: str | None = None, cache_dir: Path = LLM_CACHE_DIR, max_retries: int = 3):
        self.model = model or settings.llm_model
        self.mode = mode or settings.llm_mode
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self._provider = provider
        self.stats = {"cache_hits": 0, "api_calls": 0, "reasoning_tokens": 0}
        self.used_keys: set[str] = set()

    @property
    def provider(self) -> Provider:
        """Создает провайдера по строке модели при первом обращении."""
        if self._provider is None:
            self._provider = make_provider(self.model)
        return self._provider

    def complete(self, system: str, user: str, schema: type[T]) -> T:
        """Возвращает ответ из кэша, а в режиме live запрашивает API и сохраняет ответ."""
        key = cache_key(self.model, system, user, schema)
        path = self.cache_dir / f"{key}.json"
        self.used_keys.add(key)
        if path.exists():
            self.stats["cache_hits"] += 1
            record = json.loads(path.read_text(encoding="utf-8"))
            return schema.model_validate_json(record["response"])
        if self.mode != "live":
            raise CacheMiss(f"нет ответа в кэше ({key}); запустите с LLM_MODE=live и ключом провайдера в .env")

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = self.provider.generate(system, user, schema)
                self.stats["api_calls"] += 1
                self.stats["reasoning_tokens"] = getattr(self.provider, "reasoning_tokens", 0)
                parsed = schema.model_validate_json(raw)
            except ValidationError as e:  # модель нарушила схему - просим ещё раз
                last_err = e
                continue
            except Exception as e:  # сеть, лимиты
                last_err = e
                time.sleep(2 ** attempt * 2)
                continue
            path.write_text(json.dumps(
                {"model": self.model, "schema": schema.__name__, "system": system, "user": user,
                 "response": raw}, ensure_ascii=False, indent=1,
            ), encoding="utf-8")
            return parsed
        raise LLMError(f"LLM не вернула валидный ответ за {self.max_retries} попытки: {last_err}")

    def prune(self) -> int:
        """Удаляет из кэша ответы, не использованные в текущем прогоне."""
        removed = 0
        for path in self.cache_dir.glob("*.json"):
            if path.stem not in self.used_keys:
                path.unlink()
                removed += 1
        return removed


class Provider(Protocol):
    def generate(self, system: str, user: str, schema: type[BaseModel]) -> str:
        """Возвращает сырой JSON-ответ модели по схеме."""


class MistralProvider:
    """Провайдер Mistral со строгим JSON-выводом по схеме Pydantic."""

    def __init__(self, model: str):
        if not settings.mistral_api_key:
            raise LLMError("MISTRAL_API_KEY не задан в .env")
        try:
            from mistralai import Mistral  # SDK 1.x
        except ImportError:
            from mistralai.client import Mistral  # SDK 2.x
        self.client = Mistral(api_key=settings.mistral_api_key)
        self.model = model
        self._last_call = 0.0

    def generate(self, system: str, user: str, schema: type[BaseModel]) -> str:
        """Отправляет запрос в Mistral с учетом лимита запросов и возвращает JSON-ответ."""
        wait = 1.1 - (time.monotonic() - self._last_call)  # бесплатный план: ~1 запрос/с
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
        resp = self.client.chat.parse(
            model=self.model,
            response_format=schema,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            random_seed=0,
        )
        return resp.choices[0].message.content


# провайдер: (базовый URL, поле Settings с ключом, переменная .env)
OPENAI_COMPATIBLE = {
    "groq": ("https://api.groq.com/openai/v1", "groq_api_key", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "openrouter_api_key", "OPENROUTER_API_KEY"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai", "google_api_key",
               "GOOGLE_AI_STUDIO_API_KEY"),
}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Разбирает строку вида "groq:qwen/qwen3.8-27b" на провайдера и модель, без префикса - Mistral."""
    provider, sep, model = spec.partition(":")
    if sep and provider in (*OPENAI_COMPATIBLE, "mistral"):
        return provider, model
    return "mistral", spec


def _nullable(node: dict) -> bool:
    """Проверяет, допускает ли узел JSON-схемы значение null."""
    return node.get("type") == "null" or any(v.get("type") == "null" for v in node.get("anyOf", []))


def strict_json_schema(schema: type[BaseModel]) -> dict:
    """Приводит схему Pydantic к строгому виду: все поля обязательны, необязательные допускают null."""
    def fix(node):
        if isinstance(node, list):
            return [fix(v) for v in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for k, v in node.items():
            if k in ("title", "default"):
                continue
            out[k] = {name: fix(sub) for name, sub in v.items()} if k in ("properties", "$defs") else fix(v)
        if out.get("type") == "object" and "properties" in out:
            required = set(out.get("required", []))
            for name, prop in out["properties"].items():
                if name not in required and not _nullable(prop):
                    out["properties"][name] = {"anyOf": [prop, {"type": "null"}]}
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out
    return fix(schema.model_json_schema())


def drop_nulls(value):
    """Удаляет ключи со значением null, чтобы Pydantic подставил значения по умолчанию."""
    if isinstance(value, dict):
        return {k: drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_nulls(v) for v in value]
    return value


class OpenAICompatProvider:
    """Провайдер с OpenAI-совместимым API (Groq, OpenRouter, Google): JSON по схеме, рассуждения отключены."""

    def __init__(self, provider: str, model: str, max_tokens: int = 3000, max_wait_s: float = 900):
        base_url, key_attr, key_env = OPENAI_COMPATIBLE[provider]
        self.api_key = getattr(settings, key_attr)
        if not self.api_key:
            raise LLMError(f"{key_env} не задан в .env")
        self.provider, self.model, self.base_url = provider, model, base_url
        self.max_tokens, self.max_wait_s = max_tokens, max_wait_s
        self.output_mode, self.no_thinking = self._capabilities()
        self.reasoning_tokens = 0

    def _capabilities(self) -> tuple[str, dict]:
        """Определяет способ получить JSON по схеме и параметр, отключающий рассуждения модели."""
        if self.provider == "groq":
            if self.model.startswith("openai/gpt-oss"):
                raise LLMError(f"{self.model}: режим рассуждений нельзя отключить")
            return "schema", {"reasoning_effort": "none"} if self.model.startswith("qwen/") else {}
        if self.provider == "google":
            # каталог Google не сообщает поддерживаемые параметры, поэтому уровень thinking берётся из роли:
            # аудитору разрешён более высокий, потому что перепроверке важнее качество
            auditor = parse_model_spec(settings.auditor_model) == ("google", self.model)
            level = settings.auditor_thinking_level if auditor else settings.llm_thinking_level
            return "schema", {"reasoning_effort": level}
        resp = requests.get(f"{self.base_url}/models", timeout=60)
        resp.raise_for_status()
        meta = next((m for m in resp.json()["data"] if m["id"] == self.model), None)
        if meta is None:
            raise LLMError(f"модели {self.model} нет в каталоге {self.provider}")
        params = set(meta.get("supported_parameters") or [])
        mode = "schema" if "structured_outputs" in params else "tools" if "tools" in params else None
        if mode is None:
            raise LLMError(f"{self.model} не поддерживает ни JSON-схему, ни вызов инструментов")
        return mode, {"reasoning": {"enabled": False}} if params & {"reasoning", "include_reasoning"} else {}

    def _post(self, body: dict) -> dict:
        """Отправляет запрос, дожидаясь сброса лимитов при ответе 429, и возвращает JSON ответа."""
        deadline = time.monotonic() + self.max_wait_s
        while True:
            resp = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=300,
                                 headers={"Authorization": f"Bearer {self.api_key}"})
            if resp.status_code != 429 and resp.status_code < 500:
                break
            wait = min(float(resp.headers.get("retry-after") or 20), 60)
            if time.monotonic() + wait > deadline:
                break
            time.sleep(wait)
        if resp.status_code != 200:
            raise LLMError(f"{self.provider} HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if "error" in data:
            raise LLMError(f"{self.provider}: {str(data['error'])[:300]}")
        return data

    def generate(self, system: str, user: str, schema: type[BaseModel]) -> str:
        """Запрашивает ответ по схеме без рассуждений и возвращает JSON без null-полей."""
        json_schema = strict_json_schema(schema)
        body = {
            "model": self.model, "temperature": 0, "seed": 0, "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self.no_thinking,
        }
        if self.provider == "google":
            body.pop("seed")  # OpenAI-совместимый API Google не знает этого поля и отвечает 400
        if self.output_mode == "schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": json_schema}}
        else:
            body["tools"] = [{"type": "function", "function": {
                "name": "answer", "description": "Вернуть ответ строго по схеме", "parameters": json_schema}}]
            body["tool_choice"] = {"type": "function", "function": {"name": "answer"}}
        data = self._post(body)
        details = (data.get("usage") or {}).get("completion_tokens_details") or {}
        self.reasoning_tokens += details.get("reasoning_tokens") or 0
        message = data["choices"][0]["message"]
        if self.output_mode == "tools":
            calls = message.get("tool_calls") or []
            if not calls:
                raise LLMError("модель не вызвала инструмент с ответом")
            raw = calls[0]["function"]["arguments"]
        else:
            raw = message.get("content") or ""
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        return json.dumps(drop_nulls(json.loads(raw)), ensure_ascii=False)


def make_provider(spec: str) -> Provider:
    """Создает провайдера по строке вида "провайдер:модель"."""
    provider, model = parse_model_spec(spec)
    return MistralProvider(model) if provider == "mistral" else OpenAICompatProvider(provider, model)


def get_llm(model: str | None = None) -> CachedLLM:
    """Создает клиент LLM с настройками из .env."""
    return CachedLLM(model=model)
