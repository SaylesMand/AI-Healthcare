from resident.llm.client import drop_nulls, parse_model_spec, strict_json_schema
from resident.schemas import LLMAuditDraft, PlanChangeDraft, PlanDraft


def test_parse_model_spec():
    assert parse_model_spec("groq:qwen/qwen3.8-27b") == ("groq", "qwen/qwen3.8-27b")
    assert parse_model_spec("openrouter:nvidia/nemotron-3-ultra-550b-a55b:free") == (
        "openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free")
    assert parse_model_spec("ministral-14b-latest") == ("mistral", "ministral-14b-latest")


def _objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            yield node
        for v in node.values():
            yield from _objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _objects(v)


def test_strict_schema_requires_all_fields():
    for schema in (PlanDraft, PlanChangeDraft, LLMAuditDraft):
        js = strict_json_schema(schema)
        objects = list(_objects(js))
        assert objects
        for obj in objects:
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"])
        assert "default" not in str(js)


def test_optional_fields_become_nullable_and_defaults_restore():
    item = strict_json_schema(PlanChangeDraft)["$defs"]["PlanItemDraft"]["properties"]["applicability"]
    assert {"type": "null"} in item["anyOf"]
    raw = {"stop": [], "add": [{"kind": "drug", "name": "НФГ", "applicability": None, "dose": None,
                                "rationale": "r", "citations": []}], "rationale": "r"}
    draft = PlanChangeDraft.model_validate(drop_nulls(raw))
    assert draft.add[0].applicability == ""
    assert draft.add[0].dose is None
