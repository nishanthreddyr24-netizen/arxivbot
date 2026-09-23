"""Schema conversion and config resolution. No network: nothing here calls out."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from arxivbot.llm import LLMConfig, LLMError, _load_dotenv, complete, gemini_schema


class Inner(BaseModel):
    name: str
    count: int


class Outer(BaseModel):
    title: str
    optional: str | None = None
    inner: Inner
    many: list[Inner] = Field(default_factory=list)


class TestSchemaConversion:
    """Pydantic's JSON Schema is not the dialect Gemini accepts."""

    def test_refs_are_inlined(self):
        schema = gemini_schema(Outer)
        assert "$defs" not in schema
        assert "$ref" not in str(schema)

    def test_nested_properties_survive(self):
        # Regression: the keyword allowlist was applied to property *names*,
        # which emptied every nested object and made Gemini reject the schema.
        inner = gemini_schema(Outer)["properties"]["inner"]
        assert set(inner["properties"]) == {"name", "count"}

    def test_properties_inside_arrays_survive(self):
        many = gemini_schema(Outer)["properties"]["many"]
        assert many["type"] == "array"
        assert set(many["items"]["properties"]) == {"name", "count"}

    def test_optional_becomes_nullable(self):
        optional = gemini_schema(Outer)["properties"]["optional"]
        assert optional.get("nullable") is True
        assert optional["type"] == "string"

    def test_required_list_is_preserved(self):
        assert set(gemini_schema(Outer)["required"]) == {"title", "inner"}

    def test_unsupported_keywords_are_dropped(self):
        assert "additionalProperties" not in str(gemini_schema(Outer))

    def test_enums_are_kept(self):
        from arxivbot.spec import Unknown

        schema = gemini_schema(Unknown)
        assert "enum" in str(schema)

    def test_the_real_spec_converts(self):
        # The whole point: ImplementationSpec must be expressible to a model.
        from arxivbot.spec import ImplementationSpec

        schema = gemini_schema(ImplementationSpec)
        assert "$ref" not in str(schema)
        assert "components" in schema["properties"]


class TestConfig:
    @pytest.fixture(autouse=True)
    def no_real_dotenv(self, tmp_path, monkeypatch):
        """Run from an empty directory.

        ``from_env`` reads ``.env`` from the working directory, so without
        this the developer's own key leaks into the test run - and into the
        failure output, which is how secrets end up in CI logs.
        """
        monkeypatch.chdir(tmp_path)
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)

    def test_env_key_is_picked_up(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        assert LLMConfig.from_env().api_key == "test-key"

    def test_google_key_is_a_fallback(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "other-key")
        assert LLMConfig.from_env().api_key == "other-key"

    def test_model_is_overridable(self, monkeypatch):
        monkeypatch.setenv("ARXIVBOT_MODEL", "some-other-model")
        assert LLMConfig.from_env().model == "some-other-model"

    def test_identity_records_provider_and_model(self):
        config = LLMConfig(provider="gemini", model="gemini-3.6-flash")
        assert config.identity == "gemini/gemini-3.6-flash"


class TestDotenv:
    def test_values_are_loaded(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
        _load_dotenv(env)
        import os

        assert os.environ["GEMINI_API_KEY"] == "from-file"

    def test_real_environment_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        env = tmp_path / ".env"
        env.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
        _load_dotenv(env)
        import os

        assert os.environ["GEMINI_API_KEY"] == "from-env"

    def test_quotes_and_comments_are_handled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ARXIVBOT_MODEL", raising=False)
        env = tmp_path / ".env"
        env.write_text('# a comment\nARXIVBOT_MODEL="quoted-model"\n', encoding="utf-8")
        _load_dotenv(env)
        import os

        assert os.environ["ARXIVBOT_MODEL"] == "quoted-model"


class TestGuards:
    def test_missing_key_explains_itself(self):
        config = LLMConfig(api_key="")
        with pytest.raises(LLMError, match="aistudio.google.com"):
            complete("hello", config=config)

    def test_unknown_provider_is_rejected(self):
        config = LLMConfig(provider="hotdog", api_key="x", use_cache=False)
        with pytest.raises(LLMError, match="unknown provider"):
            complete("hello", config=config)
