"""Lookup is exact-key by design, so the keys had better be right."""

from __future__ import annotations

import pytest

from arxivbot import store
from arxivbot.spec import Extractor, ImplementationSpec


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the store at a temp directory so tests never touch the real cache."""
    monkeypatch.setenv("ARXIVBOT_CACHE", str(tmp_path))
    yield tmp_path


def make_spec(arxiv_id: str, title: str = "T") -> ImplementationSpec:
    return ImplementationSpec(
        arxiv_id=arxiv_id,
        title=title,
        extractor=Extractor(model="test/model", arxivbot_version="0.1.0"),
    )


class TestKeys:
    def test_key_is_the_identifier(self):
        assert store._key("1706.03762v7") == "1706.03762v7"

    def test_url_style_ids_reduce_to_the_identifier(self):
        assert store._key("https://arxiv.org/abs/1706.03762v7") == "1706.03762v7"

    def test_old_style_ids_are_path_safe(self):
        # cs.CL/0108005 must not become a subdirectory.
        assert "/" not in store._key("cs.CL/0108005")

    @pytest.mark.parametrize(
        "arxiv_id,expected",
        [("1706.03762v7", True), ("1706.03762", False), ("cs.CL/0108005", False)],
    )
    def test_version_detection(self, arxiv_id, expected):
        assert store.has_version(arxiv_id) is expected


class TestUnversionedLookup:
    """Nobody types version numbers, so an unversioned id must still match."""

    def test_unversioned_id_finds_the_stored_version(self):
        store.put(make_spec("1706.03762v7"))
        hit = store.get_local("1706.03762")
        assert hit is not None
        assert hit.spec.arxiv_id == "1706.03762v7"

    def test_url_finds_the_stored_version(self):
        store.put(make_spec("1706.03762v7"))
        hit = store.get_local("https://arxiv.org/abs/1706.03762")
        assert hit is not None

    def test_newest_version_wins(self):
        store.put(make_spec("1706.03762v2", title="old"))
        store.put(make_spec("1706.03762v11", title="new"))
        hit = store.get_local("1706.03762")
        # v11 > v2 numerically, not alphabetically.
        assert hit.spec.title == "new"

    def test_explicit_version_does_not_fall_back(self):
        store.put(make_spec("1706.03762v7"))
        assert store.get_local("1706.03762v3") is None

    def test_missing_paper_is_a_miss(self):
        assert store.get_local("9999.99999") is None


class TestRoundTrip:
    def test_put_then_get(self):
        store.put(make_spec("2006.11239v2", title="DDPM"))
        hit = store.get_local("2006.11239v2")
        assert hit.spec.title == "DDPM"
        assert hit.source == "local"
        assert hit.stale is False

    def test_stale_schema_is_flagged_not_rejected(self):
        spec = make_spec("1234.56789v1")
        spec.extractor.schema_version = "0.0-ancient"
        store.put(spec)
        hit = store.get_local("1234.56789v1")
        assert hit is not None and hit.stale is True

    def test_corrupt_file_is_a_miss_not_a_crash(self):
        store.put(make_spec("1111.11111v1"))
        store.local_path("1111.11111v1").write_text("{not json", encoding="utf-8")
        assert store.get_local("1111.11111v1") is None


class TestSharedIndex:
    def test_shared_lookup_degrades_to_none(self, monkeypatch):
        # Unreachable host: the shared index is an optimisation, never a
        # dependency, so this must not raise.
        monkeypatch.setenv("ARXIVBOT_SPEC_INDEX", "http://127.0.0.1:1/specs")
        assert store.get_shared("1706.03762v7") is None

    def test_get_prefers_local_without_touching_the_network(self, monkeypatch):
        store.put(make_spec("1706.03762v7"))

        def explode(*args, **kwargs):
            raise AssertionError("network was used despite a local hit")

        monkeypatch.setattr(store.httpx, "get", explode)
        monkeypatch.setattr(store, "fetch_metadata", explode)
        assert store.get("1706.03762").source == "local"


class TestContribution:
    def test_export_writes_the_index_layout(self, tmp_path):
        store.put(make_spec("1706.03762v7"))
        out = store.export_for_contribution("1706.03762v7", tmp_path / "repo")
        assert out.parent.name == "specs"
        assert out.name == "1706.03762v7.json"

    def test_export_without_a_local_spec_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            store.export_for_contribution("0000.00000v1", tmp_path / "repo")
