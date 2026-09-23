"""The spec's integrity rules are the product, so they get tested first."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arxivbot.spec import (
    SCHEMA_VERSION,
    Component,
    Confidence,
    Extractor,
    Hyperparameter,
    ImplementationSpec,
    Provenance,
    Severity,
    TrainingSpec,
    Unknown,
)


@pytest.fixture
def prov() -> Provenance:
    return Provenance(
        section="Model Architecture", quote="d_k = 64", char_start=100, char_end=112
    )


@pytest.fixture
def spec(prov: Provenance) -> ImplementationSpec:
    return ImplementationSpec(
        arxiv_id="1706.03762v7",
        title="Attention Is All You Need",
        extractor=Extractor(model="test/model", arxivbot_version="0.1.0"),
        components=[
            Component(
                name="Scaled Dot-Product Attention",
                role="Attention scaled by 1/sqrt(d_k)",
                hyperparameters=[
                    Hyperparameter(
                        name="d_k",
                        value="64",
                        confidence=Confidence.STATED,
                        provenance=prov,
                    )
                ],
            )
        ],
        training=TrainingSpec(
            optimizer="Adam",
            hyperparameters=[
                Hyperparameter(
                    name="warmup_steps", value="4000", confidence=Confidence.CONVENTIONAL
                )
            ],
        ),
        unknowns=[
            Unknown(
                question="How are projections initialised?",
                why_it_matters="Interacts with the 1/sqrt(d_k) scaling.",
                severity=Severity.BLOCKING,
            ),
            Unknown(
                question="Dropout placement inside the block?",
                why_it_matters="Minor effect on regularisation.",
                severity=Severity.MINOR,
            ),
        ],
    )


class TestProvenanceRule:
    """A claim the paper 'states' must point at where it says it."""

    def test_stated_without_provenance_is_rejected(self):
        with pytest.raises(ValidationError, match="cites no source"):
            Hyperparameter(name="lr", value="3e-4", confidence=Confidence.STATED)

    def test_stated_with_provenance_is_accepted(self, prov):
        hp = Hyperparameter(
            name="lr", value="3e-4", confidence=Confidence.STATED, provenance=prov
        )
        assert hp.provenance is prov

    @pytest.mark.parametrize(
        "confidence",
        [Confidence.CONVENTIONAL, Confidence.GUESS, Confidence.DERIVED],
    )
    def test_unstated_confidences_need_no_provenance(self, confidence):
        # Only STATED is a claim about the paper's own words.
        hp = Hyperparameter(name="init", value="xavier", confidence=confidence)
        assert hp.provenance is None


class TestProvenanceSpans:
    def test_backwards_span_is_rejected(self):
        with pytest.raises(ValidationError, match="char_end must not precede"):
            Provenance(section="X", quote="q", char_start=500, char_end=100)

    def test_negative_offset_is_rejected(self):
        with pytest.raises(ValidationError):
            Provenance(section="X", quote="q", char_start=-1, char_end=10)

    def test_empty_span_is_allowed(self):
        # A zero-length span is odd but not wrong; don't reject it.
        assert Provenance(section="X", quote="q", char_start=7, char_end=7)

    def test_overlong_quote_is_rejected(self):
        with pytest.raises(ValidationError):
            Provenance(section="X", quote="z" * 601, char_start=0, char_end=1)


class TestQueries:
    def test_blocking_unknowns_filters_by_severity(self, spec):
        blocking = spec.blocking_unknowns
        assert len(blocking) == 1
        assert "initialised" in blocking[0].question

    def test_component_lookup_is_case_insensitive(self, spec):
        assert spec.component("scaled dot-product attention") is not None
        assert spec.component("nonexistent") is None

    def test_all_hyperparameters_spans_every_location(self, spec):
        names = {hp.name for hp in spec.all_hyperparameters()}
        assert names == {"d_k", "warmup_steps"}

    def test_confidence_breakdown_counts_every_level(self, spec):
        counts = spec.confidence_breakdown()
        assert counts["stated"] == 1
        assert counts["conventional"] == 1
        assert set(counts) == {c.value for c in Confidence}


class TestSerialisation:
    def test_round_trip_preserves_content(self, spec, tmp_path):
        path = spec.save(tmp_path / "s.json")
        loaded = ImplementationSpec.load(path)
        assert loaded.arxiv_id == spec.arxiv_id
        assert loaded.components[0].name == spec.components[0].name
        assert loaded.unknowns[0].severity is Severity.BLOCKING

    def test_extractor_records_schema_version(self, spec):
        assert spec.extractor.schema_version == SCHEMA_VERSION

    def test_json_schema_is_generatable(self):
        # This is what constrains an extractor's output, so it must build.
        assert '"ImplementationSpec"' in ImplementationSpec.json_schema()
