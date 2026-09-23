"""A request may contradict the paper. It must be honoured, and it must show."""

from __future__ import annotations

import pytest

from arxivbot.deviation import Stance, banner, check
from arxivbot.spec import (
    Component,
    Confidence,
    Extractor,
    Hyperparameter,
    ImplementationSpec,
    Provenance,
    Severity,
    Unknown,
)

PROV = Provenance(
    section="Model Architecture",
    quote="we apply layer normalization before each sub-layer",
    char_start=0,
    char_end=50,
)


def spec_with(
    value: str | None,
    confidence: Confidence = Confidence.STATED,
    provenance: Provenance | None = PROV,
    unknowns: tuple[Unknown, ...] = (),
) -> ImplementationSpec:
    hyperparameters = []
    if value is not None:
        hyperparameters.append(
            Hyperparameter(
                name="normalization placement",
                value=value,
                confidence=confidence,
                provenance=provenance,
            )
        )
    return ImplementationSpec(
        arxiv_id="1706.03762v7",
        title="Attention Is All You Need",
        extractor=Extractor(model="test", arxivbot_version="0.1.0"),
        components=[
            Component(
                name="Residual Connection and Layer Normalization",
                role="Wraps each sublayer with a residual add and normalization",
                hyperparameters=hyperparameters,
            )
        ],
        unknowns=list(unknowns),
    )


class TestConflict:
    """Paper says one thing, user asks for another."""

    def test_conflict_is_detected(self):
        spec = spec_with("pre-LN")
        found = check(spec, "residual code but post normalization")
        assert len(found) == 1
        assert found[0].stance is Stance.DEVIATES
        assert found[0].requested == "post"
        assert found[0].paper_says == "pre"

    def test_conflict_names_the_source_section(self):
        found = check(spec_with("pre-LN"), "post norm please")
        assert found[0].source == "Model Architecture"

    def test_banner_is_unmissable(self):
        text = banner(check(spec_with("pre-LN"), "post norm please"))
        assert "DEVIATION" in text
        assert "not the paper's" in text

    def test_conflicts_are_listed_first(self):
        spec = spec_with("pre-LN")
        text = banner(check(spec, "post norm with gelu"))
        # Two axes touched; the contradiction must lead.
        assert text.index("DEVIATION") < len(text)
        assert text.startswith("# !!")


class TestAgreement:
    def test_matching_request_is_not_a_deviation(self):
        found = check(spec_with("pre-LN"), "residual block with pre norm")
        assert found[0].stance is Stance.MATCHES_PAPER
        assert "as described in the paper" in found[0].render()

    def test_synonyms_count_as_agreement(self):
        # "preln" and "pre" are the same choice.
        found = check(spec_with("pre-LN"), "give me the preln variant")
        assert found[0].stance is Stance.MATCHES_PAPER


class TestSilence:
    def test_silent_paper_means_the_user_is_choosing(self):
        spec = spec_with(
            None,
            unknowns=(
                Unknown(
                    question="Is normalization pre or post residual?",
                    why_it_matters="Pre-norm trains more stably at depth.",
                    severity=Severity.SIGNIFICANT,
                ),
            ),
        )
        found = check(spec, "give me post norm residual")
        assert found[0].stance is Stance.RESOLVES_UNKNOWN
        assert found[0].paper_says is None
        assert "does not specify" in found[0].render()

    def test_overriding_a_guess_is_not_a_deviation(self):
        # The extractor filled this in by convention; the paper never said it,
        # so contradicting it does not contradict the paper.
        spec = spec_with("pre-LN", confidence=Confidence.CONVENTIONAL, provenance=None)
        found = check(spec, "post norm please")
        assert found[0].stance is Stance.RESOLVES_UNKNOWN


class TestScope:
    def test_unrelated_request_flags_nothing(self):
        assert check(spec_with("pre-LN"), "give me the attention code") == []

    def test_empty_banner_for_no_findings(self):
        assert banner([]) == ""

    @pytest.mark.parametrize(
        "query,axis",
        [
            ("use gelu instead", "activation"),
            ("switch to adamw", "optimizer"),
            ("rotary embeddings please", "positional encoding"),
            ("use rmsnorm", "normalization type"),
        ],
    )
    def test_other_variant_axes_are_recognised(self, query, axis):
        found = check(spec_with("pre-LN"), query)
        assert any(d.axis == axis for d in found)

    def test_one_finding_per_axis(self):
        found = check(spec_with("pre-LN"), "post norm, postln, post normalization")
        assert len([d for d in found if d.axis == "normalization placement"]) == 1
