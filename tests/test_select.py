"""One paper answers many different requests - that is what selection is for."""

from __future__ import annotations

import pytest

from arxivbot.select import did_you_mean, select, suggest
from arxivbot.spec import (
    Component,
    Extractor,
    ImplementationSpec,
    Provenance,
    Severity,
    Unknown,
)


def component(name: str, role: str, deps: tuple[str, ...] = ()) -> Component:
    return Component(
        name=name,
        role=role,
        depends_on=list(deps),
        provenance=Provenance(
            section="Model Architecture", quote="...", char_start=0, char_end=3
        ),
    )


@pytest.fixture
def transformer() -> ImplementationSpec:
    return ImplementationSpec(
        arxiv_id="1706.03762v7",
        title="Attention Is All You Need",
        extractor=Extractor(model="test", arxivbot_version="0.1.0"),
        components=[
            component(
                "Scaled Dot-Product Attention",
                "Softmax over QK^T scaled by 1/sqrt(d_k), applied to V",
            ),
            component(
                "Multi-Head Attention",
                "Runs h attention heads in parallel and concatenates them",
                ("Scaled Dot-Product Attention",),
            ),
            component(
                "Position-wise Feed-Forward Network",
                "Two linear layers with a ReLU between them",
            ),
            component(
                "Residual Connection and Layer Normalization",
                "Wraps each sublayer as LayerNorm(x + Sublayer(x))",
            ),
            component(
                "Encoder Layer",
                "Self-attention sublayer followed by a feed-forward sublayer",
                (
                    "Multi-Head Attention",
                    "Position-wise Feed-Forward Network",
                    "Residual Connection and Layer Normalization",
                ),
            ),
        ],
        unknowns=[
            Unknown(
                question="How are the projection matrices initialised?",
                why_it_matters="Interacts with the 1/sqrt(d_k) scaling.",
                severity=Severity.BLOCKING,
                applies_to="Multi-Head Attention",
            ),
            Unknown(
                question="Pre-LN or post-LN?",
                why_it_matters="Changes training stability substantially.",
                severity=Severity.SIGNIFICANT,
                applies_to="Residual Connection and Layer Normalization",
            ),
            Unknown(
                question="What precision was used?",
                why_it_matters="Affects reproducibility of reported numbers.",
                severity=Severity.MINOR,
            ),
        ],
    )


class TestDistinctRequests:
    """The same paper, asked two different things, gives two different answers."""

    def test_attention_request(self, transformer):
        names = select(transformer, "give me the attention code").names()
        assert "Multi-Head Attention" in names
        assert "Scaled Dot-Product Attention" in names
        assert "Residual Connection and Layer Normalization" not in names

    def test_residual_request(self, transformer):
        names = select(transformer, "residual stream").names()
        assert names == ["Residual Connection and Layer Normalization"]

    def test_feed_forward_request(self, transformer):
        names = select(transformer, "feed forward network").names()
        assert "Position-wise Feed-Forward Network" in names

    def test_requests_do_not_collide(self, transformer):
        attn = set(select(transformer, "attention").names())
        resid = set(select(transformer, "residual").names())
        assert attn.isdisjoint(resid)


class TestScopeControl:
    def test_weak_role_match_does_not_hitchhike(self, transformer):
        # "Encoder Layer" mentions attention in its role, but asking for
        # attention must not return most of the model.
        names = select(transformer, "attention").names()
        assert "Encoder Layer" not in names
        assert len(names) <= 3

    def test_asking_for_a_composite_does_expand(self, transformer):
        names = select(transformer, "encoder layer").names()
        assert "Encoder Layer" in names
        assert "Multi-Head Attention" in names
        # Transitive: encoder -> multi-head -> scaled dot-product.
        assert "Scaled Dot-Product Attention" in names

    def test_dependencies_are_not_duplicated(self, transformer):
        names = select(transformer, "encoder layer").names()
        assert len(names) == len(set(names))


class TestUnknowns:
    def test_only_relevant_gaps_surface(self, transformer):
        questions = [u.question for u in select(transformer, "residual").unknowns]
        assert "Pre-LN or post-LN?" in questions
        assert "How are the projection matrices initialised?" not in questions

    def test_global_gaps_always_surface(self, transformer):
        for query in ("residual", "attention", "feed forward"):
            questions = [u.question for u in select(transformer, query).unknowns]
            assert "What precision was used?" in questions


class TestMisses:
    def test_unrelated_request_matches_nothing(self, transformer):
        assert select(transformer, "convolutional kernel stride").empty

    def test_stopwords_alone_match_nothing(self, transformer):
        assert select(transformer, "give me the code for it").empty

    def test_suggest_offers_component_names(self, transformer):
        assert "Multi-Head Attention" in suggest(transformer)


class TestTransparency:
    def test_every_match_explains_itself(self, transformer):
        # A wrong match must be visible to the user, not silent.
        for match in select(transformer, "attention").matched:
            assert match.reason


class TestLooseMatching:
    """People do not type the paper's exact words."""

    @pytest.fixture
    def spec(self, transformer):
        return transformer

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("attention block", "Multi-Head Attention"),
            ("layer norm", "Residual Connection and Layer Normalization"),
            ("layernorm", "Residual Connection and Layer Normalization"),
            ("normalisation", "Residual Connection and Layer Normalization"),
            ("feedforward", "Position-wise Feed-Forward Network"),
            ("feed-forward", "Position-wise Feed-Forward Network"),
            ("residual", "Residual Connection and Layer Normalization"),
        ],
    )
    def test_variant_spellings_resolve(self, spec, query, expected):
        assert expected in select(spec, query).names()

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("ffn", "Position-wise Feed-Forward Network"),
            ("mha", "Multi-Head Attention"),
        ],
    )
    def test_acronyms_resolve(self, spec, query, expected):
        # The initials must come from the ordered name; a token set spells
        # nothing and the acronym would never match.
        assert expected in select(spec, query).names()

    def test_loose_matching_does_not_match_everything(self, spec):
        assert select(spec, "convolutional kernel stride").empty


class TestDidYouMean:
    def test_near_miss_gets_suggestions(self, transformer):
        names = did_you_mean(transformer, "attn")
        assert names and "Attention" in names[0]

    def test_suggestions_are_real_component_names(self, transformer):
        actual = {c.name for c in transformer.components}
        assert set(did_you_mean(transformer, "mlp")) <= actual

    def test_empty_query_suggests_nothing(self, transformer):
        assert did_you_mean(transformer, "   ") == []

    def test_suggestions_are_capped(self, transformer):
        assert len(did_you_mean(transformer, "thing", limit=2)) <= 2


class TestPromptDiscipline:
    """The generated body must be as trustworthy as the derived header."""

    def _prompt(self, spec, query):
        from arxivbot.deviation import check
        from arxivbot.select import select
        from arxivbot.synthesize import prompt

        selection = select(spec, query)
        return prompt(spec, selection, check(spec, query, selection.components), query)

    def test_omitted_components_are_named(self, transformer):
        # Given one half of a pair the paper defines together, a model with no
        # way to know the other exists reports the hole as an ambiguity.
        text = self._prompt(transformer, "scaled dot-product attention")
        assert "COMPONENTS NOT INCLUDED" in text
        assert "Position-wise Feed-Forward Network" in text

    def test_gapless_selection_forbids_todos(self, transformer):
        text = self._prompt(transformer, "feed forward")
        if "WHAT THE PAPER NEVER STATES:" in text:
            assert "no TODO(paper-silent) markers at all" in text

    def test_listed_gaps_are_the_only_ones_allowed(self, transformer):
        text = self._prompt(transformer, "attention")
        assert "and none besides" in text

    def test_citable_sections_come_from_provenance(self, transformer):
        text = self._prompt(transformer, "attention")
        assert "SECTIONS YOU MAY CITE" in text
        assert "Model Architecture" in text
