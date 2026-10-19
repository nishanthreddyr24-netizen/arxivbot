"""Composing one paper's component into another paper's architecture.

The conflict ledger is the part of this a model could not have produced by
being asked nicely, so it is the part worth testing: given two sets of
assumptions, the same contradictions must come out every time.
"""

from __future__ import annotations

import pytest

from arxivbot.merge import (
    Resolution,
    _overlap,
    _same_thing,
    conflicts,
    find_component,
    merge,
    precedents,
    render,
)
from arxivbot.spec import (
    Assumptions,
    Component,
    Extractor,
    ImplementationSpec,
    Modality,
    NormPosition,
    Ordering,
    Regime,
    Severity,
)


def component(name: str, **assumptions) -> Component:
    return Component(
        name=name,
        role=f"the {name}",
        assumptions=Assumptions(**assumptions),
    )


def spec(arxiv_id: str, title: str, *components: Component) -> ImplementationSpec:
    return ImplementationSpec(
        arxiv_id=arxiv_id,
        title=title,
        extractor=Extractor(model="test", arxivbot_version="0"),
        components=list(components),
    )


# ---------------------------------------------------------------- conflicts --


def test_an_unspecified_axis_conflicts_with_nothing():
    """Most axes come back unspecified; reporting those would bury the rest."""
    host = Assumptions(modality=Modality.SEQUENCE)
    donor = Assumptions()
    assert conflicts(host, donor) == []


def test_image_into_a_sequence_model_asks_for_patches():
    """The substitution behind ViT, derived from the assumptions alone."""
    host = Assumptions(modality=Modality.SEQUENCE)
    donor = Assumptions(modality=Modality.IMAGE)
    (found,) = conflicts(host, donor)
    assert found.axis == "modality"
    assert found.severity is Severity.BLOCKING
    assert found.resolution is Resolution.ADAPTER
    assert "patches" in found.detail


def test_an_unmapped_modality_pair_says_so_rather_than_inventing_one():
    host = Assumptions(modality=Modality.STATE)
    donor = Assumptions(modality=Modality.GRAPH)
    (found,) = conflicts(host, donor)
    assert found.resolution is Resolution.ADAPTER
    assert "no standard conversion" in found.detail


def test_permutation_invariance_against_positional_input():
    host = Assumptions(ordering=Ordering.ORDERED)
    donor = Assumptions(ordering=Ordering.PERMUTATION_INVARIANT)
    (found,) = conflicts(host, donor)
    assert found.severity is Severity.BLOCKING
    assert "positional" in found.detail


def test_incompatible_regimes_cannot_be_adapted_away():
    host = Assumptions(regime=Regime.RL_ON_POLICY)
    donor = Assumptions(regime=Regime.RL_OFF_POLICY)
    (found,) = conflicts(host, donor)
    assert found.resolution is Resolution.RETHINK
    assert found.severity is Severity.BLOCKING


def test_merely_different_regimes_are_a_choice_not_a_wall():
    host = Assumptions(regime=Regime.SUPERVISED)
    donor = Assumptions(regime=Regime.SELF_SUPERVISED)
    (found,) = conflicts(host, donor)
    assert found.resolution is Resolution.CHOOSE
    assert found.severity is Severity.SIGNIFICANT


def test_pre_against_post_normalisation_is_reported_but_survivable():
    host = Assumptions(norm_position=NormPosition.POST)
    donor = Assumptions(norm_position=NormPosition.PRE)
    (found,) = conflicts(host, donor)
    assert found.axis == "norm_position"
    assert found.resolution is Resolution.CHOOSE


# -------------------------------------------------------------------- merge --


def test_merge_substitutes_in_place_and_keeps_the_order():
    host = spec(
        "1",
        "Host",
        component("embedding", modality=Modality.SEQUENCE),
        component("encoder", modality=Modality.SEQUENCE),
        component("decoder", modality=Modality.SEQUENCE),
    )
    donor = spec("2", "Donor", component("patch embedding", modality=Modality.IMAGE))

    hybrid = merge(host, donor, "patch embedding", replaces="embedding")

    assert [c.name for c in hybrid.components] == [
        "patch embedding",
        "encoder",
        "decoder",
    ]
    assert hybrid.replaced == "embedding"
    assert hybrid.conflicts and hybrid.conflicts[0].axis == "modality"


def test_merge_without_a_replacement_adds_alongside():
    host = spec("1", "Host", component("encoder"))
    donor = spec("2", "Donor", component("rotary embedding"))
    hybrid = merge(host, donor, "rotary embedding")
    assert hybrid.replaced is None
    assert [c.name for c in hybrid.components] == ["encoder", "rotary embedding"]


def test_a_dependency_on_the_replaced_component_is_flagged():
    encoder = Component(name="encoder", role="r", depends_on=["Embedding"])
    host = spec("1", "Host", component("embedding"), encoder)
    donor = spec("2", "Donor", component("patch embedding"))
    hybrid = merge(host, donor, "patch embedding", replaces="embedding")
    assert any("encoder depended on embedding" in note for note in hybrid.notes)


def test_the_hosts_dominant_assumption_wins_over_one_odd_component():
    """One component calling itself an image model must not redefine a paper."""
    host = spec(
        "1",
        "Host",
        component("a", modality=Modality.SEQUENCE),
        component("b", modality=Modality.SEQUENCE),
        component("c", modality=Modality.IMAGE),
    )
    donor = spec("2", "Donor", component("d", modality=Modality.SEQUENCE))
    assert merge(host, donor, "d").conflicts == []


def test_an_unknown_component_names_what_is_available():
    host = spec("1", "Host", component("encoder"))
    donor = spec("2", "Donor", component("rotary embedding"))
    with pytest.raises(LookupError, match="rotary embedding"):
        merge(host, donor, "no such thing")


def test_buildable_is_false_only_for_a_contradiction_no_adapter_fixes():
    host = spec("1", "Host", component("a", regime=Regime.RL_ON_POLICY))
    donor = spec("2", "Donor", component("b", regime=Regime.RL_OFF_POLICY))
    assert merge(host, donor, "b").buildable is False

    host = spec("1", "Host", component("a", modality=Modality.SEQUENCE))
    donor = spec("2", "Donor", component("b", modality=Modality.IMAGE))
    assert merge(host, donor, "b").buildable is True


# --------------------------------------------------------------- precedents --


@pytest.mark.parametrize(
    "left,right",
    [
        ("layer normalization", "Layernorm"),
        ("Attention", "Self-attention"),
        ("feed-forward network", "Feedforward layer"),
        ("residual connection", "Residual connections"),
        ("positional encodings", "Position embeddings"),
    ],
)
def test_the_same_component_named_two_ways(left, right):
    """Exact matching scores the Transformer against ViT at zero."""
    assert _same_thing(left, right)


def test_unrelated_components_do_not_match():
    assert not _same_thing("convolutional layer", "noise schedule")
    assert not _same_thing("decoder", "discriminator")


def test_overlap_uses_each_name_once():
    left = ["attention", "attention block"]
    right = ["self-attention"]
    assert _overlap(left, right) == pytest.approx(1 / 2)


def test_precedents_exclude_the_papers_being_merged():
    known = [spec("1", "Host", component("encoder"))]
    found = precedents(["encoder"], known, exclude={"1"})
    assert found == []


def test_render_is_plain_text_and_names_the_conflict():
    host = spec("1", "Host", component("embedding", modality=Modality.SEQUENCE))
    donor = spec("2", "Donor", component("patch embedding", modality=Modality.IMAGE))
    text = render(merge(host, donor, "patch embedding", replaces="embedding"))
    assert "patches" in text
    assert "buildable" in text


def test_find_component_matches_loosely():
    paper = spec("1", "P", component("Multi-Head Attention"))
    assert find_component(paper, "multi-head attention") is not None
    assert find_component(paper, "attention") is not None
    assert find_component(paper, "head multi") is not None
    assert find_component(paper, "convolution") is None
