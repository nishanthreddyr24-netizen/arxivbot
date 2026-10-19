"""Compose one paper's component into another paper's architecture.

Most work called novel is a substitution: keep a known architecture, replace
one part of it with a part from somewhere else. The Vision Transformer is the
Transformer with image patches where tokens were. Latent Diffusion is DDPM
run inside a VAE's latent space. RoFormer is the Transformer with rotary
position encoding. GAT is GCN with attention doing the weighting.

Two rules keep this honest.

**It never claims novelty.** Novelty is unfalsifiable and the moment a tool
asserts it, nothing else it says can be trusted either. What this produces is
a buildable arrangement, the contradictions it inherits, and a note of which
papers already look like it. Whether the result is new is the researcher's
judgement, made with those three things in hand.

**The conflicts are computed, not written.** Each component carries a small
closed set of assumptions. A merge compares them axis by axis, in code. The
output is therefore reproducible, and it is the one part of this a language
model could not have produced by being asked nicely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from arxivbot.spec import (
    Assumptions,
    Component,
    ImplementationSpec,
    Modality,
    NormPosition,
    Ordering,
    Regime,
    Severity,
    Unknown,
)


class Resolution(str, Enum):
    """What has to happen before two components can sit together."""

    ADAPTER = "adapter"
    """Something must convert between them - a real piece of code to write."""

    CHOOSE = "choose"
    """Both conventions work; the implementer picks one and stays consistent."""

    RETHINK = "rethink"
    """The two were designed under settings that do not coexist."""


@dataclass(frozen=True, slots=True)
class Conflict:
    """One axis on which the host and the donor disagree."""

    axis: str
    host: str
    donor: str
    severity: Severity
    resolution: Resolution
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.axis}: {self.host} vs {self.donor} - {self.detail}"


@dataclass(slots=True)
class Hybrid:
    """The result of a merge: an arrangement, its costs, and its precedents."""

    host_id: str
    host_title: str
    donor_id: str
    donor_title: str
    replaced: str | None
    donated: str
    components: list[Component] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    unknowns: list[Unknown] = field(default_factory=list)
    precedents: list[tuple[str, str, float]] = field(default_factory=list)
    """Known papers whose component set resembles this one: (id, title, overlap)."""

    notes: list[str] = field(default_factory=list)

    @property
    def buildable(self) -> bool:
        """No conflict that no adapter can bridge."""
        return not any(c.resolution is Resolution.RETHINK for c in self.conflicts)

    @property
    def blocking(self) -> list[Conflict]:
        return [c for c in self.conflicts if c.severity is Severity.BLOCKING]


# --------------------------------------------------------------- conflicts ---

# Modalities that a documented, ordinary adapter converts between. An image
# becomes a sequence by cutting it into patches, which is the whole of ViT's
# contribution to the Transformer. A graph becomes a set of node features.
# Pairs absent here are not impossible, only unmapped, so they are reported
# as needing an adapter nobody has specified rather than as a dead end.
_ADAPTABLE: dict[frozenset[Modality], str] = {
    frozenset({Modality.IMAGE, Modality.SEQUENCE}): (
        "cut the image into fixed-size patches and embed each as one token"
    ),
    frozenset({Modality.GRAPH, Modality.SET}): (
        "drop the edges and treat the node features as a set"
    ),
    frozenset({Modality.GRAPH, Modality.SEQUENCE}): (
        "serialise the nodes in a fixed order and supply the adjacency "
        "separately, as an attention mask or an edge-feature term"
    ),
    frozenset({Modality.IMAGE, Modality.LATENT}): (
        "encode the image with a trained autoencoder and operate on the code"
    ),
    frozenset({Modality.SEQUENCE, Modality.LATENT}): (
        "encode the sequence and operate on the pooled or per-step code"
    ),
    frozenset({Modality.SET, Modality.SEQUENCE}): (
        "impose an order, and add positional information if the order means "
        "anything"
    ),
}

# Regimes that cannot be run at the same time on the same parameters.
_INCOMPATIBLE_REGIMES: frozenset[frozenset[Regime]] = frozenset(
    {
        frozenset({Regime.RL_ON_POLICY, Regime.RL_OFF_POLICY}),
        frozenset({Regime.ADVERSARIAL, Regime.DENOISING}),
    }
)


def _vague(value: object) -> bool:
    """Unspecified and not-applicable axes carry no claim, so they conflict
    with nothing. Reporting them would bury the real conflicts in noise."""
    return getattr(value, "value", value) in {"unspecified", "not_applicable", "any"}


def conflicts(host: Assumptions, donor: Assumptions) -> list[Conflict]:
    """Compare two sets of assumptions, axis by axis."""
    found: list[Conflict] = []

    if not _vague(host.modality) and not _vague(donor.modality):
        if host.modality is not donor.modality:
            how = _ADAPTABLE.get(frozenset({host.modality, donor.modality}))
            found.append(
                Conflict(
                    axis="modality",
                    host=host.modality.value,
                    donor=donor.modality.value,
                    severity=Severity.BLOCKING,
                    resolution=Resolution.ADAPTER,
                    detail=(
                        f"the donor expects {donor.modality.value} and the host "
                        f"supplies {host.modality.value}; "
                        + (how or "no standard conversion exists - you are inventing one")
                    ),
                )
            )

    if not _vague(host.ordering) and not _vague(donor.ordering):
        if host.ordering is not donor.ordering:
            invariant_donor = donor.ordering is Ordering.PERMUTATION_INVARIANT
            found.append(
                Conflict(
                    axis="ordering",
                    host=host.ordering.value,
                    donor=donor.ordering.value,
                    severity=Severity.BLOCKING,
                    resolution=Resolution.ADAPTER,
                    detail=(
                        "the donor must not depend on input order, but the host "
                        "encodes position - remove the positional term or the "
                        "donor's guarantee is void"
                        if invariant_donor
                        else "the host treats its input as unordered while the "
                        "donor reads meaning into position - you must supply an "
                        "ordering and justify it"
                    ),
                )
            )

    if not _vague(host.regime) and not _vague(donor.regime):
        if host.regime is not donor.regime:
            pair = frozenset({host.regime, donor.regime})
            hard = pair in _INCOMPATIBLE_REGIMES
            found.append(
                Conflict(
                    axis="regime",
                    host=host.regime.value,
                    donor=donor.regime.value,
                    severity=Severity.BLOCKING if hard else Severity.SIGNIFICANT,
                    resolution=Resolution.RETHINK if hard else Resolution.CHOOSE,
                    detail=(
                        "these training settings cannot run over the same "
                        "parameters at once"
                        if hard
                        else "the donor was tuned under a different training "
                        "setting; its hyperparameters probably do not transfer"
                    ),
                )
            )

    if not _vague(host.norm_position) and not _vague(donor.norm_position):
        if host.norm_position is not donor.norm_position:
            found.append(
                Conflict(
                    axis="norm_position",
                    host=host.norm_position.value,
                    donor=donor.norm_position.value,
                    severity=Severity.SIGNIFICANT,
                    resolution=Resolution.CHOOSE,
                    detail=(
                        "pre- and post-normalisation differ in training stability "
                        "and warmup needs; pick one for the whole model rather "
                        "than keeping each part as its paper had it"
                    ),
                )
            )

    return found


def _dominant(spec: ImplementationSpec) -> Assumptions:
    """The assumptions the host architecture holds as a whole.

    Taken as the most common decided value on each axis. One component
    declaring something unusual should not redefine the paper around it.
    """
    chosen: dict[str, object] = {}
    for axis, empty in (
        ("modality", Modality.UNSPECIFIED),
        ("ordering", Ordering.UNSPECIFIED),
        ("regime", Regime.UNSPECIFIED),
        ("norm_position", NormPosition.UNSPECIFIED),
    ):
        counts: dict[object, int] = {}
        for component in spec.components:
            value = getattr(component.assumptions, axis)
            if _vague(value):
                continue
            counts[value] = counts.get(value, 0) + 1
        chosen[axis] = max(counts, key=lambda k: counts[k]) if counts else empty
    return Assumptions(**chosen)  # type: ignore[arg-type]


# ------------------------------------------------------------- precedents ---


# Words that say what kind of thing something is, not which thing it is.
# "Feed-Forward Network" and "Feedforward Layer" are the same component named
# by two papers; what distinguishes them is the part these words are not.
_GENERIC = frozenset(
    """layer layers network networks module modules block blocks mechanism
    function functions head heads model models component unit units stack
    operation sublayer""".split()
)


def _stem(word: str) -> str:
    """Enough stemming to see past how two papers inflect the same word.

    "positional encodings" and "position embeddings" name one thing; so do
    "convolutional layer" and "convolution". Nothing cleverer is needed,
    and anything cleverer would start merging words that differ.
    """
    if len(word) > 4 and word.endswith("s"):
        word = word[:-1]
    if len(word) > 6 and word.endswith("al"):
        word = word[:-2]
    return word


def _key(name: str) -> tuple[str, frozenset[str]]:
    """A component name reduced to something comparable across papers."""
    words = [_stem(w) for w in re.findall(r"[a-z0-9]+", name.lower())]
    distinctive = frozenset(w for w in words if w not in _GENERIC)
    return "".join(words), distinctive or frozenset(words)


def _same_thing(a: str, b: str) -> bool:
    """Whether two papers are naming the same component.

    Exact matching scores the Transformer against ViT at zero: one writes
    "layer normalization" and "Attention", the other "Layernorm" and
    "Self-attention". Comparing on the distinctive words, with the kind-of
    words dropped, is what makes a resemblance measurable at all.
    """
    left, left_words = _key(a)
    right, right_words = _key(b)
    if left == right or left.startswith(right) or right.startswith(left):
        return True
    if left_words & right_words:
        return True
    # One paper's "feed-forward network" is another's "Feedforward layer":
    # the same word, split in two by a hyphen in one of them.
    joined_left = "".join(sorted(left_words))
    joined_right = "".join(sorted(right_words))
    return joined_left == joined_right


def _overlap(left: list[str], right: list[str]) -> float:
    """Jaccard over names, matched loosely and each name used only once."""
    unmatched = list(right)
    shared = 0
    for name in left:
        for index, other in enumerate(unmatched):
            if _same_thing(name, other):
                shared += 1
                unmatched.pop(index)
                break
    union = len(left) + len(right) - shared
    return shared / union if union else 0.0


def _names(spec: ImplementationSpec) -> set[str]:
    return {c.name.strip().lower() for c in spec.components}


def precedents(
    hybrid_names: list[str],
    known: list[ImplementationSpec],
    *,
    exclude: set[str],
    threshold: float = 0.15,
    limit: int = 3,
) -> list[tuple[str, str, float]]:
    """Papers whose components already look like this arrangement.

    Not a novelty check - a published combination can be missing from any
    corpus, so finding nothing proves nothing. It is the weaker claim the tool
    can actually support: here is what this resembles among the papers we hold.

    The scores are low in absolute terms even for a pair as close as the
    Transformer and ViT, because two papers describing the same component
    rarely spell it the same way. What survives is the ranking, so this
    returns an ordered shortlist rather than a verdict.
    """
    out = []
    for spec in known:
        if spec.arxiv_id in exclude:
            continue
        other = [c.name for c in spec.components]
        if not other:
            continue
        overlap = _overlap(list(hybrid_names), other)
        if overlap >= threshold:
            out.append((spec.arxiv_id, spec.title, round(overlap, 2)))
    return sorted(out, key=lambda row: row[2], reverse=True)[:limit]


# ------------------------------------------------------------------ merge ---


def find_component(spec: ImplementationSpec, name: str) -> Component | None:
    """Match a component by name, loosely enough to be usable from a CLI."""
    wanted = name.strip().lower()
    for component in spec.components:
        if component.name.strip().lower() == wanted:
            return component
    for component in spec.components:
        if wanted in component.name.strip().lower():
            return component
    words = [w for w in wanted.split() if len(w) > 2]
    for component in spec.components:
        haystack = component.name.strip().lower()
        if words and all(word in haystack for word in words):
            return component
    return None


def merge(
    host: ImplementationSpec,
    donor: ImplementationSpec,
    donated: str,
    *,
    replaces: str | None = None,
    known: list[ImplementationSpec] | None = None,
) -> Hybrid:
    """Put `donated` from `donor` into `host`, and report what it costs.

    `replaces` names the host component it stands in for. Left out, the donor
    component is added alongside and anything that depended on a host
    component of a similar name is left alone - the arrangement is reported,
    not silently rewired.
    """
    taken = find_component(donor, donated)
    if taken is None:
        raise LookupError(
            f"{donor.arxiv_id} has no component matching {donated!r}; "
            f"it has: {', '.join(c.name for c in donor.components) or 'none'}"
        )

    displaced = find_component(host, replaces) if replaces else None
    if replaces and displaced is None:
        raise LookupError(
            f"{host.arxiv_id} has no component matching {replaces!r}; "
            f"it has: {', '.join(c.name for c in host.components) or 'none'}"
        )

    hybrid = Hybrid(
        host_id=host.arxiv_id,
        host_title=host.title,
        donor_id=donor.arxiv_id,
        donor_title=donor.title,
        replaced=displaced.name if displaced else None,
        donated=taken.name,
    )

    for component in host.components:
        if displaced is not None and component.name == displaced.name:
            hybrid.components.append(taken)
            continue
        hybrid.components.append(component)
    if displaced is None:
        hybrid.components.append(taken)

    hybrid.conflicts = conflicts(_dominant(host), taken.assumptions)

    # A dependency that pointed at the replaced component now points at
    # nothing. Saying so is cheaper than letting someone discover it in code.
    if displaced is not None:
        gone = displaced.name.strip().lower()
        for component in host.components:
            if component.name == displaced.name:
                continue
            if any(d.strip().lower() == gone for d in component.depends_on):
                hybrid.notes.append(
                    f"{component.name} depended on {displaced.name}, which "
                    f"{taken.name} now stands in for - check the interface matches"
                )

    # The host's open questions survive the merge, and so do the donor's for
    # the part being donated. A hybrid cannot be better specified than either
    # paper it came from, and pretending otherwise is the failure this whole
    # project exists to avoid.
    hybrid.unknowns = list(host.unknowns)
    donated_lower = taken.name.strip().lower()
    for unknown in donor.unknowns:
        applies = (unknown.applies_to or "").strip().lower()
        if applies and applies in donated_lower:
            hybrid.unknowns.append(unknown)

    if known:
        hybrid.precedents = precedents(
            [c.name for c in hybrid.components],
            known,
            exclude={host.arxiv_id, donor.arxiv_id},
        )

    return hybrid


def render(hybrid: Hybrid) -> str:
    """The merge as text, for a terminal or a commit message."""
    lines = [
        f"{hybrid.host_title} ({hybrid.host_id})",
        f"  with {hybrid.donated} from {hybrid.donor_title} ({hybrid.donor_id})",
    ]
    lines.append(
        f"  replacing {hybrid.replaced}" if hybrid.replaced else "  added alongside"
    )

    lines.append("")
    lines.append(f"components ({len(hybrid.components)}):")
    for component in hybrid.components:
        mark = "+" if component.name == hybrid.donated else " "
        lines.append(f"  {mark} {component.name}")

    lines.append("")
    if hybrid.conflicts:
        lines.append(f"conflicts ({len(hybrid.conflicts)}):")
        for conflict in hybrid.conflicts:
            lines.append(f"  {conflict}")
            lines.append(f"      -> {conflict.resolution.value}")
    else:
        lines.append("conflicts: none on the recorded axes")
        lines.append("  (which means the assumptions agree where they were recorded,")
        lines.append("   not that the two parts are known to fit)")

    if hybrid.notes:
        lines.append("")
        lines.append("rewiring:")
        lines.extend(f"  {note}" for note in hybrid.notes)

    if hybrid.precedents:
        lines.append("")
        lines.append("closest in the corpus (ranking, not a novelty check):")
        for arxiv_id, title, overlap in hybrid.precedents:
            lines.append(f"  {overlap:.0%}  {arxiv_id}  {title}")

    lines.append("")
    lines.append(f"open questions carried over: {len(hybrid.unknowns)}")
    lines.append(
        "buildable: yes" if hybrid.buildable else "buildable: no - see the rethink items"
    )
    return "\n".join(lines)
