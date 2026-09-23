"""When a request contradicts the paper.

A user may legitimately want a variant: the paper uses post-norm, they want
pre-norm. Refusing is unhelpful - people reimplement papers in order to change
them. Silently complying is worse, because the output then claims to be the
paper and is not.

So a request that conflicts with the spec is honoured *and recorded*. Three
things can happen when a request names a choice:

``MATCHES_PAPER``
    The request agrees with what the paper says. Cite it and move on.

``DEVIATES``
    The paper states one thing, the request asks for another. Implement the
    request, and say loudly that this is not what the paper describes.

``RESOLVES_UNKNOWN``
    The paper never said. The request is not a deviation at all - it is the
    user answering an open question, which is exactly what :class:`Unknown`
    exists to surface.

The check here is a cheap mechanical pre-filter over values the spec already
records. It catches the common architectural switches; it is not a substitute
for the synthesis step re-checking the full request against the spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from arxivbot.spec import Component, Confidence, Hyperparameter, ImplementationSpec

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Stance(str, Enum):
    MATCHES_PAPER = "matches_paper"
    DEVIATES = "deviates"
    RESOLVES_UNKNOWN = "resolves_unknown"


# Mutually exclusive choices that reimplementations commonly swap. Each family
# is a set of alternatives for one decision; naming one implies rejecting the
# others. Extend freely - this is a convenience list, not a closed taxonomy.
VARIANT_FAMILIES: dict[str, tuple[str, ...]] = {
    "normalization placement": ("pre", "post", "prenorm", "postnorm", "preln", "postln"),
    "normalization type": ("layernorm", "rmsnorm", "batchnorm", "groupnorm"),
    "activation": ("relu", "gelu", "silu", "swish", "glu", "swiglu", "geglu"),
    "positional encoding": ("sinusoidal", "learned", "rotary", "rope", "alibi", "relative"),
    "optimizer": ("adam", "adamw", "sgd", "adafactor", "lion", "rmsprop"),
    "attention variant": ("causal", "bidirectional", "masked", "cross", "self"),
}

# Spellings that mean the same choice.
_SYNONYMS = {
    "prenorm": "pre",
    "preln": "pre",
    "postnorm": "post",
    "postln": "post",
    "rope": "rotary",
}


@dataclass(slots=True)
class Deviation:
    """A point where the request and the paper part ways."""

    axis: str
    """Which decision this is about, e.g. 'normalization placement'."""

    requested: str
    """What the user asked for."""

    paper_says: str | None
    """What the paper states, or None when it is silent."""

    stance: Stance
    source: str | None = None
    """Section of the paper backing ``paper_says``, when there is one."""

    @property
    def is_conflict(self) -> bool:
        return self.stance is Stance.DEVIATES

    def render(self) -> str:
        """A banner to sit at the top of generated code."""
        if self.stance is Stance.MATCHES_PAPER:
            where = f" ({self.source})" if self.source else ""
            return f"# {self.axis}: {self.requested} - as described in the paper{where}"
        if self.stance is Stance.RESOLVES_UNKNOWN:
            return (
                f"# {self.axis}: {self.requested} - YOUR CHOICE. "
                f"The paper does not specify this."
            )
        where = f" ({self.source})" if self.source else ""
        return (
            f"# !! DEVIATION - {self.axis}\n"
            f"#    you asked for : {self.requested}\n"
            f"#    paper states  : {self.paper_says}{where}\n"
            f"#    This code implements your choice, not the paper's."
        )


def _normalise(token: str) -> str:
    return _SYNONYMS.get(token, token)


def _tokens(text: str) -> set[str]:
    return {_normalise(t) for t in _TOKEN_RE.findall(text.lower())}


def _family_of(token: str) -> str | None:
    for axis, members in VARIANT_FAMILIES.items():
        if token in {_normalise(m) for m in members}:
            return axis
    return None


def _stated_choice(
    hyperparameters: list[Hyperparameter], axis: str
) -> tuple[str, Hyperparameter] | None:
    """The alternative from ``axis`` that the spec records, if any."""
    for hp in hyperparameters:
        haystack = f"{hp.name} {hp.value or ''}"
        for token in _tokens(haystack):
            if _family_of(token) == axis:
                return token, hp
    return None


def check(
    spec: ImplementationSpec,
    query: str,
    components: list[Component] | None = None,
) -> list[Deviation]:
    """Compare a request against what the paper says.

    ``components`` narrows the comparison to the parts the request selected,
    so an unrelated choice elsewhere in the paper is not dragged in.
    """
    scope = components if components is not None else spec.components

    available: list[Hyperparameter] = list(spec.hyperparameters)
    for component in scope:
        available.extend(component.hyperparameters)
    if spec.training:
        available.extend(spec.training.hyperparameters)

    # What open questions cover, so a silent paper is reported as such.
    open_axes: dict[str, str] = {}
    for unknown in spec.unknowns:
        for token in _tokens(unknown.question):
            if axis := _family_of(token):
                open_axes.setdefault(axis, unknown.question)

    found: list[Deviation] = []
    for token in _tokens(query):
        axis = _family_of(token)
        if axis is None:
            continue
        if any(d.axis == axis for d in found):
            continue

        stated = _stated_choice(available, axis)

        if stated is None:
            found.append(
                Deviation(
                    axis=axis,
                    requested=token,
                    paper_says=None,
                    stance=Stance.RESOLVES_UNKNOWN,
                )
            )
            continue

        paper_token, hp = stated
        source = hp.provenance.section if hp.provenance else None

        if paper_token == token:
            stance = Stance.MATCHES_PAPER
        elif hp.confidence in (Confidence.CONVENTIONAL, Confidence.GUESS):
            # The paper did not actually say this; the extractor filled it in.
            # Overriding a guess is not a deviation from the paper.
            stance = Stance.RESOLVES_UNKNOWN
        else:
            stance = Stance.DEVIATES

        found.append(
            Deviation(
                axis=axis,
                requested=token,
                paper_says=paper_token if stance is not Stance.RESOLVES_UNKNOWN else None,
                stance=stance,
                source=source,
            )
        )
    return found


def banner(deviations: list[Deviation]) -> str:
    """The header block for generated code, conflicts first."""
    if not deviations:
        return ""
    ordered = sorted(deviations, key=lambda d: not d.is_conflict)
    return "\n".join(d.render() for d in ordered)
