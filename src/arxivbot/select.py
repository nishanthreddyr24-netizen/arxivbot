"""Choosing which parts of a paper a request is actually about.

A paper describes many components; a request wants a few of them. Extraction
runs once per paper and is expensive. Selection runs once per request, is
cheap, and involves no model call at all - it is matching and graph traversal
over a spec that already exists.

That split is the reason the spec is the intermediate representation. Ask the
same paper for the attention block and then for the residual stream and the
paper is read once, not twice.

Matching a request to component names is fuzzy, which is fine *here* and was
not fine for caching answers: the result is shown to the user, who can see
what was picked and say otherwise. A selection that guesses wrong is visible;
a cache that guesses wrong is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from arxivbot.spec import Component, ImplementationSpec, Unknown

# Words that carry no signal when matching a request to a component.
_STOPWORDS = frozenset(
    """a an the of for to in on and or with code give me show write implement
    implementation how does do i want need block layer module part get using
    use its it is are that this""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


@dataclass(slots=True)
class Match:
    component: Component
    score: float
    reason: str
    """Why this was picked - shown to the user so a bad match is obvious."""


@dataclass(slots=True)
class Selection:
    """What a request resolved to."""

    query: str
    matched: list[Match] = field(default_factory=list)
    """Components the request named, best first."""

    dependencies: list[Component] = field(default_factory=list)
    """Components pulled in because the matched ones compose them."""

    unknowns: list[Unknown] = field(default_factory=list)
    """Gaps that apply to anything in this selection."""

    @property
    def components(self) -> list[Component]:
        """Everything needed to implement the request, dependencies last."""
        return [m.component for m in self.matched] + self.dependencies

    @property
    def empty(self) -> bool:
        return not self.matched

    def names(self) -> list[str]:
        return [c.name for c in self.components]


def _score(component: Component, wanted: set[str]) -> tuple[float, str]:
    """How well a component answers the request, and why."""
    if not wanted:
        return 0.0, ""

    name_tokens = _tokens(component.name)
    role_tokens = _tokens(component.role)

    name_hits = wanted & name_tokens
    role_hits = wanted & role_tokens - name_hits

    # A name match is the strong signal: components are named the way papers
    # name them, which is the vocabulary a reader will use.
    score = len(name_hits) / len(wanted) * 2.0
    score += len(role_hits) / len(wanted) * 0.5

    # Equations and tensor names catch requests phrased mathematically.
    other = set()
    for eq in component.equations:
        other |= _tokens(eq.description)
    for tensor in list(component.inputs) + list(component.outputs):
        other |= _tokens(tensor.name)
    other_hits = wanted & other - name_hits - role_hits
    score += len(other_hits) / len(wanted) * 0.25

    reasons = []
    if name_hits:
        reasons.append(f"name matches {sorted(name_hits)}")
    if role_hits:
        reasons.append(f"role mentions {sorted(role_hits)}")
    if other_hits:
        reasons.append(f"equations/tensors mention {sorted(other_hits)}")
    return score, "; ".join(reasons)


def _closure(spec: ImplementationSpec, roots: list[Component]) -> list[Component]:
    """Components the roots depend on, transitively, excluding the roots."""
    by_name = {c.name.lower(): c for c in spec.components}
    seen = {c.name.lower() for c in roots}
    out: list[Component] = []
    queue = list(roots)

    while queue:
        current = queue.pop(0)
        for dep_name in current.depends_on:
            key = dep_name.lower()
            if key in seen:
                continue
            seen.add(key)
            dep = by_name.get(key)
            if dep is None:
                # The paper referred to something it never defined. That is a
                # gap in the paper, not an error here.
                continue
            out.append(dep)
            queue.append(dep)
    return out


def _relevant_unknowns(spec: ImplementationSpec, chosen: list[Component]) -> list[Unknown]:
    """Gaps attached to the chosen components, plus any that apply globally."""
    names = {c.name.lower() for c in chosen}
    out = []
    for unknown in spec.unknowns:
        if unknown.applies_to is None or unknown.applies_to.lower() in names:
            out.append(unknown)
    return out


def select(
    spec: ImplementationSpec,
    query: str,
    *,
    limit: int = 3,
    threshold: float = 0.3,
    relative_cutoff: float = 0.5,
) -> Selection:
    """Resolve a request against a spec.

    Returns the components the request names, whatever those compose, and the
    open questions that apply to the result.

    ``relative_cutoff`` discards matches far weaker than the best one. Without
    it, asking for attention also returns the encoder layer - whose *role*
    happens to mention attention - and then everything the encoder composes.
    A request for one block should not quietly return most of the model.
    """
    wanted = _tokens(query)

    scored = []
    for component in spec.components:
        score, reason = _score(component, wanted)
        if score >= threshold:
            scored.append(Match(component=component, score=score, reason=reason))
    scored.sort(key=lambda m: -m.score)

    if scored:
        floor = scored[0].score * relative_cutoff
        scored = [m for m in scored if m.score >= floor]
    matched = scored[:limit]

    dependencies = _closure(spec, [m.component for m in matched])
    chosen = [m.component for m in matched] + dependencies

    return Selection(
        query=query,
        matched=matched,
        dependencies=dependencies,
        unknowns=_relevant_unknowns(spec, chosen),
    )


def suggest(spec: ImplementationSpec, limit: int = 12) -> list[str]:
    """What this paper can be asked about - for when a request matches nothing."""
    return [c.name for c in spec.components[:limit]]
