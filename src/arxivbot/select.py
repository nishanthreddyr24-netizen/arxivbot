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
from difflib import SequenceMatcher

from arxivbot.spec import Component, ImplementationSpec, Unknown

# Words that carry no signal when matching a request to a component. Words
# like "layer", "block" and "module" are deliberately NOT here: they read as
# filler in "the attention block", but they are also half the names papers
# give things - Encoder Layer, Layer Normalization - so dropping them loses
# the signal they carry.
_STOPWORDS = frozenset(
    """a an the of for to in on and or with code give me show write implement
    implementation how does do i want need get using use its it is are that
    this please just some""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Below this length a token matches too much to be trusted loosely.
_FUZZY_MIN = 4


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _canon(token: str) -> str:
    """Fold British and American spellings together.

    "normalisation" and "normalization" diverge at the seventh character, so
    no amount of prefix matching connects them - and papers and readers do
    not agree on which to use.
    """
    for british, american in (("isation", "ization"), ("ising", "izing"),
                              ("ised", "ized"), ("ise", "ize")):
        if token.endswith(british):
            return token[: -len(british)] + american
    return token


def _initials(text: str) -> str:
    """First letters of a name, in order: "Feed-Forward Networks" -> "ffn".

    Must come from the text, not from the token set - a set has no order, so
    its initials spell nothing and the acronym never matches.
    """
    words = [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2]
    return "".join(word[0] for word in words)


def _hits(wanted: set[str], candidate: set[str], initials: str = "") -> set[str]:
    """Which request tokens are answered by ``candidate``'s tokens.

    Exact equality alone is too brittle for the way people actually type:
    "norm" never equals "normalization", "embedding" never equals
    "embeddings", and "layernorm" is one token where the paper wrote two.
    So a token also counts when it shares a prefix with a candidate token, or
    when it appears inside the candidate's name with the spaces taken out -
    one rule that covers stemming, plurals and compounds together.
    """
    if not candidate:
        return set()
    candidate = {_canon(word) for word in candidate}
    wanted = {_canon(word) for word in wanted}
    squashed = "".join(sorted(candidate))
    joined = "".join(candidate)

    found = set()
    for token in wanted:
        if token in candidate:
            found.add(token)
        elif len(token) >= _FUZZY_MIN and (
            any(
                len(other) >= _FUZZY_MIN
                and (other.startswith(token) or token.startswith(other))
                for other in candidate
            )
            or token in joined
            or token in squashed
        ):
            found.add(token)
        elif 2 <= len(token) <= 5 and len(initials) >= 2 and token in initials:
            found.add(token)
    return found


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

    name_hits = _hits(wanted, name_tokens, _initials(component.name))
    role_hits = _hits(wanted, role_tokens) - name_hits

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
    other_hits = _hits(wanted, other) - name_hits - role_hits
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


def did_you_mean(
    spec: ImplementationSpec, query: str, *, limit: int = 4, floor: float = 0.22
) -> list[str]:
    """The nearest component names to a request that matched nothing.

    Token matching cannot reach an abbreviation or a synonym: nothing about
    "ffn" resembles "Position-wise Feed-Forward Networks" as a string, and
    "mlp" is a different word for the same thing. Rather than guess - which
    would put a wrong answer under a confident heading - offer the closest
    names and let the reader choose. A shortlist the user picks from is
    honest in a way a silent best guess is not.

    Similarity is character-level against the name and the role, so a request
    that echoes the description still surfaces the right component.
    """
    wanted = query.lower().strip()
    if not wanted:
        return []

    scored: list[tuple[float, str]] = []
    for component in spec.components:
        name = component.name.lower()
        best = SequenceMatcher(None, wanted, name).ratio()
        # A short query against a long name scores badly on whole-string
        # similarity, so also try the request against each word of the name.
        for word in name.split():
            best = max(best, SequenceMatcher(None, wanted, word).ratio())
        if component.role:
            best = max(best, SequenceMatcher(None, wanted, component.role.lower()).ratio() * 0.8)
        scored.append((best, component.name))

    scored.sort(key=lambda pair: -pair[0])
    return [name for score, name in scored[:limit] if score >= floor]
