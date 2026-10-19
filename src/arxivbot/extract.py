"""Turning a paper's sections into an :class:`ImplementationSpec`.

Extraction is decomposed into several narrow calls rather than one large one:
an inventory of components, then detail per component, then training, then a
pass that looks for what the paper never says. Small tasks with small output
schemas are where free-tier models hold up; asking one call to produce an
entire spec produces mush.

**Quotes are verified, not trusted.** The model returns the sentence it is
relying on, and this module finds that sentence in the source itself to derive
the character offsets. A quote that cannot be found did not come from the
paper, and the claim resting on it is demoted rather than recorded as fact.
That check costs nothing and catches the failure mode that matters most here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

from pydantic import BaseModel, Field

from arxivbot import __version__
from arxivbot.ingest.latex import (
    ALGORITHM_ENVS,
    EQUATION_ENVS,
    Document,
    Section,
    extract_environments,
)
from arxivbot.llm import LLMConfig, LLMError, complete_json
from arxivbot.spec import (
    Assumptions,
    Component,
    Confidence,
    Equation,
    Extractor,
    Hyperparameter,
    ImplementationSpec,
    Modality,
    NormPosition,
    Ordering,
    Provenance,
    Regime,
    Severity,
    TensorSpec,
    TrainingSpec,
    Unknown,
)

SYSTEM = """You extract implementation details from machine learning papers.

Rules you must follow:
- Report only what the paper says. Never invent a value the paper does not give.
- When you mark something as stated, quote the sentence from the paper verbatim.
  The quote is checked against the source; a quote that cannot be found is
  discarded along with the claim it supports.
- If the paper does not specify something, say so. Omissions are the most
  valuable thing you can report - do not paper over them with what is typical.
- Use the paper's own names for things.
- The text is LaTeX. Ignore markup; read the content."""

# A single call is capped well under the model's context window so that a long
# paper degrades by covering fewer components rather than by failing outright.
MAX_CHARS_PER_CALL = 60_000

# Shortest quote accepted as provenance, measured after markup is stripped.
MIN_QUOTE_CHARS = 12


# ------------------------------------------------------- call-level models ----
# Deliberately small and flat. Each is one narrow question, and none of them
# asks the model for a character offset - those are computed here.


class RawTensor(BaseModel):
    name: str
    shape: list[str] = Field(
        default_factory=list, description="Symbolic dimensions, e.g. batch, seq, d_model"
    )
    description: str | None = None


class RawEquation(BaseModel):
    latex: str = Field(description="The equation in LaTeX, as the paper writes it")
    description: str = Field(description="What it computes, one sentence")
    quote: str | None = Field(
        default=None, description="Verbatim sentence from the paper introducing it"
    )


class RawHyperparameter(BaseModel):
    name: str
    value: str | None = Field(
        default=None, description="Exactly as written, or null if not given"
    )
    confidence: Confidence = Field(
        description="stated if the paper gives it; conventional if you are "
        "supplying a community default; guess otherwise"
    )
    quote: str | None = Field(
        default=None, description="Verbatim sentence from the paper. Required if stated."
    )
    applies_to: str | None = Field(
        default=None,
        description="The condition this value holds under, if the paper gives "
        "one - a model size, a dataset, a phase of training. A value quoted "
        "without its condition reads as universal when it is not.",
    )


class ComponentSketch(BaseModel):
    name: str = Field(description="The paper's own name for this component")
    role: str = Field(description="What it does, one sentence")
    depends_on: list[str] = Field(
        default_factory=list, description="Names of components this one is built from"
    )


class Inventory(BaseModel):
    summary: str = Field(description="What the paper proposes, 2-4 sentences")
    official_repo: str | None = Field(
        default=None, description="Implementation URL if the paper gives one"
    )
    components: list[ComponentSketch]


class UnitComponents(BaseModel):
    """What one section of the method needs implemented.

    Deliberately not `Inventory`: a per-unit call reusing that schema spends
    its output budget writing a summary of a single paragraph, and a longer
    section then truncates mid-string and fails to parse.
    """

    components: list[ComponentSketch] = Field(default_factory=list)


class Missing(BaseModel):
    components: list[ComponentSketch] = Field(default_factory=list)


class RawAssumptions(BaseModel):
    """The assumption axes, as the model must answer them.

    Deliberately separate from :class:`Assumptions`: the stored model gives
    every axis a default so old specs still load, and those same defaults let
    the model skip the question entirely.
    """

    modality: Modality = Field(
        default=Modality.UNSPECIFIED,
        description="What this component expects to be handed",
    )
    ordering: Ordering = Field(
        default=Ordering.UNSPECIFIED,
        description="Whether reordering the input changes the output",
    )
    regime: Regime = Field(
        default=Regime.UNSPECIFIED,
        description="The training setting it was designed inside",
    )
    norm_position: NormPosition = Field(
        default=NormPosition.UNSPECIFIED,
        description="Normalisation relative to the residual branch",
    )

    # Pydantic defaults keep the fallback path constructible, but the schema
    # sent to the model lists every axis as required, so it cannot skip them.
    model_config = {"json_schema_extra": {"required": ["modality", "ordering", "regime", "norm_position"]}}


class ComponentDetail(BaseModel):
    inputs: list[RawTensor] = Field(default_factory=list)
    outputs: list[RawTensor] = Field(default_factory=list)
    equations: list[RawEquation] = Field(default_factory=list)
    hyperparameters: list[RawHyperparameter] = Field(default_factory=list)
    # Required, with no default on any axis. Given one, the model simply
    # omitted the field and pydantic filled in "unspecified" for everything -
    # the schema, not the prompt, was deciding the answer.
    assumptions: RawAssumptions


class RawTraining(BaseModel):
    objective: str | None = None
    optimizer: str | None = None
    schedule: str | None = None
    datasets: list[str] = Field(default_factory=list)
    hardware: str | None = None
    hyperparameters: list[RawHyperparameter] = Field(default_factory=list)


class RawUnknown(BaseModel):
    question: str = Field(description="What the paper fails to specify")
    why_it_matters: str = Field(description="What a wrong choice here would do")
    severity: Severity
    conventional_default: str | None = Field(
        default=None, description="What the literature usually does, if anything"
    )
    applies_to: str | None = None


class Unknowns(BaseModel):
    items: list[RawUnknown] = Field(default_factory=list)


# ------------------------------------------------------------- provenance ----


_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?")

# Markup a reader does not see, and so does not reproduce when quoting.
_MARKUP = frozenset("${}~^_&")

# Commands whose argument is invisible in the rendered paper too. A citation
# sits in the middle of a sentence, and leaving its key behind turns
# "dropout \citep{srivastava14a} to the output" into text no quote can match.
_DROP_ARGUMENT = frozenset(
    """cite citep citet citealp citealt citeyear nocite label ref eqref cref
    autoref pageref footnote url href""".split()
)


@lru_cache(maxsize=8)
def _strip_markup(text: str) -> tuple[str, tuple[int, ...]]:
    """Reduce LaTeX to readable characters, keeping a map back to the source.

    ``map[i]`` is the offset in ``text`` that produced normalised character
    ``i``, which is what lets a match in the stripped text be reported as a
    span in the original.
    """
    chars: list[str] = []
    offsets: list[int] = []
    i, n = 0, len(text)

    while i < n:
        char = text[i]
        if char == "\\":
            if match := _COMMAND_RE.match(text, i):
                i = match.end()  # \textbf, \frac, ... carry no read text
                if match.group(0).lstrip("\\").rstrip("*") in _DROP_ARGUMENT:
                    j = i
                    while j < n and text[j] in " \t":
                        j += 1
                    if j < n and text[j] == "[":  # optional \citep[see]{...}
                        depth = 1
                        j += 1
                        while j < n and depth:
                            depth += {"[": 1, "]": -1}.get(text[j], 0)
                            j += 1
                    if j < n and text[j] == "{":
                        depth = 1
                        j += 1
                        while j < n and depth:
                            depth += {"{": 1, "}": -1}.get(text[j], 0)
                            j += 1
                        i = j
            else:
                i += 1
            continue
        if char in _MARKUP:
            i += 1
            continue
        if char.isspace():
            # Dropped entirely, not collapsed. LaTeX spaces where it likes and
            # maths carries none at all: the source writes `$d_{model}=512$`
            # while a reader quotes "d_model = 512". Comparing without any
            # whitespace is what makes those the same string.
            i += 1
            continue
        chars.append(char.lower())
        offsets.append(i)
        i += 1

    return "".join(chars), tuple(offsets)


def locate(tex: str, quote: str | None) -> tuple[int, int] | None:
    """Find ``quote`` in the source, seeing through LaTeX markup.

    A model reads ``$d_k = 64$`` and quotes it as "d_k = 64", which no literal
    search will find. Both sides are stripped of markup, case and whitespace
    before comparison, and the match is mapped back to real offsets.

    This is the check that separates a claim grounded in the paper from one
    the model made up, so it must not reject quotes that are genuinely there:
    a false rejection demotes a correct claim and makes the output look less
    trustworthy than it is.
    """
    if not quote or not quote.strip():
        return None

    needle, _ = _strip_markup(quote)
    needle = needle.strip()
    if not needle:
        return None

    # A short quote proves nothing: "h=8" occurs in thousands of papers, so
    # finding it says only that the string is common, not that this paper
    # backs the claim. The floor applies before the exact-match path too -
    # guarding only the fuzzy path let trivial quotes through by coincidence.
    if len(needle) < MIN_QUOTE_CHARS:
        return None

    if (index := tex.find(quote)) != -1:
        return index, index + len(quote)

    haystack, offsets = _strip_markup(tex)
    if not offsets:
        return None

    position = haystack.find(needle)
    if position == -1:
        return None

    end = min(position + len(needle) - 1, len(offsets) - 1)
    return offsets[position], offsets[end] + 1


def _provenance(tex: str, section: str, quote: str | None) -> Provenance | None:
    span = locate(tex, quote)
    if span is None or quote is None:
        return None
    return Provenance(
        section=section,
        quote=quote[:600],
        char_start=span[0],
        char_end=span[1],
    )


@dataclass(slots=True)
class Report:
    """What happened during an extraction, for the user and for debugging."""

    calls: int = 0
    verified_quotes: int = 0
    rejected_quotes: int = 0
    demoted: list[str] = field(default_factory=list)
    """Claims marked stated whose quote was not found in the paper."""

    recovered: list[str] = field(default_factory=list)
    """Components the first pass missed and the second pass found."""

    rejected: list[str] = field(default_factory=list)
    """Proposals consolidation judged not to be implementable units."""

    refuted_gaps: list[str] = field(default_factory=list)
    """Gaps withdrawn because the answer was found elsewhere in the paper."""

    gaps_read: bool = True
    """False when the gap pass could not run at all.

    Without this an empty `unknowns` is ambiguous: it means either that the
    paper specifies everything or that the provider refused the call. A
    corpus run hit a quota mid-way and recorded "0 gaps" for both, which is
    the project's cardinal failure - a failure that looks like a result."""

    warnings: list[str] = field(default_factory=list)

    @property
    def quote_accuracy(self) -> float | None:
        total = self.verified_quotes + self.rejected_quotes
        return self.verified_quotes / total if total else None


def _convert_hyperparameters(
    raws: list[RawHyperparameter], tex: str, section: str, report: Report
) -> list[Hyperparameter]:
    """Attach verified provenance, demoting claims whose quote does not check out."""
    out: list[Hyperparameter] = []
    for raw in raws:
        provenance = _provenance(tex, section, raw.quote)
        confidence = raw.confidence

        if confidence is Confidence.STATED:
            if provenance is None:
                # The model claimed the paper says this but could not produce a
                # sentence that appears in it. Record the value, not the claim.
                report.rejected_quotes += 1
                report.demoted.append(raw.name)
                confidence = Confidence.GUESS
            else:
                report.verified_quotes += 1
        elif provenance is not None:
            report.verified_quotes += 1

        out.append(
            Hyperparameter(
                name=raw.name,
                value=raw.value,
                confidence=confidence,
                provenance=provenance,
                applies_to=raw.applies_to,
            )
        )
    return out


# ------------------------------------------------------------ the passes ----


def _text_of(sections: list[Section], doc: Document, limit: int = MAX_CHARS_PER_CALL) -> str:
    chunks = []
    total = 0
    for section in sections:
        body = f"## {section.title}\n{doc.content(section)}"
        if total + len(body) > limit:
            break
        chunks.append(body)
        total += len(body)
    return "\n\n".join(chunks)


def landmarks(doc: Document, sections: list[Section]) -> dict[str, list[str]]:
    """The structural features of a paper's method, from the source itself.

    Subsection headings, algorithm blocks and display equations exist in every
    paper in every field. Using them rather than a list of expected part names
    is what keeps this from being tuned to one kind of paper: a diffusion
    paper has noise schedules where a transformer has attention heads, but
    both write equations and both cut their method into subsections.
    """
    # Expand to the children of each matched section. `methodology()` matches
    # headings by keyword, so it returns "Model Architecture" but not the
    # "Encoder and Decoder Stacks" beneath it - and that is exactly where a
    # paper puts the residual connections and the normalisation. Listing only
    # the matched headings points the model at the wrong parts of its own
    # structure, and the pieces defined in the unlisted ones go missing.
    expanded: list[Section] = []
    for section in sections:
        if section not in expanded:
            expanded.append(section)
        for child in doc.children(section):
            if child not in expanded:
                expanded.append(child)

    span = (
        (min(s.start for s in expanded), max(s.end for s in expanded))
        if expanded
        else (0, len(doc.tex))
    )
    inside = lambda start: span[0] <= start < span[1]  # noqa: E731

    return {
        "subsections": [s.title for s in expanded if s.level > 1],
        "algorithms": [
            env.body.strip().splitlines()[0][:80] if env.body.strip() else env.name
            for env in extract_environments(doc.tex, ALGORITHM_ENVS)
            if inside(env.start)
        ],
        "equations": [
            " ".join(env.body.split())[:110]
            for env in extract_environments(doc.tex, EQUATION_ENVS)
            if inside(env.start)
        ],
    }


def _landmarks(doc: Document, sections: list[Section]) -> str:
    """Those features, written into the prompt as things to account for."""
    found = landmarks(doc, sections)
    lines: list[str] = []
    for label, items in (
        ("Subsections of the method", found["subsections"]),
        ("Algorithm blocks", found["algorithms"]),
        ("Equations", found["equations"]),
    ):
        if items:
            lines.append(f"{label} ({len(items)}):")
            lines.extend(f"  - {item}" for item in items[:20])
    if not lines:
        return ""
    return (
        "The method contains the following. Every one of them belongs to some "
        "component; if something here is not covered by a component you list, "
        "you have missed a component.\n\n" + "\n".join(lines) + "\n\n"
    )


def _inventory(doc: Document, config: LLMConfig, report: Report) -> Inventory:
    sections = doc.methodology() or doc.sections[:6]
    prompt = (
        f"Paper: {doc.meta.title}\n\n"
        "List the components of the proposed method - the pieces someone would "
        "implement as separate units. Use the paper's own names. Set depends_on "
        "where one component is built from another.\n\n"
        "Be exhaustive, and include the structural pieces as well as the "
        "headline ones: anything that wraps, connects or feeds the others is "
        "still something a reimplementer has to write. A component you leave "
        "out is reported to the reader as something the paper does not cover.\n\n"
        f"{_landmarks(doc, sections)}"
        f"{_text_of(sections, doc)}"
    )
    report.calls += 1
    return complete_json(prompt, Inventory, system=SYSTEM, config=config)


# A unit smaller than this is a stub heading, not something to read.
MIN_UNIT_CHARS = 180


def units(doc: Document, sections: list[Section]) -> list[Section]:
    """The smallest pieces of the method worth reading on their own.

    One call over sixteen thousand characters asked "what is in here?" loses
    things: residual connections survive in a single clause, and a reader
    skimming for headline contributions walks past it. The same clause inside
    a six-hundred-character paragraph has nowhere to hide.

    Recall is a function of scope, so the unit is a leaf - a subsection or a
    paragraph with no children of its own - rather than a whole section.
    """
    out: list[Section] = []
    for section in sections:
        children = doc.children(section)
        leaves = [c for c in children if not doc.children(c)]
        for candidate in leaves or [section]:
            if candidate in out:
                continue
            if len(doc.content(candidate)) >= MIN_UNIT_CHARS:
                out.append(candidate)
    return out


def _unit_inventory(
    unit: Section, doc: Document, config: LLMConfig, report: Report
) -> UnitComponents:
    prompt = (
        f"Paper: {doc.meta.title}\n"
        f"Section: {unit.title}\n\n"
        "Read only the text below. What would someone reimplementing this "
        "paper have to write in order to realise it? Name each piece the way "
        "this text names it.\n\n"
        "Include anything applied to, around or between the others - a "
        "connection, a normalisation, a mask, a projection - even where it is "
        "mentioned in passing rather than given a definition of its own. "
        "Mentioned in passing still has to be written.\n\n"
        "Nothing here may be left out on the grounds that it is minor, and "
        "nothing may be added that this text does not support.\n\n"
        f"{doc.content(unit)[:12000]}"
    )
    report.calls += 1
    return complete_json(prompt, UnitComponents, system=SYSTEM, config=config)


_REPO_RE = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/"
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
)


def official_repo(doc: Document) -> str | None:
    """The implementation URL the paper prints, found by reading it.

    A model asked for this will sometimes supply the repository the paper
    *ought* to have. A regular expression can only return a string that is
    actually in the source.
    """
    for match in _REPO_RE.finditer(doc.tex):
        url = match.group(0).rstrip(".,);")
        if not url.endswith((".sty", ".cls")):
            return url
    return None


class Judgement(BaseModel):
    name: str = Field(description="The proposed name, copied exactly")
    implementable: bool = Field(
        description="True if this is a unit someone writes as a function, "
        "class or module of its own"
    )
    canonical_name: str = Field(
        description="The clearest name for this thing, as the paper would "
        "put it. Proposals that are the same thing must share one."
    )
    defining_quote: str | None = Field(
        default=None,
        description="The sentence in the paper that introduces this thing, "
        "verbatim. Leave null if the paper never introduces it as a thing.",
    )


class Consolidated(BaseModel):
    components: list[Judgement] = Field(default_factory=list)


def _consolidate(
    proposals: list[ComponentSketch], doc: Document, config: LLMConfig, report: Report
) -> list[ComponentSketch]:
    """Prune generous proposals down to the units a person would write.

    Reading each section separately recovers what a single pass misses, and
    also proposes a variable from an equation, the name of an operation, a
    positional reference like "third sub-layer", and the model as a whole.
    Recall was the hard problem; precision is recoverable afterwards, and this
    is where it is recovered.

    The model is asked to judge a fixed list rather than produce one. A
    decision per item has a far smaller space to vary in than a list has,
    which is the same reason the quote check works: a narrow question against
    fixed material is answerable, where an open one is a performance.
    """
    if len(proposals) < 2:
        return proposals

    # Names with a clipped role. The full roles are near-verbatim paper
    # sentences, and echoing a list of them back makes the reply look to the
    # provider like a copy of the source - it refuses the call outright
    # (finishReason RECITATION) and pruning is skipped in silence. Names
    # alone dodge that but leave nothing to judge by, and operations like
    # "Concat" then read as plausible components. A clipped phrase is enough
    # context and short enough not to trip the filter.
    listed = "\n".join(f"  {s.name} - {s.role[:70]}" for s in proposals)
    prompt = (
        f"Paper: {doc.meta.title}\n\n"
        "Each line below was proposed as a component of this paper's method. "
        "Reading them together, decide two things for each.\n\n"
        "IMPLEMENTABLE - would someone reimplementing this paper write this "
        "as a function, class or module of its own?\n"
        "  yes: a block, a layer, a transformation, a schedule, a loss, a "
        "sampler, anything with inputs and outputs of its own\n"
        "  no:  a symbol from an equation; a single tensor operation such as "
        "a concatenation, a reshape, a projection or an activation, which is "
        "a line inside whatever uses it rather than a unit of its own; a "
        "position in a list rather than a thing; a setting or a constant; the "
        "complete model, which is all of these together\n\n"
        "When a proposal could be read either way, keep it. A spurious "
        "component is visible to the reader and can be ignored; a missing one "
        "is reported to them as the paper not covering it, which is the "
        "failure this whole pipeline exists to prevent. Instructed to be "
        "strict here, this step removed the residual connection and the "
        "multi-head attention while leaving the word \"sub-layer\" in place.\n\n"
        "CANONICAL_NAME - the clearest name for the thing itself. Proposals "
        "that name one thing must be given the same canonical name, however "
        "differently they were phrased. Use the paper's wording.\n\n"
        "DEFINING_QUOTE - the sentence where the paper introduces this as a "
        "thing in its own right, copied verbatim. The quote is checked "
        "against the source, so a sentence that is not in the paper is worse "
        "than none. A thing the paper never introduces has no quote - say so "
        "with null rather than finding something nearby.\n\n"
        "Judge every line. Do not add any.\n\n"
        f"{listed}"
    )
    report.calls += 1
    try:
        verdicts = complete_json(prompt, Consolidated, system=SYSTEM, config=config)
    except LLMError as exc:
        report.warnings.append(f"consolidation: {exc}")
        return proposals

    by_name = {s.name.strip().lower(): s for s in proposals}
    kept: dict[str, ComponentSketch] = {}
    for verdict in verdicts.components:
        source = by_name.get(verdict.name.strip().lower())
        if source is None:
            continue
        if not verdict.implementable:
            report.rejected.append(source.name)
            continue
        # A defining sentence was tried as a hard gate here - drop anything
        # the paper does not introduce - on the reasoning that "head_i" is a
        # symbol and "concat" an operation, while a residual connection has a
        # sentence of its own. Measured, it cost far more than it bought:
        # components per run went from 13-20 to 3-16 and residual connection
        # itself fell to 2/3, because a real component whose quote the model
        # paraphrases is indistinguishable from an invented one. The quote is
        # kept as evidence where it checks out, and never used to delete.
        if verdict.defining_quote and locate(doc.tex, verdict.defining_quote):
            report.verified_quotes += 1
        key = _canonical(verdict.canonical_name) or _canonical(source.name)
        existing = kept.get(key)
        if existing is None:
            source.name = verdict.canonical_name.strip() or source.name
            kept[key] = source
            continue
        if len(source.role) > len(existing.role):
            existing.role = source.role
        for dependency in source.depends_on:
            if dependency not in existing.depends_on:
                existing.depends_on.append(dependency)

    # A refusal to judge is not a reason to throw the paper away.
    return list(kept.values()) or proposals


def _inventory_units(doc: Document, config: LLMConfig, report: Report) -> Inventory:
    """Read the method one unit at a time and merge what each turns up.

    The summary and the repository link do not need a model: the abstract is
    the authors' own summary, and a URL either appears in the source or does
    not. Spending a call on either invites an answer that sounds right.
    """
    sketches: list[ComponentSketch] = []
    for unit in units(doc, doc.methodology() or doc.sections[:6]):
        try:
            sketches.extend(_unit_inventory(unit, doc, config, report).components)
        except LLMError as exc:
            report.warnings.append(f"unit {unit.title!r}: {exc}")

    merged = _consolidate(_merge(sketches), doc, config, report)
    return Inventory(
        summary=doc.meta.abstract,
        official_repo=official_repo(doc),
        components=_remap(merged),
    )


def _canonical(name: str) -> str:
    """A name reduced to what two spellings of the same thing share.

    Runs disagree on wording - Embeddings, Embeddings and Softmax, Softmax -
    and three names for one component is three components as far as anything
    downstream can tell.
    """
    text = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    words = [w for w in text.split() if w not in _NAME_NOISE]
    return " ".join(sorted(w.rstrip("s") for w in words))


_NAME_NOISE = frozenset(
    "the a an of and or for to in on with its module layer block component "
    "mechanism function networks network".split()
)


def _remap(components: list[ComponentSketch]) -> list[ComponentSketch]:
    """Point every depends_on at the name its target ended up with.

    Merging and consolidation both rename things, and a dependency still
    spelled the old way resolves to nothing. The graph walk then drops it in
    silence, so asking for the decoder returns the decoder by itself.
    """
    canon = {_canonical(c.name): c.name for c in components}
    for component in components:
        fixed: list[str] = []
        for dependency in component.depends_on:
            key = _canonical(dependency)
            target = canon.get(key)
            if target is None:
                for existing, name in canon.items():
                    if _subsumes(existing, key):
                        target = name
                        break
            if target and target != component.name and target not in fixed:
                fixed.append(target)
        component.depends_on = fixed
    return components


def _subsumes(a: str, b: str) -> bool:
    """Whether two canonical names are qualified versions of one thing.

    Reading nine sections separately produces "embeddings", "input
    embeddings", "output embeddings" and "learned embeddings" - one component
    under four names, which string similarity will not fold together because
    they are genuinely different strings. What they share is that one name's
    words contain the other's.
    """
    first, second = set(a.split()), set(b.split())
    if not first or not second:
        return False
    return first <= second or second <= first


def _merge(sketches: list[ComponentSketch]) -> list[ComponentSketch]:
    """Fold duplicate and near-duplicate names into one component each."""
    kept: list[ComponentSketch] = []
    keys: list[str] = []
    for sketch in sketches:
        key = _canonical(sketch.name)
        if not key:
            continue
        match = None
        for index, existing in enumerate(keys):
            if (
                existing == key
                or _subsumes(existing, key)
                or SequenceMatcher(None, existing, key).ratio() > 0.86
            ):
                match = index
                break
        if match is None:
            keys.append(key)
            kept.append(sketch)
            continue
        # Keep the longer role; two readings of one component rarely say the
        # same amount, and the fuller one is the more useful.
        if len(sketch.role) > len(kept[match].role):
            kept[match].role = sketch.role
        # Prefer the plainer name. Of "embeddings" and "learned input
        # embeddings", the first is what a reader will type.
        if len(sketch.name) < len(kept[match].name):
            kept[match].name = sketch.name
            keys[match] = _canonical(sketch.name)
        for dependency in sketch.depends_on:
            if dependency not in kept[match].depends_on:
                kept[match].depends_on.append(dependency)
    return kept


def _missing(
    doc: Document,
    found: list[ComponentSketch],
    config: LLMConfig,
    report: Report,
) -> Missing:
    """Ask what the first pass left out.

    Landmarks are headings, equations and algorithm blocks, so a piece the
    paper describes only in a sentence has no landmark and structural
    grounding cannot reach it - which is how residual connections and layer
    normalisation went missing from ten consecutive runs. They are introduced
    by a clause: "we employ a residual connection around each of the two
    sub-layers, followed by layer normalization".

    This pass asks about the relationship rather than the thing. Nothing here
    names a part of any particular architecture, so it reads the same way over
    a sampler, a target network or a message-passing step.
    """
    sections = doc.methodology() or doc.sections[:6]
    listed = "\n".join(f"  - {c.name}: {c.role}" for c in found)
    prompt = (
        f"Paper: {doc.meta.title}\n\n"
        "These components have already been recorded from this paper:\n\n"
        f"{listed}\n\n"
        "Read the method text again and look for what is not in that list. "
        "In particular, anything applied to, around, or between the recorded "
        "components, or that every one of them passes through. A paper often "
        "introduces such a piece in a clause rather than under a heading or an "
        "equation of its own, which is exactly how it escapes a first reading "
        "- but a reimplementer still has to write it.\n\n"
        "List only what is genuinely absent above. Returning an empty list is "
        "a correct answer when nothing is missing; do not pad it, and do not "
        "restate something already recorded under a different name.\n\n"
        f"{_text_of(sections, doc)}"
    )
    report.calls += 1
    return complete_json(prompt, Missing, system=SYSTEM, config=config)


def _detail(
    sketch: ComponentSketch, doc: Document, config: LLMConfig, report: Report
) -> ComponentDetail:
    relevant = doc.find(*sketch.name.split()[:3]) or doc.methodology()
    prompt = (
        f"Paper: {doc.meta.title}\n"
        f"Component: {sketch.name} - {sketch.role}\n\n"
        "Extract this component's input and output tensors (symbolic shapes), the "
        "equations defining it, and its hyperparameters. Quote the paper verbatim "
        "for anything you mark as stated. Omit anything the paper does not give.\n\n"
        "Also record what this component assumes about the model around it. "
        "Unlike the fields above, these are NOT quoted claims about what the "
        "paper says - no paper writes them down. They are what the component "
        "requires in order to work, which you infer from how it is used and "
        "what kind of model this paper builds. Infer them.\n"
        "- modality: what it is handed. A paper about language hands its blocks "
        "sequences; one about images hands them images; one that first encodes "
        "its input with another trained model hands them latents. Use `any` "
        "only for something genuinely shape-agnostic, like a normalisation "
        "layer or an optimiser.\n"
        "- ordering: would reordering the input change the output? If the "
        "component needs position to be told to it, it is ordered. If it must "
        "give the same answer whatever the order, it is permutation-invariant.\n"
        "- regime: the training setting it was designed inside.\n"
        "- norm_position: `pre` if normalisation is applied inside the residual "
        "branch before the transformation, `post` if it is applied after the "
        "branch is added back, `none` if this component has no normalisation.\n"
        "Use `unspecified` only when you genuinely cannot tell from the paper - "
        "not merely because the paper does not say it outright, since it never "
        "will.\n\n"
        f"{_text_of(relevant, doc, limit=30_000)}"
    )
    report.calls += 1
    return complete_json(prompt, ComponentDetail, system=SYSTEM, config=config)


def _training(doc: Document, config: LLMConfig, report: Report) -> RawTraining | None:
    sections = doc.find("training", "experiment", "setup", "implementation detail")
    if not sections:
        return None
    prompt = (
        f"Paper: {doc.meta.title}\n\n"
        "Extract the training setup: objective, optimizer, schedule, datasets, "
        "hardware and hyperparameters. Quote verbatim for stated values.\n\n"
        f"{_text_of(sections, doc, limit=30_000)}"
    )
    report.calls += 1
    return complete_json(prompt, RawTraining, system=SYSTEM, config=config)


# --------------------------------------------------------- gap refutation ----

# Words that carry no retrieval signal. Most are the scaffolding of a question
# about absence - "what exact value is specified" - and matching on them pulls
# back the whole paper.
_GAP_STOP = frozenset(
    """a an the is are was were be been being of for to in on by with and or
    not does do did what which how when where many much any their its it this
    that from at as if then than used use using specified stated state states
    paper authors report reported given unclear unspecified whether exact
    specific precise other each both such during across between within same
    different""".split()
)

# A term found this often is a theme of the paper, not a locator for one fact.
# Terms below it are still weighted down by how common they are, so the cutoff
# only has to catch the hopeless ones.
_TOO_COMMON = 120

# How far apart two matches can be and still describe the same passage.
_CLUSTER = 400
_WINDOW = 500
_MAX_PASSAGES = 6


class GapCheck(BaseModel):
    """Whether the passages answer the question, decided over short text."""

    answered: bool = Field(
        description="True only if a passage states the answer outright"
    )
    answer: str | None = Field(
        default=None, description="The answer, as briefly as it can be put"
    )
    quote: str | None = Field(
        default=None,
        description="The verbatim sentence stating it, copied exactly",
    )


def _gap_terms(question: str, applies_to: str | None) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", f"{question} {applies_to or ''}")
    terms, seen = [], set()
    for word in words:
        lower = word.lower()
        if lower in _GAP_STOP or lower in seen or len(lower) < 4:
            continue
        seen.add(lower)
        terms.append(lower)
    return terms


def _passages(doc: Document, terms: list[str]) -> list[tuple[str, str]]:
    """Windows from anywhere in the paper where several terms occur together.

    The gap pass only ever sees the method and the appendix, which is why it
    reports hyperparameters as unspecified when they are sitting in the
    experiments section. This searches the whole document, bibliography
    excluded, so a gap can be refuted by the part of the paper that answers it.
    """
    tex = doc.tex

    def in_bibliography(offset: int) -> bool:
        # Skipped, but only the span itself: the appendix usually sits after
        # the reference list, and that is exactly where the hyperparameter
        # tables that refute a gap tend to be.
        return 0 <= doc.bib_start <= offset < doc.bib_end

    hits: list[tuple[int, str]] = []
    weight: dict[str, float] = {}
    for term in terms:
        found = [
            m.start()
            for m in re.finditer(re.escape(term), tex, re.I)
            if not in_bibliography(m.start())
        ]
        # A term in every other paragraph locates nothing.
        if not found or len(found) > _TOO_COMMON:
            continue
        # "activation" appears everywhere in a deep learning paper and
        # "warmup" appears where the answer is. Ranking on the count of
        # distinct terms treats those two alike, which is how a window
        # holding four vague words outranked the one holding the number.
        weight[term] = 1.0 / math.log(2 + len(found))
        hits.extend((offset, term) for offset in found)

    if not hits:
        return []
    hits.sort()

    clusters: list[list[tuple[int, str]]] = [[hits[0]]]
    for hit in hits[1:]:
        if hit[0] - clusters[-1][-1][0] <= _CLUSTER:
            clusters[-1].append(hit)
        else:
            clusters.append([hit])

    # Distinct terms, each worth what it is worth: a window holding one rare
    # term beats one holding three ubiquitous ones.
    clusters.sort(
        key=lambda c: (sum(weight[t] for t in {t for _, t in c}), len(c)),
        reverse=True,
    )

    passages = []
    for cluster in clusters[:_MAX_PASSAGES]:
        start = max(0, cluster[0][0] - _WINDOW)
        end = min(len(tex), cluster[-1][0] + _WINDOW)
        passages.append((_section_at(doc, cluster[0][0]), tex[start:end]))

    titled = _section_titled(doc, terms)
    if titled is not None:
        passages.insert(0, titled)
    return passages


def _section_titled(doc: Document, terms: list[str]) -> tuple[str, str] | None:
    """The section whose heading answers the question, if a paper has one.

    Papers routinely name a section after the very thing a gap asks about -
    LoRA has "Which Weight Matrices in Transformer Should We Apply LoRA to?".
    Offsets alone miss it when the answer sits a paragraph past the nearest
    keyword, and widening the window to reach it would bloat every other
    lookup. A heading is the cheaper and more structural signal.
    """
    wanted = set(terms)
    best, score = None, 0
    for section in doc.sections:
        words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", section.title)}
        overlap = len(words & wanted)
        if overlap > score and overlap >= 2:
            best, score = section, overlap
    if best is None:
        return None
    return best.title, doc.content(best)[: _WINDOW * 4]


def _section_at(doc: Document, offset: int) -> str:
    title = ""
    for section in doc.sections:
        if section.start <= offset:
            title = section.title
        else:
            break
    return title or doc.meta.title


def _refute(
    doc: Document,
    raw: RawUnknown,
    config: LLMConfig,
    report: Report,
) -> tuple[str, Provenance] | None:
    """Try to find, anywhere in the paper, the answer a gap says is absent.

    A gap is the one claim the pipeline makes with nothing mechanical behind
    it, and an audit of twenty found seven asserting the paper omits something
    it states plainly. So the model proposes the gap and this disposes of it:
    retrieve, ask one short question over the retrieved text only, and require
    a quote that can be located in the source. Reasoning about absence over a
    whole paper is the hardest possible framing; over four paragraphs it is an
    ordinary reading question.
    """
    passages = _passages(doc, _gap_terms(raw.question, raw.applies_to))
    if not passages:
        return None

    body = "\n\n".join(f"## {title}\n{text}" for title, text in passages)
    prompt = (
        f"Paper: {doc.meta.title}\n\n"
        "Someone claims this paper never specifies the following. Passages "
        "from the paper that mention the same terms are below.\n\n"
        f"Question: {raw.question}\n\n"
        "Answer `answered: true` if one of these passages states the answer "
        "outright, and then copy the text that states it verbatim. That text "
        "may be a sentence, a row of a table, or a line of an equation - "
        "hyperparameters are usually given in tables, and a table row counts. "
        "If the question asks about several things at once, answer true when "
        "the passages state it for the main one.\n\n"
        "A passage that merely discusses the topic, cites another paper for "
        "it, or states something similar for a different part of the model is "
        "NOT an answer - say false. Do not infer, and do not fill the answer "
        "in from what papers usually do.\n\n"
        f"{body}"
    )
    report.calls += 1
    try:
        check = complete_json(prompt, GapCheck, system=SYSTEM, config=config)
    except LLMError as exc:
        report.warnings.append(f"gap check: {exc}")
        return None

    if not check.answered or not check.answer:
        return None

    # The same rule every other claim lives under: a quote that cannot be
    # found in the source does not count as evidence, so the gap stands.
    provenance = _provenance(doc.tex, _section_at(doc, 0), check.quote)
    if provenance is None:
        report.warnings.append(
            f"gap {raw.question[:50]!r}: answer offered without a locatable quote"
        )
        return None

    span = locate(doc.tex, check.quote)
    if span:
        provenance = Provenance(
            section=_section_at(doc, span[0]),
            quote=(check.quote or "")[:600],
            char_start=span[0],
            char_end=span[1],
        )
    return check.answer, provenance


def _unknowns(
    doc: Document,
    components: list[Component],
    training: TrainingSpec | None,
    config: LLMConfig,
    report: Report,
) -> Unknowns:
    lines = [
        f"- {c.name}: " + ", ".join(f"{h.name}={h.value}" for h in c.hyperparameters)
        for c in components
    ]
    if training:
        # Without this the pass reports the optimizer and schedule as missing
        # even though the previous call just extracted them.
        settings = ", ".join(
            f"{h.name}={h.value}" for h in training.hyperparameters if h.value
        )
        lines.append(
            f"- Training: objective={training.objective}, "
            f"optimizer={training.optimizer}, schedule={training.schedule}, "
            f"datasets={', '.join(training.datasets)}, hardware={training.hardware}"
            + (f", {settings}" if settings else "")
        )
    known = "\n".join(lines)
    sections = (doc.methodology() or []) + doc.appendix()
    prompt = (
        f"Paper: {doc.meta.title}\n\n"
        "Someone is reimplementing this paper from scratch. Identify what the paper "
        "does NOT specify but that they must decide - initialisation, normalisation "
        "placement, masking, tokenisation, ordering of operations, exact shapes, and "
        "so on.\n\n"
        "Do not list things the paper does state. For each gap, say why a wrong "
        "choice would matter, and grade it: blocking (cannot proceed), significant "
        "(results likely change), minor.\n\n"
        f"Already extracted:\n{known}\n\n"
        f"{_text_of(sections, doc)}"
    )
    report.calls += 1
    return complete_json(prompt, Unknowns, system=SYSTEM, config=config)


# ------------------------------------------------------------ entry point ----


def extract(
    doc: Document,
    *,
    config: LLMConfig | None = None,
    max_components: int = 8,
    with_training: bool = True,
    with_unknowns: bool = True,
    with_missing: bool = True,
    per_unit: bool = True,
) -> tuple[ImplementationSpec, Report]:
    """Read a paper and build a spec from it.

    Returns the spec and a :class:`Report` describing how the extraction went,
    including how many of the model's quotes were actually found in the paper.
    """
    config = config or LLMConfig.from_env()
    report = Report()
    tex = doc.tex

    inventory = (
        _inventory_units(doc, config, report)
        if per_unit
        else _inventory(doc, config, report)
    )

    sketches = list(inventory.components)
    if with_missing and sketches:
        try:
            seen = {s.name.strip().lower() for s in sketches}
            for sketch in _missing(doc, sketches, config, report).components:
                if sketch.name.strip().lower() not in seen:
                    seen.add(sketch.name.strip().lower())
                    sketches.append(sketch)
                    report.recovered.append(sketch.name)
        except LLMError as exc:
            report.warnings.append(f"missing-components pass: {exc}")

    components: list[Component] = []
    for sketch in sketches[:max_components]:
        try:
            detail = _detail(sketch, doc, config, report)
        except LLMError as exc:
            report.warnings.append(f"{sketch.name}: {exc}")
            # No answer means no assumptions read, not assumptions absent.
            detail = ComponentDetail(assumptions=RawAssumptions())

        section = (doc.find(*sketch.name.split()[:3]) or [None])[0]
        section_title = section.title if section else "Model"

        components.append(
            Component(
                name=sketch.name,
                role=sketch.role,
                depends_on=sketch.depends_on,
                assumptions=Assumptions(**detail.assumptions.model_dump()),
                inputs=[TensorSpec(**t.model_dump()) for t in detail.inputs],
                outputs=[TensorSpec(**t.model_dump()) for t in detail.outputs],
                equations=[
                    Equation(
                        latex=e.latex,
                        description=e.description,
                        provenance=_provenance(tex, section_title, e.quote),
                        implements=sketch.name,
                    )
                    for e in detail.equations
                ],
                hyperparameters=_convert_hyperparameters(
                    detail.hyperparameters, tex, section_title, report
                ),
                provenance=Provenance(
                    section=section_title,
                    quote=sketch.role[:600],
                    char_start=section.start if section else 0,
                    char_end=section.end if section else 0,
                )
                if section
                else None,
            )
        )

    training = None
    if with_training:
        try:
            if raw := _training(doc, config, report):
                training = TrainingSpec(
                    objective=raw.objective,
                    optimizer=raw.optimizer,
                    schedule=raw.schedule,
                    datasets=raw.datasets,
                    hardware=raw.hardware,
                    hyperparameters=_convert_hyperparameters(
                        raw.hyperparameters, tex, "Training", report
                    ),
                )
        except LLMError as exc:
            report.warnings.append(f"training: {exc}")

    unknowns: list[Unknown] = []
    if with_unknowns:
        try:
            found = _unknowns(doc, components, training, config, report)
            searched = [s.title for s in (doc.methodology() or [])[:5]]
            for u in found.items:
                # Each candidate is challenged against the whole paper before
                # it is published as a gap. Saying a paper omits something it
                # states plainly is the worst output this tool can produce.
                refuted = _refute(doc, u, config, report)
                if refuted is not None:
                    answer, provenance = refuted
                    report.refuted_gaps.append(
                        f"{u.question[:70]} - answered in "
                        f"{provenance.section!r}: {answer[:70]}"
                    )
                    continue
                unknowns.append(
                    Unknown(
                        question=u.question,
                        why_it_matters=u.why_it_matters,
                        severity=u.severity,
                        conventional_default=u.conventional_default,
                        applies_to=u.applies_to,
                        searched=searched,
                    )
                )
        except LLMError as exc:
            report.gaps_read = False
            report.warnings.append(f"unknowns: {exc}")

    # A depends_on naming something that was never extracted is an artifact of
    # this run, not a property of the paper. The graph walk drops it silently,
    # so it has to be recorded here or it goes unnoticed - "Residual Connection
    # depends on Sublayer" looked fine until someone asked why the residual
    # came back without the layer normalisation it is defined with.
    known = {component.name.lower() for component in components}
    for component in components:
        for dependency in component.depends_on:
            if dependency.lower() not in known:
                report.warnings.append(
                    f"{component.name}: depends on {dependency!r}, which was not "
                    "extracted as a component"
                )

    spec = ImplementationSpec(
        arxiv_id=doc.meta.arxiv_id,
        title=doc.meta.title,
        extractor=Extractor(model=config.identity, arxivbot_version=__version__),
        summary=inventory.summary,
        official_repo=inventory.official_repo,
        components=components,
        training=training,
        unknowns=unknowns,
    )
    return spec, report
