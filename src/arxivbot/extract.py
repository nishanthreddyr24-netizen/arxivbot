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

import re
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import BaseModel, Field

from arxivbot import __version__
from arxivbot.ingest.latex import Document, Section
from arxivbot.llm import LLMConfig, LLMError, complete_json
from arxivbot.spec import (
    Component,
    Confidence,
    Equation,
    Extractor,
    Hyperparameter,
    ImplementationSpec,
    Provenance,
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
    applies_to: str | None = None


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


class ComponentDetail(BaseModel):
    inputs: list[RawTensor] = Field(default_factory=list)
    outputs: list[RawTensor] = Field(default_factory=list)
    equations: list[RawEquation] = Field(default_factory=list)
    hyperparameters: list[RawHyperparameter] = Field(default_factory=list)


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


def _inventory(doc: Document, config: LLMConfig, report: Report) -> Inventory:
    sections = doc.methodology() or doc.sections[:6]
    prompt = (
        f"Paper: {doc.meta.title}\n\n"
        "List the components of the proposed method - the pieces someone would "
        "implement as separate units. Use the paper's own names. Set depends_on "
        "where one component is built from another.\n\n"
        f"{_text_of(sections, doc)}"
    )
    report.calls += 1
    return complete_json(prompt, Inventory, system=SYSTEM, config=config)


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
) -> tuple[ImplementationSpec, Report]:
    """Read a paper and build a spec from it.

    Returns the spec and a :class:`Report` describing how the extraction went,
    including how many of the model's quotes were actually found in the paper.
    """
    config = config or LLMConfig.from_env()
    report = Report()
    tex = doc.tex

    inventory = _inventory(doc, config, report)

    components: list[Component] = []
    for sketch in inventory.components[:max_components]:
        try:
            detail = _detail(sketch, doc, config, report)
        except LLMError as exc:
            report.warnings.append(f"{sketch.name}: {exc}")
            detail = ComponentDetail()

        section = (doc.find(*sketch.name.split()[:3]) or [None])[0]
        section_title = section.title if section else "Model"

        components.append(
            Component(
                name=sketch.name,
                role=sketch.role,
                depends_on=sketch.depends_on,
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
            unknowns = [
                Unknown(
                    question=u.question,
                    why_it_matters=u.why_it_matters,
                    severity=u.severity,
                    conventional_default=u.conventional_default,
                    applies_to=u.applies_to,
                    searched=[s.title for s in (doc.methodology() or [])[:5]],
                )
                for u in found.items
            ]
        except LLMError as exc:
            report.warnings.append(f"unknowns: {exc}")

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
