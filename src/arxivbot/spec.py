"""The ``ImplementationSpec`` - what a paper actually says, and what it does not.

This is the project's central artifact. Code generation is a rendering of a
spec; the spec itself is the thing worth storing, sharing and correcting.

Two rules shape every model here:

**Nothing is asserted without provenance.** Every extracted claim carries the
span of LaTeX it came from, so a reader can check it. A claim with no source
is a bug, not a convenience.

**Silence is recorded, not filled.** Papers are always underspecified. Where a
paper does not say, the spec says *that it does not say* - see :class:`Unknown`.
An extractor that quietly invents a plausible value defeats the entire purpose.

These are pydantic models rather than dataclasses because they do double duty:
they validate what a model returns, and their JSON Schema constrains what it is
allowed to return in the first place.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Bump when a change would make previously extracted specs wrong or unreadable.
# Stored specs record the version they were built under so a shared index can
# tell a current spec from one that needs regenerating.
SCHEMA_VERSION = "0.1"


class Confidence(str, Enum):
    """How much the paper itself backs a claim."""

    STATED = "stated"
    """Written in the paper. Provenance is required."""

    DERIVED = "derived"
    """Not written, but forced by something that is - e.g. a shape implied by
    two stated dimensions. Provenance points at what it was derived from."""

    CONVENTIONAL = "conventional"
    """The paper is silent and this is standard practice in the surrounding
    literature. A defensible default, but still a guess - it belongs in
    :class:`Unknown` too."""

    GUESS = "guess"
    """Neither stated nor conventional. Should be rare; prefer an Unknown."""


class Severity(str, Enum):
    """How badly a gap hurts someone trying to reimplement the paper."""

    BLOCKING = "blocking"
    """Cannot proceed without choosing - e.g. the core update rule is absent."""

    SIGNIFICANT = "significant"
    """Can proceed, but the choice plausibly changes reported results."""

    MINOR = "minor"
    """Unlikely to change results materially."""


class Provenance(BaseModel):
    """Where in the source a claim came from."""

    section: str = Field(description="Title of the section containing the claim")
    quote: str = Field(
        max_length=600,
        description="The supporting text, verbatim from the paper",
    )
    char_start: int = Field(ge=0, description="Offset into the flattened LaTeX")
    char_end: int = Field(ge=0)
    label: str | None = Field(
        default=None, description=r"The \label of the section or float, if any"
    )

    @field_validator("char_end")
    @classmethod
    def _end_after_start(cls, end: int, info):
        start = info.data.get("char_start")
        if start is not None and end < start:
            raise ValueError("char_end must not precede char_start")
        return end


class TensorSpec(BaseModel):
    """A tensor flowing in or out of a component.

    Shapes are symbolic, not numeric: ``["batch", "seq_len", "d_model"]``.
    Papers describe architectures in terms of named dimensions, and resolving
    those to integers is the user's job, not the paper's.
    """

    name: str
    shape: list[str] = Field(
        default_factory=list,
        description="Symbolic dimensions, outermost first",
    )
    dtype: str | None = None
    description: str | None = None


class Equation(BaseModel):
    """A numbered or displayed equation that the implementation must realise."""

    latex: str = Field(description="The equation as written, in LaTeX")
    description: str = Field(description="What it computes, in one sentence")
    provenance: Provenance | None = None
    implements: str | None = Field(
        default=None,
        description="Name of the component this equation defines, if any",
    )


class Hyperparameter(BaseModel):
    """A single configurable value.

    ``value`` is a string on purpose - papers write "3e-4", "1e-5 to 1e-3",
    "4 for base / 8 for large". Coercing that to a float loses information the
    implementer needs.
    """

    name: str
    value: str | None = Field(
        default=None, description="As written in the paper; None if unspecified"
    )
    unit: str | None = None
    applies_to: str | None = Field(
        default=None, description="Component or phase this applies to"
    )
    confidence: Confidence
    provenance: Provenance | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _stated_needs_source(self) -> Hyperparameter:
        # Must be a model validator, not a field validator: a field validator
        # on `provenance` does not run when the field is simply omitted, which
        # is exactly the case this rule exists to catch.
        if self.confidence is Confidence.STATED and self.provenance is None:
            raise ValueError(
                f"hyperparameter {self.name!r} is marked STATED but cites no source"
            )
        return self


class Component(BaseModel):
    """One named piece of the architecture, at the granularity a reader would
    reimplement as a unit - an attention block, a noise schedule, a loss."""

    name: str = Field(description="As the paper names it, e.g. 'Multi-Head Attention'")
    role: str = Field(description="What it does, in one sentence")
    inputs: list[TensorSpec] = Field(default_factory=list)
    outputs: list[TensorSpec] = Field(default_factory=list)
    equations: list[Equation] = Field(default_factory=list)
    hyperparameters: list[Hyperparameter] = Field(default_factory=list)
    depends_on: list[str] = Field(
        default_factory=list,
        description="Names of other components this one composes",
    )
    provenance: Provenance | None = None


class TrainingSpec(BaseModel):
    """How the thing was trained. Usually the most underspecified part."""

    objective: str | None = Field(default=None, description="The loss being minimised")
    optimizer: str | None = None
    schedule: str | None = Field(default=None, description="Learning rate schedule")
    hyperparameters: list[Hyperparameter] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    hardware: str | None = None
    equations: list[Equation] = Field(default_factory=list)


class Unknown(BaseModel):
    """Something the paper does not specify.

    The point of the project. A reimplementer must choose these; the paper
    leaves them open; pretending otherwise produces code that looks right and
    does not reproduce.
    """

    question: str = Field(
        description="The open question, e.g. 'How are attention weights initialised?'"
    )
    why_it_matters: str = Field(
        description="What a wrong choice here would do to results"
    )
    severity: Severity
    searched: list[str] = Field(
        default_factory=list,
        description="Sections checked before concluding the paper is silent",
    )
    conventional_default: str | None = Field(
        default=None,
        description="What the surrounding literature usually does, if anything",
    )
    applies_to: str | None = Field(default=None, description="Component or phase")


class Extractor(BaseModel):
    """Who produced this spec, so a consumer can judge and regenerate it."""

    schema_version: str = SCHEMA_VERSION
    model: str = Field(description="Model id used, e.g. 'gemini/gemini-2.5-flash'")
    arxivbot_version: str
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    notes: str | None = None


class ImplementationSpec(BaseModel):
    """Everything recovered from one paper, plus everything it left open."""

    arxiv_id: str = Field(description="Version-pinned, e.g. '1706.03762v7'")
    title: str
    extractor: Extractor

    summary: str = Field(
        default="", description="What the paper proposes, in a few sentences"
    )
    components: list[Component] = Field(default_factory=list)
    hyperparameters: list[Hyperparameter] = Field(
        default_factory=list, description="Global values not owned by a component"
    )
    training: TrainingSpec | None = None
    unknowns: list[Unknown] = Field(default_factory=list)

    official_repo: str | None = Field(
        default=None, description="Implementation URL found in the paper, if any"
    )

    # ---- convenience ----

    @property
    def blocking_unknowns(self) -> list[Unknown]:
        return [u for u in self.unknowns if u.severity is Severity.BLOCKING]

    def component(self, name: str) -> Component | None:
        """Case-insensitive lookup by component name."""
        target = name.lower()
        for comp in self.components:
            if comp.name.lower() == target:
                return comp
        return None

    def all_hyperparameters(self) -> list[Hyperparameter]:
        """Global, per-component and training hyperparameters together."""
        out = list(self.hyperparameters)
        for comp in self.components:
            out.extend(comp.hyperparameters)
        if self.training:
            out.extend(self.training.hyperparameters)
        return out

    def confidence_breakdown(self) -> dict[str, int]:
        """How much of this spec the paper actually backs."""
        counts = {c.value: 0 for c in Confidence}
        for hp in self.all_hyperparameters():
            counts[hp.confidence.value] += 1
        return counts

    # ---- serialisation ----

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=True)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ImplementationSpec:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def json_schema(cls) -> str:
        """The JSON Schema an extractor is constrained to produce."""
        return json.dumps(cls.model_json_schema(), indent=2)


__all__ = [
    "SCHEMA_VERSION",
    "Component",
    "Confidence",
    "Equation",
    "Extractor",
    "Hyperparameter",
    "ImplementationSpec",
    "Provenance",
    "Severity",
    "TensorSpec",
    "TrainingSpec",
    "Unknown",
]
