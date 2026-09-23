"""Turning a selected part of a spec into a code skeleton.

The division of labour here is the point. The honest parts of the output are
built from the spec by this module and cannot be argued away by a model: the
deviation banner, the citations, and the TODO for every unknown. The model is
asked only to write the code body around them.

A model told "mention the gaps" will sometimes decide the code looks tidier
without them. A header assembled from the spec has no opinion.
"""

from __future__ import annotations

from arxivbot.deviation import Deviation, Stance
from arxivbot.llm import LLMConfig, stream
from arxivbot.select import Selection
from arxivbot.spec import Component, Confidence, ImplementationSpec, Severity

SYSTEM = """You write implementation skeletons for machine learning papers.

You are given what a paper states about a component, and what it leaves
unsaid. Write PyTorch unless told otherwise.

Rules:
- Implement only what the spec supports. Where the paper is silent, leave a
  `# TODO(paper-silent):` comment naming the decision - never quietly pick a
  value and move on.
- Cite the paper in comments for shapes, constants and equations, using the
  section given in the spec.
- Annotate tensor shapes on every forward pass, using the paper's own
  dimension names.
- Prefer clear, plain code over clever code. No training loop unless asked.
- Output only code in a single ```python block. No prose before or after."""


def _shape(names: list[str]) -> str:
    return " x ".join(names) if names else "?"


def _describe(component: Component) -> str:
    lines = [f"COMPONENT: {component.name}", f"  role: {component.role}"]
    if component.depends_on:
        lines.append(f"  built from: {', '.join(component.depends_on)}")
    for tensor in component.inputs:
        lines.append(f"  input  {tensor.name}: {_shape(tensor.shape)}")
    for tensor in component.outputs:
        lines.append(f"  output {tensor.name}: {_shape(tensor.shape)}")
    for equation in component.equations:
        lines.append(f"  equation: {equation.latex}")
        if equation.description:
            lines.append(f"    ({equation.description})")
    for hp in component.hyperparameters:
        mark = "stated" if hp.confidence is Confidence.STATED else hp.confidence.value
        where = f", {hp.provenance.section}" if hp.provenance else ""
        lines.append(f"  {hp.name} = {hp.value}  [{mark}{where}]")
    return "\n".join(lines)


def header(
    spec: ImplementationSpec, selection: Selection, deviations: list[Deviation]
) -> str:
    """The part of the output that is derived, not generated."""
    lines: list[str] = []

    conflicts = [d for d in deviations if d.stance is Stance.DEVIATES]
    for deviation in conflicts:
        lines.append(f"# !! DEVIATION - {deviation.axis}")
        lines.append(f"#    you asked for : {deviation.requested}")
        source = f" ({deviation.source})" if deviation.source else ""
        lines.append(f"#    paper states  : {deviation.paper_says}{source}")
        lines.append("#    This code implements your choice, not the paper's.")
        lines.append("#")

    for deviation in deviations:
        if deviation.stance is Stance.RESOLVES_UNKNOWN:
            lines.append(
                f"# {deviation.axis}: {deviation.requested} - your choice; "
                "the paper does not specify this."
            )

    lines.append(f"# {spec.title}")
    lines.append(f"# arXiv:{spec.arxiv_id}")
    lines.append(f"# Components: {', '.join(selection.names())}")

    cited = [
        hp
        for component in selection.components
        for hp in component.hyperparameters
        if hp.confidence is Confidence.STATED and hp.provenance
    ]
    if cited:
        lines.append("#")
        lines.append("# Values taken from the paper:")
        for hp in cited[:8]:
            lines.append(f"#   {hp.name} = {hp.value}  ({hp.provenance.section})")

    blocking = [u for u in selection.unknowns if u.severity is Severity.BLOCKING]
    other = [u for u in selection.unknowns if u.severity is not Severity.BLOCKING]
    if selection.unknowns:
        lines.append("#")
        lines.append("# The paper does NOT specify the following. Each is your decision:")
        for unknown in blocking + other:
            lines.append(f"#   [{unknown.severity.value}] {unknown.question}")
            if unknown.conventional_default:
                lines.append(f"#       commonly: {unknown.conventional_default}")
    return "\n".join(lines)


def prompt(
    spec: ImplementationSpec,
    selection: Selection,
    deviations: list[Deviation],
    request: str,
) -> str:
    parts = [
        f"PAPER: {spec.title} (arXiv:{spec.arxiv_id})",
        f"REQUEST: {request}",
        "",
        "WHAT THE PAPER STATES:",
    ]
    parts.extend(_describe(c) for c in selection.components)

    if selection.unknowns:
        parts.append("")
        parts.append("WHAT THE PAPER NEVER STATES - leave a TODO for each, do not guess:")
        for unknown in selection.unknowns:
            parts.append(f"  [{unknown.severity.value}] {unknown.question}")
            if unknown.conventional_default:
                parts.append(f"      commonly: {unknown.conventional_default}")

    conflicts = [d for d in deviations if d.stance is Stance.DEVIATES]
    if conflicts:
        parts.append("")
        parts.append("THE REQUEST DEPARTS FROM THE PAPER - implement the request:")
        for deviation in conflicts:
            parts.append(
                f"  {deviation.axis}: use {deviation.requested}; "
                f"the paper uses {deviation.paper_says}"
            )

    parts.append("")
    parts.append("Write the skeleton now.")
    return "\n".join(parts)


def generate(
    spec: ImplementationSpec,
    selection: Selection,
    deviations: list[Deviation],
    request: str,
    *,
    config: LLMConfig | None = None,
):
    """Yield the skeleton: the derived header first, then generated code."""
    yield header(spec, selection, deviations) + "\n\n"
    body = prompt(spec, selection, deviations, request)
    for chunk in stream(body, system=SYSTEM, config=config):
        yield chunk
