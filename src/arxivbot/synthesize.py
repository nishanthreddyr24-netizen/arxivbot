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
- Implement only what the spec supports.

- `# TODO(paper-silent):` is reserved. Use it ONLY for the gaps listed under
  WHAT THE PAPER NEVER STATES, one marker per listed gap, and invent no
  others. These markers are the reader's list of decisions they must make,
  and a marker on something the paper does specify makes the whole list
  untrustworthy. If you are unsure about anything not on that list, write an
  ordinary comment - never a TODO.

- Cite sections by the titles given in the spec, copied exactly. Do not write
  section numbers: you do not know them, and a wrong one is worse than none.

- You are given only part of the paper. Components listed as NOT INCLUDED are
  real and were simply not requested - refer to one by name where the code
  needs it and move on. Never describe an absent component as unspecified,
  ambiguous or missing from the paper.

- Annotate tensor shapes on every forward pass, using the paper's own
  dimension names. A shape annotation is not a decision - do not mark it.

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

    # Name what was left out. Given one component of a pair the paper defines
    # together, a model with no way to know the other exists will describe the
    # hole as an ambiguity in the paper - inventing a gap where there is none.
    chosen = {c.name for c in selection.components}
    omitted = [c.name for c in spec.components if c.name not in chosen]
    if omitted:
        parts.append("")
        parts.append(
            "COMPONENTS NOT INCLUDED - these exist in the paper and were simply "
            "not requested. Refer to one by name if the code needs it. Do not "
            "call any of them unspecified:"
        )
        parts.extend(f"  {name}" for name in omitted)

    parts.append("")
    if selection.unknowns:
        parts.append(
            "WHAT THE PAPER NEVER STATES - write exactly one "
            "`# TODO(paper-silent):` for each of these, and none besides:"
        )
        for unknown in selection.unknowns:
            parts.append(f"  [{unknown.severity.value}] {unknown.question}")
            if unknown.conventional_default:
                parts.append(f"      commonly: {unknown.conventional_default}")
    else:
        parts.append(
            "WHAT THE PAPER NEVER STATES: nothing recorded for these "
            "components. Write no TODO(paper-silent) markers at all."
        )

    conflicts = [d for d in deviations if d.stance is Stance.DEVIATES]
    if conflicts:
        parts.append("")
        parts.append("THE REQUEST DEPARTS FROM THE PAPER - implement the request:")
        for deviation in conflicts:
            parts.append(
                f"  {deviation.axis}: use {deviation.requested}; "
                f"the paper uses {deviation.paper_says}"
            )

    # The sections it is allowed to name. Left to itself the model writes
    # section numbers from memory of the paper's layout, and gets them wrong:
    # it put the residual formula in "Section 5.4" when the paper states it
    # under Encoder and Decoder Stacks.
    sections = sorted(
        {
            hp.provenance.section
            for component in selection.components
            for hp in component.hyperparameters
            if hp.provenance
        }
        | {
            eq.provenance.section
            for component in selection.components
            for eq in component.equations
            if eq.provenance
        }
        | {
            component.provenance.section
            for component in selection.components
            if component.provenance
        }
    )
    if sections:
        parts.append("")
        parts.append("SECTIONS YOU MAY CITE, copied exactly, without numbers:")
        parts.extend(f"  {title}" for title in sections)

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
