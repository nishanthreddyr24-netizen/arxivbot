"""Retrieval behind gap refutation.

A gap says the paper never specifies something. An audit of twenty found
seven that named things the paper states plainly, every one of them outside
the method section. These cover the retrieval that now challenges a gap
before it is published; the judgement itself needs a model and is measured
in the evaluation harness rather than here.
"""

from __future__ import annotations

from arxivbot.extract import _gap_terms, _passages
from arxivbot.ingest import build_document
from arxivbot.ingest.fetch import PaperMeta

# Comfortably past MIN_BODY_CHARS, so a fixture counts as a real paper.
FILLER = "We propose a model that is trained end to end. " * 40


def doc_with(*parts: str):
    meta = PaperMeta(arxiv_id="0000.00000", title="T", authors=[], abstract="")
    return build_document(meta, "\n".join(parts).encode("utf-8"))


def test_gap_terms_drop_question_scaffolding():
    terms = _gap_terms("What is the exact warmup schedule used?", None)
    assert "warmup" in terms
    assert "what" not in terms
    assert "exact" not in terms


def test_passages_reach_outside_the_method_section():
    """The answer to a gap is usually in the experiments section."""
    doc = doc_with(
        r"\section{Method}",
        FILLER,
        r"\section{Experiments}",
        "We use early stopping with a window of 10 epochs.",
    )
    terms = _gap_terms("What early stopping window is used?", None)
    assert any("window of 10 epochs" in text for _, text in _passages(doc, terms))


def test_a_term_that_is_everywhere_locates_nothing():
    doc = doc_with(r"\section{Method}", "attention " * 400)
    assert _passages(doc, ["attention"]) == []


def test_a_section_named_after_the_question_is_included():
    """Papers routinely title a section with the very question being asked."""
    doc = doc_with(
        r"\section{Method}",
        FILLER,
        r"\subsection{Which Weight Matrices Do We Adapt}",
        "We adapt the query and value projections only.",
    )
    terms = _gap_terms("Which weight matrices does the method adapt?", None)
    found = _passages(doc, terms)
    assert any("query and value projections" in text for _, text in found)


def test_the_bibliography_is_not_searched():
    """A title in the reference list is not the paper stating something."""
    doc = doc_with(
        r"\section{Method}",
        FILLER,
        r"\begin{thebibliography}{9}",
        r"\bibitem{x} A paper about cosine learning rate schedules. 2019.",
        r"\end{thebibliography}",
    )
    terms = _gap_terms("What cosine schedule is used?", None)
    assert all("cosine learning rate schedules" not in t for _, t in _passages(doc, terms))
