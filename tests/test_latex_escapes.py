"""LaTeX inside JSON, where the backslash means something else.

A model asked for structured output writes `"$\\beta_t$"` as `"$\beta_t$"`,
which is valid JSON for a backspace followed by "eta_t". Nothing raises. The
spec then carries a control character where a Greek letter should be, the
equation never renders, and - worse - the quote that was supposed to be
verbatim can no longer be found in the paper, so a real claim is reported as
one the authors never made.
"""

from __future__ import annotations

import json

import pytest

from arxivbot.llm import _repair, repair_latex_escapes


@pytest.mark.parametrize(
    "escaped,expected",
    [
        ("$\x08eta_t$", r"$\beta_t$"),
        ("\x0crac{1}{2}", r"\frac{1}{2}"),
        ("\tilde\x08mu_t", r"\tilde\bmu_t"),
        ("\theta", r"\theta"),
        ("\rho_t", r"\rho_t"),
    ],
)
def test_the_eaten_backslash_comes_back(escaped, expected):
    assert repair_latex_escapes(escaped) == expected


def test_real_whitespace_survives():
    """Tab and newline are only LaTeX when a letter follows."""
    assert repair_latex_escapes("a\t b") == "a\t b"
    assert repair_latex_escapes("one\ntwo") == "one\ntwo"
    assert repair_latex_escapes("col\t\tcol") == "col\t\tcol"


def test_a_sentence_with_no_latex_is_untouched():
    plain = "The encoder is composed of a stack of N=6 identical layers."
    assert repair_latex_escapes(plain) == plain


def test_the_whole_reply_is_repaired_not_just_the_top_level():
    """The damage is wherever the model put LaTeX, which is deep in the tree."""
    reply = json.loads(r'{"components": [{"equations": [{"latex": "$\beta_t$"}]}]}')
    assert _repair(reply)["components"][0]["equations"][0]["latex"] == r"$\beta_t$"


def test_the_round_trip_that_actually_bit():
    """Exactly what the provider sent back for the diffusion paper."""
    sent = r'{"role": "variances $\beta_t$ and \frac{1}{2}"}'
    assert _repair(json.loads(sent))["role"] == r"variances $\beta_t$ and \frac{1}{2}"
