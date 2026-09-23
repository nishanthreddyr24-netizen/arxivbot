"""Quote verification is the hallucination check, so it gets tested hard.

Nothing here calls a model. These cover the deterministic half of extraction:
finding a claimed quote in the source, and demoting claims when it is absent.
"""

from __future__ import annotations

import pytest

from arxivbot.extract import Report, _convert_hyperparameters, locate
from arxivbot.extract import RawHyperparameter
from arxivbot.spec import Confidence

SOURCE = r"""
\section{Model Architecture}
In this work we employ $h=8$ parallel attention layers, or heads. For each of
these we use $d_k=d_v=d_{\text{model}}/h=64$. We apply dropout
\citep{JMLR:v15:srivastava14a} to the output of each sub-layer, before it is
added to the sub-layer input. The inner-layer has dimensionality $d_{ff}=2048$.
"""


class TestFindingRealQuotes:
    """A quote the model copied correctly must be found, or correct claims
    get demoted and the output looks less trustworthy than it is."""

    @pytest.mark.parametrize(
        "quote",
        [
            "we employ h = 8 parallel attention layers",
            "we employ $h=8$ parallel attention layers",  # verbatim LaTeX
            "We apply dropout to the output of each sub-layer",  # citation elided
            "the inner-layer has dimensionality d_ff = 2048",
            "WE EMPLOY H = 8 PARALLEL ATTENTION LAYERS",  # case differs
        ],
    )
    def test_quote_is_located(self, quote):
        assert locate(SOURCE, quote) is not None

    def test_span_points_at_the_real_text(self):
        start, end = locate(SOURCE, "the inner-layer has dimensionality d_ff = 2048")
        assert "inner-layer" in SOURCE[start:end]
        assert "2048" in SOURCE[start:end]

    def test_line_wrapped_quote_is_located(self):
        # LaTeX wraps wherever it likes; the model quotes one flowing sentence.
        assert locate(SOURCE, "before it is added to the sub-layer input") is not None


class TestRejectingInventedQuotes:
    @pytest.mark.parametrize(
        "quote",
        [
            "we used a batch size of 9999 and a magic constant",
            "weights are initialised with orthogonal initialisation",
            "We apply dropout to the input of each sub-layer",  # output -> input
            "we employ h = 16 parallel attention layers",  # 8 -> 16
        ],
    )
    def test_quote_is_rejected(self, quote):
        assert locate(SOURCE, quote) is None

    @pytest.mark.parametrize("quote", ["", "   ", None, "the"])
    def test_empty_or_trivial_quotes_are_rejected(self, quote):
        assert locate(SOURCE, quote) is None

    def test_short_quotes_are_rejected(self):
        # Without whitespace, a few characters would match almost anything.
        assert locate(SOURCE, "h=8") is None


class TestDemotion:
    def test_unverifiable_stated_claim_is_demoted(self):
        report = Report()
        raw = [
            RawHyperparameter(
                name="warmup_steps",
                value="4000",
                confidence=Confidence.STATED,
                quote="we warmed up over the first 4000 steps of training",
            )
        ]
        out = _convert_hyperparameters(raw, SOURCE, "Training", report)
        # The quote is not in SOURCE, so the claim about the paper is dropped.
        assert out[0].confidence is Confidence.GUESS
        assert out[0].value == "4000"  # the value itself is kept
        assert "warmup_steps" in report.demoted
        assert report.rejected_quotes == 1

    def test_verifiable_stated_claim_survives(self):
        report = Report()
        raw = [
            RawHyperparameter(
                name="h",
                value="8",
                confidence=Confidence.STATED,
                quote="we employ h = 8 parallel attention layers",
            )
        ]
        out = _convert_hyperparameters(raw, SOURCE, "Model Architecture", report)
        assert out[0].confidence is Confidence.STATED
        assert out[0].provenance is not None
        assert report.verified_quotes == 1
        assert report.demoted == []

    def test_conventional_claim_needs_no_quote(self):
        report = Report()
        raw = [
            RawHyperparameter(
                name="init", value="xavier", confidence=Confidence.CONVENTIONAL
            )
        ]
        out = _convert_hyperparameters(raw, SOURCE, "Model", report)
        assert out[0].confidence is Confidence.CONVENTIONAL
        assert report.rejected_quotes == 0

    def test_accuracy_is_reported(self):
        report = Report()
        _convert_hyperparameters(
            [
                RawHyperparameter(
                    name="good",
                    value="8",
                    confidence=Confidence.STATED,
                    quote="we employ h = 8 parallel attention layers",
                ),
                RawHyperparameter(
                    name="bad",
                    value="1",
                    confidence=Confidence.STATED,
                    quote="a sentence that is absolutely not in the paper at all",
                ),
            ],
            SOURCE,
            "Model",
            report,
        )
        assert report.quote_accuracy == 0.5

    def test_accuracy_is_none_when_nothing_was_checked(self):
        assert Report().quote_accuracy is None
