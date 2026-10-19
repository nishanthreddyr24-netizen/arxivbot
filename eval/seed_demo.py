"""Warm the caches a read-only deployment will serve from.

A demo in ARXIVBOT_DEMO mode answers only from the spec store and the model
cache, so whatever is not seeded here does not exist to a visitor. Run this
once, with a working key and no demo flag, before deploying.

    python eval/seed_demo.py           # every pair below
    python eval/seed_demo.py 1706.03762

It is resumable and cheap to repeat: a pair already in the cache costs
nothing, so a run interrupted by a quota limit can simply be run again.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arxivbot import store  # noqa: E402
from arxivbot.deviation import check as deviation_check  # noqa: E402
from arxivbot.llm import LLMConfig, LLMError  # noqa: E402
from arxivbot.select import select  # noqa: E402
from arxivbot.synthesize import generate  # noqa: E402

# Chosen to show the product rather than to flatter it: each paper has a
# component whose evidence is strong, and the skeletons carry real
# TODO(paper-silent) lines.
DEMO: dict[str, list[str]] = {
    "1706.03762": [
        "multi-head attention",
        "residual connection",
        "positional encoding",
        "layer normalization",
    ],
    "1512.03385": ["residual block", "shortcut connection"],
    "2010.11929": ["patch embedding", "classification token"],
    "1609.02907": ["graph convolution", "propagation rule"],
}


def warm(arxiv_id: str, query: str, config: LLMConfig) -> str:
    hit = store.get(arxiv_id)
    if hit is None:
        return "no spec - run: arxivbot spec " + arxiv_id

    selection = select(hit.spec, query)
    if selection.empty:
        return "no component matches"

    deviations = deviation_check(hit.spec, query, selection.components)
    try:
        produced = "".join(generate(hit.spec, selection, deviations, query, config=config))
    except LLMError as exc:
        return f"failed: {str(exc)[:70]}"
    return f"{len(produced)} chars"


def main() -> None:
    wanted = sys.argv[1:] or list(DEMO)
    config = LLMConfig.from_env()
    config.timeout = 300
    if not config.api_key:
        raise SystemExit("no API key: set GEMINI_API_KEY in .env")

    for arxiv_id in wanted:
        queries = DEMO.get(arxiv_id)
        if not queries:
            print(f"{arxiv_id}: not a demo paper")
            continue
        print(arxiv_id, flush=True)
        for query in queries:
            print(f"  {query:28} {warm(arxiv_id, query, config)}", flush=True)


if __name__ == "__main__":
    main()
