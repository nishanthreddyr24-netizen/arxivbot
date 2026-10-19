"""Turn the evaluation results into the page the site serves.

The numbers that justify this project live in eval/results.json and
eval/gap_audit.md, which no visitor will ever read. This writes them into
site/lib/evidence.json so the site can show them.

Generated, never hand-edited: a page claiming to be evidence must not be
able to drift from the run it describes. Re-run it after any corpus run.

    python eval/publish.py
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results.json"
OUT = ROOT / "site" / "lib" / "evidence.json"

# The hand audit of twenty gaps. Counts come from eval/gap_audit.md, which
# records the per-gap verdicts and the reasoning behind each one.
AUDIT = {
    "sampled": 20,
    "true": 11,
    "false": 7,
    "partial": 2,
    "note": (
        "Twenty gaps, sampled at most two per paper, judged by hand against "
        "the LaTeX source. Seven asserted the paper omits something it states "
        "plainly."
    ),
}

# Measured after the refutation pass was added, on the same audited gaps.
AFTER = {
    "retrieval_hit": [6, 7],
    "end_to_end": [13, 16],
    "real_kept": [9, 9],
    "false_dropped": [4, 7],
    "note": (
        "A candidate gap is now challenged against the whole paper and "
        "withdrawn if the answer is found. Errors fall the safe way: a gap "
        "wrongly kept changes nothing, a gap wrongly dropped would hide a "
        "real hole."
    ),
}


def mean(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else 0.0


def commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    breadth = data["breadth"]
    ok = [e for e in breadth.values() if e.get("ok")]
    refused = [e for e in breadth.values() if not e.get("ok")]

    by_area: dict[str, list[dict]] = defaultdict(list)
    for entry in ok:
        by_area[entry["area"]].append(entry)

    areas = [
        {
            "area": area,
            "papers": len(entries),
            "components": round(mean([e.get("n_components") for e in entries]), 1),
            "gaps": round(mean([e.get("n_gaps") for e in entries]), 1),
            "quotes": round(mean([e.get("quote_accuracy") for e in entries]), 3),
        }
        for area, entries in sorted(by_area.items())
    ]

    with_algo = [e for e in ok if e.get("algorithm_blocks", 0) > 0]
    without = [e for e in ok if not e.get("algorithm_blocks", 0)]

    evidence = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "commit": commit(),
        "corpus": {
            "papers": len(breadth),
            "extracted": len(ok),
            "refused": len(refused),
            "components": round(mean([e.get("n_components") for e in ok]), 1),
            "gaps": round(mean([e.get("n_gaps") for e in ok]), 1),
            "quotes": round(mean([e.get("quote_accuracy") for e in ok]), 3),
        },
        "areas": areas,
        # Papers that ship a PDF wrapped in LaTeX. Refusing them is the
        # result; they used to return "0 components" and look like an answer.
        "refused": [
            {"id": e.get("id", ""), "short": e.get("short", ""), "why": e.get("error", "")}
            for e in refused
        ],
        "algorithm_blocks": {
            "with": {
                "papers": len(with_algo),
                "components": round(mean([e.get("n_components") for e in with_algo]), 1),
                "quotes": round(mean([e.get("quote_accuracy") for e in with_algo]), 3),
            },
            "without": {
                "papers": len(without),
                "components": round(mean([e.get("n_components") for e in without]), 1),
                "quotes": round(mean([e.get("quote_accuracy") for e in without]), 3),
            },
        },
        "audit": AUDIT,
        "after": AFTER,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT.relative_to(ROOT)}: {len(ok)} extracted, {len(refused)} refused, "
        f"quotes {evidence['corpus']['quotes']:.0%}"
    )


if __name__ == "__main__":
    main()
