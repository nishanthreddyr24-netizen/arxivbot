"""Extract and cache full specs, so merge experiments do not re-pay for them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arxivbot.extract import extract  # noqa: E402
from arxivbot.ingest import build_document, load  # noqa: E402
from arxivbot.llm import LLMConfig  # noqa: E402
from arxivbot.spec import ImplementationSpec  # noqa: E402

OUT = ROOT / "eval" / "specs"


def spec_for(arxiv_id: str, *, refresh: bool = False) -> ImplementationSpec:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{arxiv_id}.json"
    if path.exists() and not refresh:
        return ImplementationSpec.model_validate_json(path.read_text(encoding="utf-8"))

    config = LLMConfig.from_env()
    config.timeout = 300
    meta, raw = load(arxiv_id)
    spec, report = extract(build_document(meta, raw), config=config, max_components=12)
    path.write_text(spec.model_dump_json(indent=1), encoding="utf-8")
    print(
        f"  {arxiv_id}: {len(spec.components)} components, "
        f"{len(spec.unknowns)} gaps, quotes {report.quote_accuracy or 0:.0%}",
        flush=True,
    )
    return spec


def main() -> None:
    for arxiv_id in sys.argv[1:]:
        print(arxiv_id, flush=True)
        spec_for(arxiv_id)


if __name__ == "__main__":
    main()
