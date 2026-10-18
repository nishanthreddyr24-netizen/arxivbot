"""Phase 1: measure the pipeline across papers instead of across one paper.

Everything known about this pipeline's behaviour comes from the Transformer
paper, which is one sample and a famously well-written one. This runs a corpus
and records, per paper, what was found, how much of it was found again on a
second reading, and how much of what it claims the paper says is actually
there.

Two stages, because they cost very differently:

  breadth    one full extraction per paper. Gives components, gaps, the
             quote-verification rate and wall time.
  stability  the inventory pass only, run again. Inventory is the unstable
             step, and it is a third the cost of a full extraction.

The run is resumable. Every paper is checkpointed as it finishes, a free tier
will refuse requests long before thirty papers are done, and a night's progress
should not be lost to a quota that resets in the morning. Re-running skips
whatever is already recorded.

    python eval/run.py breadth      # one pass over every paper
    python eval/run.py stability    # extra inventory runs, for agreement
    python eval/run.py report       # CSV and summary from what exists
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arxivbot.extract import Report, _inventory_units, extract, units  # noqa: E402
from arxivbot.ingest import build_document, load  # noqa: E402
from arxivbot.ingest.fetch import FetchError  # noqa: E402
from arxivbot.ingest.latex import ALGORITHM_ENVS, UnpackError, extract_environments  # noqa: E402
from arxivbot.llm import LLMConfig, LLMError  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPERS = json.loads((HERE / "papers.json").read_text(encoding="utf-8"))
RESULTS = HERE / "results.json"

# A free tier refuses a burst long before it refuses a day's work, so the
# backoff is generous and the run keeps going rather than dying on one paper.
BACKOFF = [30, 90, 240]


def load_results() -> dict:
    if RESULTS.exists():
        try:
            return json.loads(RESULTS.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {"breadth": {}, "stability": {}}


def save(results: dict) -> None:
    RESULTS.write_text(json.dumps(results, indent=1), encoding="utf-8")


def fresh_config() -> LLMConfig:
    config = LLMConfig.from_env()
    config.timeout = 300
    return config


def quota_exhausted(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "quota" in text or "resource_exhausted" in text


def with_retries(work, label: str):
    """Run `work`, backing off when the provider refuses for rate reasons."""
    for attempt, pause in enumerate([*BACKOFF, None]):
        try:
            return work()
        except LLMError as exc:
            if not quota_exhausted(exc) or pause is None:
                raise
            print(f"    rate limited ({label}); waiting {pause}s", flush=True)
            time.sleep(pause)
    raise LLMError("exhausted retries")


def run_breadth(results: dict, limit: int | None) -> None:
    done = results["breadth"]
    todo = [p for p in PAPERS if p["id"] not in done][:limit]
    print(f"breadth: {len(done)} recorded, {len(todo)} to run\n")

    for index, paper in enumerate(todo, 1):
        started = time.time()
        print(f"[{index}/{len(todo)}] {paper['id']} {paper['short']}", flush=True)
        entry: dict = {**paper}

        try:
            meta, raw = load(paper["id"])
            doc = build_document(meta, raw)
        except (FetchError, UnpackError, ValueError) as exc:
            entry.update(ok=False, error=f"ingest: {exc}")
            done[paper["id"]] = entry
            save(results)
            print(f"    ingest failed: {exc}", flush=True)
            continue

        # Structural facts, free and deterministic - useful on their own for
        # asking whether papers with algorithm blocks extract better.
        entry.update(
            title=meta.title,
            tex_chars=len(doc.tex),
            sections=len(doc.sections),
            units=len(units(doc, doc.methodology() or doc.sections[:6])),
            algorithm_blocks=len(extract_environments(doc.tex, ALGORITHM_ENVS)),
        )

        try:
            spec, report = with_retries(
                lambda: extract(doc, config=fresh_config(), max_components=12),
                paper["id"],
            )
        except Exception as exc:  # noqa: BLE001 - a failed paper is a result
            entry.update(ok=False, error=f"extract: {type(exc).__name__}: {exc}")
            done[paper["id"]] = entry
            save(results)
            print(f"    extract failed: {exc}", flush=True)
            if quota_exhausted(exc):
                print("\nquota appears exhausted; stopping cleanly.", flush=True)
                return
            continue

        counts = spec.confidence_breakdown()
        entry.update(
            ok=True,
            seconds=round(time.time() - started, 1),
            calls=report.calls,
            components=[c.name for c in spec.components],
            n_components=len(spec.components),
            gaps=[
                {"question": u.question, "severity": u.severity.value}
                for u in spec.unknowns
            ],
            n_gaps=len(spec.unknowns),
            stated=counts["stated"],
            unverified=counts["guess"] + counts["conventional"],
            quote_accuracy=report.quote_accuracy,
            recovered=report.recovered,
            rejected=report.rejected,
            warnings=report.warnings[:6],
        )
        done[paper["id"]] = entry
        save(results)
        print(
            f"    {len(spec.components)} components, {len(spec.unknowns)} gaps, "
            f"quotes {report.quote_accuracy or 0:.0%}, {entry['seconds']}s",
            flush=True,
        )


def run_stability(results: dict, repeats: int, limit: int | None) -> None:
    """Extra inventory passes, to see how much of a reading survives a reread."""
    store = results["stability"]
    eligible = [p for p in PAPERS if results["breadth"].get(p["id"], {}).get("ok")]
    todo = [p for p in eligible if len(store.get(p["id"], [])) < repeats][:limit]
    print(f"stability: {len(todo)} paper(s) need more runs\n")

    for index, paper in enumerate(todo, 1):
        print(f"[{index}/{len(todo)}] {paper['id']} {paper['short']}", flush=True)
        try:
            meta, raw = load(paper["id"])
            doc = build_document(meta, raw)
        except Exception as exc:  # noqa: BLE001
            print(f"    skipped: {exc}", flush=True)
            continue

        runs = store.setdefault(paper["id"], [])
        while len(runs) < repeats:
            try:
                config = fresh_config()
                config.use_cache = False  # a cached reply is not a second reading
                found = with_retries(
                    lambda: _inventory_units(doc, config, Report()).components,
                    paper["id"],
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    run failed: {exc}", flush=True)
                if quota_exhausted(exc):
                    save(results)
                    print("\nquota appears exhausted; stopping cleanly.", flush=True)
                    return
                break
            runs.append(sorted({c.name.strip().lower() for c in found}))
            save(results)
            print(f"    run {len(runs)}: {len(runs[-1])} components", flush=True)


def report(results: dict) -> None:
    breadth = [e for e in results["breadth"].values() if e.get("ok")]
    failed = [e for e in results["breadth"].values() if not e.get("ok")]

    rows = ["id,short,area,tex_chars,units,algorithm_blocks,components,gaps,stated,unverified,quote_accuracy,calls,seconds,agreement"]
    agreements: dict[str, float] = {}

    for entry in breadth:
        runs = results["stability"].get(entry["id"], [])
        agreement = ""
        if len(runs) >= 2:
            seen: Counter[str] = Counter()
            for run in runs:
                seen.update(run)
            always = sum(1 for c in seen.values() if c == len(runs))
            agreement = f"{always / max(1, len(seen)):.2f}"
            agreements[entry["id"]] = always / max(1, len(seen))
        rows.append(
            ",".join(
                str(x)
                for x in [
                    entry["id"],
                    '"' + entry["short"] + '"',
                    entry["area"],
                    entry.get("tex_chars", ""),
                    entry.get("units", ""),
                    entry.get("algorithm_blocks", ""),
                    entry.get("n_components", ""),
                    entry.get("n_gaps", ""),
                    entry.get("stated", ""),
                    entry.get("unverified", ""),
                    f"{entry['quote_accuracy']:.2f}" if entry.get("quote_accuracy") else "",
                    entry.get("calls", ""),
                    entry.get("seconds", ""),
                    agreement,
                ]
            )
        )
    (HERE / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"\n{'=' * 66}\nPHASE 1 — {len(breadth)} papers extracted, {len(failed)} failed\n")

    if failed:
        print("failed:")
        for entry in failed:
            print(f"  {entry['id']:12} {entry['short']:18} {entry.get('error','')[:60]}")
        print()

    if not breadth:
        return

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else 0.0

    print(f"{'area':14}{'papers':>7}{'comps':>7}{'gaps':>6}{'quotes':>8}{'algo':>6}{'agree':>7}")
    by_area = defaultdict(list)
    for entry in breadth:
        by_area[entry["area"]].append(entry)
    for area, entries in sorted(by_area.items()):
        area_agree = [agreements[e["id"]] for e in entries if e["id"] in agreements]
        print(
            f"{area:14}{len(entries):>7}"
            f"{mean([e.get('n_components') for e in entries]):>7.1f}"
            f"{mean([e.get('n_gaps') for e in entries]):>6.1f}"
            f"{mean([e.get('quote_accuracy') for e in entries]):>8.0%}"
            f"{mean([e.get('algorithm_blocks') for e in entries]):>6.1f}"
            f"{(f'{mean(area_agree):.2f}' if area_agree else '-'):>7}"
        )

    print(
        f"\n{'overall':14}{len(breadth):>7}"
        f"{mean([e.get('n_components') for e in breadth]):>7.1f}"
        f"{mean([e.get('n_gaps') for e in breadth]):>6.1f}"
        f"{mean([e.get('quote_accuracy') for e in breadth]):>8.0%}"
        f"{mean([e.get('algorithm_blocks') for e in breadth]):>6.1f}"
        f"{(f'{mean(list(agreements.values())):.2f}' if agreements else '-'):>7}"
    )

    # Does having algorithm blocks help? The structural grounding leans on them.
    withalgo = [e for e in breadth if e.get("algorithm_blocks", 0) > 0]
    without = [e for e in breadth if not e.get("algorithm_blocks", 0)]
    if withalgo and without:
        print(
            f"\nwith algorithm blocks   n={len(withalgo):<3} "
            f"components {mean([e.get('n_components') for e in withalgo]):.1f}  "
            f"quotes {mean([e.get('quote_accuracy') for e in withalgo]):.0%}"
        )
        print(
            f"prose only              n={len(without):<3} "
            f"components {mean([e.get('n_components') for e in without]):.1f}  "
            f"quotes {mean([e.get('quote_accuracy') for e in without]):.0%}"
        )

    print(f"\nwritten to {HERE / 'results.csv'}")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    results = load_results()

    if command == "breadth":
        run_breadth(results, limit)
    elif command == "stability":
        run_stability(results, repeats=3, limit=limit)
    elif command != "report":
        raise SystemExit(f"unknown command {command!r}")

    save(results)
    report(results)


if __name__ == "__main__":
    main()
