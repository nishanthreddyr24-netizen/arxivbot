"""Command line entry point.

Right now this exposes the ingestion layer only - enough to confirm that a
paper comes back structurally intact before any model is involved.
"""

from __future__ import annotations

import argparse
import sys

from arxivbot.ingest import build_document, load
from arxivbot.ingest.fetch import FetchError
from arxivbot.ingest.latex import ALGORITHM_ENVS, TABLE_ENVS, UnpackError


def _inspect(args: argparse.Namespace) -> int:
    try:
        meta, raw = load(args.arxiv_id, refresh=args.refresh)
        doc = build_document(meta, raw)
    except (FetchError, UnpackError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{meta.title}")
    print(f"  {', '.join(meta.authors[:4])}{' et al.' if len(meta.authors) > 4 else ''}")
    print(f"  {meta.arxiv_id}  [{', '.join(meta.categories[:3])}]")
    print(f"  source: {len(raw):,} bytes -> {len(doc.files)} tex files -> {len(doc.tex):,} chars")
    print()

    print(f"sections ({len(doc.sections)}):")
    for section in doc.sections:
        if section.level > args.depth:
            continue
        indent = "  " * section.level
        size = len(doc.content(section))
        print(f"{indent}{section.title}  ({size:,} chars)")

    algos = doc.environments(*ALGORITHM_ENVS)
    tables = doc.environments(*TABLE_ENVS)
    print(f"\nalgorithm blocks: {len(algos)}   tables: {len(tables)}")

    method = doc.methodology()
    print(f"methodology candidates: {[s.title for s in method]}")
    appendix = doc.appendix()
    print(f"appendix sections: {[s.title for s in appendix][:6]}")

    if args.show:
        wanted = doc.find(args.show)
        if not wanted:
            print(f"\nno section matching {args.show!r}", file=sys.stderr)
            return 1
        section = wanted[0]
        print(f"\n--- {section.title} ---")
        print(doc.content(section)[: args.chars])
    return 0


def _spec(args: argparse.Namespace) -> int:
    from arxivbot import store
    from arxivbot.extract import extract
    from arxivbot.llm import LLMConfig, LLMError

    if not args.refresh:
        if hit := store.get(args.arxiv_id):
            print(f"found an existing spec ({hit.source}) - pass --refresh to redo it\n")
            _print_spec(hit.spec, source=hit.source)
            return 0

    try:
        meta, raw = load(args.arxiv_id)
        doc = build_document(meta, raw)
    except (FetchError, UnpackError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    config = LLMConfig.from_env()
    if args.model:
        config.model = args.model

    print(f"{meta.title}")
    print(f"extracting with {config.identity} - this takes a few minutes\n")

    try:
        spec, report = extract(doc, config=config, max_components=args.max_components)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    path = store.put(spec)
    _print_spec(spec)

    print(f"\n{report.calls} model calls")
    if (accuracy := report.quote_accuracy) is not None:
        print(f"quotes verified against the source: {accuracy:.0%}")
    if report.demoted:
        print(f"claimed as stated but unverifiable: {', '.join(report.demoted[:6])}")
    for warning in report.warnings:
        print(f"warning: {warning}")
    print(f"\nsaved to {path}")
    return 0


def _print_spec(spec, source: str | None = None) -> None:
    if source == "shared":
        print(f"[from the shared index, extracted by {spec.extractor.model}]")
    if spec.summary:
        print(f"{spec.summary}\n")
    if spec.official_repo:
        print(f"official implementation: {spec.official_repo}\n")

    print(f"components ({len(spec.components)}):")
    for component in spec.components:
        print(f"  {component.name}")
        print(f"      {component.role}")
        if component.depends_on:
            print(f"      built from: {', '.join(component.depends_on)}")
        for hp in component.hyperparameters[:4]:
            mark = "*" if hp.confidence.value == "stated" else "?"
            print(f"      {mark} {hp.name} = {hp.value}")

    counts = spec.confidence_breakdown()
    print(f"\nvalues: {counts['stated']} stated, {counts['conventional']} conventional, "
          f"{counts['guess']} unverified")

    if spec.unknowns:
        print(f"\nthe paper does not specify ({len(spec.unknowns)}):")
        for unknown in sorted(spec.unknowns, key=lambda u: u.severity.value):
            print(f"  [{unknown.severity.value:11}] {unknown.question}")


def _ask(args: argparse.Namespace) -> int:
    from arxivbot import store
    from arxivbot.deviation import banner, check
    from arxivbot.select import select, suggest

    hit = store.get(args.arxiv_id)
    if hit is None:
        print(
            f"no spec for {args.arxiv_id} yet - run: arxivbot spec {args.arxiv_id}",
            file=sys.stderr,
        )
        return 1

    selection = select(hit.spec, args.query)
    if selection.empty:
        print(f"nothing in this paper matches {args.query!r}.", file=sys.stderr)
        print("it covers:", ", ".join(suggest(hit.spec)), file=sys.stderr)
        return 1

    if text := banner(check(hit.spec, args.query, selection.components)):
        print(text + "\n")

    for match in selection.matched:
        print(f"{match.component.name}   ({match.reason})")
        print(f"    {match.component.role}")
        for tensor in match.component.inputs:
            print(f"    in  {tensor.name}: {' x '.join(tensor.shape) or '?'}")
        for tensor in match.component.outputs:
            print(f"    out {tensor.name}: {' x '.join(tensor.shape) or '?'}")
        for equation in match.component.equations:
            print(f"    eq  {equation.latex}")
        for hp in match.component.hyperparameters:
            mark = "*" if hp.confidence.value == "stated" else "?"
            print(f"    {mark}   {hp.name} = {hp.value}")
    for dep in selection.dependencies:
        print(f"\n+ needs {dep.name}: {dep.role}")

    if selection.unknowns:
        print("\nyou will have to decide:")
        for unknown in selection.unknowns:
            print(f"  [{unknown.severity.value}] {unknown.question}")
            if unknown.conventional_default:
                print(f"      usually: {unknown.conventional_default}")
    return 0


def _cache(args: argparse.Namespace) -> int:
    from arxivbot.ingest.fetch import cache_dir, cache_entries, cache_size

    entries = cache_entries()
    path = cache_dir(create=False)

    if args.clear:
        for entry in entries:
            entry.unlink()
        print(f"removed {len(entries)} cached papers from {path}")
        return 0

    print(f"{path}")
    print(f"{len(entries)} papers, {cache_size() / 1e6:.1f} MB")
    for entry in entries[: args.limit]:
        print(f"  {entry.stem:<18} {entry.stat().st_size / 1e6:>6.2f} MB")
    if len(entries) > args.limit:
        print(f"  ... and {len(entries) - args.limit} more")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arxivbot",
        description="Reconstruct implementation scaffolding from arXiv papers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="fetch a paper and show its structure")
    inspect.add_argument("arxiv_id", help="arXiv id or URL, e.g. 1706.03762")
    inspect.add_argument("--refresh", action="store_true", help="bypass the download cache")
    inspect.add_argument("--depth", type=int, default=2, help="max heading level to print")
    inspect.add_argument("--show", metavar="KEYWORD", help="print the body of a matching section")
    inspect.add_argument("--chars", type=int, default=2000, help="how much of it to print")
    inspect.set_defaults(func=_inspect)

    spec = sub.add_parser("spec", help="extract an implementation spec from a paper")
    spec.add_argument("arxiv_id", help="arXiv id or URL")
    spec.add_argument("--refresh", action="store_true", help="re-extract even if cached")
    spec.add_argument("--model", help="override the model, e.g. gemini-3.6-flash")
    spec.add_argument("--max-components", type=int, default=8)
    spec.set_defaults(func=_spec)

    ask = sub.add_parser("ask", help="ask a paper's spec about one part of it")
    ask.add_argument("arxiv_id", help="arXiv id or URL")
    ask.add_argument("query", help="what you want, e.g. 'the attention block'")
    ask.set_defaults(func=_ask)

    cache = sub.add_parser("cache", help="show or clear the downloaded-paper cache")
    cache.add_argument("--clear", action="store_true", help="delete every cached paper")
    cache.add_argument("--limit", type=int, default=15, help="how many entries to list")
    cache.set_defaults(func=_cache)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
