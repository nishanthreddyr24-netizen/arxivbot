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

    cache = sub.add_parser("cache", help="show or clear the downloaded-paper cache")
    cache.add_argument("--clear", action="store_true", help="delete every cached paper")
    cache.add_argument("--limit", type=int, default=15, help="how many entries to list")
    cache.set_defaults(func=_cache)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
