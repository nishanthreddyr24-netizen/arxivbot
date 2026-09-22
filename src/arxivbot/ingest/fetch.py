"""Fetch arXiv metadata and LaTeX e-print source, with on-disk caching.

The e-print endpoint returns the author's original submission - real LaTeX,
with intact section boundaries, algorithm environments and hyperparameter
tables. That is dramatically more useful than PDF text extraction, so it is
the primary path here; ``pdf_url`` on the metadata is left for a fallback.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

ATOM = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
API = "http://export.arxiv.org/api/query"
EPRINT = "https://arxiv.org/e-print/{arxiv_id}"

# arXiv asks automated clients to identify themselves and to stay under
# roughly one request every three seconds.
USER_AGENT = "arxivbot/0.1 (+https://github.com/; contact via repo issues)"
MIN_INTERVAL = 3.0

_ID_PATTERNS = (
    re.compile(r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)"),
    re.compile(r"(?P<id>[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"),
)

_last_request = 0.0


class FetchError(RuntimeError):
    """Raised when arXiv cannot be reached or returns something unusable."""


@dataclass(slots=True)
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str] = field(default_factory=list)
    published: str = ""
    updated: str = ""
    pdf_url: str = ""
    comment: str | None = None
    doi: str | None = None
    journal_ref: str | None = None

    @property
    def short_id(self) -> str:
        """The identifier without its version suffix."""
        return re.sub(r"v\d+$", "", self.arxiv_id)


def parse_id(raw: str) -> str:
    """Pull a bare arXiv identifier out of an id, an abs/pdf URL, or a DOI.

    >>> parse_id("https://arxiv.org/abs/1706.03762v5")
    '1706.03762v5'
    >>> parse_id("cs.CL/0108005")
    'cs.CL/0108005'
    """
    raw = raw.strip()
    for pattern in _ID_PATTERNS:
        if match := pattern.search(raw):
            return match.group("id")
    raise ValueError(f"could not parse an arXiv id out of {raw!r}")


def cache_dir() -> Path:
    """Where downloaded sources live. Override with ``ARXIVBOT_CACHE``."""
    root = os.environ.get("ARXIVBOT_CACHE")
    path = Path(root) if root else Path.home() / ".cache" / "arxivbot"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _throttle() -> None:
    global _last_request
    elapsed = time.monotonic() - _last_request
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_request = time.monotonic()


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def fetch_metadata(arxiv_id: str, *, timeout: float = 30.0) -> PaperMeta:
    """Look up title, authors, abstract and categories via the arXiv API."""
    ident = parse_id(arxiv_id)
    _throttle()
    try:
        response = httpx.get(
            API,
            params={"id_list": ident, "max_results": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise FetchError(f"arXiv API request failed for {ident}: {exc}") from exc

    root = ET.fromstring(response.text)
    entry = root.find(f"{{{ATOM}}}entry")
    if entry is None:
        raise FetchError(f"no arXiv entry found for {ident}")

    # A deleted or nonexistent id still returns an entry, but with this title.
    title = _text(entry.find(f"{{{ATOM}}}title"))
    if title.lower() == "error":
        raise FetchError(f"arXiv reports no such paper: {ident}")

    pdf_url = ""
    for link in entry.findall(f"{{{ATOM}}}link"):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")

    canonical = _text(entry.find(f"{{{ATOM}}}id")).rsplit("/abs/", 1)[-1]

    return PaperMeta(
        arxiv_id=canonical or ident,
        title=title,
        authors=[
            _text(a.find(f"{{{ATOM}}}name"))
            for a in entry.findall(f"{{{ATOM}}}author")
        ],
        abstract=_text(entry.find(f"{{{ATOM}}}summary")),
        categories=[
            c.get("term", "") for c in entry.findall(f"{{{ATOM}}}category")
        ],
        published=_text(entry.find(f"{{{ATOM}}}published")),
        updated=_text(entry.find(f"{{{ATOM}}}updated")),
        pdf_url=pdf_url,
        comment=_text(entry.find(f"{{{ARXIV_NS}}}comment")) or None,
        doi=_text(entry.find(f"{{{ARXIV_NS}}}doi")) or None,
        journal_ref=_text(entry.find(f"{{{ARXIV_NS}}}journal_ref")) or None,
    )


def fetch_source(arxiv_id: str, *, refresh: bool = False, timeout: float = 60.0) -> bytes:
    """Download the raw e-print submission, caching it under :func:`cache_dir`.

    The bytes are returned untouched - they may be a gzipped tarball, a single
    gzipped ``.tex`` file, or occasionally a bare PDF for submissions that had
    no source. :func:`arxivbot.ingest.latex.unpack` sorts that out.
    """
    ident = parse_id(arxiv_id)
    blob = cache_dir() / f"{ident.replace('/', '_')}.eprint"

    if blob.exists() and not refresh:
        return blob.read_bytes()

    _throttle()
    try:
        response = httpx.get(
            EPRINT.format(arxiv_id=ident),
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise FetchError(f"could not download source for {ident}: {exc}") from exc

    if not response.content:
        raise FetchError(f"arXiv returned an empty source archive for {ident}")

    blob.write_bytes(response.content)
    return response.content


def load(arxiv_id: str, *, refresh: bool = False) -> tuple[PaperMeta, bytes]:
    """Fetch metadata and source together - the usual entry point."""
    meta = fetch_metadata(arxiv_id)
    return meta, fetch_source(meta.arxiv_id, refresh=refresh)
