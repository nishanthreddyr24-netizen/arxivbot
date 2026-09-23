"""Where specs come from: this machine first, then the shared public index.

Extraction is the expensive step, so nobody should pay for the same paper
twice. Papers are immutable once versioned - ``1706.03762v7`` is the same
document forever - which means a plain exact key is enough and there is no
need to guess at similarity between requests.

The shared index is an ordinary git repository of JSON files served over
HTTPS. That choice is deliberate: it costs nothing to host, works offline once
fetched, and a wrong spec is a pull request someone can fix rather than an
opaque cache entry nobody can correct.

Nothing here uploads anything. Contributing a spec back is an explicit action
by the user - see :func:`export_for_contribution`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from arxivbot.ingest.fetch import (
    USER_AGENT,
    FetchError,
    cache_dir,
    fetch_metadata,
    parse_id,
)
from arxivbot.spec import SCHEMA_VERSION, ImplementationSpec

DEFAULT_INDEX = (
    "https://raw.githubusercontent.com/"
    "nishanthreddyr24-netizen/arxivbot-specs/main/specs"
)

Source = Literal["local", "shared"]


@dataclass(slots=True)
class Hit:
    """A spec and where it came from.

    Callers should surface ``source`` to the user. A spec pulled from the
    shared index was produced by someone else's extraction run, possibly with a
    different model - ``spec.extractor`` says which - and deserves to be
    labelled rather than presented as locally verified.
    """

    spec: ImplementationSpec
    source: Source
    stale: bool = False
    """True when the spec was built under an older schema version."""


def index_url() -> str:
    """Base URL of the shared spec index. Override with ``ARXIVBOT_SPEC_INDEX``."""
    return os.environ.get("ARXIVBOT_SPEC_INDEX", DEFAULT_INDEX).rstrip("/")


def specs_dir() -> Path:
    """Local directory holding extracted and downloaded specs."""
    path = cache_dir() / "specs"
    path.mkdir(parents=True, exist_ok=True)
    return path


_VERSION_RE = re.compile(r"v(\d+)$")


def _key(arxiv_id: str) -> str:
    return parse_id(arxiv_id).replace("/", "_")


def has_version(arxiv_id: str) -> bool:
    """Whether an id pins a specific version, e.g. ``1706.03762v7``."""
    return _VERSION_RE.search(parse_id(arxiv_id)) is not None


def local_path(arxiv_id: str) -> Path:
    return specs_dir() / f"{_key(arxiv_id)}.json"


def _newest_local_version(arxiv_id: str) -> Path | None:
    """Highest-numbered stored version of an unversioned id.

    Specs are filed under the version they describe, but almost nobody types
    a version number. Without this, the ordinary `1706.03762` never matches
    the stored `1706.03762v7.json`.
    """
    matches = []
    for path in specs_dir().glob(f"{_key(arxiv_id)}v*.json"):
        if match := _VERSION_RE.search(path.stem):
            matches.append((int(match.group(1)), path))
    if not matches:
        return None
    return max(matches)[1]


def resolve_version(arxiv_id: str) -> str:
    """Return a version-pinned id, asking arXiv only when necessary.

    An id that already names a version is returned untouched, so the common
    path costs nothing.
    """
    if has_version(arxiv_id):
        return parse_id(arxiv_id)
    return fetch_metadata(arxiv_id).arxiv_id


def _check(spec: ImplementationSpec, source: Source) -> Hit:
    stale = spec.extractor.schema_version != SCHEMA_VERSION
    return Hit(spec=spec, source=source, stale=stale)


def get_local(arxiv_id: str) -> Hit | None:
    """Look for a spec already on this machine.

    An unversioned id matches the newest stored version of that paper.
    """
    path = local_path(arxiv_id)
    if not path.exists():
        if has_version(arxiv_id):
            return None
        path = _newest_local_version(arxiv_id)
        if path is None:
            return None
    try:
        return _check(ImplementationSpec.load(path), "local")
    except (ValueError, OSError):
        # A corrupt or schema-incompatible file should not be fatal; treat it
        # as a miss so the caller can re-extract over it.
        return None


def get_shared(arxiv_id: str, *, timeout: float = 15.0) -> Hit | None:
    """Look the paper up in the public spec index.

    A miss - including no network - returns None rather than raising. The
    shared index is an optimisation, never a dependency.
    """
    url = f"{index_url()}/{_key(arxiv_id)}.json"
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None

    try:
        spec = ImplementationSpec.model_validate_json(response.text)
    except ValueError:
        return None

    # Keep it, so this paper works offline from now on.
    try:
        spec.save(local_path(arxiv_id))
    except OSError:
        pass
    return _check(spec, "shared")


def get(arxiv_id: str, *, use_shared: bool = True) -> Hit | None:
    """This machine first, then the shared index. None if neither has it.

    The shared index is keyed by exact version, so an unversioned id has to be
    resolved against arXiv before it can be looked up there. That costs a
    request, which is why the local check - which can match an unversioned id
    on its own - happens first.
    """
    if hit := get_local(arxiv_id):
        return hit
    if not use_shared:
        return None

    try:
        pinned = resolve_version(arxiv_id)
    except (FetchError, ValueError):
        return None
    return get_shared(pinned)


def put(spec: ImplementationSpec) -> Path:
    """Store a freshly extracted spec locally."""
    return spec.save(local_path(spec.arxiv_id))


def list_local() -> list[Path]:
    return sorted(specs_dir().glob("*.json"))


def export_for_contribution(arxiv_id: str, dest: Path) -> Path:
    """Write a spec out in the layout the shared index expects.

    Contributing is a deliberate act: this only writes a file the user can
    inspect and open a pull request with. Nothing is transmitted.
    """
    hit = get_local(arxiv_id)
    if hit is None:
        raise FileNotFoundError(f"no local spec for {arxiv_id}")
    target = dest / "specs" / f"{_key(arxiv_id)}.json"
    return hit.spec.save(target)
