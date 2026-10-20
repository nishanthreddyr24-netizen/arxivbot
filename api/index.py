"""The pipeline behind an HTTP API, shaped to run as a Vercel Function.

Vercel detects FastAPI from the dependencies and loads the module-level `app`,
so the same file runs locally under uvicorn and in production as a function.

Two constraints come from the host and are visible throughout:

*No filesystem.* Everything the pipeline caches goes through
:mod:`arxivbot.storage`, pointed at a database here. Set ARXIVBOT_STORE=sqlite
with ARXIVBOT_STORE_PATH on a mounted volume, or point it at a hosted database.

*A request cannot run forever.* Extraction of a cold paper is twenty-odd model
calls and can exceed the 300s a function gets, so it is a job: a POST starts
it, the client polls, and the work survives in the store rather than in the
memory of one invocation.
"""

from __future__ import annotations

import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Importable both as a Vercel function and from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from arxivbot import store  # noqa: E402
from arxivbot.deviation import banner as deviation_banner  # noqa: E402
from arxivbot.deviation import check as deviation_check  # noqa: E402
from arxivbot.extract import extract  # noqa: E402
from arxivbot.ingest import build_document, load  # noqa: E402
from arxivbot.ingest.fetch import BadId, FetchError, fetch_metadata  # noqa: E402
from arxivbot.ingest.latex import UnpackError  # noqa: E402
from arxivbot.llm import CacheMiss, LLMConfig, LLMError  # noqa: E402
from arxivbot.select import did_you_mean, select, suggest  # noqa: E402
from arxivbot.synthesize import generate  # noqa: E402

app = FastAPI(title="arxiv→code", docs_url="/api/docs", openapi_url="/api/openapi.json")

# The frontend may be served from a different origin than the function.
# A deployment that must not spend the operator's quota. Extraction is
# limited to papers already in the store and generation to replies already
# cached, so a public demo costs nothing and cannot be drained. Enforced in
# the model layer by `cache_only`, not by asking endpoints to behave.
DEMO = os.environ.get("ARXIVBOT_DEMO", "").lower() in ("1", "true", "yes", "on")


app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.exception_handler(BadId)
def _unparseable_id(request: Request, exc: BadId) -> JSONResponse:
    """A typo in the id is the caller's mistake, so say so with a 400.

    Four endpoints were answering 500 for `not-a-real-id`, which told the
    reader the server was broken when the id simply was not one. The handler
    is registered once rather than wrapped around each endpoint, so an
    endpoint added later cannot forget it.
    """
    return JSONResponse(status_code=400, content={"detail": str(exc)})


STAGES = ("fetch", "parse", "read", "consolidate", "gaps")


@dataclass
class Job:
    state: str = "running"
    stage: str = "fetch"
    done: list[str] = field(default_factory=list)
    error: str | None = None
    payload: dict | None = None

    def advance(self, stage: str) -> None:
        if self.stage and self.stage not in self.done:
            self.done.append(self.stage)
        self.stage = stage


# In-process only. A function instance may be recycled between polls, so a
# lost job is a normal outcome: the client re-POSTs and the store means the
# work already done is not repeated.
JOBS: dict[str, Job] = {}


class ExtractRequest(BaseModel):
    paper: str
    refresh: bool = False
    max_components: int = 10


def _spec_payload(spec, report=None, cached: bool = False) -> dict:
    import json as _json

    return {
        "spec": _json.loads(spec.to_json()),
        "counts": spec.confidence_breakdown(),
        "cached": cached,
        "report": {
            "calls": getattr(report, "calls", 0),
            "quote_accuracy": getattr(report, "quote_accuracy", None),
            "recovered": getattr(report, "recovered", []),
            "rejected": getattr(report, "rejected", []),
            "warnings": getattr(report, "warnings", []),
        },
    }


def _work(job_id: str, request: ExtractRequest) -> None:
    job = JOBS[job_id]
    try:
        meta, raw = load(request.paper, refresh=request.refresh)
        job.advance("parse")
        doc = build_document(meta, raw)

        job.advance("read")
        spec, report = extract(
            doc, config=LLMConfig.from_env(), max_components=request.max_components
        )

        job.advance("gaps")
        store.put(spec)
        job.payload = _spec_payload(spec, report)
        job.advance("done")
        job.state = "done"
    except (FetchError, UnpackError, LLMError, ValueError) as exc:
        job.state, job.error = "error", str(exc)
    except Exception as exc:  # noqa: BLE001 - the browser should see anything
        job.state, job.error = "error", f"{type(exc).__name__}: {exc}"


@app.get("/api/health")
def health() -> dict:
    config = LLMConfig.from_env()
    return {
        "ok": True,
        "model": config.identity if config.api_key else None,
        "key": bool(config.api_key),
        "store": os.environ.get("ARXIVBOT_STORE", "disk"),
        "demo": DEMO,
    }


@app.get("/api/paper")
def paper(id: str = Query(..., description="arXiv id or URL")) -> dict:
    """Title, authors and categories. Cheap, and enough to render a header."""
    try:
        meta = fetch_metadata(id)
    except (FetchError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "arxiv_id": meta.arxiv_id,
        "title": meta.title,
        "authors": meta.authors,
        "categories": meta.categories,
        "abstract": meta.abstract,
        "pdf_url": meta.pdf_url,
    }


@app.get("/api/recent")
def recent(limit: int = Query(12, ge=1, le=50)) -> dict:
    """Papers already extracted, for the homepage listing.

    Read from the same store the pipeline writes to. A second cache in the
    frontend would be a second answer to the same question, free to disagree
    with this one.
    """
    from arxivbot.spec import ImplementationSpec
    from arxivbot.storage import SPEC
    from arxivbot.storage import store as backend

    papers = []
    for name in backend().keys(SPEC)[:limit]:
        raw = backend().get_text(SPEC, name)
        if raw is None:
            continue
        try:
            spec = ImplementationSpec.model_validate_json(raw)
        except ValueError:
            continue
        counts = spec.confidence_breakdown()
        papers.append(
            {
                "arxiv_id": spec.arxiv_id,
                "title": spec.title,
                "components": [c.name for c in spec.components],
                "gaps": len(spec.unknowns),
                "stated": counts["stated"],
                "model": spec.extractor.model,
                "extracted_at": spec.extractor.extracted_at.isoformat(),
            }
        )
    return {"papers": papers}


@app.get("/api/spec")
def read_spec(id: str = Query(...)) -> dict:
    hit = store.get(id)
    if hit is None:
        raise HTTPException(status_code=404, detail="no spec for that paper yet")
    return _spec_payload(hit.spec, cached=True)


@app.post("/api/extract")
def start_extract(request: ExtractRequest) -> dict:
    config = LLMConfig.from_env()
    if not config.api_key:
        raise HTTPException(
            status_code=400,
            detail="no API key configured on the server (set GEMINI_API_KEY)",
        )

    if DEMO and request.refresh:
        raise HTTPException(
            status_code=403,
            detail="this is a read-only demo; it cannot re-extract a paper",
        )

    if not request.refresh and (hit := store.get(request.paper)):
        job_id = uuid.uuid4().hex
        JOBS[job_id] = Job(
            state="done",
            stage="done",
            done=list(STAGES),
            payload=_spec_payload(hit.spec, cached=True),
        )
        return {"job": job_id, "cached": True}

    if DEMO:
        raise HTTPException(
            status_code=403,
            detail=(
                "this is a read-only demo and holds a fixed set of papers, so "
                "it will not read a new one. Run it yourself with your own "
                "free API key to extract any paper on arXiv."
            ),
        )

    job_id = uuid.uuid4().hex
    JOBS[job_id] = Job()
    threading.Thread(target=_work, args=(job_id, request), daemon=True).start()
    return {"job": job_id, "cached": False}


@app.get("/api/job")
def read_job(id: str = Query(...)) -> dict:
    job = JOBS.get(id)
    if job is None:
        # An instance was recycled, or the id is wrong. Either way the caller
        # should start again; the store makes that cheap.
        raise HTTPException(status_code=404, detail="unknown job")
    body = {"state": job.state, "stage": job.stage, "done": job.done, "error": job.error}
    if job.payload:
        body.update(job.payload)
    return body


@app.get("/api/generate")
def generate_code(id: str = Query(...), q: str = Query(...)) -> StreamingResponse:
    hit = store.get(id)
    if hit is None:
        raise HTTPException(status_code=404, detail="extract this paper first")

    selection = select(hit.spec, q)
    if selection.empty:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no component in the extracted spec matches that",
                "did_you_mean": did_you_mean(hit.spec, q),
                "available": suggest(hit.spec),
            },
        )

    deviations = deviation_check(hit.spec, q, selection.components)

    # In demo mode the header is still derived from the spec - honest work
    # the model had no part in - but the body must already be paid for.
    config = LLMConfig.from_env()
    config.cache_only = DEMO

    def body():
        try:
            yield from generate(hit.spec, selection, deviations, q, config=config)
        except CacheMiss:
            yield (
                "\n\n# This demo streams only skeletons it has generated before,\n"
                "# and this request is not one of them. Everything above was\n"
                "# derived from the spec rather than written by a model, so it\n"
                "# still holds.\n"
                "#\n"
                "# Run it yourself with a free API key to generate any component\n"
                "# of any paper on arXiv.\n"
            )
        except LLMError as exc:
            # Mid-stream, the failure has to arrive as part of the document.
            yield f"\n\n# generation failed: {exc}\n"

    return StreamingResponse(
        body(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ask")
def ask(id: str = Query(...), q: str = Query(...)) -> dict:
    hit = store.get(id)
    if hit is None:
        raise HTTPException(status_code=404, detail="extract this paper first")

    selection = select(hit.spec, q)
    if selection.empty:
        return {
            "matched": [],
            "did_you_mean": did_you_mean(hit.spec, q),
            "available": suggest(hit.spec),
        }

    deviations = deviation_check(hit.spec, q, selection.components)
    return {
        "banner": deviation_banner(deviations),
        "conflict": any(d.is_conflict for d in deviations),
        "matched": [
            {"name": m.component.name, "role": m.component.role, "reason": m.reason}
            for m in selection.matched
        ],
        "dependencies": [c.name for c in selection.dependencies],
        "unknowns": [
            {"question": u.question, "severity": u.severity.value}
            for u in selection.unknowns
        ],
    }


@app.get("/api/components")
def components_of(id: str = Query(...)) -> dict:
    """The component names of an already-extracted paper, for the merge form.

    A merge needs you to name a component of the donor paper, and guessing
    what a given paper calls its parts is the hard bit. This lets the page
    offer them instead of asking.
    """
    hit = store.get(id)
    if hit is None:
        raise HTTPException(status_code=404, detail="extract this paper first")
    return {
        "arxiv_id": hit.spec.arxiv_id,
        "title": hit.spec.title,
        "components": [
            {
                "name": c.name,
                "role": c.role,
                "modality": c.assumptions.modality.value,
                "ordering": c.assumptions.ordering.value,
                "regime": c.assumptions.regime.value,
                "norm_position": c.assumptions.norm_position.value,
            }
            for c in hit.spec.components
        ],
    }


@app.get("/api/merge")
def merge_papers(
    host: str = Query(...),
    donor: str = Query(...),
    component: str = Query(...),
    replaces: str | None = Query(None),
) -> dict:
    """Put a donor paper's component into a host paper's architecture.

    Both specs must already exist. The conflicts come out of code comparing
    the two sets of assumptions, not out of a model, so this endpoint costs
    nothing and returns the same answer every time.
    """
    from arxivbot.merge import merge as compose

    specs = {}
    for arxiv_id in (host, donor):
        hit = store.get(arxiv_id)
        if hit is None:
            raise HTTPException(
                status_code=404, detail=f"extract {arxiv_id} first"
            )
        specs[arxiv_id] = hit.spec

    known = []
    try:
        from arxivbot.storage import SPEC
        from arxivbot.storage import store as backend
        from arxivbot.spec import ImplementationSpec

        for name in backend().keys(SPEC):
            if name in (host, donor):
                continue
            raw = backend().get_text(SPEC, name)
            if raw:
                known.append(ImplementationSpec.model_validate_json(raw))
    except Exception:  # noqa: BLE001 - precedents are a nicety, not the answer
        known = []

    try:
        hybrid = compose(
            specs[host], specs[donor], component, replaces=replaces, known=known
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "host": {"id": hybrid.host_id, "title": hybrid.host_title},
        "donor": {"id": hybrid.donor_id, "title": hybrid.donor_title},
        "donated": hybrid.donated,
        "replaced": hybrid.replaced,
        "buildable": hybrid.buildable,
        "components": [
            {"name": c.name, "donated": c.name == hybrid.donated}
            for c in hybrid.components
        ],
        "conflicts": [
            {
                "axis": c.axis,
                "host": c.host,
                "donor": c.donor,
                "severity": c.severity.value,
                "resolution": c.resolution.value,
                "detail": c.detail,
            }
            for c in hybrid.conflicts
        ],
        "notes": hybrid.notes,
        "precedents": [
            {"id": i, "title": t, "overlap": o} for i, t, o in hybrid.precedents
        ],
        "unknowns": [
            {"question": u.question, "severity": u.severity.value}
            for u in hybrid.unknowns
        ],
    }
