"""A local web front end for the CLI.

Runs on this machine, against the user's own key, and talks to nothing but
arXiv and whichever model provider is configured. Built on ``http.server`` so
the project keeps its two dependencies.

Extraction takes minutes, so a request starts a job and the page polls it.
That also lets the page show which stage is running, which matters when the
alternative is several silent minutes.
"""

from __future__ import annotations

import json
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from arxivbot import store
from arxivbot.deviation import banner as deviation_banner
from arxivbot.deviation import check as deviation_check
from arxivbot.extract import extract
from arxivbot.ingest import build_document, load
from arxivbot.ingest.fetch import FetchError, fetch_pdf
from arxivbot.ingest.latex import UnpackError
from arxivbot.llm import LLMConfig, LLMError
from arxivbot.select import select as select_components
from arxivbot.spec import Confidence, ImplementationSpec
from arxivbot.synthesize import generate

ASSETS = Path(__file__).parent / "assets"
STAGES = ("fetch", "parse", "inventory", "detail", "unknowns")


@dataclass
class Job:
    state: str = "running"
    stage: str = "fetch"
    done_stages: list[str] = field(default_factory=list)
    error: str | None = None
    spec: ImplementationSpec | None = None
    calls: int = 0
    quote_accuracy: float | None = None

    def advance(self, stage: str) -> None:
        if self.stage and self.stage not in self.done_stages:
            self.done_stages.append(self.stage)
        self.stage = stage


JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def _payload(spec: ImplementationSpec) -> dict:
    """The spec as the page needs it, with counts precomputed."""
    data = json.loads(spec.to_json())
    data["counts"] = spec.confidence_breakdown()
    return data


def _work(job_id: str, arxiv_id: str, max_components: int) -> None:
    job = JOBS[job_id]
    try:
        meta, raw = load(arxiv_id)
        job.advance("parse")
        doc = build_document(meta, raw)

        job.advance("inventory")
        config = LLMConfig.from_env()

        # The extractor does not report progress, so move the page on once the
        # inventory call can plausibly have finished. Honest enough: the stages
        # describe what is happening, not a measured percentage.
        def nudge() -> None:
            job.advance("detail")

        timer = threading.Timer(35.0, nudge)
        timer.daemon = True
        timer.start()

        spec, report = extract(doc, config=config, max_components=max_components)
        timer.cancel()

        job.advance("unknowns")
        store.put(spec)

        job.spec = spec
        job.calls = report.calls
        job.quote_accuracy = report.quote_accuracy
        job.advance("done")
        job.state = "done"
    except (FetchError, UnpackError, LLMError, ValueError) as exc:
        job.state = "error"
        job.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - surface anything to the browser
        job.state = "error"
        job.error = f"{type(exc).__name__}: {exc}"


class Handler(BaseHTTPRequestHandler):
    server_version = "arxivbot"

    def log_message(self, fmt, *args):  # quieter console
        return

    # ---- helpers ----

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    # ---- routes ----

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            page = (ASSETS / "index.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if route.path == "/api/job":
            return self._job(parse_qs(route.query))
        if route.path == "/api/ask":
            return self._ask(parse_qs(route.query))
        if route.path == "/api/pdf":
            return self._pdf(parse_qs(route.query))
        if route.path == "/api/generate":
            return self._generate(parse_qs(route.query))
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/extract":
            return self._json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json({"error": "bad request"}, 400)

        arxiv_id = (body.get("id") or "").strip()
        if not arxiv_id:
            return self._json({"error": "give an arXiv id"}, 400)

        config = LLMConfig.from_env()
        if not config.api_key:
            return self._json(
                {
                    "error": "No API key found. Put GEMINI_API_KEY in a .env file "
                    "next to the project, or in your environment."
                },
                400,
            )

        # An already-extracted paper answers immediately.
        if hit := store.get_local(arxiv_id):
            job_id = uuid.uuid4().hex
            job = Job(state="done", stage="done", done_stages=list(STAGES), spec=hit.spec)
            with _LOCK:
                JOBS[job_id] = job
            return self._json({"job": job_id, "cached": True})

        job_id = uuid.uuid4().hex
        with _LOCK:
            JOBS[job_id] = Job()
        thread = threading.Thread(
            target=_work, args=(job_id, arxiv_id, self.server.max_components), daemon=True
        )
        thread.start()
        self._json({"job": job_id})

    def _job(self, query: dict) -> None:
        job = JOBS.get((query.get("id") or [""])[0])
        if job is None:
            return self._json({"error": "unknown job"}, 404)

        out = {
            "state": job.state,
            "stage": job.stage,
            "done_stages": job.done_stages,
            "error": job.error,
        }
        if job.state == "done" and job.spec is not None:
            out["spec"] = _payload(job.spec)
            out["report"] = {"calls": job.calls, "quote_accuracy": job.quote_accuracy}
        self._json(out)

    def _pdf(self, query: dict) -> None:
        arxiv_id = (query.get("id") or [""])[0]
        try:
            data = fetch_pdf(arxiv_id)
        except (FetchError, ValueError) as exc:
            return self._json({"error": str(exc)}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", "inline")
        self.end_headers()
        self.wfile.write(data)

    def _generate(self, query: dict) -> None:
        """Stream a code skeleton as it is produced.

        Written straight to the socket with no Content-Length: the point is
        that the reader sees tokens arrive, so nothing may buffer the body.
        """
        arxiv_id = (query.get("id") or [""])[0]
        request = (query.get("q") or [""])[0]
        hit = store.get_local(arxiv_id)
        if hit is None:
            return self._json({"error": "extract this paper first"}, 404)
        if not request:
            return self._json({"error": "say what you want"}, 400)

        selection = select_components(hit.spec, request)
        if selection.empty:
            return self._json(
                {
                    "error": "Nothing in this paper matches that. It covers: "
                    + ", ".join(c.name for c in hit.spec.components)
                },
                404,
            )
        deviations = deviation_check(hit.spec, request, selection.components)

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for chunk in generate(hit.spec, selection, deviations, request):
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except LLMError as exc:
            # The reader is mid-stream, so the failure has to arrive as part
            # of the document rather than as a status code.
            self.wfile.write(f"\n\n# generation failed: {exc}\n".encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            pass  # reader navigated away mid-stream

    def _ask(self, query: dict) -> None:
        arxiv_id = (query.get("id") or [""])[0]
        question = (query.get("q") or [""])[0]
        hit = store.get_local(arxiv_id)
        if hit is None:
            return self._json({"error": "extract this paper first"}, 404)

        selection = select_components(hit.spec, question)
        deviations = deviation_check(hit.spec, question, selection.components)
        self._json(
            {
                "banner": deviation_banner(deviations),
                "conflict": any(d.is_conflict for d in deviations),
                "matched": [
                    {
                        "name": m.component.name,
                        "role": m.component.role,
                        "reason": m.reason,
                    }
                    for m in selection.matched
                ],
                "dependencies": [c.name for c in selection.dependencies],
            }
        )


def serve(port: int = 8000, *, open_browser: bool = True, max_components: int = 6) -> None:
    """Run the local front end until interrupted."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.max_components = max_components  # type: ignore[attr-defined]

    url = f"http://127.0.0.1:{port}"
    config = LLMConfig.from_env()
    print(f"arxivbot -> {url}")
    print(f"model    -> {config.identity}" if config.api_key else "model    -> no API key found")
    print("ctrl-c to stop")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
