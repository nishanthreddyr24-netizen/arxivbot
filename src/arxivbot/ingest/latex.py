"""Turn a raw arXiv e-print archive into addressable sections of LaTeX.

The pipeline is: unpack the archive, find the root ``.tex`` file, splice in
every ``\\input``/``\\include``, drop comments, then cut the result into
sections. Everything downstream addresses the paper through :class:`Document`.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from dataclasses import dataclass, field

from arxivbot.ingest.fetch import PaperMeta

TEX_SUFFIXES = (".tex", ".ltx", ".cls", ".sty", ".bbl")

# Headings we cut on, most significant first.
_HEADINGS = ("section", "subsection", "subsubsection", "paragraph")
_HEADING_RE = re.compile(
    r"\\(?P<kind>" + "|".join(_HEADINGS) + r")\*?\s*(?:\[[^\]]*\])?\s*\{",
    re.MULTILINE,
)
_LABEL_RE = re.compile(r"\\label\s*\{(?P<label>[^}]*)\}")
_INPUT_RE = re.compile(r"(?<!\\)\\(?:input|include)\s*\{\s*(?P<path>[^}]+?)\s*\}")
_DOCUMENTCLASS_RE = re.compile(r"(?<!%)\\documentclass")

# Environments worth surfacing on their own: these carry the implementation.
ALGORITHM_ENVS = ("algorithm", "algorithmic", "algorithm2e", "algo", "lstlisting", "verbatim")
EQUATION_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*", "multline")
TABLE_ENVS = ("table", "table*", "tabular", "tabularx")


class UnpackError(RuntimeError):
    """Raised when an e-print archive yields no usable LaTeX."""


@dataclass(slots=True)
class Section:
    """One heading and the body text that follows it.

    ``start``, ``body_start`` and ``end`` are absolute offsets into the
    flattened source, which is what lets a downstream claim cite the exact
    span of LaTeX it came from.
    """

    kind: str
    title: str
    body: str
    label: str | None = None
    start: int = 0
    body_start: int = 0
    end: int = 0

    @property
    def level(self) -> int:
        return _HEADINGS.index(self.kind) + 1 if self.kind in _HEADINGS else 0

    def __repr__(self) -> str:
        return f"<Section {self.kind} {self.title!r} {len(self.body)} chars>"


@dataclass(slots=True)
class Environment:
    """A captured LaTeX environment, e.g. an algorithm block or a table."""

    name: str
    body: str
    start: int


@dataclass(slots=True)
class Document:
    meta: PaperMeta
    tex: str
    sections: list[Section] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    bib_start: int = -1
    bib_end: int = -1

    def children(self, section: Section) -> list[Section]:
        """Every section nested under ``section``, at any depth."""
        try:
            index = self.sections.index(section)
        except ValueError:
            return []
        out = []
        for later in self.sections[index + 1 :]:
            if later.level <= section.level:
                break
            out.append(later)
        return out

    def content(self, section: Section) -> str:
        """``section`` plus everything nested under it.

        A heading like ``\\subsection{Encoder and Decoder Stacks}`` often has an
        empty body of its own because a ``\\paragraph`` follows immediately; the
        text a reader thinks of as belonging to it lives in the descendants.
        """
        parts = [section.body]
        for child in self.children(section):
            parts.append(f"\\{child.kind}{{{child.title}}}\n{child.body}")
        return "\n\n".join(p for p in parts if p.strip())

    def find(self, *keywords: str, level: int | None = None) -> list[Section]:
        """Sections whose title contains any of ``keywords`` (case-insensitive)."""
        wanted = [k.lower() for k in keywords]
        return [
            s
            for s in self.sections
            if (level is None or s.level == level)
            and any(k in s.title.lower() for k in wanted)
        ]

    def methodology(self) -> list[Section]:
        """Best-effort guess at the sections describing how the thing works."""
        hits = self.find(
            "method", "approach", "model", "architecture", "algorithm",
            "framework", "implementation", "our ", "design", "formulation",
        )
        return hits or [s for s in self.sections if s.level == 1]

    def appendix(self) -> list[Section]:
        """Sections from the appendix onward - where the real details hide.

        Tried in order: an explicitly titled heading, the ``\\appendix``
        command, then anything after the reference list. The last rule catches
        the many papers that just keep adding sections past the bibliography.
        """
        for i, section in enumerate(self.sections):
            title = section.title.lower()
            if title.startswith("appendix") or title.startswith("supplement"):
                return self.sections[i:]

        marker = self.tex.find("\\appendix")
        if marker != -1:
            return [s for s in self.sections if s.start > marker]

        if self.bib_end > 0:
            return [s for s in self.sections if s.start >= self.bib_end]
        return []

    def environments(self, *names: str) -> list[Environment]:
        return extract_environments(self.tex, names or ALGORITHM_ENVS)


def unpack(raw: bytes) -> dict[str, str]:
    """Extract the text files from an e-print archive.

    Handles the three shapes arXiv actually serves: a gzipped tarball, a
    single gzipped ``.tex``, and (rarely) plain uncompressed LaTeX.
    """
    if raw[:5] == b"%PDF-":
        raise UnpackError("submission has no LaTeX source, only a PDF")

    # Try tarball first - the common case.
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            files = {}
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if not member.name.lower().endswith(TEX_SUFFIXES):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                files[member.name.lstrip("./")] = _decode(handle.read())
            if files:
                return files
    except tarfile.TarError:
        pass

    # Single gzipped file, or raw bytes.
    try:
        body = _decode(gzip.decompress(raw))
    except (OSError, EOFError):
        body = _decode(raw)

    if "\\documentclass" not in body and "\\section" not in body:
        raise UnpackError("archive contained no recognisable LaTeX")
    return {"main.tex": body}


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def find_main(files: dict[str, str]) -> str:
    """Pick the root document: the ``.tex`` file that declares a class."""
    candidates = [
        name
        for name, body in files.items()
        if name.lower().endswith((".tex", ".ltx")) and _DOCUMENTCLASS_RE.search(body)
    ]
    if not candidates:
        tex = [n for n in files if n.lower().endswith((".tex", ".ltx"))]
        if not tex:
            raise UnpackError("no .tex file in archive")
        # Fall back to whichever file has the most section headings.
        return max(tex, key=lambda n: len(_HEADING_RE.findall(files[n])))

    # Prefer a shallow path, then a conventional name, then the largest file.
    def rank(name: str) -> tuple[int, int, int]:
        stem = name.rsplit("/", 1)[-1].lower()
        conventional = stem in {"main.tex", "paper.tex", "ms.tex", "root.tex", "arxiv.tex"}
        return (name.count("/"), 0 if conventional else 1, -len(files[name]))

    return min(candidates, key=rank)


def strip_comments(tex: str) -> str:
    """Remove ``%`` comments while preserving escaped ``\\%``."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.splitlines())


def flatten(files: dict[str, str], main: str, *, max_depth: int = 12) -> str:
    """Splice ``\\input`` and ``\\include`` targets into the root document."""
    lookup = {name.lstrip("./"): body for name, body in files.items()}
    # Also index by basename so \input{sections/intro} finds intro.tex anywhere.
    by_stem: dict[str, str] = {}
    for name, body in lookup.items():
        stem = name.rsplit("/", 1)[-1]
        by_stem.setdefault(stem, body)
        by_stem.setdefault(stem.rsplit(".", 1)[0], body)

    def resolve(path: str) -> str | None:
        path = path.strip().lstrip("./")
        base = path.rsplit("/", 1)[-1]
        for key in (path, f"{path}.tex", base, f"{base}.tex"):
            if key in lookup:
                return lookup[key]
            if key in by_stem:
                return by_stem[key]
        return None

    seen: set[str] = set()

    def expand(body: str, depth: int) -> str:
        if depth > max_depth:
            return body

        def replace(match: re.Match[str]) -> str:
            path = match.group("path")
            if path in seen:
                return ""
            child = resolve(path)
            if child is None:
                return ""
            seen.add(path)
            return "\n" + expand(child, depth + 1) + "\n"

        return _INPUT_RE.sub(replace, body)

    return expand(lookup.get(main, files[main]), 0)


def _balanced(tex: str, open_at: int) -> tuple[str, int]:
    """Read a brace group starting at ``open_at`` (the ``{``). Returns (inner, end)."""
    depth = 0
    for i in range(open_at, len(tex)):
        char = tex[i]
        if char == "{" and (i == 0 or tex[i - 1] != "\\"):
            depth += 1
        elif char == "}" and tex[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return tex[open_at + 1 : i], i + 1
    return tex[open_at + 1 :], len(tex)


def segment(tex: str) -> list[Section]:
    """Cut flattened LaTeX into sections at each heading command.

    ``Section.start`` is an offset into ``tex`` itself, not into the trimmed
    body, so it stays comparable with plain ``tex.find(...)`` lookups.
    """
    base = 0
    start = tex.find("\\begin{document}")
    if start != -1:
        base = start + len("\\begin{document}")
    body = tex[base:]

    marks: list[tuple[int, str, str, int]] = []  # (pos, kind, title, body_start)
    for match in _HEADING_RE.finditer(body):
        title, after = _balanced(body, match.end() - 1)
        marks.append((match.start(), match.group("kind"), _clean_title(title), after))

    sections: list[Section] = []
    for i, (pos, kind, title, body_start) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(body)
        chunk = body[body_start:end]
        label = None
        if label_match := _LABEL_RE.search(chunk[:400]):
            label = label_match.group("label")
        sections.append(
            Section(
                kind=kind,
                title=title,
                body=chunk.strip(),
                label=label,
                start=base + pos,
                body_start=base + body_start,
                end=base + end,
            )
        )
    return sections


def bibliography_span(tex: str) -> tuple[int, int] | None:
    """Locate the reference list, as ``(start, end)`` offsets into ``tex``.

    Papers end a section by starting the next one, so without this the last
    section before the references swallows the entire bibliography.
    """
    begin = re.search(r"\\begin\s*\{thebibliography\}", tex)
    if begin:
        close = re.search(r"\\end\s*\{thebibliography\}", tex[begin.end() :])
        end = begin.end() + close.end() if close else len(tex)
        return begin.start(), end

    # BibTeX style: \bibliography{refs}, optionally preceded by \bibliographystyle.
    cite = re.search(r"\\(?:printbibliography|bibliography)\s*(?:\{[^}]*\})?", tex)
    if cite:
        return cite.start(), cite.end()
    return None


def _clean_title(title: str) -> str:
    """Strip common markup out of a heading so it reads as plain text."""
    title = re.sub(r"\\(?:texttt|textbf|textit|emph|textsc|mbox)\s*\{", "", title)
    title = re.sub(r"\\label\s*\{[^}]*\}", "", title)
    title = title.replace("{", "").replace("}", "").replace("\\", "")
    return " ".join(title.split())


def extract_environments(tex: str, names: tuple[str, ...]) -> list[Environment]:
    """Capture ``\\begin{name}...\\end{name}`` blocks, nesting-aware."""
    found: list[Environment] = []
    for name in names:
        escaped = re.escape(name)
        opener = re.compile(r"\\begin\s*\{" + escaped + r"\}")
        closer = re.compile(r"\\end\s*\{" + escaped + r"\}")
        for match in opener.finditer(tex):
            depth, cursor = 1, match.end()
            while depth and cursor < len(tex):
                nxt_open = opener.search(tex, cursor)
                nxt_close = closer.search(tex, cursor)
                if nxt_close is None:
                    break
                if nxt_open is not None and nxt_open.start() < nxt_close.start():
                    depth += 1
                    cursor = nxt_open.end()
                else:
                    depth -= 1
                    cursor = nxt_close.end()
            found.append(
                Environment(name=name, body=tex[match.end() : cursor].strip(), start=match.start())
            )
    return sorted(found, key=lambda e: e.start)


def build_document(meta: PaperMeta, raw: bytes) -> Document:
    """Full ingestion: archive bytes in, addressable :class:`Document` out."""
    files = unpack(raw)
    main = find_main(files)
    tex = strip_comments(flatten(files, main))
    sections = segment(tex)

    span = bibliography_span(tex)
    if span:
        bib_start, bib_end = span
        kept = []
        for section in sections:
            # Drop headings that are part of the reference list itself.
            if bib_start <= section.start < bib_end:
                continue
            # Truncate the section that runs into the references.
            if section.body_start < bib_start < section.end:
                section.end = bib_start
                section.body = tex[section.body_start : bib_start].strip()
            kept.append(section)
        sections = kept
    else:
        bib_start = bib_end = -1

    return Document(
        meta=meta,
        tex=tex,
        sections=sections,
        files=files,
        bib_start=bib_start,
        bib_end=bib_end,
    )
