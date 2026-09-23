# arxivbot

Reconstruct implementation scaffolding from arXiv papers.

Point it at a paper, describe what you need, and get back a code skeleton for
that component — with every design decision cited back to the span of the paper
it came from, and with the things the paper **never specifies** marked as open
questions instead of silently invented.

> **Status: early.** The ingestion layer is built and working. The model-driven
> extraction and synthesis layers are not written yet. See [Roadmap](#roadmap)
> for what is real today and what is not.

## Why

Most paper-to-code tools optimise for producing *complete* code. Papers are
always underspecified, so "complete" means the model invents the 30-40% the
authors left out — and says nothing about which 30-40% that was. The output
looks authoritative and is quietly fabricated.

arxivbot inverts that. The primary artifact is an **`ImplementationSpec`**: a
structured, machine-readable record of what the paper actually states, with
citations, plus an explicit list of what it does not. Code generation is a
rendering of that spec, not a replacement for it.

Two design choices follow from this:

**LaTeX source, not PDF.** arXiv serves the author's original submission at
`arxiv.org/e-print/<id>`. That gives real `\section{}` boundaries, `algorithm`
environments, equations as LaTeX and hyperparameters as `tabular` — all of which
PDF text extraction destroys. Roughly 90% of submissions ship source; PDF is a
fallback, not the main path.

**Structural retrieval, not embeddings.** For a single paper the corpus is
smaller than a model's context window, and LaTeX already tells you which chunk
is the methodology. Chunking that into fixed-size pieces and retrieving by
cosine similarity replaces ground truth with a guess. There is no vector
database here, and for the single-paper path there does not need to be one.

## Install

```bash
git clone https://github.com/nishanthreddyr24-netizen/arxivbot
cd arxivbot
pip install -e .
```

Requires Python 3.11+. The ingestion layer has one dependency (`httpx`) and
needs no API key of any kind.

## Usage

Inspect a paper's structure:

```bash
arxivbot inspect 1706.03762
```

```
Attention Is All You Need
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit et al.
  1706.03762v7  [cs.CL, cs.LG]
  source: 1,150,988 bytes -> 11 tex files -> 53,878 chars

sections (30):
  Introduction  (2,180 chars)
  Model Architecture  (11,174 chars)
    Encoder and Decoder Stacks  (1,415 chars)
    Attention  (5,820 chars)
    ...

algorithm blocks: 0   tables: 10
appendix sections: ['Attention Visualizations', ...]
```

Print a section's text:

```bash
arxivbot inspect 2006.11239 --show "diffusion" --chars 3000
```

Accepts bare ids, versioned ids, and `arxiv.org/abs/...` URLs. Requests are
throttled to the rate arXiv asks of automated clients.

### Caching

Every download is cached locally so repeated runs cost nothing and do not hit
arXiv again. The cache lives on your own machine, per user — nothing is shared
or uploaded:

| Platform | Location |
|---|---|
| Windows | `%LOCALAPPDATA%\arxivbot\Cache` |
| macOS | `~/Library/Caches/arxivbot` |
| Linux | `$XDG_CACHE_HOME/arxivbot`, else `~/.cache/arxivbot` |

Set `ARXIVBOT_CACHE` to override. Inspect or empty it with:

```bash
arxivbot cache            # where it is, how big, what is in it
arxivbot cache --clear    # delete everything cached
```

Papers average a few MB each and nothing is evicted automatically, so run
`--clear` if it grows beyond what you want to keep.

## Design

```
fetch          arXiv Atom API + e-print tarball, disk-cached
  |
segment        unpack -> find root .tex -> splice \input -> strip comments
  |            -> cut into sections -> locate bibliography and appendix
  v
extract        [planned] sections -> ImplementationSpec, many narrow
  |            schema-constrained calls rather than one large one
  v
ground         [planned] reconcile against the official repo, when one exists
  |
synthesise     [planned] user's request -> relevant spec subset -> skeleton
  |
critique       [planned] coverage + shape consistency + AST validity
```

The model layer will be provider-agnostic (`litellm`), so any backend works and
none is required — a free tier, a paid key, or a local model via Ollama. No
component of this project will ever require a paid account to run.

## Roadmap

- [x] arXiv metadata and e-print source retrieval, cached and rate-limited
- [x] LaTeX unpacking, `\input` flattening, section segmentation
- [x] Bibliography and appendix boundary detection
- [x] Algorithm, equation and table environment extraction
- [ ] `ImplementationSpec` schema
- [ ] Provider-agnostic model layer with disk-cached responses
- [ ] Spec extraction from methodology and appendix
- [ ] Underspecification report — what the paper does not tell you
- [ ] Skeleton synthesis from a user's request
- [ ] Critic loop (spec coverage, shape consistency, AST validity)
- [ ] Evaluation against papers with official implementations

## Evaluation

Planned, and the part that decides whether any of this is trustworthy. Papers
with public official implementations provide ground truth: extract a spec, diff
it against the real repo, and measure both how much is recovered and how
accurately underspecification is flagged. Until those numbers exist, treat the
output as a starting point for a human, not as a reproduction.

## Contributing

Issues and PRs welcome. The ingestion layer is the easiest place to help —
LaTeX in the wild is endlessly creative, and papers that segment badly are
useful bug reports. Please include the arXiv id.

## License

MIT — see [LICENSE](LICENSE).
