# arxivbot

Reconstruct implementation scaffolding from arXiv papers.

Point it at a paper, describe what you need, and get back a code skeleton for
that component — with every design decision cited back to the span of the paper
it came from, and with the things the paper **never specifies** marked as open
questions instead of silently invented.

> **Status: early but working end to end.** A paper can be fetched, read,
> turned into a spec, and rendered as an annotated code skeleton. Nothing
> here has been evaluated at scale — see [Roadmap](#roadmap) and
> [Limitations](#limitations) for what is real today and what is not.

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

### The web interface

```bash
arxivbot serve
```

Opens a local workspace at `127.0.0.1:8000`. The paper renders on the left;
you ask for a component on the right and watch the skeleton being written.

Runs entirely on your machine against your own key — it talks to arXiv and
your model provider and nothing else.

### Extracting a spec

This is the part that needs a model. Set `GEMINI_API_KEY` in your environment or
a `.env` file — a free key takes a minute to get from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey), no card needed.

```bash
arxivbot spec 1706.03762
```

```
components (4):
  Multi-Head Attention
      Performs multiple attention functions in parallel on linearly projected
      versions of queries, keys, and values, then concatenates the results.
      built from: Scaled Dot-Product Attention
      * h = 8
      * d_k = 64

values: 11 stated, 0 conventional, 1 unverified

the paper does not specify (4):
  [blocking   ] What is the weight initialization scheme for the linear layers?
  [blocking   ] What loss function is used for training?
  [minor      ] How are biases initialized throughout the network?

7 model calls
quotes verified against the source: 92%
```

A `*` marks a value the paper states and whose supporting sentence was found in
the source. A `?` marks one that could not be verified.

### Asking about one part

Extraction happens once per paper. Asking costs nothing and involves no model
call — the spec is already on disk.

```bash
arxivbot ask 1706.03762 "the attention block"
arxivbot ask 1706.03762 "the residual stream"
```

Requests that contradict the paper are honoured, and labelled:

```bash
$ arxivbot ask 1706.03762 "feed forward but with gelu instead of relu"

# !! DEVIATION - activation
#    you asked for : gelu
#    paper states  : relu (Position-wise Feed-Forward Networks)
#    This code implements your choice, not the paper's.
```

Any model works. Point it elsewhere with `ARXIVBOT_MODEL`, or at any
OpenAI-compatible endpoint — Groq, OpenRouter, a local Ollama — with
`ARXIVBOT_PROVIDER=openai` and `ARXIVBOT_BASE_URL`.

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
extract        sections -> ImplementationSpec, over several narrow
  |            schema-constrained calls rather than one large one.
  |            Every quote is checked against the source; a claim whose
  |            evidence cannot be found is demoted, not recorded as fact.
  v
select         a request -> the components it names, plus what they
  |            compose. No model call: the spec is already on disk.
  v
synthesise     [planned] selected components -> code skeleton
  |
critique       [planned] coverage + shape consistency + AST validity
```

The model layer is provider-agnostic and written directly against two HTTP
APIs rather than taking a framework dependency: Gemini, and any
OpenAI-compatible endpoint, which covers Groq, OpenRouter, vLLM and Ollama at
once. No component of this project will ever require a paid account to run.

## Roadmap

- [x] arXiv metadata and e-print source retrieval, cached and rate-limited
- [x] LaTeX unpacking, `\input` flattening, section segmentation
- [x] Bibliography and appendix boundary detection
- [x] Algorithm, equation and table environment extraction
- [x] `ImplementationSpec` schema with per-claim provenance and confidence
- [x] Provider-agnostic model layer with disk-cached responses
- [x] Spec extraction from methodology and appendix
- [x] Quote verification — claims whose evidence is absent are demoted
- [x] Underspecification report — what the paper does not tell you
- [x] Request routing: one spec answers many different questions
- [x] Deviation detection when a request contradicts the paper
- [x] Skeleton synthesis — emit actual code, not just the spec
- [ ] Critic loop (spec coverage, shape consistency, AST validity)
- [ ] The shared spec index
- [ ] Grounding against the official implementation, when one exists
- [ ] Evaluation against papers with official implementations

## Limitations

Known and worth stating plainly:

- **Barely evaluated.** Tested on a handful of papers. No numbers exist yet for
  how much it recovers or how reliably it spots a gap. Until they do, treat
  output as a starting point for a human.
- **LaTeX macros defeat quote verification.** A paper defining
  `\newcommand{\dmodel}{d_{\text{model}}}` writes `$\dmodel=512$`, so the
  literal sentence a reader would quote is not in the source and the claim is
  demoted even though it is correct.
- **False gaps happen.** The underspecification pass occasionally reports
  something the paper does state.
- **No PDF fallback.** Roughly one submission in ten ships no LaTeX source.
- **Structure-aware, so unusual LaTeX hurts.** Papers that avoid `\section`
  or build headings from custom macros will segment poorly.

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
