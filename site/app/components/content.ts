/**
 * Homepage copy that is not in `lib/sample.ts`.
 *
 * The generated skeleton lives here rather than in the shared sample, because
 * the page has to show both halves of the product and the shared sample only
 * carries the first: it has derived lines but no `TODO(paper-silent)`. The
 * paper's own text - title, authors, excerpt, equation - still comes from
 * `lib/sample.ts`, so nothing quoted is duplicated and nothing can drift.
 */

/** The skeleton, as the generator emits it. 1-indexed line maps below. */
export const SKELETON = `class SlidingWindowAttention(nn.Module):
    """Sliding-window attention from "Mistral 7B" (2310.06825).

    Section: Sliding Window Attention
    """

    def __init__(self, d_model: int, n_heads: int, window: int):
        super().__init__()
        self.window = window          # W, Sliding Window Attention
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_model]
        B, S, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Position i attends to [i - W, i]: a band, not the full sequence.
        rows = torch.arange(S, device=x.device).unsqueeze(1)
        cols = torch.arange(S, device=x.device).unsqueeze(0)
        band = (cols <= rows) & (cols > rows - self.window)

        # TODO(paper-silent): the paper does not state how the first W
        # positions are handled, nor whether the band includes i - W.
        # Both are the implementer's call. This file picks one.

        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(~band, float("-inf"))
        return self.out(scores.softmax(-1) @ v)`;

/** Lines the marked sentence produced. */
export const DERIVED_LINES = [19, 20, 21, 22];

/** Lines standing in for something the paper never decided. */
export const SILENT_LINES = [24, 25, 26];

/**
 * The gap report for this one section, stated in the hero.
 *
 * This is the half of the product nothing else does: a model that has read the
 * paper will answer all three of these without hesitating, and the paper does
 * not answer any of them.
 */
export const SILENCES: { subject: string; body: string }[] = [
  {
    subject: "the first W positions",
    body: "Whether early tokens see a shorter window or a padded one is never stated.",
  },
  {
    subject: "the band's lower bound",
    body: "The text reads \u201cbetween i \u2212 W and i\u201d. Inclusive of i \u2212 W, or not, changes the mask by one column.",
  },
  {
    subject: "the masked value",
    body: "\u2212inf is the obvious choice and overflows in fp16. The paper names no value.",
  },
];

/** The real pipeline, in its own order. */
export const STEPS: { title: string; body: string }[] = [
  {
    title: "Fetch the source, not the PDF",
    body: "arXiv serves the authors' original LaTeX. Every claim is read from it, so a quote can be checked character for character against what they wrote.",
  },
  {
    title: "Extract a spec, then verify every claim",
    body: "Components, hyperparameters and equations, each carrying the sentence it came from. A claim whose quote cannot be located in the source is thrown away rather than shown.",
  },
  {
    title: "Name what the paper never decided",
    body: "Initialisation, padding, decoding: the choices an implementer has to make and the paper does not state. Reported as gaps, not guessed at.",
  },
  {
    title: "Generate a skeleton that admits both",
    body: "Code with the section behind each value, and a marked TODO wherever the paper was silent.",
  },
];

/**
 * The step between the quote and the code: what the sentence actually fixed,
 * and where each decision landed in the file. Without this the panel asks the
 * reader to take the jump from prose to a line number on trust.
 */
export const DERIVATIONS: { name: string; value: string; lines: string }[] = [
  { name: "window", value: "W", lines: "line 9" },
  { name: "attention band", value: "[i − W, i]", lines: "lines 19–22" },
  { name: "mask", value: "cols ≤ rows & cols > rows − W", lines: "line 22" },
];
