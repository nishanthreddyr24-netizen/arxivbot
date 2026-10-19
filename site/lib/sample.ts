/**
 * Hardcoded sample data for the homepage.
 *
 * Real papers, real sections, real arXiv ids. Nothing here is a usage count,
 * a user number or a testimonial, because none of those exist yet and a
 * homepage for a tool about unstated assumptions should not open with one.
 */

export const TRACE = {
  arxivId: "2310.06825",
  title: "Mistral 7B",
  authors:
    "Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, " +
    "Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, et al.",
  category: "cs.CL",
  section: "Sliding Window Attention",
  /** The excerpt. `mark` is the sentence the code was derived from. */
  excerpt: [
    {
      kind: "text" as const,
      body:
        "Vanilla attention operates over the full sequence, so the number of " +
        "operations and the memory required both grow with sequence length. ",
    },
    {
      kind: "mark" as const,
      body:
        "SWA exploits the stacked layers of a transformer to attend information " +
        "beyond the window size W: the hidden state in position i of the layer k " +
        "attends to all hidden states from the previous layer with positions " +
        "between i − W and i.",
    },
    {
      kind: "text" as const,
      body:
        " Recursively, the hidden state can access tokens from the input layer " +
        "at a distance of up to W × k tokens.",
    },
  ],
  /** Rendered server-side by KaTeX. */
  equation: "\\text{attn}(i, k) \\;=\\; \\{\\, j \\;:\\; i - W \\le j \\le i \\,\\}",
  equationNote: "the positions a token at i may attend to in layer k",
  code: `class SlidingWindowAttention(nn.Module):
    """Sliding-window attention from "Mistral 7B".

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

        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(~band, float("-inf"))
        return self.out(scores.softmax(-1) @ v)`,
  /** Which generated lines the marked sentence produced, 1-indexed. */
  derivedLines: [19, 20, 21, 22],
};

export type RecentPaper = {
  id: string;
  title: string;
  category: string;
  authors: string;
  extracted: string[];
};

// The fabricated "recent extractions" list lived here. It was invented
// activity on a page whose whole claim is that it only shows what it can
// evidence, so it was cut rather than relabelled. `RecentPaper` stays: it
// is the shape of /api/recent, which returns real rows.

