# Gap spot-check — 20 gaps, judged by hand against the LaTeX source

Sampled with `seed=7`, at most 2 gaps per paper, 24 papers eligible. PPO and
Adam were excluded: their recorded rows predate the fix that correctly refuses
them, so their gaps are artifacts.

A gap asserts *"the paper does not state X"*. That is falsifiable in one
direction: if the source plainly states X, the gap is false. "I could not find
it" is weaker evidence, so every verdict below was read in context.

| # | paper | severity | gap | verdict | where the answer actually is |
|---|---|---|---|---|---|
| 1 | T5 | blocking | tokenisation + vocabulary size | **FALSE** | §Vocabulary: SentencePiece, 32,000 wordpieces |
| 2 | LoRA | significant | which modules get LoRA beyond q,v | **FALSE** | Table `weight_type`: Wq,Wk,Wv,Wo combinations |
| 3 | BatchNorm | blocking | weight initialisation scheme | true | — |
| 4 | Mistral 7B | blocking | normalisation type and placement | true | no mention of normalisation anywhere |
| 5 | BERT | blocking | weight init distribution | true | — |
| 6 | RoBERTa | significant | peak LR and warmup per setting | **FALSE** | appendix table: warmup 30k/24k, peak 4e-4/6e-4 |
| 7 | DDPM | significant | number of groups in GroupNorm | true | "group normalization throughout", count never given |
| 8 | LoRA | significant | LR schedule for RoBERTa/DeBERTa | **FALSE** | appendix: "AdamW with a linear learning rate decay schedule" |
| 9 | SAC | significant | weight initialisation | true | hyperparameter table omits it |
| 10 | Transformer | minor | how biases are initialised | true | — |
| 11 | U-Net | significant | input image dims + tile sizes | *partial* | 512x512 images stated; tile size never given |
| 12 | Latent Diffusion | significant | patch discriminator architecture | true | deferred to VQGAN by citation |
| 13 | U-Net | minor | mean of the init Gaussian | *partial* | std sqrt(2/N) stated, mean omitted (pedantic) |
| 14 | MAML | blocking | network architectures | **FALSE** | regression 2x40 ReLU; RL policy 2x100 ReLU |
| 15 | MAML | blocking | env params, horizons, rewards | **FALSE** | H=100 and H=200, reward functions stated |
| 16 | ALBERT | significant | weight init distributions | true | — |
| 17 | GCN | minor | exact early-stopping criterion | **FALSE** | "window size of 10 ... 10 consecutive epochs" |
| 18 | RoFormer | minor | AdamW weight decay | true | only LR 1e-5 given |
| 19 | GAN | blocking | LR and momentum | true | "We used momentum"; values only in the repo |
| 20 | RoFormer | significant | LR schedule for pre-training | true | no schedule stated |

## Result

    true gaps      11 / 20   55%
    FALSE gaps      7 / 20   35%
    partial         2 / 20   10%

A false gap is the project's worst failure mode: a confident, well-formatted
claim that the paper omits something it states plainly. It is strictly worse
than silence, because the whole point of the feature is to be trusted.

## Why — measured, not guessed

`_unknowns()` is handed `doc.methodology() + doc.appendix()` in a single call
and asked what the paper fails to specify. Two things go wrong.

**The experiments section is never in scope.** Hyperparameters live there.

    paper         tex     seen by the gap pass   never seen
    T5         215,947    45%                    Introduction, Experiments, Reflection
    MAML        69,855    49%                    Experimental Evaluation, ...
    GCN         47,038    60%                    Experiments, Results, ...
    RoBERTa     48,089    69%                    Introduction, Background, ...

GCN's early-stopping rule and MAML's architectures are both in `Experiments`.

**Where scope was not the problem, size was.** LoRA reports `seen = 161%` of
the paper - `appendix()` returns the whole document, so the method sections are
sent twice, 171,660 characters in one prompt. RoBERTa's appendix *was* in
scope and the model still missed the table. Asking a model to reason about
*absence* across 170k characters is the hardest possible framing: one missing
sentence is invisible, and silence reads the same as omission.

This is the lesson the extractor already learned - recall is a function of
scope, and per-unit beats one big call - arriving a second time at a pass that
never got the treatment.

## The fix: make a gap falsifiable, like a quote

Gaps are currently the one model output with no mechanical check behind them.
Every other claim in the spec must survive one. The same medicine applies:

1. The model proposes candidate gaps, leniently, as now.
2. For each candidate, code searches the **whole** document - not the method
   section - for terms drawn from the question.
3. The retrieved passages, and only those, go back in a small focused call:
   *does this passage state X?* Short context, one question, answerable.
4. A gap whose answer is found is **dropped**, and where possible promoted
   into a hyperparameter with the provenance that was just located.

Model proposes, code disposes - applied to the one claim that so far escaped
it. Target: false gaps below 10%, re-measured on this same 20.

Also to fix: `Document.appendix()` returning the entire document (LoRA).

## Measured after the fix

Retrieval alone, asking only whether the refuting sentence is reachable:
**6 of 7**. The miss is LoRA's table of adapted weight matrices, where the
question and the section that answers it share almost no vocabulary.

End to end against all sixteen audited gaps that could be re-run, with the
model making the judgement:

    13 / 16 correct
    real gaps kept       9 / 9    (first run)   8 / 9  (second run)
    false gaps dropped   4 / 7    (first run)   5 / 7  (second run)

The direction of the errors matters more than the count. Keeping a gap that
should have been dropped leaves today's behaviour unchanged; dropping one
that should have been kept hides a real hole in the paper. Over two runs
there was one drop of a gap the audit called real - BatchNorm's weight
initialisation - and the quote it produced is genuine text from the paper:
*"the weights initialized to small random Gaussian values"*. That is stated
for the MNIST network and not for the ImageNet experiments the question was
about, so the audit and the checker disagree on a question that is genuinely
ambiguous rather than on a fact.

Two cautions on these numbers. The twenty-gap audit is itself a hand
judgement and not ground truth. And the two runs disagree with each other on
three gaps, which is the same non-determinism the extractor has, now visible
in a second place.

Prompt detail that moved the result most: hyperparameters live in tables, and
asking for *"the sentence that states it"* made the model reject a table row
as evidence. Allowing a row or an equation line fixed RoBERTa and both MAML
gaps in one change.
