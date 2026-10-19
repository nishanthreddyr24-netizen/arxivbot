"use client";

/** The pipeline's own stage keys, with what each one is actually doing. */
const STAGES: [string, string, string][] = [
  ["fetch", "fetch", "the authors' LaTeX source from arXiv"],
  ["parse", "parse", "cutting the source into sections"],
  ["read", "read", "every section, one at a time"],
  ["consolidate", "merge", "one spec from many readings"],
  ["gaps", "gaps", "what the paper leaves undecided"],
];

export function Progress({
  paper,
  stage,
  done,
  seconds,
}: {
  paper: string;
  stage: string;
  done: string[];
  seconds: number;
}) {
  // The service does not report every stage it passes through — it jumps from
  // `read` to `gaps` without ever naming `consolidate` — so a row left to the
  // `done` list alone sits dim for the whole run and reads as stuck. Anything
  // the current stage is past is finished, whether or not it was announced.
  const at = STAGES.findIndex(([key]) => key === stage);
  const reached = (index: number, key: string) =>
    done.includes(key) || (at >= 0 && index < at);
  const finished = STAGES.filter(([key], i) => reached(i, key)).length;

  return (
    <div className="flex-1 px-4 py-6 sm:px-6">
      <p className="font-mono text-2xs uppercase tracking-[0.11em] text-muted">
        Reading {paper}
      </p>

      <ol className="mt-4 max-w-[30rem] border-t border-hairline">
        {STAGES.map(([key, name, what], index) => {
          const state = reached(index, key) ? "done" : stage === key ? "now" : "waiting";
          return (
            <li
              key={key}
              className="flex items-baseline gap-3 border-b border-hairline py-2.5"
            >
              <span
                aria-hidden
                className={[
                  "size-[7px] shrink-0 translate-y-[-1px]",
                  state === "done"
                    ? "bg-red-fill"
                    : state === "now"
                      ? "animate-pulse bg-red-ink"
                      : "border border-hairline-strong",
                ].join(" ")}
              />
              <span
                className={[
                  "w-[4.5rem] shrink-0 font-mono text-xs",
                  state === "waiting" ? "text-faint" : "text-ink",
                ].join(" ")}
              >
                {name}
              </span>
              <span
                className={[
                  "min-w-0 text-sm",
                  state === "waiting" ? "text-faint" : "text-muted",
                ].join(" ")}
              >
                {what}
              </span>
              <span className="ml-auto shrink-0 font-mono text-2xs text-faint">
                {state === "done" ? "ok" : state === "now" ? "···" : ""}
              </span>
            </li>
          );
        })}
      </ol>

      <p className="mt-4 font-mono text-2xs text-faint">
        {finished}/{STAGES.length} · {seconds}s
      </p>

      {/* The warning about minutes is itself alarming, so it waits until the
          run has actually been slow rather than greeting every paper. */}
      {seconds > 8 ? (
        <p className="mt-4 max-w-[46ch] text-sm text-muted">
          A new paper is read section by section against the model, which takes
          a few minutes. The result is stored, so the next visit is immediate.
        </p>
      ) : null}
    </div>
  );
}
