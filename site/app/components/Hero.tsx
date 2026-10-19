"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { SILENCES } from "./content";

/** Sample requests, filled into the form on click. Real papers, real sections. */
const EXAMPLES = [
  // `label` is what the chip shows; `query` is what goes into the field. The
  // full queries are 30-40 characters each and wrapped the row three deep.
  { paper: "2310.06825", label: "SWA block", query: "the sliding-window attention block" },
  { paper: "2106.09685", label: "LoRA injection", query: "the low-rank adapter injection" },
  { paper: "2006.11239", label: "noise schedule", query: "the noise schedule and the reverse step" },
];

/** Accepts a bare id, an abs/pdf URL, or a versioned id. */
function parsePaperId(raw: string): string | null {
  const match = raw
    .trim()
    .match(/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?\/\d{7}(?:v\d+)?)/);
  return match ? match[1] : null;
}

export function Hero() {
  const router = useRouter();
  const paperRef = useRef<HTMLInputElement>(null);
  const queryRef = useRef<HTMLInputElement>(null);

  const [paper, setPaper] = useState("");
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  // "/" jumps to the paper field, the way a listing page does. Ignored while
  // the caller is already typing somewhere.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const tag = (event.target as HTMLElement | null)?.tagName;
      const typing = tag === "INPUT" || tag === "TEXTAREA";
      if (event.key === "/" && !typing) {
        event.preventDefault();
        paperRef.current?.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function go() {
    const id = parsePaperId(paper);
    if (!id) {
      setError(
        paper.trim()
          ? "that does not look like an arXiv id"
          : "enter an arXiv id or URL",
      );
      paperRef.current?.focus();
      return;
    }
    setError(null);
    const params = new URLSearchParams({ paper: id, q: query.trim() });
    router.push(`/app?${params.toString()}`);
  }

  function onFormKeyDown(event: React.KeyboardEvent) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      go();
    }
  }

  const field =
    "w-full min-w-0 bg-transparent px-3 py-3 font-mono text-sm text-ink " +
    "placeholder:text-faint focus:outline-none";

  // Below `sm` the label sits above the field. Side by side it was a fixed
  // 80px column, and "component" at 12px mono is wider than that: on a 360px
  // phone the longer of the two labels was clipped mid-word.
  const label =
    "flex items-center border-hairline px-3 pt-3 pb-1 font-mono text-xs " +
    "text-muted sm:w-[6.5rem] sm:shrink-0 sm:border-r sm:py-0 sm:pb-0";

  return (
    <section className="hero-aura mx-auto max-w-[1240px] px-4 pb-12 pt-8 sm:px-6 lg:pt-12">
      <div className="grid gap-12 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] lg:gap-14">
        <div className="min-w-0">
          <p className="font-mono text-2xs uppercase tracking-[0.14em] text-faint">
            LaTeX source
            <span className="px-2 text-red-ink" aria-hidden>
              &rarr;
            </span>
            verified spec
            <span className="px-2 text-red-ink" aria-hidden>
              &rarr;
            </span>
            skeleton
          </p>

          <h1 className="display mt-4 max-w-[16ch] font-normal text-ink">
            Implement the paper, not your memory of it.
          </h1>

          <p className="mt-5 max-w-[54ch] text-lg text-muted">
            Paste an arXiv id. This reads the authors&rsquo; own LaTeX
            &mdash; not the PDF, not a summary &mdash; and returns every
            component, hyperparameter and equation{" "}
            <span className="text-ink">
              with the verbatim sentence that supports it
            </span>
            .
          </p>

          <form
            className="mt-7 max-w-[760px] border border-hairline bg-surface"
            onSubmit={(event) => {
              event.preventDefault();
              go();
            }}
            onKeyDown={onFormKeyDown}
          >
            <div className="flex flex-col border-b border-hairline sm:flex-row sm:items-stretch">
              <label htmlFor="paper" className={label}>
                paper
              </label>
              <input
                id="paper"
                ref={paperRef}
                value={paper}
                onChange={(event) => {
                  setPaper(event.target.value);
                  if (error) setError(null);
                }}
                className={field}
                placeholder="2310.06825 or arxiv.org/abs/..."
                autoComplete="off"
                spellCheck={false}
                aria-invalid={Boolean(error)}
                aria-describedby={error ? "paper-error" : undefined}
              />
            </div>

            <div className="flex flex-col sm:flex-row sm:items-stretch">
              <label htmlFor="component" className={label}>
                component
              </label>
              <input
                id="component"
                ref={queryRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                className={field}
                placeholder="the sliding-window attention block"
                autoComplete="off"
                spellCheck={false}
              />
            </div>

            <div className="flex border-t border-hairline p-2">
              <button
                type="submit"
                className="w-full bg-red-fill px-5 py-3 font-mono text-xs font-bold text-white transition-opacity hover:opacity-90 active:opacity-80 sm:ml-auto sm:w-auto"
              >
                Extract <span aria-hidden>&crarr;</span>
              </button>
            </div>
          </form>

          {error ? (
            <p
              id="paper-error"
              role="alert"
              className="mt-3 font-mono text-sm text-red-ink"
            >
              {error}
            </p>
          ) : null}

          {/* Examples were a three-row list under its own rubric; as chips they
              say the same thing in a quarter of the height, which is what gets
              the worked example below into the first screenful. */}
          <div id="examples" className="mt-5 flex flex-wrap items-center gap-2">
            <span className="font-mono text-2xs text-faint">or try</span>
            {EXAMPLES.map((example) => (
              <button
                key={example.paper}
                type="button"
                onClick={() => {
                  setPaper(example.paper);
                  setQuery(example.query);
                  setError(null);
                  queryRef.current?.focus();
                }}
                className="border border-hairline px-2.5 py-2 font-mono text-2xs text-muted transition-colors hover:border-hairline-strong hover:text-red-ink"
              >
                {example.paper}
                <span className="pl-2 text-faint">{example.label}</span>
              </button>
            ))}
            {/* Trails the chips rather than being pushed right: at 760px
                `ml-auto` sent it onto a row of its own. */}
            <span className="hidden pl-2 font-mono text-2xs text-faint lg:inline">
              <kbd className="text-muted">/</kbd> focus &middot;{" "}
              <kbd className="text-muted">&#8984;&crarr;</kbd> run
            </span>
          </div>
        </div>

        <Silence />
      </div>
    </section>
  );
}

/**
 * The half of the product that nothing else does, stated before the demo.
 *
 * This slot held a legend defining three marks the visitor had not met yet. A
 * legend explains the notation; this makes the argument. Every item is a
 * question a model that has read the paper answers instantly, and the paper
 * answers none of them.
 */
function Silence() {
  return (
    <aside className="min-w-0 self-start border border-hairline bg-surface-raised">
      <div className="flex items-baseline gap-3 border-b border-hairline-strong px-4 py-3 sm:px-5">
        <h2 className="font-mono text-2xs uppercase tracking-[0.12em] text-ink">
          What the paper never says
        </h2>
        <span className="ml-auto shrink-0 font-mono text-2xs text-faint">
          2310.06825 &sect;SWA
        </span>
      </div>

      <dl>
        {SILENCES.map((item) => (
          <div
            key={item.subject}
            className="flex gap-3 border-b border-hairline px-4 py-3.5 sm:px-5"
          >
            {/* The tool's own mark for a gap, so the notation is learned by
                meeting it rather than by being defined. */}
            <span aria-hidden className="mt-[0.55em] flex shrink-0 gap-[2px]">
              <span className="size-[6px] bg-red-fill" />
              <span className="size-[6px] bg-red-fill" />
              <span className="size-[6px] bg-red-fill" />
            </span>
            <div className="min-w-0">
              <dt className="font-mono text-xs text-ink">{item.subject}</dt>
              <dd className="mt-1 text-sm text-muted">{item.body}</dd>
            </div>
          </div>
        ))}
      </dl>

      <p className="bg-sunken px-4 py-3 font-mono text-2xs leading-[1.6] text-faint sm:px-5">
        Each is reported as a gap, and lands in the generated file as{" "}
        <span className="text-red-ink">TODO(paper-silent)</span>. A model that
        has read this paper will answer all three without hesitating.
      </p>
    </aside>
  );
}
