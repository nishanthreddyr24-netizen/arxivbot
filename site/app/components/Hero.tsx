"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

/** Sample requests, filled into the form on click. Real papers, real sections. */
const EXAMPLES = [
  { paper: "2310.06825", query: "the sliding-window attention block" },
  { paper: "2106.09685", query: "the low-rank adapter injection" },
  { paper: "2006.11239", query: "the noise schedule and the reverse step" },
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
    "w-full bg-transparent px-3 py-3 font-mono text-sm text-ink " +
    "placeholder:text-faint focus:outline-none";

  return (
    <section className="mx-auto max-w-[1120px] px-4 pb-12 pt-8 sm:px-6 sm:pt-10">
      <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_17rem] lg:gap-12">
        <div className="min-w-0">
      <h1 className="max-w-[22ch] text-[clamp(1.9rem,4.6vw,2.75rem)] font-normal leading-[1.12] tracking-[-0.02em] text-ink">
        Implement the paper, not your memory of it.
      </h1>
      <p className="mt-5 max-w-[58ch] text-lg text-muted">
        Paste an arXiv id. This reads the authors&rsquo; own LaTeX source and
        returns the components, hyperparameters and equations with the
        sentence each one came from &mdash; plus a list of everything the paper
        never decided.
      </p>

      <form
        className="mt-8 max-w-[760px] border border-hairline bg-surface"
        onSubmit={(event) => {
          event.preventDefault();
          go();
        }}
        onKeyDown={onFormKeyDown}
      >
        <div className="flex items-stretch border-b border-hairline">
          <label
            htmlFor="paper"
            className="flex w-[5rem] shrink-0 sm:w-[6.5rem] items-center border-r border-hairline px-3 font-mono text-xs text-muted"
          >
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

        <div className="flex items-stretch">
          <label
            htmlFor="component"
            className="flex w-[5rem] shrink-0 sm:w-[6.5rem] items-center border-r border-hairline px-3 font-mono text-xs text-muted"
          >
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
          <button
            type="submit"
            className="m-2 shrink-0 bg-red-fill px-5 font-mono text-xs font-bold text-white transition-opacity hover:opacity-90 active:opacity-80"
          >
            Go <span aria-hidden>&crarr;</span>
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

      <p className="mt-4 font-mono text-2xs text-faint">
        <kbd className="text-muted">/</kbd> to focus &middot;{" "}
        <kbd className="text-muted">&#8984;&crarr;</kbd> to run
      </p>

      <div id="examples" className="mt-8 max-w-[760px]">
        <p className="border-b border-hairline-strong pb-2 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
          Or start from one of these
        </p>
        <ul>
          {EXAMPLES.map((example) => (
            <li key={example.paper} className="border-b border-hairline">
              <button
                type="button"
                onClick={() => {
                  setPaper(example.paper);
                  setQuery(example.query);
                  setError(null);
                  queryRef.current?.focus();
                }}
                className="group flex w-full items-baseline gap-4 py-2.5 text-left"
              >
                <span className="w-[5.5rem] shrink-0 font-mono text-xs text-red-ink">
                  {example.paper}
                </span>
                <span className="min-w-0 font-mono text-xs text-muted transition-colors group-hover:text-ink">
                  {example.query}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
        </div>

        <Legend />
      </div>
    </section>
  );
}

/**
 * The notation the tool uses, stated once before anyone meets it. The hero
 * was two-thirds empty, and a legend is the one thing that belongs there:
 * it is the product's whole vocabulary in three rows.
 */
function Legend() {
  const rows: [React.ReactNode, string, string][] = [
    [
      <span key="m" className="mt-[0.4em] size-[7px] shrink-0 bg-red-fill" />,
      "quoted",
      "a verbatim sentence from the source backs this value",
    ],
    [
      <span key="m" className="mt-[0.4em] size-[7px] shrink-0 border border-faint" />,
      "unquoted",
      "asserted, but no sentence could be located for it",
    ],
    [
      <span key="m" aria-hidden className="mt-[0.4em] flex shrink-0 gap-[2px]">
        <span className="size-[7px] bg-red-fill" />
        <span className="size-[7px] bg-red-fill" />
        <span className="size-[7px] bg-red-fill" />
      </span>,
      "gap",
      "the paper never decides this; you have to",
    ],
  ];

  return (
    <aside className="lg:border-l lg:border-hairline lg:pl-6">
      <p className="border-b border-hairline-strong pb-2 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
        What comes back
      </p>
      <dl>
        {rows.map(([mark, term, body]) => (
          <div key={term} className="flex gap-2.5 border-b border-hairline py-3">
            <span aria-hidden className="flex w-[27px] shrink-0 justify-start">
              {mark}
            </span>
            <div className="min-w-0">
              <dt className="font-mono text-xs text-ink">{term}</dt>
              <dd className="mt-0.5 text-sm text-muted">{body}</dd>
            </div>
          </div>
        ))}
      </dl>
    </aside>
  );
}
