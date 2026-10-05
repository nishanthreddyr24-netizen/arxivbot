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
    "w-full bg-transparent px-3 py-3 font-mono text-[14px] text-ink " +
    "placeholder:text-faint focus:outline-none";

  return (
    <section className="mx-auto max-w-[1120px] px-4 pb-14 pt-12 sm:px-6 sm:pt-16">
      <h1 className="max-w-[20ch] text-[clamp(2rem,5.5vw,3.25rem)] font-normal leading-[1.14] tracking-[-0.02em] text-ink">
        Paste a paper. Ask for a component. Get the code.
      </h1>
      <p className="mt-5 max-w-[48ch] font-mono text-[14px] leading-[1.7] text-muted">
        Every snippet links back to the paragraph or equation it came from.
      </p>

      <form
        className="mt-10 max-w-[760px] border border-hairline bg-surface"
        onSubmit={(event) => {
          event.preventDefault();
          go();
        }}
        onKeyDown={onFormKeyDown}
      >
        <div className="flex items-stretch border-b border-hairline">
          <label
            htmlFor="paper"
            className="flex w-[84px] shrink-0 items-center border-r border-hairline px-3 font-mono text-[12px] text-muted"
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
            className="flex w-[84px] shrink-0 items-center border-r border-hairline px-3 font-mono text-[12px] text-muted"
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
            className="m-2 shrink-0 bg-red-fill px-5 font-mono text-[13px] font-bold text-white transition-opacity hover:opacity-90 active:opacity-80"
          >
            Go <span aria-hidden>&crarr;</span>
          </button>
        </div>
      </form>

      {error ? (
        <p
          id="paper-error"
          role="alert"
          className="mt-3 font-mono text-[13px] text-red-ink"
        >
          {error}
        </p>
      ) : null}

      <p className="mt-5 font-mono text-[12px] text-faint">
        <kbd className="text-muted">/</kbd> to focus &middot;{" "}
        <kbd className="text-muted">&#8984;&crarr;</kbd> to run
      </p>

      <ul id="examples" className="mt-10 max-w-[760px] border-t border-hairline">
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
              className="group flex w-full items-baseline gap-4 py-3 text-left"
            >
              <span className="font-mono text-[13px] text-red-ink">
                {example.paper}
              </span>
              <span className="font-mono text-[13px] text-muted transition-colors group-hover:text-ink">
                {example.query}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
