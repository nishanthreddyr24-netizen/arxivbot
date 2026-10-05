"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";

import {
  ApiError,
  ask,
  generate,
  job as readJob,
  startExtract,
  type SpecResult,
} from "@/lib/api";

const STAGES: [string, string][] = [
  ["fetch", "fetching LaTeX source"],
  ["parse", "cutting into sections"],
  ["read", "reading each section"],
  ["consolidate", "merging and pruning"],
  ["gaps", "finding what the paper omits"],
];

type Phase = "idle" | "extracting" | "ready" | "error";

export function Tool({ paper, query }: { paper: string; query: string }) {
  const [phase, setPhase] = useState<Phase>(paper ? "extracting" : "idle");
  const [stage, setStage] = useState("fetch");
  const [done, setDone] = useState<string[]>([]);
  const [result, setResult] = useState<SpecResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [request, setRequest] = useState(query);
  const [code, setCode] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [suggestions, setSuggestions] = useState<string[] | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const codeRef = useRef<HTMLPreElement>(null);

  // ---- extraction ----
  useEffect(() => {
    if (!paper) return;
    let cancelled = false;

    (async () => {
      try {
        const started = await startExtract(paper);
        for (;;) {
          if (cancelled) return;
          const state = await readJob(started.job);
          setStage(state.stage);
          setDone(state.done ?? []);
          if (state.state === "error") throw new Error(state.error ?? "extraction failed");
          if (state.state === "done") {
            setResult(state as SpecResult);
            setPhase("ready");
            return;
          }
          await new Promise((resolve) => setTimeout(resolve, 1200));
        }
      } catch (cause) {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : String(cause));
        setPhase("error");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [paper]);

  // ---- generation ----
  const run = useCallback(
    async (text: string) => {
      if (!text.trim() || !result) return;
      setStreaming(true);
      setSuggestions(null);
      setCode("");
      setElapsed(0);

      const started = Date.now();
      const ticker = setInterval(
        () => setElapsed(Math.round((Date.now() - started) / 1000)),
        1000,
      );

      try {
        const reader = await generate(result.spec.arxiv_id, text);
        const decoder = new TextDecoder();
        let collected = "";
        for (;;) {
          const { value, done: finished } = await reader.read();
          if (finished) break;
          collected += decoder.decode(value, { stream: true });
          setCode(collected);
          codeRef.current?.scrollTo({ top: codeRef.current.scrollHeight });
        }
      } catch (cause) {
        if (cause instanceof ApiError && cause.detail && typeof cause.detail === "object") {
          const detail = cause.detail as { did_you_mean?: string[]; available?: string[] };
          setSuggestions([...(detail.did_you_mean ?? []), ...(detail.available ?? [])]);
          setCode("");
        } else {
          setCode(`# ${cause instanceof Error ? cause.message : String(cause)}`);
        }
      } finally {
        clearInterval(ticker);
        setStreaming(false);
      }
    },
    [result],
  );

  // Run the query carried from the homepage, once the spec is ready.
  const auto = useRef(false);
  useEffect(() => {
    if (phase === "ready" && query && !auto.current) {
      auto.current = true;
      void run(query);
    }
  }, [phase, query, run]);

  if (!paper) {
    return (
      <Empty>
        <p>No paper given.</p>
        <Link href="/" className="text-red-ink underline-offset-4 hover:underline">
          Start from the homepage
        </Link>
      </Empty>
    );
  }

  const gaps = (code.match(/TODO\(paper-silent\)/g) ?? []).length;

  return (
    <div className="grid min-h-[calc(100dvh-3rem)] lg:grid-cols-2">
      {/* ---- left: the paper ---- */}
      <section className="flex min-w-0 flex-col border-hairline lg:border-r">
        <PaneBar label="Paper">
          {result ? (
            <span className="truncate text-ink">{result.spec.title}</span>
          ) : null}
        </PaneBar>

        {phase === "extracting" ? (
          <div className="flex-1 p-6">
            <p className="font-mono text-[13px] text-muted">reading {paper}</p>
            <ol className="mt-5 max-w-[22rem] border-t border-hairline">
              {STAGES.map(([key, label]) => {
                const state = done.includes(key)
                  ? "done"
                  : stage === key
                    ? "now"
                    : "waiting";
                return (
                  <li
                    key={key}
                    className="flex gap-3 border-b border-hairline py-2.5 font-mono text-[12.5px]"
                  >
                    <span
                      className={
                        state === "done"
                          ? "text-red-ink"
                          : state === "now"
                            ? "animate-pulse text-red-ink"
                            : "text-faint"
                      }
                    >
                      {state === "done" ? "done" : state === "now" ? " >>" : "  ."}
                    </span>
                    <span className={state === "waiting" ? "text-faint" : "text-muted"}>
                      {label}
                    </span>
                  </li>
                );
              })}
            </ol>
            <p className="mt-5 font-mono text-[11.5px] text-faint">
              A paper not seen before takes a few minutes. One already extracted
              answers at once.
            </p>
          </div>
        ) : null}

        {phase === "error" ? (
          <div className="flex-1 p-6">
            <p className="border-l-2 border-red-ink py-1 pl-4 font-mono text-[13px] text-red-ink">
              {error}
            </p>
          </div>
        ) : null}

        {phase === "ready" && result ? <SpecView result={result} onPick={run} /> : null}
      </section>

      {/* ---- right: the code ---- */}
      <section className="flex min-w-0 flex-col">
        <PaneBar label="Skeleton">
          {result ? (
            <span className="truncate text-faint">{result.spec.extractor.model}</span>
          ) : null}
        </PaneBar>

        <form
          className="flex gap-2 border-b border-hairline p-3"
          onSubmit={(event) => {
            event.preventDefault();
            void run(request);
          }}
        >
          <input
            id="request"
            value={request}
            onChange={(event) => setRequest(event.target.value)}
            disabled={phase !== "ready" || streaming}
            placeholder="the attention block"
            className="min-w-0 flex-1 bg-ground px-3 py-2.5 font-mono text-[13px] text-ink placeholder:text-faint focus:outline-none disabled:opacity-50"
            autoComplete="off"
            spellCheck={false}
          />
          <button
            type="submit"
            disabled={phase !== "ready" || streaming || !request.trim()}
            className="shrink-0 bg-red-fill px-4 font-mono text-[12.5px] font-bold text-white disabled:opacity-40"
          >
            {streaming ? "…" : "Go"}
          </button>
        </form>

        {suggestions ? (
          <div className="p-6">
            <p className="text-[15px] text-ink">
              No component in the extracted spec matches that.
            </p>
            <p className="mt-1 font-mono text-[12px] text-faint">Did you mean:</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {suggestions.map((name) => (
                <button
                  key={name}
                  type="button"
                  onClick={() => {
                    setRequest(name);
                    void run(name);
                  }}
                  className="border border-hairline px-3 py-1.5 font-mono text-[12px] text-muted hover:border-red-ink hover:text-red-ink"
                >
                  {name}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {!suggestions && code ? (
          <>
            <pre
              ref={codeRef}
              className="tool-code min-w-0 flex-1 overflow-auto whitespace-pre-wrap break-words p-5 font-mono text-[12.5px] leading-[1.7] text-ink"
            >
              {highlight(code)}
              {streaming ? (
                <span className="ml-0.5 inline-block h-[1em] w-[0.5em] translate-y-[0.16em] animate-pulse bg-red-ink" />
              ) : null}
            </pre>
            <div className="flex items-center gap-4 border-t border-hairline px-5 py-2.5 font-mono text-[11.5px] text-faint">
              <span>
                {streaming
                  ? `generating · ${elapsed}s`
                  : gaps === 0
                    ? "nothing left undecided"
                    : `${gaps} decision${gaps === 1 ? "" : "s"} left to you`}
              </span>
              <button
                type="button"
                className="ml-auto hover:text-red-ink"
                onClick={() => navigator.clipboard?.writeText(code).catch(() => {})}
              >
                Copy
              </button>
            </div>
          </>
        ) : null}

        {!suggestions && !code && phase === "ready" ? (
          <Empty>
            <p className="font-mono text-[13px]">Ask for a part of this paper.</p>
          </Empty>
        ) : null}
      </section>
    </div>
  );
}

function PaneBar({ label, children }: { label: string; children?: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 border-b border-hairline px-5 py-2.5">
      <span className="font-mono text-[11px] uppercase tracking-[0.11em] text-faint">
        {label}
      </span>
      <div className="ml-auto min-w-0 font-mono text-[12px]">{children}</div>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-2 p-10 text-center text-muted">
      {children}
    </div>
  );
}

function SpecView({
  result,
  onPick,
}: {
  result: SpecResult;
  onPick: (query: string) => void;
}) {
  const { spec, counts, report } = result;
  return (
    <div className="min-w-0 flex-1 overflow-y-auto p-6">
      <div className="flex flex-wrap gap-x-5 gap-y-1 border-b border-hairline pb-4 font-mono text-[11.5px] text-faint">
        <span className="text-red-ink">{spec.arxiv_id}</span>
        <span>
          <b className="font-bold text-ink">{counts.stated ?? 0}</b> verified
        </span>
        <span>
          <b className="font-bold text-ink">{spec.unknowns.length}</b> gaps
        </span>
        {report.quote_accuracy != null ? (
          <span>
            <b className="font-bold text-ink">
              {Math.round(report.quote_accuracy * 100)}%
            </b>{" "}
            quotes located
          </span>
        ) : null}
      </div>

      <p className="mt-5 font-mono text-[11px] uppercase tracking-[0.11em] text-muted">
        Components
      </p>
      <ul className="mt-2">
        {spec.components.map((component) => (
          <li key={component.name} className="border-b border-hairline py-3">
            <button
              type="button"
              onClick={() => onPick(component.name)}
              className="group w-full text-left"
            >
              <span className="text-[15px] text-ink underline-offset-4 group-hover:underline">
                {component.name}
              </span>
              <span className="mt-0.5 block text-[13.5px] leading-snug text-muted">
                {component.role}
              </span>
              {component.hyperparameters?.length ? (
                <span className="mt-1.5 flex flex-wrap gap-x-4 font-mono text-[11.5px]">
                  {component.hyperparameters.slice(0, 4).map((hp) => (
                    <span key={hp.name} className="text-faint">
                      <span
                        className={
                          hp.confidence === "stated" ? "text-red-ink" : "text-faint"
                        }
                      >
                        {hp.confidence === "stated" ? "*" : "?"}
                      </span>{" "}
                      {hp.name} = {hp.value ?? "—"}
                    </span>
                  ))}
                </span>
              ) : null}
            </button>
          </li>
        ))}
      </ul>

      {spec.unknowns.length ? (
        <>
          <p className="mt-7 font-mono text-[11px] uppercase tracking-[0.11em] text-muted">
            The paper does not specify
          </p>
          <ul className="mt-2">
            {spec.unknowns.map((unknown) => (
              <li
                key={unknown.question}
                className="flex gap-4 border-b border-hairline py-3"
              >
                <span
                  className={[
                    "shrink-0 font-mono text-[10.5px] uppercase tracking-[0.09em]",
                    unknown.severity === "minor" ? "text-faint" : "text-red-ink",
                  ].join(" ")}
                >
                  {unknown.severity}
                </span>
                <span className="text-[14px] leading-snug text-muted">
                  {unknown.question}
                </span>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}

/**
 * Lightweight highlighting for the streaming panel.
 *
 * Shiki renders the homepage at build time, but re-running a full highlighter
 * on every chunk of a stream is wasted work. The markers that matter here are
 * the honesty ones: a gap, a deviation, a citation.
 */
function highlight(source: string) {
  return source.split("\n").map((line, index) => {
    const key = `${index}-${line.slice(0, 12)}`;
    if (/TODO\(paper-silent\)/.test(line)) {
      return (
        <span key={key} className="block bg-red-dim/25 font-medium text-red-ink">
          {line || " "}
        </span>
      );
    }
    if (/^#\s*!!/.test(line)) {
      return (
        <span key={key} className="block bg-red-dim/25 font-bold text-red-ink">
          {line || " "}
        </span>
      );
    }
    if (/^\s*#/.test(line) || /^\s*"""/.test(line)) {
      return (
        <span key={key} className="block text-faint">
          {line || " "}
        </span>
      );
    }
    return (
      <span key={key} className="block">
        {line || " "}
      </span>
    );
  });
}
