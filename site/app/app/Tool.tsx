"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";

import {
  ApiError,
  generate,
  job as readJob,
  startExtract,
  type SpecResult,
} from "@/lib/api";
import { CodePane, usedComponents } from "./CodePane";
import { Progress } from "./Progress";
import { SourcePane } from "./SourcePane";
import { SpecPane } from "./SpecPane";
import type { Claim } from "./claims";

type Phase = "idle" | "extracting" | "ready" | "error";

// Generous: a long paper read for the first time can take minutes, and the
// hosted function is allowed 300 seconds. This only has to catch a service
// that has stopped answering, not a slow one.
const EXTRACT_TIMEOUT_MS = 6 * 60 * 1000;
/** Which of the three views the phone is showing. */
type View = "spec" | "source" | "skeleton";

export function Tool({ paper, query }: { paper: string; query: string }) {
  const [phase, setPhase] = useState<Phase>(paper ? "extracting" : "idle");
  const [stage, setStage] = useState("fetch");
  const [done, setDone] = useState<string[]>([]);
  // null until the POST says whether this paper was read before: the ladder
  // must not flash for a cached paper.
  const [stored, setStored] = useState<boolean | null>(null);
  const [result, setResult] = useState<SpecResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [request, setRequest] = useState(query);
  const [code, setCode] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [suggestions, setSuggestions] = useState<string[] | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [reading, setReading] = useState(0);

  const [view, setView] = useState<View>("spec");
  const [right, setRight] = useState<"skeleton" | "source">("skeleton");
  const [focused, setFocused] = useState<string | null>(null);

  // ---- extraction ----
  useEffect(() => {
    if (!paper) return;
    let cancelled = false;

    const started = Date.now();
    const ticker = setInterval(
      () => setReading(Math.round((Date.now() - started) / 1000)),
      1000,
    );

    (async () => {
      try {
        const begun = await startExtract(paper);
        // A paper read before returns instantly. Running the five-stage ladder
        // for it showed 0/5 and "this takes a few minutes" for the half second
        // before the spec appeared — the one state that looked like a hang.
        if (!cancelled) setStored(begun.cached);
        for (;;) {
          if (cancelled) return;
          // A server that answers "still running" forever is indistinguishable
          // from a hung page, and the page is the one the reader blames. An
          // error that names the wait is worth more than a spinner that never
          // resolves - this is the same reason the extractor reports gaps.
          if (Date.now() - started > EXTRACT_TIMEOUT_MS) {
            throw new Error(
              `extraction has been running for ${Math.round(
                EXTRACT_TIMEOUT_MS / 60000,
              )} minutes without finishing - the service may be stuck. ` +
                "Reload to try again.",
            );
          }
          const state = await readJob(begun.job);
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
      clearInterval(ticker);
    };
  }, [paper]);

  // ---- generation ----
  const run = useCallback(
    async (text: string) => {
      if (!text.trim() || !result) return;
      setRequest(text);
      setStreaming(true);
      setSuggestions(null);
      setCode("");
      setElapsed(0);
      setRight("skeleton");
      setView("skeleton");

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

  const locate = useCallback((claim: Claim) => {
    setFocused(claim.id);
    setRight("source");
    setView("source");
  }, []);

  const used = useMemo(() => usedComponents(code), [code]);
  const names = result?.spec.components.map((c) => c.name) ?? [];

  if (!paper) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 px-4 py-16 text-center sm:px-6">
        <p className="text-base">No paper given.</p>
        <Link
          href="/"
          className="font-mono text-sm text-red-ink underline-offset-4 hover:underline"
        >
          Start from the homepage
        </Link>
      </div>
    );
  }

  return (
    <>
      {/* A two-pane tool does not fit a phone, so on a phone it is not two
          panes: it is three views and a switch. Each one gets the full width
          it needs, which is what the quotes and the code both want. */}
      <nav className="sticky top-0 z-30 flex h-10 shrink-0 border-b border-hairline bg-ground lg:hidden">
        {(
          [
            ["spec", "Spec"],
            ["source", "Source"],
            ["skeleton", "Skeleton"],
          ] as [View, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => {
              setView(key);
              if (key !== "spec") setRight(key);
            }}
            aria-current={view === key}
            className={[
              "min-w-0 flex-1 border-r border-hairline font-mono text-2xs uppercase tracking-[0.11em] last:border-r-0",
              view === key
                ? "border-b-2 border-b-red-fill bg-surface text-ink"
                : "text-muted",
            ].join(" ")}
          >
            {label}
          </button>
        ))}
      </nav>

      <div className="flex h-[calc(100dvh-5.5rem)] min-h-0 flex-col lg:grid lg:h-[calc(100dvh-3rem)] lg:grid-cols-2">
        {/* ---- left: what the paper says ---- */}
        <section
          className={[
            view === "spec" ? "flex" : "hidden",
            "min-h-0 min-w-0 flex-1 flex-col lg:flex lg:h-full lg:overflow-hidden lg:border-r lg:border-hairline",
          ].join(" ")}
        >
          <PaneBar label="Spec">
            {result ? (
              <span className="text-faint">
                {result.spec.components.length} components
              </span>
            ) : (
              <span className="text-faint">{paper}</span>
            )}
          </PaneBar>

          {phase === "extracting" ? (
            stored === false || (stored === null && reading > 2) ? (
              <Progress paper={paper} stage={stage} done={done} seconds={reading} />
            ) : (
              <p className="px-4 py-6 font-mono text-2xs text-muted sm:px-6">
                {stored === null
                  ? "asking whether this paper has been read"
                  : "read before · opening the stored extraction"}
              </p>
            )
          ) : null}

          {phase === "error" ? (
            <div className="flex-1 px-4 py-6 sm:px-6">
              <p className="font-mono text-2xs uppercase tracking-[0.11em] text-red-ink">
                Extraction failed
              </p>
              <p className="mt-3 border-l-2 border-red-fill pl-4 font-mono text-sm text-ink">
                {error}
              </p>
              <Link
                href="/"
                className="mt-5 inline-block font-mono text-xs text-muted underline-offset-4 hover:text-red-ink hover:underline"
              >
                Try another paper
              </Link>
            </div>
          ) : null}

          {phase === "ready" && result ? (
            <div className="min-w-0 flex-1 overflow-y-auto">
              <SpecPane
                result={result}
                onPick={run}
                onLocate={locate}
                selected={request}
                used={used}
                focusedClaim={focused}
              />
            </div>
          ) : null}
        </section>

        {/* ---- right: the paper's own words, or the code ---- */}
        <section
          className={[
            view === "spec" ? "hidden" : "flex",
            "min-h-0 min-w-0 flex-1 flex-col lg:flex lg:h-full lg:overflow-hidden",
          ].join(" ")}
        >
          <div className="hidden shrink-0 items-center border-b border-hairline lg:flex">
            {(
              [
                ["skeleton", "Skeleton"],
                ["source", "Source"],
              ] as ["skeleton" | "source", string][]
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setRight(key)}
                disabled={!result}
                aria-current={right === key}
                className={[
                  "border-r border-hairline px-4 py-2.5 font-mono text-2xs uppercase tracking-[0.11em] disabled:opacity-40 sm:px-6",
                  right === key
                    ? "border-b-2 border-b-red-fill text-ink"
                    : "text-faint hover:text-muted",
                ].join(" ")}
              >
                {label}
              </button>
            ))}
            <span className="ml-auto min-w-0 truncate px-4 font-mono text-2xs text-faint sm:px-6">
              {result?.spec.extractor.model}
            </span>
          </div>

          {right === "source" && result ? (
            <SourcePane spec={result.spec} focused={focused} onPick={run} />
          ) : null}

          {right === "skeleton" ? (
            <>
              <form
                className="flex shrink-0 items-stretch border-b border-hairline"
                onSubmit={(event) => {
                  event.preventDefault();
                  void run(request);
                }}
              >
                <label
                  htmlFor="request"
                  className="flex shrink-0 items-center border-r border-hairline px-3 font-mono text-2xs uppercase tracking-[0.09em] text-faint"
                >
                  build
                </label>
                <input
                  id="request"
                  value={request}
                  onChange={(event) => setRequest(event.target.value)}
                  disabled={phase !== "ready" || streaming}
                  placeholder={
                    phase === "ready" ? "a part of this paper" : "reading the paper…"
                  }
                  className="min-w-0 flex-1 bg-transparent px-3 py-3 font-mono text-sm text-ink placeholder:text-faint focus:outline-none disabled:opacity-50"
                  autoComplete="off"
                  spellCheck={false}
                />
                {/* `white` rather than `ink` only because no token names the
                    text colour that belongs on top of `red-fill`: ink on it is
                    3.8:1, white is 4.8:1. */}
                <button
                  type="submit"
                  disabled={phase !== "ready" || streaming || !request.trim()}
                  className="m-2 shrink-0 bg-red-fill px-5 font-mono text-xs font-bold text-white transition-opacity hover:opacity-90 disabled:opacity-30"
                >
                  {streaming ? "···" : "Go"}
                </button>
              </form>

              {suggestions ? (
                <Picker
                  title="Nothing in the extracted spec matches that."
                  names={suggestions}
                  onPick={run}
                />
              ) : null}

              {!suggestions && code ? (
                <CodePane code={code} streaming={streaming} elapsed={elapsed} />
              ) : null}

              {/* Idle, the panel used to be a paragraph. The components are
                  the actual answer to "what can I ask for", so they are here
                  as well as on the left. */}
              {!suggestions && !code && phase === "ready" ? (
                <Picker
                  title="Pick a part of this paper to write a skeleton for."
                  names={names}
                  onPick={run}
                  note="Every value it uses cites the section it came from, and each point the paper left undecided arrives as a marked TODO rather than a guess."
                />
              ) : null}
            </>
          ) : null}
        </section>
      </div>
    </>
  );
}

function PaneBar({ label, children }: { label: string; children?: React.ReactNode }) {
  return (
    <div className="flex shrink-0 items-center gap-3 border-b border-hairline px-4 py-2.5 sm:px-6">
      <span className="font-mono text-2xs uppercase tracking-[0.11em] text-faint">
        {label}
      </span>
      <div className="ml-auto min-w-0 font-mono text-2xs">{children}</div>
    </div>
  );
}

/** Named components as targets, not as prose. */
function Picker({
  title,
  names,
  note,
  onPick,
}: {
  title: string;
  names: string[];
  note?: string;
  onPick: (name: string) => void;
}) {
  return (
    <div className="min-w-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6">
      <p className="max-w-[52ch] text-sm text-ink">{title}</p>
      <div className="mt-4 flex flex-wrap gap-2">
        {names.map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => onPick(name)}
            className="border border-hairline-strong px-3 py-2 text-left font-mono text-xs text-muted hover:border-red-ink hover:text-red-ink"
          >
            {name}
          </button>
        ))}
      </div>
      {note ? <p className="mt-5 max-w-[52ch] text-sm text-muted">{note}</p> : null}
    </div>
  );
}
