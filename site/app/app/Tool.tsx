"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";

import {
  ApiError,
  generate,
  job as readJob,
  startExtract,
  type Component,
  type Hyperparameter,
  type Provenance,
  type SpecResult,
  type Unknown,
} from "@/lib/api";
import { TeX } from "./TeX";

/** The pipeline's own stage keys, with what each one is actually doing. */
const STAGES: [string, string, string][] = [
  ["fetch", "fetch", "the authors' LaTeX source from arXiv"],
  ["parse", "parse", "cutting the source into sections"],
  ["read", "read", "every section, one at a time"],
  ["consolidate", "merge", "one spec from many readings"],
  ["gaps", "gaps", "what the paper leaves undecided"],
];

type Phase = "idle" | "extracting" | "ready" | "error";

/** A claim is verified exactly when the service located a verbatim quote. */
const verified = (p?: Provenance | null) => Boolean(p?.quote);

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
  const [reading, setReading] = useState(0);
  const codeRef = useRef<HTMLPreElement>(null);

  // ---- extraction ----
  useEffect(() => {
    if (!paper) return;
    let cancelled = false;

    // An uncached paper takes minutes. Without a clock on screen the page
    // looks hung, so the wait is counted out loud.
    const started = Date.now();
    const ticker = setInterval(
      () => setReading(Math.round((Date.now() - started) / 1000)),
      1000,
    );

    (async () => {
      try {
        const begun = await startExtract(paper);
        for (;;) {
          if (cancelled) return;
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
        <p className="text-base">No paper given.</p>
        <Link
          href="/"
          className="font-mono text-sm text-red-ink underline-offset-4 hover:underline"
        >
          Start from the homepage
        </Link>
      </Empty>
    );
  }

  const gaps = (code.match(/TODO\(paper-silent\)/g) ?? []).length;

  return (
    <div className="lg:grid lg:h-[calc(100dvh-3rem)] lg:grid-cols-2">
      {/* ---- left: what the paper says ---- */}
      <section className="flex min-w-0 flex-col border-b border-hairline lg:h-full lg:overflow-hidden lg:border-b-0 lg:border-r">
        <PaneBar label="Source">
          {result ? (
            <span className="truncate text-ink">{result.spec.title}</span>
          ) : (
            <span className="text-faint">{paper}</span>
          )}
        </PaneBar>

        {phase === "extracting" ? (
          <Progress paper={paper} stage={stage} done={done} seconds={reading} />
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
          <div className="min-w-0 flex-1 lg:overflow-y-auto">
            <SpecView result={result} onPick={run} active={request} />
          </div>
        ) : null}
      </section>

      {/* ---- right: what it generated ---- */}
      <section className="flex min-w-0 flex-col lg:h-full lg:overflow-hidden">
        <PaneBar label="Skeleton">
          {result ? (
            <span className="truncate text-faint">{result.spec.extractor.model}</span>
          ) : null}
        </PaneBar>

        <form
          className="flex items-stretch border-b border-hairline"
          onSubmit={(event) => {
            event.preventDefault();
            void run(request);
          }}
        >
          <label
            htmlFor="request"
            className="flex shrink-0 items-center border-r border-hairline px-3 font-mono text-xs text-muted"
          >
            component
          </label>
          <input
            id="request"
            value={request}
            onChange={(event) => setRequest(event.target.value)}
            disabled={phase !== "ready" || streaming}
            placeholder={phase === "ready" ? "the attention block" : "reading the paper…"}
            className="min-w-0 flex-1 bg-transparent px-3 py-3 font-mono text-sm text-ink placeholder:text-faint focus:outline-none disabled:opacity-50"
            autoComplete="off"
            spellCheck={false}
          />
          <button
            type="submit"
            disabled={phase !== "ready" || streaming || !request.trim()}
            className="m-2 shrink-0 bg-red-fill px-5 font-mono text-xs font-bold text-white transition-opacity hover:opacity-90 disabled:opacity-30"
          >
            {streaming ? "···" : "Go"}
          </button>
        </form>

        {suggestions ? (
          <div className="px-4 py-6 sm:px-6">
            <p className="text-base text-ink">
              Nothing in the extracted spec matches that.
            </p>
            <p className="mt-4 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
              What it did find
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {suggestions.map((name) => (
                <button
                  key={name}
                  type="button"
                  onClick={() => void run(name)}
                  className="border border-hairline px-3 py-2 font-mono text-xs text-muted hover:border-red-ink hover:text-red-ink"
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
              className="tool-code code-lines min-w-0 flex-1 overflow-auto whitespace-pre-wrap bg-sunken px-4 py-5 font-mono text-xs leading-[1.75] text-ink sm:px-6"
            >
              {highlight(code)}
              {streaming ? (
                <span className="ml-0.5 inline-block h-[1em] w-[0.5em] translate-y-[0.16em] animate-pulse bg-red-ink" />
              ) : null}
            </pre>
            <CodeBar code={code} gaps={gaps} streaming={streaming} elapsed={elapsed} />
          </>
        ) : null}

        {!suggestions && !code && phase === "ready" ? (
          <div className="flex-1 px-4 py-5 sm:px-6">
            <p className="max-w-[52ch] text-sm text-muted">
              Name a part of this paper above, or pick a component on the left.
              The skeleton cites the section behind every value it uses, and
              leaves a marked <code className="font-mono text-xs text-red-ink">TODO</code>{" "}
              wherever the paper decided nothing.
            </p>
          </div>
        ) : null}

      </section>
    </div>
  );
}

function PaneBar({ label, children }: { label: string; children?: React.ReactNode }) {
  return (
    <div className="flex shrink-0 items-center gap-3 border-b border-hairline px-4 py-2.5 sm:px-6">
      <span className="font-mono text-2xs uppercase tracking-[0.11em] text-faint">
        {label}
      </span>
      <div className="ml-auto min-w-0 font-mono text-xs">{children}</div>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 px-4 py-16 text-center sm:px-6">
      {children}
    </div>
  );
}

/** Section label. The one heading style the tool has. */
function Rubric({ children, count }: { children: React.ReactNode; count?: number }) {
  return (
    <h2 className="flex items-baseline gap-3 border-b border-hairline-strong pb-2 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
      {children}
      {count != null ? <span className="text-faint">{count}</span> : null}
    </h2>
  );
}

/**
 * Verified or not, said twice: a filled square against a hollow one, and the
 * word itself. Colour carries neither cue on its own.
 */
function Mark({ ok }: { ok: boolean }) {
  return (
    <span
      aria-hidden
      className={[
        "mt-[0.45em] size-[7px] shrink-0",
        ok ? "bg-red-fill" : "border border-faint",
      ].join(" ")}
    />
  );
}

/** The quote, under the claim it backs. This is the whole product. */
function Evidence({ provenance }: { provenance?: Provenance | null }) {
  if (!verified(provenance) || !provenance) {
    return (
      <p className="mt-1.5 border-l border-dashed border-faint pl-3 font-mono text-2xs text-faint">
        no quote located in the source
      </p>
    );
  }
  return (
    <figure className="mt-1.5 border-l border-red-dim pl-3">
      <blockquote className="text-sm text-muted">
        &ldquo;{provenance.quote}&rdquo;
      </blockquote>
      <figcaption className="mt-0.5 font-mono text-2xs text-faint">
        {provenance.section}
      </figcaption>
    </figure>
  );
}

function Param({ hp }: { hp: Hyperparameter }) {
  const ok = verified(hp.provenance);
  return (
    <li className="py-2">
      <div className="flex items-start gap-2.5">
        <Mark ok={ok} />
        <code className="min-w-0 font-mono text-xs text-ink">
          {hp.name} = <span className="text-red-ink">{hp.value ?? "—"}</span>
        </code>
        <span className="ml-auto shrink-0 font-mono text-2xs uppercase tracking-[0.09em] text-faint">
          {ok ? hp.confidence : `${hp.confidence} · unquoted`}
        </span>
      </div>
      {hp.applies_to ? (
        <p className="mt-0.5 pl-[17px] font-mono text-2xs text-faint">{hp.applies_to}</p>
      ) : null}
      <div className="pl-[17px]">
        <Evidence provenance={hp.provenance} />
      </div>
    </li>
  );
}

function ComponentRow({
  component,
  active,
  onPick,
}: {
  component: Component;
  active: boolean;
  onPick: (query: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const params = component.hyperparameters ?? [];
  const equations = component.equations ?? [];
  const evidence = params.length + equations.length;

  return (
    <li
      className={[
        "border-b border-hairline py-3",
        active ? "border-l-2 border-l-red-fill pl-3" : "pl-0",
      ].join(" ")}
    >
      <div className="flex items-baseline gap-3">
        <button
          type="button"
          onClick={() => onPick(component.name)}
          title={`Generate a skeleton for ${component.name}`}
          className="group flex min-w-0 items-baseline gap-2 text-left"
        >
          {/* The arrow is the affordance: a name on its own looked like prose. */}
          <span aria-hidden className="font-mono text-xs text-faint group-hover:text-red-ink">
            &rarr;
          </span>
          <span className="min-w-0 text-base text-ink underline decoration-hairline-strong decoration-1 underline-offset-4 group-hover:decoration-red-ink">
            {component.name}
          </span>
        </button>
        <span className="ml-auto shrink-0 font-mono text-2xs text-faint">
          {params.length ? `${params.length}p` : null}
          {params.length && equations.length ? " · " : null}
          {equations.length ? `${equations.length}eq` : null}
        </span>
      </div>

      <p className="mt-0.5 pl-[18px] text-sm text-muted">{component.role}</p>

      {component.depends_on?.length ? (
        <p className="mt-1 pl-[18px] font-mono text-2xs text-faint">
          needs {component.depends_on.join(", ")}
        </p>
      ) : null}

      {evidence ? (
        <>
          <button
            type="button"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            className="mt-1.5 ml-[18px] font-mono text-2xs uppercase tracking-[0.09em] text-muted underline-offset-4 hover:text-red-ink hover:underline"
          >
            {open ? "hide evidence" : `evidence (${evidence})`}
          </button>

          {open ? (
            <div className="mt-1 pl-[18px]">
              <ul className="divide-y divide-hairline">
                {params.map((hp) => (
                  <Param key={hp.name} hp={hp} />
                ))}
              </ul>
              {equations.map((equation) => (
                <div key={equation.latex} className="border-t border-hairline py-2.5">
                  <div className="flex items-start gap-2.5">
                    <Mark ok={verified(equation.provenance)} />
                    <div className="min-w-0 flex-1">
                      <TeX tex={equation.latex} />
                      <p className="mt-1 text-sm text-muted">{equation.description}</p>
                      <Evidence provenance={equation.provenance} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : null}
        </>
      ) : (
        <p className="mt-1.5 pl-[18px] font-mono text-2xs text-faint">
          no values or equations attached
        </p>
      )}
    </li>
  );
}

/** Three marks, two, one: severity you can count without reading the word. */
function Severity({ level }: { level: Unknown["severity"] }) {
  const weight = level === "blocking" ? 3 : level === "significant" ? 2 : 1;
  return (
    <span className="flex shrink-0 items-center gap-2">
      <span aria-hidden className="flex gap-[2px]">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className={[
              "size-[7px]",
              i < weight ? "bg-red-fill" : "border border-hairline-strong",
            ].join(" ")}
          />
        ))}
      </span>
      <span className="font-mono text-2xs uppercase tracking-[0.09em] text-muted">
        {level}
      </span>
    </span>
  );
}

function SpecView({
  result,
  onPick,
  active,
}: {
  result: SpecResult;
  onPick: (query: string) => void;
  active: string;
}) {
  const { spec, report, cached } = result;
  const training = spec.training?.hyperparameters ?? [];

  // Counted from the claims themselves rather than from the extractor's
  // confidence labels, so the two numbers at the top are exactly the two
  // kinds of mark the reader can count further down the page.
  const claims = [
    ...spec.components.flatMap((c) => [
      c,
      ...(c.hyperparameters ?? []),
      ...(c.equations ?? []),
    ]),
    ...training,
  ];
  const quoted = claims.filter((claim) => verified(claim.provenance)).length;

  // The order is the argument: what was checked, what was asserted anyway,
  // and what nobody can know from this paper.
  return (
    <div className="px-4 pb-16 sm:px-6">
      {/* masthead */}
      <div className="border-b border-hairline py-4">
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 font-mono text-xs">
          <a
            href={`https://arxiv.org/abs/${spec.arxiv_id}`}
            className="text-red-ink underline-offset-4 hover:underline"
          >
            arXiv:{spec.arxiv_id}
          </a>
          {spec.official_repo ? (
            <a
              href={spec.official_repo}
              className="min-w-0 truncate text-muted underline-offset-4 hover:text-red-ink hover:underline"
            >
              authors&rsquo; code
            </a>
          ) : null}
        </div>
        <h1 className="mt-2 text-xl text-ink">{spec.title}</h1>
      </div>

      {/* the ledger: the honest summary of the whole run */}
      <dl className="grid grid-cols-3 border-b border-hairline">
        <Tally n={quoted} label="quoted" tone="fill" />
        <Tally n={claims.length - quoted} label="unquoted" tone="hollow" />
        <Tally n={spec.unknowns.length} label="gaps" tone="fill" />
      </dl>

      <section className="pt-7">
        <Rubric count={spec.components.length}>Components</Rubric>
        <ul>
          {spec.components.map((component) => (
            <ComponentRow
              key={component.name}
              component={component}
              active={active.trim().toLowerCase() === component.name.toLowerCase()}
              onPick={onPick}
            />
          ))}
        </ul>
      </section>

      {training.length ? (
        <section className="pt-7">
          <Rubric count={training.length}>Training</Rubric>
          {spec.training?.optimizer ? (
            <p className="mt-2 font-mono text-xs text-muted">
              optimizer <span className="text-ink">{spec.training.optimizer}</span>
            </p>
          ) : null}
          <ul className="mt-1 divide-y divide-hairline">
            {training.map((hp) => (
              <Param key={hp.name} hp={hp} />
            ))}
          </ul>
        </section>
      ) : null}

      {spec.unknowns.length ? (
        <section className="pt-7">
          <Rubric count={spec.unknowns.length}>What the paper does not say</Rubric>
          <p className="mt-2 max-w-[56ch] text-sm text-muted">
            Not errors. These are the decisions the authors left to whoever
            implements this, and they are yours to make.
          </p>
          <ul className="mt-3">
            {spec.unknowns.map((unknown) => (
              <li key={unknown.question} className="border-b border-hairline py-3">
                <Severity level={unknown.severity} />
                <p className="mt-1.5 text-base text-ink">{unknown.question}</p>
                <p className="mt-0.5 text-sm text-muted">{unknown.why_it_matters}</p>
                {unknown.conventional_default ? (
                  <p className="mt-1.5 border-l border-dashed border-hairline-strong pl-3 font-mono text-2xs text-muted">
                    <span className="text-faint">commonly </span>
                    {unknown.conventional_default}
                  </p>
                ) : null}
                {unknown.searched?.length ? (
                  <p className="mt-1 font-mono text-2xs text-faint">
                    searched {unknown.searched.join(", ")}
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {report.warnings.length ? (
        <section className="pt-7">
          <Rubric count={report.warnings.length}>Warnings from the run</Rubric>
          <ul className="mt-1">
            {report.warnings.map((warning) => (
              <li
                key={warning}
                className="border-b border-hairline py-2 font-mono text-2xs text-muted"
              >
                {warning}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <p className="pt-7 font-mono text-2xs text-faint">
        extracted {new Date(spec.extractor.extracted_at).toISOString().slice(0, 10)} by{" "}
        {spec.extractor.model} · schema {spec.extractor.schema_version}
        {report.quote_accuracy != null
          ? ` · ${Math.round(report.quote_accuracy * 100)}% of quotes located`
          : null}
        {cached ? " · served from cache" : null}
      </p>
    </div>
  );
}

function Tally({
  n,
  label,
  tone,
}: {
  n: number;
  label: string;
  tone: "fill" | "hollow";
}) {
  return (
    <div className="min-w-0 border-r border-hairline py-3 pr-3 last:border-r-0">
      <dt className="flex items-center gap-2 font-mono text-2xs uppercase tracking-[0.09em] text-faint">
        <span
          aria-hidden
          className={[
            "size-[7px] shrink-0",
            tone === "fill" ? "bg-red-fill" : "border border-faint",
          ].join(" ")}
        />
        {label}
      </dt>
      <dd className="mt-1 font-mono text-lg text-ink">{n}</dd>
    </div>
  );
}

function Progress({
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
  return (
    <div className="flex-1 px-4 py-6 sm:px-6">
      <p className="font-mono text-2xs uppercase tracking-[0.11em] text-muted">
        Reading {paper}
      </p>

      <ol className="mt-4 max-w-[30rem] border-t border-hairline">
        {STAGES.map(([key, name, what]) => {
          const state = done.includes(key) ? "done" : stage === key ? "now" : "waiting";
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
        {done.length}/{STAGES.length} · {seconds}s
      </p>
      <p className="mt-4 max-w-[46ch] text-sm text-muted">
        A paper read before comes back at once. A new one is read section by
        section, which takes a few minutes.
      </p>
    </div>
  );
}

function CodeBar({
  code,
  gaps,
  streaming,
  elapsed,
}: {
  code: string;
  gaps: number;
  streaming: boolean;
  elapsed: number;
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="flex shrink-0 items-center gap-4 border-t border-hairline px-4 py-2.5 font-mono text-2xs text-faint sm:px-6">
      <span className="min-w-0 truncate">
        {streaming ? (
          `generating · ${elapsed}s`
        ) : gaps === 0 ? (
          "nothing left undecided"
        ) : (
          <>
            <span className="text-red-ink">{gaps}</span> decision
            {gaps === 1 ? "" : "s"} left to you
          </>
        )}
      </span>
      <button
        type="button"
        onClick={copy}
        disabled={streaming}
        className="ml-auto shrink-0 py-1 hover:text-red-ink disabled:opacity-40"
      >
        {copied ? "copied" : "copy"}
      </button>
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
