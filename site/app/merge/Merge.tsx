"use client";

import { useCallback, useEffect, useState } from "react";

import { componentsOf, mergePapers, type MergeResult } from "@/lib/api";

const SEVERITY_RANK = { blocking: 3, significant: 2, minor: 1 } as const;

/** Severity said by shape as well as colour, as everywhere else on the site. */
function Pips({ severity }: { severity: string }) {
  const filled = SEVERITY_RANK[severity as keyof typeof SEVERITY_RANK] ?? 1;
  return (
    <span aria-hidden className="inline-flex gap-[3px]">
      {[1, 2, 3].map((n) => (
        <span
          key={n}
          className={`size-[7px] ${
            n <= filled ? "bg-red-fill" : "border border-hairline-strong"
          }`}
        />
      ))}
    </span>
  );
}

function PaperField({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: string;
  onChange: (next: string) => void;
  hint: string;
}) {
  return (
    <label className="block">
      <span className="font-mono text-2xs uppercase tracking-[0.11em] text-faint">
        {label}
      </span>
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={hint}
        spellCheck={false}
        className="mt-2 w-full border border-hairline-strong bg-surface px-3 py-2 font-mono text-sm text-ink placeholder:text-faint focus:border-red-dim focus:outline-none"
      />
    </label>
  );
}

export function Merge({
  initialHost,
  initialDonor,
}: {
  initialHost: string;
  initialDonor: string;
}) {
  const [host, setHost] = useState(initialHost);
  const [donor, setDonor] = useState(initialDonor);
  const [component, setComponent] = useState("");
  const [replaces, setReplaces] = useState("");

  const [donorParts, setDonorParts] = useState<string[] | null>(null);
  const [hostParts, setHostParts] = useState<string[] | null>(null);
  const [result, setResult] = useState<MergeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Offering the names is most of the usability here: nobody can guess what
  // a given paper calls its parts, and a wrong guess is just an error.
  useEffect(() => {
    let cancelled = false;
    if (!donor.trim()) return setDonorParts(null);
    componentsOf(donor.trim())
      .then((d) => !cancelled && setDonorParts(d.components.map((c) => c.name)))
      .catch(() => !cancelled && setDonorParts(null));
    return () => {
      cancelled = true;
    };
  }, [donor]);

  useEffect(() => {
    let cancelled = false;
    if (!host.trim()) return setHostParts(null);
    componentsOf(host.trim())
      .then((d) => !cancelled && setHostParts(d.components.map((c) => c.name)))
      .catch(() => !cancelled && setHostParts(null));
    return () => {
      cancelled = true;
    };
  }, [host]);

  const run = useCallback(async () => {
    if (!host.trim() || !donor.trim() || !component.trim()) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const data = await mergePapers(
        host.trim(),
        donor.trim(),
        component.trim(),
        replaces.trim() || undefined,
      );
      setResult(data);
      const url = new URL(window.location.href);
      url.searchParams.set("host", host.trim());
      url.searchParams.set("donor", donor.trim());
      window.history.replaceState(null, "", url);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }, [host, donor, component, replaces]);

  return (
    <div className="mx-auto max-w-[1240px] px-4 pb-20 pt-12 sm:px-6">
      <header className="max-w-[46rem]">
        <h1 className="text-2xl leading-tight text-ink">
          Put one paper&rsquo;s component into another&rsquo;s architecture
        </h1>
        <p className="mt-4 text-lg leading-relaxed text-muted">
          Most work called novel is a substitution. This performs one and,
          more usefully, tells you what it costs: the assumptions the two
          papers disagree on, computed by comparing them axis by axis rather
          than asked of a model.
        </p>
        <p className="mt-3 text-base leading-relaxed text-faint">
          It will not tell you whether the result is new. That claim cannot be
          checked, and a tool that made it could not be trusted about anything
          else.
        </p>
      </header>

      <section className="mt-10 grid max-w-[900px] gap-5 sm:grid-cols-2">
        <PaperField
          label="host — the architecture to keep"
          value={host}
          onChange={setHost}
          hint="1706.03762"
        />
        <PaperField
          label="donor — the paper to take a part from"
          value={donor}
          onChange={setDonor}
          hint="1512.03385"
        />
        <PaperField
          label="component to take"
          value={component}
          onChange={setComponent}
          hint="convolutional layers"
        />
        <PaperField
          label="host component it replaces (optional)"
          value={replaces}
          onChange={setReplaces}
          hint="embedding layers"
        />
      </section>

      {donorParts?.length ? (
        <p className="mt-4 max-w-[900px] font-mono text-2xs leading-relaxed text-faint">
          donor has:{" "}
          {donorParts.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setComponent(name)}
              className="mr-2 underline underline-offset-4 hover:text-red-ink"
            >
              {name}
            </button>
          ))}
        </p>
      ) : null}

      {hostParts?.length ? (
        <p className="mt-2 max-w-[900px] font-mono text-2xs leading-relaxed text-faint">
          host has:{" "}
          {hostParts.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setReplaces(name)}
              className="mr-2 underline underline-offset-4 hover:text-red-ink"
            >
              {name}
            </button>
          ))}
        </p>
      ) : null}

      <button
        type="button"
        onClick={run}
        disabled={busy || !host.trim() || !donor.trim() || !component.trim()}
        className="mt-6 border border-hairline-strong px-5 py-2 font-mono text-sm text-red-ink transition-colors hover:border-red-dim disabled:opacity-40"
      >
        {busy ? "composing…" : "Compose"}
      </button>

      {error ? (
        <div className="mt-8 max-w-[46rem] border-l-2 border-red-fill pl-5">
          <p className="font-mono text-2xs uppercase tracking-[0.11em] text-red-ink">
            Cannot compose
          </p>
          <p className="mt-2 font-mono text-sm text-ink">{error}</p>
          <p className="mt-2 text-base text-muted">
            Both papers must have been extracted first — open each in the tool
            once, then come back.
          </p>
        </div>
      ) : null}

      {result ? (
        <div className="mt-12 grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <section>
            <h2 className="border-b border-hairline-strong pb-3 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
              The arrangement
            </h2>
            <p className="mt-4 text-base leading-relaxed text-muted">
              <span className="text-ink">{result.host.title}</span>, with{" "}
              <span className="text-red-ink">{result.donated}</span> from{" "}
              <span className="text-ink">{result.donor.title}</span>
              {result.replaced ? (
                <>
                  , replacing{" "}
                  <span className="text-ink">{result.replaced}</span>
                </>
              ) : (
                ", added alongside"
              )}
              .
            </p>
            <ul className="mt-5">
              {result.components.map((c) => (
                <li
                  key={c.name}
                  className="flex items-baseline gap-3 border-b border-hairline py-2 font-mono text-sm"
                >
                  <span
                    className={
                      c.donated ? "text-red-ink" : "text-transparent"
                    }
                    aria-hidden
                  >
                    +
                  </span>
                  <span className={c.donated ? "text-red-ink" : "text-ink"}>
                    {c.name}
                  </span>
                </li>
              ))}
            </ul>
          </section>

          <section>
            <h2 className="border-b border-hairline-strong pb-3 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
              What it costs
            </h2>

            {result.conflicts.length === 0 ? (
              <p className="mt-4 text-base leading-relaxed text-muted">
                No conflict on the recorded axes. That means the assumptions
                agree where they were recorded — not that the two parts are
                known to fit.
              </p>
            ) : (
              <ul className="mt-4">
                {result.conflicts.map((c) => (
                  <li key={c.axis} className="border-b border-hairline py-4">
                    <div className="flex flex-wrap items-center gap-3">
                      <Pips severity={c.severity} />
                      <span className="font-mono text-2xs uppercase tracking-[0.11em] text-red-ink">
                        {c.severity}
                      </span>
                      <span className="font-mono text-sm text-ink">
                        {c.axis}
                      </span>
                      <span className="font-mono text-2xs text-faint">
                        {c.host} vs {c.donor}
                      </span>
                      <span className="ml-auto border border-hairline-strong px-2 py-0.5 font-mono text-2xs text-muted">
                        {c.resolution}
                      </span>
                    </div>
                    <p className="mt-2 text-base leading-relaxed text-muted">
                      {c.detail}
                    </p>
                  </li>
                ))}
              </ul>
            )}

            {result.notes.length ? (
              <div className="mt-6">
                <p className="font-mono text-2xs uppercase tracking-[0.11em] text-faint">
                  rewiring
                </p>
                {result.notes.map((note) => (
                  <p key={note} className="mt-2 text-base text-muted">
                    {note}
                  </p>
                ))}
              </div>
            ) : null}

            {result.precedents.length ? (
              <div className="mt-8">
                <p className="font-mono text-2xs uppercase tracking-[0.11em] text-faint">
                  closest in the corpus — a ranking, not a novelty check
                </p>
                {result.precedents.map((p) => (
                  <p key={p.id} className="mt-2 font-mono text-sm">
                    <span className="text-red-ink">
                      {Math.round(p.overlap * 100)}%
                    </span>{" "}
                    <span className="text-faint">{p.id}</span>{" "}
                    <span className="text-ink">{p.title}</span>
                  </p>
                ))}
              </div>
            ) : null}

            <p className="mt-8 border-t border-hairline-strong pt-4 font-mono text-2xs text-faint">
              {result.unknowns.length} open question
              {result.unknowns.length === 1 ? "" : "s"} carried over ·{" "}
              {result.buildable
                ? "no contradiction that an adapter cannot bridge"
                : "contains a contradiction no adapter fixes"}
            </p>
          </section>
        </div>
      ) : null}
    </div>
  );
}
