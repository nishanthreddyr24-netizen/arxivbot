"use client";

import { useMemo, useState } from "react";

import type { Component, SpecResult, Unknown } from "@/lib/api";
import { ClaimRow, Strip } from "./Evidence";
import { allClaims, claimsOf, trainingClaims, verified, type Claim } from "./claims";

/** Section label. The one heading style the tool has. */
function Rubric({
  children,
  count,
  id,
}: {
  children: React.ReactNode;
  count?: number;
  id?: string;
}) {
  return (
    <h2
      id={id}
      className="flex items-baseline gap-3 border-b border-hairline-strong pb-2 font-mono text-2xs uppercase tracking-[0.11em] text-muted"
    >
      {children}
      {count != null ? <span className="text-faint">{count}</span> : null}
    </h2>
  );
}

/**
 * The run's honesty, in one bar.
 *
 * Three stacked numbers were three things to read. A proportion is one thing
 * to look at, and the sentence under it says the same in words for anyone the
 * bar does not reach.
 */
function Ledger({
  claims,
  gaps,
  weakOnly,
  onWeakOnly,
}: {
  claims: Claim[];
  gaps: number;
  weakOnly: boolean;
  onWeakOnly: (on: boolean) => void;
}) {
  const quoted = claims.filter(verified).length;
  const weak = claims.length - quoted;
  const share = claims.length ? (quoted / claims.length) * 100 : 0;

  return (
    <section className="border-b border-hairline py-4">
      <div className="flex h-2 w-full gap-[2px]" aria-hidden>
        <span className="bg-red-fill" style={{ width: `${share}%` }} />
        <span
          className="flex-1 border border-dashed border-hairline-strong"
          style={{ minWidth: weak ? undefined : 0 }}
        />
      </div>

      <p className="mt-2.5 font-mono text-2xs text-muted">
        <span className="text-ink">{quoted}</span> of {claims.length} claims carry a
        sentence located in the authors&rsquo; source
        {weak ? (
          <>
            {" · "}
            <span className="text-red-ink">{weak} unverified</span>
          </>
        ) : null}
      </p>

      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-2">
        <a
          href="#gaps"
          className="font-mono text-2xs text-muted underline-offset-4 hover:text-red-ink hover:underline"
        >
          {gaps} question{gaps === 1 ? "" : "s"} the paper never answers &darr;
        </a>
        {weak ? (
          <button
            type="button"
            onClick={() => onWeakOnly(!weakOnly)}
            aria-pressed={weakOnly}
            className={[
              "ml-auto border px-2.5 py-1.5 font-mono text-2xs tracking-[0.06em]",
              weakOnly
                ? "border-red-ink text-red-ink"
                : "border-hairline-strong text-muted hover:border-red-ink hover:text-red-ink",
            ].join(" ")}
          >
            {weakOnly ? "showing unverified only" : "show unverified only"}
          </button>
        ) : null}
      </div>
    </section>
  );
}

/**
 * A component, and the whole row is the button.
 *
 * Before this the affordance was a small arrow next to a name that otherwise
 * read as prose; nothing said the name was the thing to press. Now the header
 * is one target with a verb in it, and the evidence hangs underneath.
 */
function ComponentCard({
  component,
  selected,
  used,
  weakOnly,
  onPick,
  onLocate,
  focusedClaim,
}: {
  component: Component;
  selected: boolean;
  used: boolean;
  weakOnly: boolean;
  onPick: (query: string) => void;
  onLocate: (claim: Claim) => void;
  focusedClaim: string | null;
}) {
  const claims = useMemo(() => claimsOf(component), [component]);
  const shown = weakOnly ? claims.filter((c) => !verified(c)) : claims;
  // Opened by default when filtering: the point of the filter is to read the
  // weak claims, not to hunt for them behind another click.
  const [open, setOpen] = useState(false);
  const expanded = open || weakOnly;

  return (
    <li
      className={[
        "border-b border-hairline",
        selected ? "border-l-2 border-l-red-fill pl-3" : "",
      ].join(" ")}
    >
      <button
        type="button"
        onClick={() => onPick(component.name)}
        className="group flex w-full items-start gap-3 py-3 text-left"
      >
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="min-w-0 text-base text-ink underline decoration-hairline-strong decoration-1 underline-offset-4 group-hover:decoration-red-ink">
              {component.name}
            </span>
            {used ? (
              <span className="shrink-0 font-mono text-2xs uppercase tracking-[0.09em] text-red-ink">
                in skeleton
              </span>
            ) : null}
            {/* Always present, not hover-revealed: a phone has no hover, and
                the affordance was the thing this list was missing. */}
            <span className="ml-auto shrink-0 font-mono text-2xs text-faint group-hover:text-red-ink">
              generate &rarr;
            </span>
          </div>
          <p className="mt-0.5 text-sm text-muted">{component.role}</p>
          {component.depends_on?.length ? (
            <p className="mt-1 font-mono text-2xs text-faint">
              needs {component.depends_on.join(", ")}
            </p>
          ) : null}
        </div>
      </button>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pb-3">
        <Strip claims={claims} />
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={expanded}
          disabled={weakOnly}
          className="font-mono text-2xs uppercase tracking-[0.09em] text-muted underline-offset-4 hover:text-red-ink hover:underline disabled:no-underline disabled:opacity-60"
        >
          {expanded ? "evidence" : `evidence (${claims.length})`}
        </button>
      </div>

      {expanded && shown.length ? (
        <ul className="divide-y divide-hairline border-t border-hairline pb-2">
          {shown.map((claim) => (
            <ClaimRow
              key={claim.id}
              claim={claim}
              onLocate={onLocate}
              focused={focusedClaim === claim.id}
            />
          ))}
        </ul>
      ) : null}
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

export function SpecPane({
  result,
  onPick,
  onLocate,
  selected,
  used,
  focusedClaim,
}: {
  result: SpecResult;
  onPick: (query: string) => void;
  onLocate: (claim: Claim) => void;
  selected: string;
  used: string[];
  focusedClaim: string | null;
}) {
  const { spec, report, cached } = result;
  const [weakOnly, setWeakOnly] = useState(false);

  const claims = useMemo(() => allClaims(spec), [spec]);
  const training = useMemo(() => trainingClaims(spec), [spec]);
  const trainingShown = weakOnly ? training.filter((c) => !verified(c)) : training;

  const usedSet = new Set(used.map((name) => name.toLowerCase()));
  const components = weakOnly
    ? spec.components.filter((c) => claimsOf(c).some((claim) => !verified(claim)))
    : spec.components;

  return (
    <div className="px-4 pb-20 sm:px-6">
      <div className="border-b border-hairline py-4">
        <h1 className="text-xl text-ink">{spec.title}</h1>
        <div className="mt-1.5 flex flex-wrap items-baseline gap-x-4 gap-y-1 font-mono text-xs">
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
          ) : (
            <span className="text-faint">no code linked from the paper</span>
          )}
        </div>
      </div>

      <Ledger
        claims={claims}
        gaps={spec.unknowns.length}
        weakOnly={weakOnly}
        onWeakOnly={setWeakOnly}
      />

      <section className="pt-7">
        <Rubric count={components.length}>
          {weakOnly ? "Components with unverified claims" : "Components"}
        </Rubric>
        {components.length ? (
          <ul>
            {components.map((component) => (
              <ComponentCard
                key={component.name}
                component={component}
                selected={selected.trim().toLowerCase() === component.name.toLowerCase()}
                used={usedSet.has(component.name.toLowerCase())}
                weakOnly={weakOnly}
                onPick={onPick}
                onLocate={onLocate}
                focusedClaim={focusedClaim}
              />
            ))}
          </ul>
        ) : (
          <p className="pt-3 text-sm text-muted">
            Every claim in every component carries a located sentence.
          </p>
        )}
      </section>

      {trainingShown.length ? (
        <section className="pt-7">
          <Rubric count={trainingShown.length}>Training</Rubric>
          {spec.training?.optimizer ? (
            <p className="mt-2 font-mono text-xs text-muted">
              optimizer <span className="text-ink">{spec.training.optimizer}</span>
            </p>
          ) : null}
          <ul className="mt-1 divide-y divide-hairline">
            {trainingShown.map((claim) => (
              <ClaimRow
                key={claim.id}
                claim={claim}
                onLocate={onLocate}
                focused={focusedClaim === claim.id}
              />
            ))}
          </ul>
        </section>
      ) : null}

      {spec.unknowns.length ? (
        <section className="pt-7">
          <Rubric id="gaps" count={spec.unknowns.length}>
            What the paper does not say
          </Rubric>
          <p className="mt-2 max-w-[56ch] text-sm text-muted">
            Not errors. Each was challenged against the whole paper and survived;
            they are the decisions the authors left to whoever implements this.
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
