"use client";

import { useEffect, useMemo, useRef } from "react";

import type { Spec } from "@/lib/api";
import { Mark } from "./Evidence";
import { allClaims, passages, verified, type Claim, type Passage } from "./claims";

/**
 * The paper's own words, in the paper's own order.
 *
 * The API returns the located sentence and its character offset but not the
 * LaTeX itself, so this cannot be the full source — and says so. What it can
 * be is every sentence the extractor stood on, laid out in document order
 * with the claims that lean on each one. Reading down this pane is reading
 * the argument the way the authors made it, with the derived claims attached.
 */
export function SourcePane({
  spec,
  focused,
  onPick,
}: {
  spec: Spec;
  focused: string | null;
  onPick: (query: string) => void;
}) {
  const claims = useMemo(() => allClaims(spec), [spec]);
  const found = useMemo(() => passages(claims), [claims]);
  const missing = useMemo(() => claims.filter((c) => !verified(c)), [claims]);
  const scroller = useRef<HTMLDivElement>(null);

  // The claim that was clicked on the left names a passage here; bring it
  // into view rather than making the reader find it again.
  useEffect(() => {
    if (!focused) return;
    const target = found.find((p) => p.cites.some((c) => c.id === focused));
    if (!target) return;
    const node = scroller.current?.querySelector(`#passage-${target.id}`);
    const motion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    node?.scrollIntoView({ behavior: motion ? "auto" : "smooth", block: "center" });
  }, [focused, found]);

  // Sections, in the order their first located sentence appears.
  const sections: [string, Passage[]][] = [];
  for (const passage of found) {
    const last = sections[sections.length - 1];
    if (last && last[0] === passage.section) last[1].push(passage);
    else sections.push([passage.section, [passage]]);
  }

  return (
    <div ref={scroller} className="min-w-0 flex-1 overflow-y-auto px-4 pb-20 sm:px-6">
      <p className="max-w-[60ch] border-b border-hairline py-4 text-sm text-muted">
        Every sentence the extractor could locate in the authors&rsquo; LaTeX, in the
        order the paper says them. Not the whole paper — only what something in
        the spec stands on.
      </p>

      {sections.map(([section, group]) => (
        <section key={section} className="pt-6">
          <h3 className="sticky top-0 z-10 bg-ground py-2 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
            {section}
          </h3>
          <ul className="border-t border-hairline-strong">
            {group.map((passage) => {
              const here = passage.cites.some((c) => c.id === focused);
              return (
                <li
                  key={passage.id}
                  id={`passage-${passage.id}`}
                  className={[
                    "border-b border-hairline py-3.5",
                    here ? "-ml-3 border-l-2 border-l-red-fill bg-surface pl-3" : "",
                  ].join(" ")}
                >
                  <p className="text-base text-ink">&ldquo;{passage.quote}&rdquo;</p>
                  <ul className="mt-2 flex flex-col gap-1">
                    {passage.cites.map((claim) => (
                      <li
                        key={claim.id}
                        className="flex items-baseline gap-2 font-mono text-2xs text-faint"
                      >
                        <Mark ok className="translate-y-[0.1em]" />
                        <button
                          type="button"
                          onClick={() => onPick(claim.owner)}
                          className="min-w-0 truncate text-left underline-offset-4 hover:text-red-ink hover:underline"
                        >
                          {claim.owner}
                        </button>
                        <span className="min-w-0 truncate text-muted">{claim.label}</span>
                      </li>
                    ))}
                  </ul>
                </li>
              );
            })}
          </ul>
        </section>
      ))}

      {missing.length ? (
        <section className="pt-8">
          <h3 className="border-b border-hairline-strong pb-2 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
            Not found in the source <span className="text-faint">{missing.length}</span>
          </h3>
          <p className="mt-2 max-w-[60ch] text-sm text-muted">
            These claims appear in the spec with no sentence behind them. They
            sit outside the document because nothing in it could be located to
            support them.
          </p>
          <ul className="mt-3">
            {missing.map((claim: Claim) => (
              <li
                key={claim.id}
                className="flex items-baseline gap-2.5 border-b border-dashed border-hairline py-2.5"
              >
                <Mark ok={false} className="translate-y-[0.15em]" />
                <span className="min-w-0 flex-1 font-mono text-2xs text-faint">
                  <button
                    type="button"
                    onClick={() => onPick(claim.owner)}
                    className="underline-offset-4 hover:text-red-ink hover:underline"
                  >
                    {claim.owner}
                  </button>{" "}
                  <span className="text-muted">{claim.label}</span>
                </span>
                <span className="shrink-0 font-mono text-2xs uppercase tracking-[0.09em] text-red-ink">
                  unverified
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {!found.length && !missing.length ? (
        <p className="py-10 text-sm text-muted">This spec makes no claims to locate.</p>
      ) : null}
    </div>
  );
}
