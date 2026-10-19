"use client";

import { TeX } from "./TeX";
import { verified, type Claim } from "./claims";

/**
 * Verified or not, said three ways: a solid square against a dashed hollow
 * one, a solid rule against a dashed one, and the word itself. Red reinforces
 * the first of those and carries none of them.
 */
export function Mark({ ok, className = "" }: { ok: boolean; className?: string }) {
  return (
    <span
      aria-hidden
      className={[
        "size-[7px] shrink-0",
        ok ? "bg-red-fill" : "border border-dashed border-faint",
        className,
      ].join(" ")}
    />
  );
}

/**
 * The evidence of a whole group at a glance: one mark per claim, solid for
 * located, dashed for not. A component whose row of marks is mostly dashed is
 * visible from across the page, which is the thing a long spec never showed.
 */
export function Strip({ claims }: { claims: Claim[] }) {
  if (!claims.length) return null;
  const weak = claims.filter((c) => !verified(c)).length;
  return (
    <span
      className="flex items-center gap-[3px]"
      title={`${claims.length - weak} of ${claims.length} claims carry a located quote`}
    >
      {claims.map((claim) => (
        <Mark key={claim.id} ok={verified(claim)} />
      ))}
    </span>
  );
}

/** The word, for anyone who cannot see the shape. */
export function Verdict({ claim }: { claim: Claim }) {
  const ok = verified(claim);
  return (
    <span className="shrink-0 font-mono text-2xs uppercase tracking-[0.09em]">
      {ok ? (
        <span className="text-muted">{claim.confidence ?? "quoted"}</span>
      ) : (
        // The extractor's own label is kept but demoted: a claim it called
        // "stated" with no quote behind it is exactly the case to flag.
        <span className="text-faint">
          {claim.confidence ? `${claim.confidence} · ` : ""}
          <span className="text-red-ink">unverified</span>
        </span>
      )}
    </span>
  );
}

/**
 * One claim and the sentence behind it.
 *
 * `onLocate` is what makes this product's argument legible: it puts the
 * authors' sentence next to every other sentence the extractor found, in the
 * order the paper says them.
 */
export function ClaimRow({
  claim,
  onLocate,
  focused = false,
}: {
  claim: Claim;
  onLocate?: (claim: Claim) => void;
  focused?: boolean;
}) {
  const ok = verified(claim);
  const provenance = claim.provenance;

  return (
    <li
      id={`claim-${claim.id}`}
      className={[
        "py-2.5",
        focused ? "-ml-3 border-l-2 border-l-red-fill pl-3" : "",
      ].join(" ")}
    >
      <div className="flex items-start gap-2.5">
        <Mark ok={ok} className="mt-[0.5em]" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            {claim.kind === "value" ? (
              <code className="min-w-0 font-mono text-xs break-all text-ink">
                {claim.label.split(" = ")[0]} ={" "}
                <span className="text-red-ink">{claim.label.split(" = ").slice(1).join(" = ")}</span>
              </code>
            ) : claim.kind === "equation" ? null : (
              <span className="min-w-0 text-sm text-ink">{claim.label}</span>
            )}
            <span className="ml-auto">
              <Verdict claim={claim} />
            </span>
          </div>

          {claim.latex ? (
            <div className="mt-1">
              <TeX tex={claim.latex} />
            </div>
          ) : null}

          {claim.kind === "equation" ? (
            <p className="mt-1 text-sm text-muted">{claim.label}</p>
          ) : null}

          {claim.detail ? (
            <p className="mt-0.5 font-mono text-2xs text-faint">{claim.detail}</p>
          ) : null}

          {ok && provenance ? (
            <figure className="mt-1.5 border-l border-red-dim pl-3">
              <blockquote className="text-sm text-muted">
                &ldquo;{provenance.quote}&rdquo;
              </blockquote>
              <figcaption className="mt-1 flex flex-wrap items-baseline gap-x-3 font-mono text-2xs text-faint">
                <span className="min-w-0 truncate">{provenance.section}</span>
                {onLocate ? (
                  <button
                    type="button"
                    onClick={() => onLocate(claim)}
                    className="shrink-0 underline-offset-4 hover:text-red-ink hover:underline"
                  >
                    in source &rarr;
                  </button>
                ) : null}
              </figcaption>
            </figure>
          ) : (
            <p className="mt-1.5 border-l border-dashed border-faint pl-3 font-mono text-2xs text-faint">
              no sentence in the source could be located for this
            </p>
          )}
        </div>
      </div>
    </li>
  );
}
