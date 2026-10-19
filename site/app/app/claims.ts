import type { Component, Provenance, Spec } from "@/lib/api";

/**
 * One flat shape for everything the extractor asserts.
 *
 * The spec nests three different kinds of claim — a component's own
 * description, a hyperparameter, an equation — and each carries the same
 * provenance. Flattening them means the evidence strip, the "only unverified"
 * filter and the source pane are each written once instead of three times.
 */
export type ClaimKind = "component" | "value" | "equation";

export type Claim = {
  /** Stable across renders: the source pane scrolls to these. */
  id: string;
  kind: ClaimKind;
  /** The assertion itself, in one line. */
  label: string;
  /** Present on equations; rendered by KaTeX rather than as text. */
  latex?: string;
  detail?: string;
  /** The extractor's own word: stated / guess / conventional. */
  confidence?: string;
  provenance?: Provenance | null;
  /** The component this hangs off, or "Training". */
  owner: string;
};

/**
 * A claim is verified exactly when the service located a verbatim quote in the
 * authors' source. Nothing else counts — an extractor that labels its own
 * guess "stated" still arrives here without a quote.
 */
export const located = (p?: Provenance | null): p is Provenance => Boolean(p?.quote);

export const verified = (claim: Claim) => located(claim.provenance);

/** Every claim a component makes, its own description first. */
export function claimsOf(component: Component): Claim[] {
  const owner = component.name;
  const out: Claim[] = [
    {
      id: `${owner}::self`,
      kind: "component",
      label: owner,
      detail: component.role,
      provenance: component.provenance,
      owner,
    },
  ];

  (component.hyperparameters ?? []).forEach((hp, i) => {
    out.push({
      id: `${owner}::value::${i}`,
      kind: "value",
      label: `${hp.name} = ${hp.value ?? "—"}`,
      detail: hp.applies_to ?? undefined,
      confidence: hp.confidence,
      provenance: hp.provenance,
      owner,
    });
  });

  (component.equations ?? []).forEach((eq, i) => {
    out.push({
      id: `${owner}::equation::${i}`,
      kind: "equation",
      label: eq.description,
      latex: eq.latex,
      detail: eq.implements ? `implements ${eq.implements}` : undefined,
      provenance: eq.provenance,
      owner,
    });
  });

  return out;
}

/** The training block is a component in everything but name. */
export function trainingClaims(spec: Spec): Claim[] {
  return (spec.training?.hyperparameters ?? []).map((hp, i) => ({
    id: `Training::value::${i}`,
    kind: "value" as const,
    label: `${hp.name} = ${hp.value ?? "—"}`,
    detail: hp.applies_to ?? undefined,
    confidence: hp.confidence,
    provenance: hp.provenance,
    owner: "Training",
  }));
}

export function allClaims(spec: Spec): Claim[] {
  return [...spec.components.flatMap(claimsOf), ...trainingClaims(spec)];
}

/**
 * The located quotes, in the order they appear in the authors' LaTeX.
 *
 * This is the closest thing to reading the paper that the API makes available:
 * the service returns character offsets but not the source text, so the only
 * honest "source view" is the passages it could actually find, in document
 * order. Quotes repeat — one sentence often backs three values — so they are
 * merged, and the claims that lean on each one travel with it.
 */
export type Passage = {
  id: string;
  quote: string;
  section: string;
  start: number;
  cites: Claim[];
};

export function passages(claims: Claim[]): Passage[] {
  const byQuote = new Map<string, Passage>();
  for (const claim of claims) {
    if (!located(claim.provenance)) continue;
    const key = `${claim.provenance.section}::${claim.provenance.quote}`;
    const existing = byQuote.get(key);
    if (existing) {
      existing.cites.push(claim);
      continue;
    }
    byQuote.set(key, {
      id: `p${byQuote.size}`,
      quote: claim.provenance.quote,
      section: claim.provenance.section,
      // Offsets are optional in the schema; an unplaced quote sorts last
      // rather than pretending to sit at the top of the paper.
      start: claim.provenance.char_start ?? Number.MAX_SAFE_INTEGER,
      cites: [claim],
    });
  }
  return [...byQuote.values()].sort((a, b) => a.start - b.start);
}
