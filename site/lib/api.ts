/**
 * The pipeline, over HTTP.
 *
 * The extraction, the quote verification and every guardrail live in the
 * Python service; this is the only thing that talks to it. On Vercel both
 * halves deploy to one domain, so the base URL is empty and `/api/...` is
 * already correct. Locally it points at uvicorn.
 *
 * Caching lives entirely on the Python side. A cache here as well would be a
 * second answer to the same question, free to disagree with the first.
 */

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export type Hyperparameter = {
  name: string;
  value: string | null;
  confidence: string;
  applies_to?: string | null;
  provenance?: { section: string; quote: string } | null;
};

export type Component = {
  name: string;
  role: string;
  depends_on?: string[];
  hyperparameters?: Hyperparameter[];
  equations?: { latex: string; description: string }[];
};

export type Unknown = {
  question: string;
  why_it_matters: string;
  severity: "blocking" | "significant" | "minor";
  conventional_default?: string | null;
};

export type Spec = {
  arxiv_id: string;
  title: string;
  summary?: string;
  official_repo?: string | null;
  components: Component[];
  unknowns: Unknown[];
  extractor: { model: string; schema_version: string; extracted_at: string };
};

export type SpecResult = {
  spec: Spec;
  counts: Record<string, number>;
  cached: boolean;
  report: {
    calls: number;
    quote_accuracy: number | null;
    recovered: string[];
    rejected: string[];
    warnings: string[];
  };
};

export type RecentPaper = {
  arxiv_id: string;
  title: string;
  components: string[];
  gaps: number;
  stated: number;
  model: string;
  extracted_at: string;
};

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
  }
}

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = (body as { detail?: unknown } | null)?.detail;
    throw new ApiError(
      typeof detail === "string" ? detail : `request failed (${response.status})`,
      response.status,
      detail,
    );
  }
  return (await response.json()) as T;
}

export const health = () =>
  get<{ ok: boolean; model: string | null; key: boolean; store: string }>(
    "/api/health",
  );

export const paper = (id: string) =>
  get<{
    arxiv_id: string;
    title: string;
    authors: string[];
    categories: string[];
    abstract: string;
    pdf_url: string;
  }>(`/api/paper?id=${encodeURIComponent(id)}`);

export const recent = (limit = 12) =>
  get<{ papers: RecentPaper[] }>(`/api/recent?limit=${limit}`);

export const spec = (id: string) =>
  get<SpecResult>(`/api/spec?id=${encodeURIComponent(id)}`);

export const startExtract = (paperId: string, refresh = false) =>
  get<{ job: string; cached: boolean }>("/api/extract", {
    method: "POST",
    body: JSON.stringify({ paper: paperId, refresh }),
  });

export const job = (id: string) =>
  get<
    {
      state: "running" | "done" | "error";
      stage: string;
      done: string[];
      error: string | null;
    } & Partial<SpecResult>
  >(`/api/job?id=${encodeURIComponent(id)}`);

export const ask = (id: string, query: string) =>
  get<{
    banner?: string;
    conflict?: boolean;
    matched: { name: string; role: string; reason: string }[];
    dependencies?: string[];
    unknowns?: { question: string; severity: string }[];
    did_you_mean?: string[];
    available?: string[];
  }>(`/api/ask?id=${encodeURIComponent(id)}&q=${encodeURIComponent(query)}`);

/** Streams a skeleton. The caller reads chunks as the model produces them. */
export async function generate(
  id: string,
  query: string,
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const response = await fetch(
    `${BASE}/api/generate?id=${encodeURIComponent(id)}&q=${encodeURIComponent(query)}`,
  );
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => null);
    const detail = (body as { detail?: unknown } | null)?.detail;
    throw new ApiError(
      typeof detail === "string" ? detail : "generation failed",
      response.status,
      detail,
    );
  }
  return response.body.getReader();
}
