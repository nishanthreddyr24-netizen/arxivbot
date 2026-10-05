import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

/**
 * The cache the site runs on.
 *
 * SQLite from `node:sqlite`, which ships inside Node and needs no native
 * build and no hosted service - the project's rule is that cloning the repo
 * and running it must not require an account anywhere.
 *
 * Three things are cached, each for a different reason:
 *
 *   papers       immutable. 1706.03762v7 is the same document forever.
 *   specs        expensive. ~22 model calls and several minutes each.
 *   generations  cheap to redo but slow to wait for, and identical requests
 *                are common because the examples on the homepage are fixed.
 */

const DB_PATH =
  process.env.ARXIVCODE_DB ??
  join(
    process.env.LOCALAPPDATA ?? join(homedir(), ".local", "share"),
    "arxivbot",
    "site.db",
  );

let handle: DatabaseSync | null = null;

export function db(): DatabaseSync {
  if (handle) return handle;

  mkdirSync(dirname(DB_PATH), { recursive: true });
  handle = new DatabaseSync(DB_PATH);

  // WAL lets a read run while the extractor is writing, which matters because
  // an extraction holds the connection for minutes.
  handle.exec("PRAGMA journal_mode = WAL");
  handle.exec("PRAGMA foreign_keys = ON");
  handle.exec(`
    CREATE TABLE IF NOT EXISTS papers (
      arxiv_id    TEXT PRIMARY KEY,
      title       TEXT NOT NULL,
      authors     TEXT NOT NULL DEFAULT '',
      categories  TEXT NOT NULL DEFAULT '',
      abstract    TEXT NOT NULL DEFAULT '',
      fetched_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS specs (
      arxiv_id       TEXT NOT NULL,
      schema_version TEXT NOT NULL,
      model          TEXT NOT NULL,
      spec_json      TEXT NOT NULL,
      components     INTEGER NOT NULL DEFAULT 0,
      gaps           INTEGER NOT NULL DEFAULT 0,
      stated         INTEGER NOT NULL DEFAULT 0,
      unverified     INTEGER NOT NULL DEFAULT 0,
      quote_accuracy REAL,
      calls          INTEGER NOT NULL DEFAULT 0,
      created_at     TEXT NOT NULL,
      PRIMARY KEY (arxiv_id, schema_version),
      FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS generations (
      arxiv_id   TEXT NOT NULL,
      query_key  TEXT NOT NULL,
      query      TEXT NOT NULL,
      target     TEXT NOT NULL,
      code       TEXT NOT NULL,
      gaps       INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      PRIMARY KEY (arxiv_id, query_key, target)
    );

    CREATE INDEX IF NOT EXISTS specs_recent ON specs(created_at DESC);
  `);

  return handle;
}

/** Requests differing only in wording should hit the same cached generation. */
export function queryKey(query: string): string {
  return query
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(" ")
    .filter((word) => word && !STOPWORDS.has(word))
    .sort()
    .join(" ");
}

const STOPWORDS = new Set(
  "a an the of for to in on and or with code give me show write implement how do i want need get using use its it is are that this please just some".split(
    " ",
  ),
);

export type PaperRow = {
  arxiv_id: string;
  title: string;
  authors: string;
  categories: string;
  abstract: string;
  fetched_at: string;
};

export type SpecRow = {
  arxiv_id: string;
  schema_version: string;
  model: string;
  spec_json: string;
  components: number;
  gaps: number;
  stated: number;
  unverified: number;
  quote_accuracy: number | null;
  calls: number;
  created_at: string;
};

export function getSpec(arxivId: string): SpecRow | null {
  const row = db()
    .prepare(
      `SELECT * FROM specs WHERE arxiv_id = ? ORDER BY created_at DESC LIMIT 1`,
    )
    .get(arxivId) as SpecRow | undefined;
  return row ?? null;
}

export function listRecent(limit = 12): (SpecRow & { title: string; categories: string })[] {
  return db()
    .prepare(
      `SELECT s.*, p.title, p.categories
         FROM specs s JOIN papers p ON p.arxiv_id = s.arxiv_id
        ORDER BY s.created_at DESC
        LIMIT ?`,
    )
    .all(limit) as (SpecRow & { title: string; categories: string })[];
}

type Spec = {
  arxiv_id: string;
  title: string;
  components: { name: string }[];
  unknowns: unknown[];
  extractor: { model: string; schema_version: string };
};

export function saveSpec(
  spec: Spec,
  counts: Record<string, number>,
  report: { calls?: number; quote_accuracy?: number | null },
  meta?: { authors?: string[]; categories?: string[]; abstract?: string },
): void {
  const now = new Date().toISOString();
  const database = db();

  database
    .prepare(
      `INSERT INTO papers (arxiv_id, title, authors, categories, abstract, fetched_at)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(arxiv_id) DO UPDATE SET title = excluded.title`,
    )
    .run(
      spec.arxiv_id,
      spec.title,
      (meta?.authors ?? []).join(", "),
      (meta?.categories ?? []).join(", "),
      meta?.abstract ?? "",
      now,
    );

  database
    .prepare(
      `INSERT INTO specs
         (arxiv_id, schema_version, model, spec_json, components, gaps,
          stated, unverified, quote_accuracy, calls, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(arxiv_id, schema_version) DO UPDATE SET
         model = excluded.model, spec_json = excluded.spec_json,
         components = excluded.components, gaps = excluded.gaps,
         stated = excluded.stated, unverified = excluded.unverified,
         quote_accuracy = excluded.quote_accuracy, calls = excluded.calls,
         created_at = excluded.created_at`,
    )
    .run(
      spec.arxiv_id,
      spec.extractor.schema_version,
      spec.extractor.model,
      JSON.stringify(spec),
      spec.components.length,
      spec.unknowns.length,
      counts.stated ?? 0,
      (counts.guess ?? 0) + (counts.conventional ?? 0),
      report.quote_accuracy ?? null,
      report.calls ?? 0,
      now,
    );
}

export function getGeneration(
  arxivId: string,
  query: string,
  target: string,
): { code: string; created_at: string } | null {
  const row = db()
    .prepare(
      `SELECT code, created_at FROM generations
        WHERE arxiv_id = ? AND query_key = ? AND target = ?`,
    )
    .get(arxivId, queryKey(query), target) as
    | { code: string; created_at: string }
    | undefined;
  return row ?? null;
}

export function saveGeneration(
  arxivId: string,
  query: string,
  target: string,
  code: string,
): void {
  const gaps = (code.match(/TODO\(paper-silent\)/g) ?? []).length;
  db()
    .prepare(
      `INSERT INTO generations
         (arxiv_id, query_key, query, target, code, gaps, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(arxiv_id, query_key, target) DO UPDATE SET
         code = excluded.code, gaps = excluded.gaps,
         created_at = excluded.created_at`,
    )
    .run(
      arxivId,
      queryKey(query),
      query,
      target,
      code,
      gaps,
      new Date().toISOString(),
    );
}
