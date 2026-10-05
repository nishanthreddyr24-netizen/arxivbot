import { spawn } from "node:child_process";
import { join } from "node:path";

/**
 * Runs the extraction pipeline.
 *
 * The pipeline stays in Python: it is where the quote verification, the
 * consolidation and every guardrail live, and porting it would mean
 * reimplementing the checks that make the output worth trusting. The site
 * drives it over stdout rather than duplicating it.
 */

const ROOT = process.env.ARXIVBOT_ROOT ?? join(process.cwd(), "..");
const PYTHON = process.env.ARXIVBOT_PYTHON ?? "python";

export type ExtractResult = {
  spec: {
    arxiv_id: string;
    title: string;
    summary?: string;
    official_repo?: string | null;
    components: {
      name: string;
      role: string;
      depends_on?: string[];
      hyperparameters?: {
        name: string;
        value: string | null;
        confidence: string;
        applies_to?: string | null;
        provenance?: { section: string; quote: string } | null;
      }[];
      equations?: { latex: string; description: string }[];
    }[];
    unknowns: {
      question: string;
      why_it_matters: string;
      severity: string;
      conventional_default?: string | null;
    }[];
    extractor: { model: string; schema_version: string };
  };
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

function run(args: string[], timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, ["-m", "arxivbot.cli", ...args], {
      cwd: ROOT,
      env: {
        ...process.env,
        PYTHONPATH: join(ROOT, "src"),
        PYTHONIOENCODING: "utf-8",
      },
    });

    let out = "";
    let err = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`pipeline timed out after ${Math.round(timeoutMs / 1000)}s`));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => (out += chunk));
    child.stderr.on("data", (chunk) => (err += chunk));
    child.on("error", (cause) => {
      clearTimeout(timer);
      reject(new Error(`could not start ${PYTHON}: ${cause.message}`));
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve(out);
      else reject(new Error(err.trim() || `pipeline exited ${code}`));
    });
  });
}

export async function extract(
  paper: string,
  options: { refresh?: boolean; maxComponents?: number } = {},
): Promise<ExtractResult> {
  const args = ["spec", paper, "--json", "--max-components", String(options.maxComponents ?? 10)];
  if (options.refresh) args.push("--refresh");

  // A cold paper is ~22 model calls; a cached one answers immediately.
  const raw = await run(args, options.refresh ? 20 * 60_000 : 12 * 60_000);
  const start = raw.indexOf("{");
  if (start === -1) throw new Error("pipeline produced no JSON");

  try {
    return JSON.parse(raw.slice(start)) as ExtractResult;
  } catch {
    throw new Error("pipeline produced unparseable JSON");
  }
}

/** Streams a generated skeleton, forwarding chunks as the model produces them. */
export function generate(
  paper: string,
  query: string,
): { stream: ReadableStream<Uint8Array>; done: Promise<string> } {
  let settle: (code: string) => void;
  let fail: (reason: Error) => void;
  const done = new Promise<string>((resolve, reject) => {
    settle = resolve;
    fail = reject;
  });

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const child = spawn(PYTHON, ["-m", "arxivbot.cli", "emit", paper, query], {
        cwd: ROOT,
        env: {
          ...process.env,
          PYTHONPATH: join(ROOT, "src"),
          PYTHONIOENCODING: "utf-8",
          PYTHONUNBUFFERED: "1",
        },
      });

      let collected = "";
      let stderr = "";

      child.stdout.on("data", (chunk: Buffer) => {
        collected += chunk.toString("utf-8");
        controller.enqueue(new Uint8Array(chunk));
      });
      child.stderr.on("data", (chunk) => (stderr += chunk));
      child.on("error", (cause) => {
        controller.error(cause);
        fail(cause as Error);
      });
      child.on("close", (code) => {
        controller.close();
        if (code === 0) settle(collected);
        else fail(new Error(stderr.trim() || `generation exited ${code}`));
      });
    },
  });

  return { stream, done };
}
