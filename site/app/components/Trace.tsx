"use client";

import { useEffect, useRef, useState } from "react";

import { DERIVATIONS } from "./content";

type Props = {
  arxivId: string;
  title: string;
  authors: string;
  category: string;
  section: string;
  excerpt: { kind: "text" | "mark"; body: string }[];
  equationHtml: string;
  equationNote: string;
  codeHtml: string;
  code: string;
  derivedLines: number[];
  silentLines: number[];
};

type Band = { top: number; height: number };

/** The marker's distance from the code pane's left edge, in pixels. */
const GUTTER = 8;


/** The span of a 1-indexed run of lines, in pixels relative to `within`. */
function bandOf(
  lines: NodeListOf<HTMLElement>,
  numbers: number[],
  within: DOMRect,
): Band | null {
  const first = lines[numbers[0] - 1];
  const last = lines[numbers[numbers.length - 1] - 1];
  if (!first || !last) return null;
  const a = first.getBoundingClientRect();
  const b = last.getBoundingClientRect();
  return { top: a.top - within.top, height: Math.max(b.bottom - a.top, 2) };
}

/**
 * The thesis, made visible: a sentence from the paper on the left, the lines it
 * became on the right, and a line drawn between them.
 *
 * Everything positional is measured from the real geometry of the real
 * elements rather than computed from the type scale. The marker used to be
 * placed at `(first - 1) * 1.7 * 11px`, which assumed a line height, a font
 * size and a padding; when the mono font loaded late it pointed three lines
 * below the code it was marking, and the connector terminated on the pane's
 * border rather than at the marker.
 */
export function Trace(props: Props) {
  const frame = useRef<HTMLDivElement>(null);
  const sentence = useRef<HTMLSpanElement>(null);
  const markLayer = useRef<HTMLDivElement>(null);
  const codeBody = useRef<HTMLDivElement>(null);

  const [path, setPath] = useState<string | null>(null);
  const [derived, setDerived] = useState<Band | null>(null);
  const [silent, setSilent] = useState<Band | null>(null);
  const [drawn, setDrawn] = useState(false);
  const [tab, setTab] = useState<"paper" | "code">("paper");

  useEffect(() => {
    function measure() {
      const layerEl = markLayer.current;
      const bodyEl = codeBody.current;
      if (!layerEl || !bodyEl) return;

      const layer = layerEl.getBoundingClientRect();
      const lines = bodyEl.querySelectorAll<HTMLElement>(".line");

      const d = bandOf(lines, props.derivedLines, layer);
      setDerived(d);
      setSilent(bandOf(lines, props.silentLines, layer));

      // Below the split breakpoint the two halves are stacked tabs, and a
      // connector between them would cross unrelated content.
      const box = frame.current?.getBoundingClientRect();
      const from = sentence.current?.getBoundingClientRect();
      if (!box || !from || !d || window.innerWidth < 1024) {
        setPath(null);
        return;
      }

      const x1 = from.right - box.left;
      const y1 = from.top + from.height / 2 - box.top;
      // GUTTER below is the marker's own offset inside the code pane, so the
      // curve lands on the marker rather than near it.
      const x2 = layer.left + GUTTER - box.left;
      const y2 = layer.top + d.top + d.height / 2 - box.top;
      // The two anchors are nearly vertically aligned, so a single curve
      // between them came out as a near-vertical squiggle pinned to the
      // divider. Routed instead: out of the sentence, down the gutter between
      // the panes, into the marker. Three straight runs and two corners.
      const gx = x1 + (x2 - x1) / 2;
      const down = y2 > y1 ? 1 : -1;
      const r = Math.min(
        10,
        Math.abs(gx - x1),
        Math.abs(x2 - gx),
        Math.abs(y2 - y1) / 2,
      );

      setPath(
        `M ${x1} ${y1} H ${gx - r} Q ${gx} ${y1} ${gx} ${y1 + r * down} ` +
          `V ${y2 - r * down} Q ${gx} ${y2} ${gx + r} ${y2} H ${x2}`,
      );
    }

    measure();
    const observer = new ResizeObserver(measure);
    if (frame.current) observer.observe(frame.current);
    if (codeBody.current) observer.observe(codeBody.current);
    window.addEventListener("resize", measure);
    // The mono face changes every line box when it swaps in.
    document.fonts?.ready.then(measure).catch(() => {});
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [props.derivedLines, props.silentLines]);

  // Draw once, when the section is actually on screen.
  useEffect(() => {
    if (!frame.current || drawn) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setDrawn(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setDrawn(true);
          observer.disconnect();
        }
      },
      { threshold: 0.25 },
    );
    observer.observe(frame.current);
    return () => observer.disconnect();
  }, [drawn]);

  return (
    <div ref={frame} className="relative border border-hairline bg-surface">
      {/* Tabs, below the split breakpoint only. */}
      <div className="flex border-b border-hairline lg:hidden">
        {(["paper", "code"] as const).map((name) => (
          <button
            key={name}
            type="button"
            aria-pressed={tab === name}
            onClick={() => setTab(name)}
            className={[
              "flex-1 border-r border-hairline px-4 py-3 font-mono text-xs last:border-r-0",
              tab === name
                ? "bg-surface-raised text-red-ink"
                : "text-muted",
            ].join(" ")}
          >
            {name === "paper" ? "The sentence" : "The code"}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        {/* ---- paper ---- */}
        <div
          className={[
            "flex min-w-0 flex-col border-hairline lg:border-r",
            tab === "paper" ? "flex" : "hidden lg:flex",
          ].join(" ")}
        >
          <div className="flex items-baseline gap-3 border-b border-hairline px-5 py-3 font-mono text-xs sm:px-6">
            <span className="text-red-ink">{props.arxivId}</span>
            <span className="text-faint">{props.category}</span>
            <span className="ml-auto shrink-0 text-faint">LaTeX source</span>
          </div>

          <div className="flex-1 px-5 py-5 sm:px-6">
            <h3 className="text-xl text-ink">{props.title}</h3>
            <p className="mt-2 truncate font-mono text-2xs text-faint">
              {props.authors}
            </p>

            <p className="mt-6 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
              &sect; {props.section}
            </p>

            <p className="mt-3 text-base text-muted">
              {props.excerpt.map((piece, index) =>
                piece.kind === "mark" ? (
                  <span
                    key={index}
                    ref={sentence}
                    className="bg-red-wash text-ink decoration-red-ink decoration-[1.5px] underline-offset-[5px] [text-decoration-line:underline]"
                  >
                    {piece.body}
                  </span>
                ) : (
                  <span key={index}>{piece.body}</span>
                ),
              )}
            </p>

            <div
              className="mt-6 overflow-x-auto text-ink"
              dangerouslySetInnerHTML={{ __html: props.equationHtml }}
            />
            <p className="mt-2 font-mono text-2xs text-faint">
              {props.equationNote}
            </p>

            {/* The missing middle step. Without it the panel asks you to take
                the jump from a sentence to a line number on trust. */}
            <p className="mt-8 border-b border-hairline pb-2 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
              What that sentence fixed
            </p>
            <dl className="font-mono text-2xs">
              {DERIVATIONS.map((item) => (
                <div
                  key={item.name}
                  className="flex flex-wrap items-baseline gap-x-4 gap-y-0.5 border-b border-hairline py-2"
                >
                  <dt className="w-[7.5rem] shrink-0 text-muted">{item.name}</dt>
                  <dd className="min-w-0 flex-1 text-ink">{item.value}</dd>
                  <dd className="shrink-0 text-faint">{item.lines}</dd>
                </div>
              ))}
            </dl>
          </div>

          {/* The pane used to end here, two hundred pixels short of the code
              beside it. The provenance line is what belonged in that space:
              it is the claim the whole product rests on. */}
          <p className="bg-sunken px-5 py-3 font-mono text-2xs text-faint sm:px-6">
            Quote located character-for-character in{" "}
            <a
              href={`https://arxiv.org/abs/${props.arxivId}`}
              className="text-muted underline-offset-4 hover:text-red-ink hover:underline"
            >
              the submitted source
            </a>
            . Nothing on this side was written by a model.
          </p>
        </div>

        {/* ---- code ---- */}
        <div
          className={[
            "min-w-0",
            tab === "code" ? "block" : "hidden lg:block",
          ].join(" ")}
        >
          <CodePanel
            codeHtml={props.codeHtml}
            code={props.code}
            layerRef={markLayer}
            bodyRef={codeBody}
            derived={derived}
            silent={silent}
          />
        </div>
      </div>

      {/* The connector. Non-interactive and ignored by assistive tech: it
          restates a relationship the text already carries. */}
      {path ? (
        <svg
          aria-hidden
          className="pointer-events-none absolute inset-0 hidden h-full w-full lg:block"
        >
          <path
            d={path}
            fill="none"
            stroke="var(--color-red-ink)"
            strokeWidth="1"
            strokeDasharray="2000"
            strokeDashoffset={drawn ? 0 : 2000}
            style={{
              transition: "stroke-dashoffset 900ms cubic-bezier(0.4, 0, 0.2, 1)",
            }}
          />
        </svg>
      ) : null}
    </div>
  );
}

function CodePanel({
  codeHtml,
  code,
  layerRef,
  bodyRef,
  derived,
  silent,
}: {
  codeHtml: string;
  code: string;
  layerRef: React.RefObject<HTMLDivElement | null>;
  bodyRef: React.RefObject<HTMLDivElement | null>;
  derived: Band | null;
  silent: Band | null;
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
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-4 border-b border-hairline px-5 py-3 sm:px-6">
        <span className="min-w-0 truncate font-mono text-xs text-muted">
          sliding_window_attention.py
        </span>
        <button
          type="button"
          onClick={copy}
          className="ml-auto shrink-0 font-mono text-xs text-muted hover:text-red-ink"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      <div ref={layerRef} className="relative min-w-0 flex-1">
        {/* The two markers. Measured, so they survive a late font swap. */}
        {derived ? (
          <span
            aria-hidden
            className="absolute w-[3px] bg-red-fill"
            style={{ top: derived.top, height: derived.height, left: GUTTER }}
          />
        ) : null}
        {silent ? (
          <span
            aria-hidden
            className="absolute w-[3px] bg-red-dim"
            style={{ top: silent.top, height: silent.height, left: GUTTER }}
          />
        ) : null}

        <div className="overflow-x-auto px-5 py-5 sm:px-6">
          <div
            ref={bodyRef}
            className="trace-code font-mono text-2xs leading-[1.7]"
            dangerouslySetInnerHTML={{ __html: codeHtml }}
          />
        </div>
      </div>

      {/* The markers mean two different things and neither is self-evident. */}
      <dl className="flex flex-wrap gap-x-6 gap-y-1 bg-sunken px-5 py-3 font-mono text-2xs sm:px-6">
        <div className="flex items-center gap-2">
          <span aria-hidden className="h-3 w-[3px] shrink-0 bg-red-fill" />
          <dt className="sr-only">solid marker</dt>
          <dd className="text-faint">derived from the marked sentence</dd>
        </div>
        <div className="flex items-center gap-2">
          <span aria-hidden className="h-3 w-[3px] shrink-0 bg-red-dim" />
          <dt className="sr-only">dim marker</dt>
          <dd className="text-faint">
            <span className="text-red-ink">TODO(paper-silent)</span> &mdash; the
            paper decides nothing here
          </dd>
        </div>
      </dl>
    </div>
  );
}
