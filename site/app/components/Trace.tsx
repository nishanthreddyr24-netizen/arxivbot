"use client";

import { useEffect, useRef, useState } from "react";

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
};

/**
 * Paper on the left, the code it produced on the right, and a line drawn
 * between the sentence and the lines it became.
 *
 * The connector is the point of the section, so it is drawn from the real
 * geometry of the two elements rather than positioned by hand: measure both,
 * draw a path between them, redraw when the layout changes. Measured rather
 * than fixed means it survives a font loading late or a window resizing.
 */
export function Trace(props: Props) {
  const frame = useRef<HTMLDivElement>(null);
  const sentence = useRef<HTMLSpanElement>(null);
  const lines = useRef<HTMLDivElement>(null);
  const [path, setPath] = useState<string | null>(null);
  const [drawn, setDrawn] = useState(false);
  const [tab, setTab] = useState<"paper" | "code">("paper");

  useEffect(() => {
    function measure() {
      const box = frame.current?.getBoundingClientRect();
      const from = sentence.current?.getBoundingClientRect();
      const to = lines.current?.getBoundingClientRect();
      if (!box || !from || !to) return;

      // Below the split breakpoint the two halves are stacked tabs and a
      // connector between them would cross unrelated content.
      if (window.innerWidth < 900) {
        setPath(null);
        return;
      }

      const x1 = from.right - box.left;
      const y1 = from.top + from.height / 2 - box.top;
      const x2 = to.left - box.left;
      const y2 = to.top + to.height / 2 - box.top;
      const mid = x1 + (x2 - x1) / 2;

      setPath(`M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`);
    }

    measure();
    const observer = new ResizeObserver(measure);
    if (frame.current) observer.observe(frame.current);
    window.addEventListener("resize", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);

  // Draw once, when the section is actually on screen.
  useEffect(() => {
    if (!frame.current || drawn) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setDrawn(true);
          observer.disconnect();
        }
      },
      { threshold: 0.3 },
    );
    observer.observe(frame.current);
    if (reduced) setDrawn(true);
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
              "flex-1 border-r border-hairline px-4 py-3 font-mono text-[12px] last:border-r-0",
              tab === name ? "text-red-ink" : "text-muted",
            ].join(" ")}
          >
            {name === "paper" ? "Paper" : "Code"}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-2">
        {/* ---- paper ---- */}
        <div
          className={[
            "min-w-0 border-hairline p-6 lg:border-r",
            tab === "paper" ? "block" : "hidden lg:block",
          ].join(" ")}
        >
          <div className="flex items-baseline gap-3 font-mono text-[12px]">
            <span className="text-red-ink">{props.arxivId}</span>
            <span className="text-faint">{props.category}</span>
          </div>
          <h3 className="mt-3 text-[1.35rem] leading-tight text-ink">{props.title}</h3>
          <p className="mt-2 truncate font-mono text-[11px] text-faint">
            {props.authors}
          </p>

          <p className="mt-5 font-mono text-[11px] uppercase tracking-[0.11em] text-muted">
            {props.section}
          </p>

          <p className="mt-3 text-[15px] leading-[1.7] text-muted">
            {props.excerpt.map((piece, index) =>
              piece.kind === "mark" ? (
                <span
                  key={index}
                  ref={sentence}
                  className="bg-red-dim/25 text-ink decoration-red-ink/70 decoration-[1.5px] underline-offset-[5px] [text-decoration-line:underline]"
                >
                  {piece.body}
                </span>
              ) : (
                <span key={index}>{piece.body}</span>
              ),
            )}
          </p>

          <div
            className="mt-5 overflow-x-auto text-ink"
            dangerouslySetInnerHTML={{ __html: props.equationHtml }}
          />
          <p className="mt-2 font-mono text-[11px] text-faint">{props.equationNote}</p>
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
            derivedLines={props.derivedLines}
            derivedRef={lines}
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
            strokeDasharray="1200"
            strokeDashoffset={drawn ? 0 : 1200}
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
  derivedLines,
  derivedRef,
}: {
  codeHtml: string;
  code: string;
  derivedLines: number[];
  derivedRef: React.RefObject<HTMLDivElement | null>;
}) {
  const [copied, setCopied] = useState(false);
  const first = Math.min(...derivedLines);
  const count = derivedLines.length;

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
      <div className="flex items-center gap-4 border-b border-hairline px-5 py-3">
        <span className="font-mono text-[12px] text-muted">
          sliding_window_attention.py
        </span>
        <div className="ml-auto flex items-center gap-4">
          <a
            href="https://arxiv.org/abs/2310.06825"
            className="font-mono text-[12px] text-muted underline-offset-4 hover:text-red-ink hover:underline"
          >
            Verify against paper
          </a>
          <button
            type="button"
            onClick={copy}
            className="font-mono text-[12px] text-muted hover:text-red-ink"
          >
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>

      <div className="relative min-w-0 flex-1 overflow-x-auto p-5">
        {/* Marks the derived lines and gives the connector something to
            terminate at. Sized from the measured line height. */}
        <div
          ref={derivedRef}
          aria-hidden
          className="pointer-events-none absolute left-0 w-1 bg-red-ink/50"
          style={{
            // 1.25rem is the panel padding; the line box is 1.7 x the 12.5px
            // type, which has to be stated in px because this element does not
            // inherit the code block's font size.
            top: `calc(1.25rem + ${(first - 1) * 1.7 * 11.5}px)`,
            height: `${count * 1.7 * 11.5}px`,
          }}
        />
        <div
          className="trace-code font-mono text-[11.5px] leading-[1.7]"
          dangerouslySetInnerHTML={{ __html: codeHtml }}
        />
      </div>
    </div>
  );
}
