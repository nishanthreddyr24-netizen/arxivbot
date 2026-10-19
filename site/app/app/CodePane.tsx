"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const GAP = /TODO\(paper-silent\)/;

/**
 * The model answers in markdown, so the stream arrives wrapped in a fenced
 * block. Shown raw, the first line of the "code" is ```python — which is not
 * code, and is the first thing the reader sees. The fences are dropped from
 * both the view and the clipboard.
 */
function clean(source: string): string[] {
  return source.split("\n").filter((line) => !/^\s*```/.test(line));
}

export function CodePane({
  code,
  streaming,
  elapsed,
}: {
  code: string;
  streaming: boolean;
  elapsed: number;
}) {
  const lines = useMemo(() => clean(code), [code]);
  const gaps = useMemo(
    () => lines.map((line, i) => (GAP.test(line) ? i : -1)).filter((i) => i >= 0),
    [lines],
  );

  const scroller = useRef<HTMLPreElement>(null);
  const [cursor, setCursor] = useState(0);
  const [copied, setCopied] = useState(false);
  const [shared, setShared] = useState(false);

  // Follow the stream, but only while it is running: once it stops the
  // reader is in charge of where they are looking.
  useEffect(() => {
    if (!streaming) return;
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [code, streaming]);

  function jump(next: number) {
    if (!gaps.length) return;
    const at = ((next % gaps.length) + gaps.length) % gaps.length;
    setCursor(at);
    const node = scroller.current?.querySelector(`#line-${gaps[at]}`);
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    node?.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "center" });
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  /** The URL already carries the paper and the request; this hands it over
   *  without making anyone go and select the address bar. */
  async function share() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setShared(true);
      setTimeout(() => setShared(false), 1400);
    } catch {
      setShared(false);
    }
  }

  return (
    <>
      <pre
        ref={scroller}
        className="tool-code code-lines min-w-0 flex-1 overflow-auto bg-sunken px-4 py-5 font-mono text-xs leading-[1.75] break-words whitespace-pre-wrap text-ink sm:px-6"
      >
        {lines.map((line, index) => {
          const key = `${index}-${line.slice(0, 12)}`;
          const id = `line-${index}`;
          if (GAP.test(line)) {
            return (
              // The one thing in this panel worth finding: marked by a filled
              // rule in the gutter as well as a wash, so it survives greyscale.
              <span
                key={key}
                id={id}
                className="block border-l-2 border-red-fill bg-red-dim/25 font-medium text-red-ink"
              >
                {line || " "}
              </span>
            );
          }
          if (/^#\s*!!/.test(line)) {
            return (
              <span key={key} id={id} className="block bg-red-dim/25 font-bold text-red-ink">
                {line || " "}
              </span>
            );
          }
          if (/^\s*#/.test(line) || /^\s*"""/.test(line)) {
            return (
              <span key={key} id={id} className="block text-faint">
                {line || " "}
              </span>
            );
          }
          return (
            <span key={key} id={id} className="block">
              {line || " "}
            </span>
          );
        })}
        {streaming ? (
          <span className="ml-0.5 inline-block h-[1em] w-[0.5em] translate-y-[0.16em] animate-pulse bg-red-ink" />
        ) : null}
      </pre>

      <div className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1 border-t border-hairline px-4 py-2.5 font-mono text-2xs text-faint sm:px-6">
        <span className="min-w-0">
          {streaming ? (
            `writing · ${elapsed}s`
          ) : gaps.length === 0 ? (
            "nothing left undecided"
          ) : (
            <>
              <span className="text-red-ink">{gaps.length}</span> decision
              {gaps.length === 1 ? "" : "s"} left to you
            </>
          )}
        </span>

        {!streaming && gaps.length ? (
          <span className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => jump(cursor - 1)}
              aria-label="previous undecided point"
              className="px-1.5 py-1 hover:text-red-ink"
            >
              &uarr;
            </button>
            <span className="tabular-nums">
              {cursor + 1}/{gaps.length}
            </span>
            <button
              type="button"
              onClick={() => jump(cursor + 1)}
              aria-label="next undecided point"
              className="px-1.5 py-1 hover:text-red-ink"
            >
              &darr;
            </button>
          </span>
        ) : null}

        <span className="ml-auto flex shrink-0 items-center gap-3">
          <button
            type="button"
            onClick={share}
            disabled={streaming}
            title="Copy a link that reopens this paper and this request"
            className="py-1 hover:text-red-ink disabled:opacity-40"
          >
            {shared ? "link copied" : "share"}
          </button>
          <button
            type="button"
            onClick={copy}
            disabled={streaming}
            className="py-1 hover:text-red-ink disabled:opacity-40"
          >
            {copied ? "copied" : "copy"}
          </button>
        </span>
      </div>
    </>
  );
}

/** The component names the generator says it used, from its own header. */
export function usedComponents(code: string): string[] {
  const line = code.split("\n").find((l) => /^#\s*Components:/.test(l));
  if (!line) return [];
  return line
    .replace(/^#\s*Components:\s*/, "")
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
}
