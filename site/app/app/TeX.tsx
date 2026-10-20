"use client";

import { useEffect, useState } from "react";

/**
 * KaTeX in the browser.
 *
 * The homepage pre-renders its one equation on the server, but here the LaTeX
 * arrives with the API response, so there is nothing to pre-render. The import
 * is dynamic: a paper with no equations never pays for the library.
 */
/**
 * Prose with `$…$` in it, as prose.
 *
 * The extractor writes a component's role in the paper's own notation, so a
 * diffusion paper describes one as "fixed constant variances $\beta_t$" and
 * the reader saw the dollar signs and the backslash. KaTeX falls back to the
 * literal command for a macro it does not know - `\bmu` stays `\bmu`, which
 * is what was on screen anyway - so rendering can only improve this.
 *
 * Deliberately not used for the quoted sentences. Those are claimed to be
 * character-for-character what the authors wrote, and typesetting them would
 * make that claim unverifiable by eye.
 */
export function Prose({ text, className }: { text: string; className?: string }) {
  const parts = text.split(/(\$[^$]+\$)/g);
  if (parts.length === 1) return <span className={className}>{text}</span>;
  return (
    <span className={className}>
      {parts.map((part, at) =>
        part.startsWith("$") && part.endsWith("$") && part.length > 2 ? (
          <TeX key={at} tex={part.slice(1, -1)} inline />
        ) : (
          <span key={at}>{part}</span>
        ),
      )}
    </span>
  );
}

export function TeX({ tex, inline = false }: { tex: string; inline?: boolean }) {
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    void import("katex")
      .then((mod) =>
        mod.default.renderToString(tex, {
          displayMode: !inline,
          throwOnError: false,
          // Papers define their own macros - \bx, \bmu, \dmodel - and KaTeX
          // cannot know them, so it prints the command itself. In its default
          // red that reads as "this tool is broken" rather than "the paper
          // wrote a macro here", which is all it means. Muted, it reads as
          // what it is: the source, standing in for itself.
          errorColor: "#8a847a",
          output: "html",
        }),
      )
      .then((out) => {
        if (live) setHtml(out);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [tex, inline]);

  // Until KaTeX lands — and if it never does — the LaTeX itself stands in. It
  // is the verbatim source, which is the thing being claimed anyway.
  if (!html) {
    return inline ? (
      <code className="font-mono">{tex}</code>
    ) : (
      <code className="font-mono text-2xs break-all text-muted">{tex}</code>
    );
  }

  if (inline) {
    return <span dangerouslySetInnerHTML={{ __html: html }} />;
  }

  return (
    <div
      className="overflow-x-auto text-ink"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
