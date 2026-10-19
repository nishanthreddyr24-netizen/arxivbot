"use client";

import { useEffect, useState } from "react";

/**
 * KaTeX in the browser.
 *
 * The homepage pre-renders its one equation on the server, but here the LaTeX
 * arrives with the API response, so there is nothing to pre-render. The import
 * is dynamic: a paper with no equations never pays for the library.
 */
export function TeX({ tex }: { tex: string }) {
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    void import("katex")
      .then((mod) =>
        mod.default.renderToString(tex, {
          displayMode: true,
          throwOnError: false,
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
  }, [tex]);

  // Until KaTeX lands — and if it never does — the LaTeX itself stands in. It
  // is the verbatim source, which is the thing being claimed anyway.
  if (!html) {
    return <code className="font-mono text-2xs break-all text-muted">{tex}</code>;
  }

  return (
    <div
      className="overflow-x-auto text-ink"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
