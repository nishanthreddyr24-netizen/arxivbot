import katex from "katex";
import { codeToHtml } from "shiki";

/**
 * Server-side rendering of maths and code.
 *
 * Both run at build time for the homepage, so the browser receives finished
 * HTML and loads neither library. KaTeX's stylesheet is inlined in globals.css
 * rather than linked, because the CSP only admits stylesheets from Google
 * Fonts.
 */

export function math(tex: string): string {
  return katex.renderToString(tex, {
    displayMode: true,
    throwOnError: false,
    output: "html",
  });
}

/** A theme built from the site's own tokens, so code sits in the same palette. */
const THEME = {
  name: "arxiv-code",
  type: "dark" as const,
  colors: { "editor.background": "#141312", "editor.foreground": "#e8e4dc" },
  settings: [
    { scope: ["comment", "punctuation.definition.comment"], settings: { foreground: "#5f5a53", fontStyle: "italic" } },
    { scope: ["keyword", "storage.type", "storage.modifier", "keyword.control"], settings: { foreground: "#ff5a4d" } },
    { scope: ["string", "string.quoted", "constant.character"], settings: { foreground: "#8a847a" } },
    { scope: ["constant.numeric", "constant.language"], settings: { foreground: "#e8e4dc" } },
    { scope: ["entity.name.function", "support.function", "meta.function-call"], settings: { foreground: "#e8e4dc" } },
    { scope: ["entity.name.class", "entity.name.type", "support.class"], settings: { foreground: "#e8e4dc" } },
    { scope: ["variable", "variable.parameter", "meta.attribute"], settings: { foreground: "#8a847a" } },
  ],
};

export async function code(source: string, lang = "python"): Promise<string> {
  return codeToHtml(source, { lang, theme: THEME });
}
