import { DERIVED_LINES, SILENT_LINES, SKELETON } from "./components/content";
import { Hero } from "./components/Hero";
import { Footer, HowItWorks, Limits, Rubric } from "./components/Sections";
import { TopBar } from "./components/TopBar";
import { Trace } from "./components/Trace";
import { code, math } from "@/lib/render";
import { TRACE } from "@/lib/sample";

export default async function Home() {
  // Rendered here, so the browser loads neither KaTeX nor Shiki.
  const [codeHtml, equationHtml] = await Promise.all([
    code(SKELETON),
    Promise.resolve(math(TRACE.equation)),
  ]);

  return (
    <>
      <TopBar />
      <main>
        <Hero />

        {/* The worked example is the argument, so it gets the page's widest
            measure and its own full-bleed rule rather than sitting as the
            third item in a stack of sections. */}
        <section className="mx-auto max-w-[1240px] px-4 sm:px-6">
          <Rubric note="worked example">One sentence, traced into code</Rubric>
          <p className="max-w-[72ch] pb-5 pt-4 text-lg text-muted">
            One sentence, quoted from the submitted LaTeX of{" "}
            <span className="text-ink">Mistral 7B</span>, and the lines it
            produced &mdash; marked, alongside the place where the paper
            stopped specifying and the implementer has to decide.
          </p>

          <Trace
            arxivId={TRACE.arxivId}
            title={TRACE.title}
            authors={TRACE.authors}
            category={TRACE.category}
            section={TRACE.section}
            excerpt={TRACE.excerpt}
            equationHtml={equationHtml}
            equationNote={TRACE.equationNote}
            codeHtml={codeHtml}
            code={SKELETON}
            derivedLines={DERIVED_LINES}
            silentLines={SILENT_LINES}
          />
        </section>

        <HowItWorks />
        <Limits />
      </main>
      <Footer />
    </>
  );
}
