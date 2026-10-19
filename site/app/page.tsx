import { Hero } from "./components/Hero";
import { Footer, HowItWorks, Limits, Listing } from "./components/Sections";
import { TopBar } from "./components/TopBar";
import { Trace } from "./components/Trace";
import { code, math } from "@/lib/render";
import { TRACE } from "@/lib/sample";

export default async function Home() {
  // Rendered here, so the browser loads neither KaTeX nor Shiki.
  const [codeHtml, equationHtml] = await Promise.all([
    code(TRACE.code),
    Promise.resolve(math(TRACE.equation)),
  ]);

  return (
    <>
      <TopBar />
      <main>
        <Hero />

        <section className="mx-auto max-w-[1120px] px-4 sm:px-6">
          {/* The panel was unlabelled, so it read as decoration rather than
              as the one worked example the page is built around. */}
          <div className="flex items-baseline gap-4 border-b border-hairline-strong pb-2">
            <h2 className="font-mono text-2xs uppercase tracking-[0.11em] text-muted">
              One sentence, traced into code
            </h2>
            <span className="font-mono text-2xs text-faint">worked example</span>
          </div>
          <div className="pt-4">
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
            code={TRACE.code}
            derivedLines={TRACE.derivedLines}
          />
          </div>
        </section>

        <Listing />
        <HowItWorks />
        <Limits />
      </main>
      <Footer />
    </>
  );
}
