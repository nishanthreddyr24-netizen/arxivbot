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

        <section className="mx-auto max-w-[1120px] px-4 pb-16 sm:px-6">
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
        </section>

        <Listing />
        <HowItWorks />
        <Limits />
      </main>
      <Footer />
    </>
  );
}
