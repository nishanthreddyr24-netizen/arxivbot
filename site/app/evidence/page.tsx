import Link from "next/link";

import { Footer } from "@/app/components/Sections";
import { TopBar } from "@/app/components/TopBar";
import evidence from "@/lib/evidence.json";

export const metadata = {
  title: "Evidence — arxiv→code",
  description:
    "What this tool was measured on, what it got wrong, and what was fixed.",
};

const pct = (value: number) => `${Math.round(value * 100)}%`;

/** One decimal, always. A mean of exactly 3 serialises as 3.0 from Python and
 *  renders as "3" in JavaScript, which put a column of averages on the page
 *  with the decimals coming and going. */
const avg = (value: number) => value.toFixed(1);

/** A number and what it means, on the monospace grid the rest of the site uses. */
function Figure({
  value,
  label,
  tone = "ink",
}: {
  value: string;
  label: string;
  tone?: "ink" | "red";
}) {
  return (
    <div className="border-t border-hairline-strong pt-3">
      <p
        className={`font-mono text-2xl leading-none ${
          tone === "red" ? "text-red-ink" : "text-ink"
        }`}
      >
        {value}
      </p>
      <p className="mt-2 font-mono text-2xs uppercase tracking-[0.11em] text-faint">
        {label}
      </p>
    </div>
  );
}

function Rule({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="border-b border-hairline-strong pb-3 font-mono text-2xs uppercase tracking-[0.11em] text-muted">
      {children}
    </h2>
  );
}

export default function EvidencePage() {
  const { corpus, areas, algorithm_blocks: algo, audit, after } = evidence;
  // Typed explicitly: a run where nothing was refused leaves this empty, and
  // an empty JSON array infers as never[].
  const refused = evidence.refused as { id: string; short: string; why: string }[];

  return (
    <div className="min-h-dvh">
      <TopBar />

      <main className="mx-auto max-w-[1240px] px-4 pb-20 pt-12 sm:px-6">
        <header className="max-w-[46rem]">
          <h1 className="text-2xl leading-tight text-ink">
            What this was measured on
          </h1>
          <p className="mt-4 text-lg leading-relaxed text-muted">
            Every claim on this site about how well the tool works comes from
            the run below. The page is generated from{" "}
            <code className="font-mono text-sm text-red-ink">
              eval/results.json
            </code>{" "}
            rather than written by hand, so it cannot drift from what was
            actually measured.
          </p>
          <p className="mt-3 font-mono text-2xs text-faint">
            generated {evidence.generated} · commit {evidence.commit}
          </p>
        </header>

        {/* ---- the corpus ---- */}
        <section className="mt-14">
          <Rule>The corpus</Rule>
          <p className="mt-4 max-w-[46rem] text-lg leading-relaxed text-muted">
            Thirty papers across seven areas, chosen before any of them was
            run. Everything known about this pipeline used to come from the
            Transformer paper, which is one sample and a famously well-written
            one.
          </p>

          <div className="mt-8 grid grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-4">
            <Figure value={String(corpus.papers)} label="papers" />
            <Figure value={pct(corpus.quotes)} label="claims with a located quote" tone="red" />
            <Figure value={corpus.components.toFixed(1)} label="components per paper" />
            <Figure value={corpus.gaps.toFixed(1)} label="gaps per paper" />
          </div>

          <div className="mt-10 overflow-x-auto">
            <table className="w-full min-w-[520px] border-collapse font-mono text-sm">
              <thead>
                <tr className="border-b border-hairline-strong text-left text-2xs uppercase tracking-[0.11em] text-faint">
                  <th className="py-2 pr-4 font-normal">area</th>
                  <th className="py-2 pr-4 text-right font-normal">papers</th>
                  <th className="py-2 pr-4 text-right font-normal">components</th>
                  <th className="py-2 pr-4 text-right font-normal">gaps</th>
                  <th className="py-2 text-right font-normal">quotes located</th>
                </tr>
              </thead>
              <tbody>
                {areas.map((row) => (
                  <tr key={row.area} className="border-b border-hairline">
                    <td className="py-2 pr-4 text-ink">{row.area}</td>
                    <td className="py-2 pr-4 text-right text-muted">{row.papers}</td>
                    <td className="py-2 pr-4 text-right text-muted">{avg(row.components)}</td>
                    <td className="py-2 pr-4 text-right text-muted">{avg(row.gaps)}</td>
                    <td className="py-2 text-right text-red-ink">{pct(row.quotes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 max-w-[46rem] font-mono text-2xs leading-relaxed text-faint">
            Quote verification is the share of claims marked as stated whose
            sentence was then located character-for-character in the submitted
            source. It varies widely by field, and the low end is a known
            weakness rather than a rounding error. Part of it was also a bug,
            found after this run and described under Still unresolved: these
            columns understate the tool as it stands, by a margin nobody has
            measured.
          </p>
          {corpus.gaps_unread.length > 0 ? (
            <p className="mt-3 max-w-[46rem] font-mono text-2xs leading-relaxed text-red-ink">
              The provider refused the gap pass partway through this run, so{" "}
              {corpus.gaps_unread.join(" and ")} carry no gap count and are left
              out of that average. A refused call is not a paper without gaps,
              and recording it as zero would have been the exact mistake this
              tool exists to catch.
            </p>
          ) : null}
        </section>

        {/* ---- the audit: the part that went wrong ---- */}
        <section className="mt-16">
          <Rule>Where it was wrong</Rule>
          <p className="mt-4 max-w-[46rem] text-lg leading-relaxed text-muted">
            Reporting what a paper never specifies is the feature that
            distinguishes this from asking a chatbot. Nobody had checked
            whether those reports were true, so twenty were sampled and judged
            by hand against the source.
          </p>

          <div className="mt-8 grid grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-4">
            <Figure value={String(audit.sampled)} label="gaps audited" />
            <Figure value={String(audit.true)} label="genuinely unspecified" />
            <Figure
              value={String(audit.false)}
              label="stated plainly in the paper"
              tone="red"
            />
            <Figure value={String(audit.partial)} label="partly answered" />
          </div>

          <div className="mt-8 max-w-[46rem] border-l-2 border-red-fill pl-5">
            <p className="text-lg leading-relaxed text-ink">
              A third of the gap reports were false. The tool was telling
              people a paper had left something open when the paper said it
              outright — T5&rsquo;s 32,000-wordpiece vocabulary, MAML&rsquo;s two
              architectures, GCN&rsquo;s early-stopping rule.
            </p>
            <p className="mt-3 text-base leading-relaxed text-muted">
              A confident wrong answer is worse than no answer, because the
              whole value of the feature is being trusted. The cause was
              measurable: the pass was only shown the method section and the
              appendix, so it never saw the experiments section, where
              hyperparameters live.
            </p>
          </div>
        </section>

        {/* ---- the fix ---- */}
        <section className="mt-16">
          <Rule>What was done about it</Rule>
          <p className="mt-4 max-w-[46rem] text-lg leading-relaxed text-muted">
            A gap is now challenged before it is published. The model proposes
            it, code searches the whole paper for an answer, and only the
            retrieved passages go back in a short focused question. A gap whose
            answer is found is withdrawn, and the section that settled it is
            recorded.
          </p>

          <div className="mt-8 grid grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-4">
            <Figure
              value={`${after.retrieval_hit[0]}/${after.retrieval_hit[1]}`}
              label="refuting sentence reached"
            />
            <Figure
              value={`${after.end_to_end[0]}/${after.end_to_end[1]}`}
              label="judged correctly"
            />
            <Figure
              value={`${after.real_kept[0]}/${after.real_kept[1]}`}
              label="real gaps kept"
              tone="red"
            />
            <Figure
              value={`${after.false_dropped[0]}/${after.false_dropped[1]}`}
              label="false gaps withdrawn"
            />
          </div>
          <p className="mt-6 max-w-[46rem] text-base leading-relaxed text-muted">
            {after.note}
          </p>
        </section>

        {/* ---- refusals ---- */}
        {refused.length > 0 ? (
          <section className="mt-16">
            <Rule>Papers it refuses</Rule>
            <p className="mt-4 max-w-[46rem] text-lg leading-relaxed text-muted">
              Some submissions ship a PDF wrapped in a few hundred characters
              of LaTeX. There is nothing to read. These used to come back as
              &ldquo;0 components&rdquo;, which reads as a statement about the
              paper rather than an admission about the tool.
            </p>
            <ul className="mt-6 max-w-[46rem]">
              {refused.map((row) => (
                <li
                  key={row.id}
                  className="border-b border-hairline py-3 font-mono text-sm"
                >
                  <span className="text-red-ink">{row.id}</span>{" "}
                  <span className="text-ink">{row.short}</span>
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        {/* ---- a result that surprised us ---- */}
        <section className="mt-16">
          <Rule>One thing the corpus overturned</Rule>
          <div className="mt-6 grid max-w-[46rem] gap-6 sm:grid-cols-2">
            <div className="border-t border-hairline-strong pt-4">
              <p className="font-mono text-2xs uppercase tracking-[0.11em] text-faint">
                papers with algorithm blocks
              </p>
              <p className="mt-3 font-mono text-xl text-ink">
                {algo.with.components.toFixed(1)}{" "}
                <span className="text-2xs text-faint">components</span>
              </p>
              <p className="mt-1 font-mono text-xl text-red-ink">
                {pct(algo.with.quotes)}{" "}
                <span className="text-2xs text-faint">quotes located</span>
              </p>
            </div>
            <div className="border-t border-hairline-strong pt-4">
              <p className="font-mono text-2xs uppercase tracking-[0.11em] text-faint">
                prose only
              </p>
              <p className="mt-3 font-mono text-xl text-ink">
                {algo.without.components.toFixed(1)}{" "}
                <span className="text-2xs text-faint">components</span>
              </p>
              <p className="mt-1 font-mono text-xl text-red-ink">
                {pct(algo.without.quotes)}{" "}
                <span className="text-2xs text-faint">quotes located</span>
              </p>
            </div>
          </div>
          <p className="mt-6 max-w-[46rem] text-base leading-relaxed text-muted">
            A paper with a numbered algorithm block yields{" "}
            <em>fewer</em> components than one written in prose, and quotes
            that are easier to verify. The prediction was the opposite.
          </p>
        </section>

        {/* ---- what is still wrong ---- */}
        <section className="mt-16">
          <Rule>Still unresolved</Rule>
          <ul className="mt-6 max-w-[46rem]">
            {[
              "Extraction is not reproducible. The same paper does not always yield the same components, and the agreement across repeated runs has not been measured.",
              "Quote verification ranges widely by field. At the low end, close to half of the provenance could not be located.",
              "These numbers were measured before a parser bug was found. LaTeX returned inside JSON was losing its backslashes — \\beta became a control character — which left the quote unfindable in the source and published a supported claim as an unsupported one. The one paper re-extracted since went from 80% of quotes located to 100%. The corpus has not been re-run, so every quote figure on this page should be read as a floor, by an unknown margin.",
              "Precision is loose. Things a paper compares against can be extracted as if the paper had built them.",
              "The gap audit is one person's judgement on twenty samples, not ground truth, and two runs of the checker disagreed on three of them.",
            ].map((line) => (
              <li
                key={line}
                className="border-b border-hairline py-4 text-base leading-relaxed text-muted"
              >
                {line}
              </li>
            ))}
          </ul>
          <p className="mt-6 max-w-[46rem] text-base leading-relaxed text-muted">
            Everything here is reproducible from the repository:{" "}
            <code className="font-mono text-sm text-red-ink">
              python eval/run.py breadth
            </code>
            .{" "}
            <Link
              href="https://github.com/nishanthreddyr24-netizen/arxivbot"
              className="text-red-ink underline underline-offset-4"
            >
              The code and the raw results
            </Link>{" "}
            are public.
          </p>
        </section>
      </main>

      <Footer />
    </div>
  );
}
