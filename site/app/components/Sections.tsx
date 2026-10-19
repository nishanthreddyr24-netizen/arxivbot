import Link from "next/link";

import { RECENT } from "@/lib/sample";

/** One heading style for the whole page, so the sections read as a sequence. */
function Rubric({ children, note }: { children: React.ReactNode; note?: string }) {
  return (
    <div className="flex items-baseline gap-4 border-b border-hairline-strong pb-2">
      <h2 className="font-mono text-2xs uppercase tracking-[0.11em] text-muted">
        {children}
      </h2>
      {note ? <span className="font-mono text-2xs text-faint">{note}</span> : null}
    </div>
  );
}

/** arXiv's listing, re-lit: id, title, category on one line; detail beneath. */
export function Listing() {
  return (
    <section className="mx-auto max-w-[1120px] px-4 pt-14 sm:px-6">
      <Rubric note="sample data">Recent extractions</Rubric>

      <ul>
        {RECENT.map((paper) => (
          <li key={paper.id} className="border-b border-hairline">
            <Link
              href={`/app?paper=${paper.id}`}
              className="group block py-3 focus-visible:outline-offset-[-2px]"
            >
              <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <span className="w-[5.5rem] shrink-0 font-mono text-xs text-red-ink">
                  {paper.id}
                </span>
                <span className="min-w-0 flex-1 text-base text-ink underline decoration-hairline-strong decoration-1 underline-offset-4 group-hover:decoration-red-ink">
                  {paper.title}
                </span>
                <span className="font-mono text-2xs text-faint">{paper.category}</span>
              </div>
              <p className="mt-0.5 truncate pl-0 font-mono text-2xs text-faint sm:pl-[6.5rem]">
                {paper.authors}
              </p>
              <p className="mt-0.5 pl-0 font-mono text-2xs text-muted sm:pl-[6.5rem]">
                {paper.extracted.join(" · ")}
              </p>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** The real pipeline, in its own order. Vague steps described no product. */
const STEPS: [string, string][] = [
  [
    "Fetch the source, not the PDF",
    "arXiv serves the authors' original LaTeX. Every claim below is read from it, so a quote can be checked character for character against what they wrote.",
  ],
  [
    "Extract a spec, and verify every claim",
    "Components, hyperparameters and equations, each carrying the sentence it came from. A claim whose quote cannot be located in the source is thrown away rather than shown.",
  ],
  [
    "Name what the paper never decided",
    "Initialisation, padding, decoding: the choices an implementer has to make and the paper does not state. These are reported as gaps, not guessed at.",
  ],
  [
    "Generate a skeleton that admits both",
    "Code with the section behind each value, and a marked TODO wherever the paper was silent.",
  ],
];

export function HowItWorks() {
  return (
    <section
      id="how-it-works"
      className="mx-auto max-w-[1120px] px-4 pt-14 sm:px-6"
    >
      <Rubric>How it works</Rubric>
      <ol className="max-w-[62ch]">
        {STEPS.map(([title, body], index) => (
          <li key={title} className="flex gap-5 border-b border-hairline py-3">
            <span className="shrink-0 pt-1 font-mono text-2xs text-faint">
              {String(index + 1).padStart(2, "0")}
            </span>
            <div className="min-w-0">
              <h3 className="text-base text-ink">{title}</h3>
              <p className="mt-0.5 text-sm text-muted">{body}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function Limits() {
  return (
    <section className="mx-auto max-w-[1120px] px-4 pt-14 pb-16 sm:px-6">
      <Rubric>Limits</Rubric>
      <div className="max-w-[62ch] pt-4 text-base text-muted">
        <p className="mb-4">
          Generated code can be wrong. It is written from what the extraction
          recorded, and the extraction can miss a component, misread a value, or
          report something as unspecified that the paper states plainly. Check
          every snippet against the section it links to before you rely on it.
        </p>
        <p>
          Extraction is also not reproducible: the same paper does not always
          yield the same components. And nothing here replaces reading the
          paper. It is a way to start implementing one, and a record of what you
          will have to decide for yourself.
        </p>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="border-t border-hairline">
      <div className="mx-auto flex max-w-[1120px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-6 font-mono text-2xs text-faint sm:px-6">
        <Link href="/app" className="hover:text-red-ink">
          Open tool
        </Link>
        <Link
          href="https://github.com/nishanthreddyr24-netizen/arxivbot"
          className="hover:text-red-ink"
        >
          GitHub
        </Link>
        <Link href="#how-it-works" className="hover:text-red-ink">
          How it works
        </Link>
        <span className="sm:ml-auto">
          Not affiliated with arXiv or Cornell University.
        </span>
      </div>
    </footer>
  );
}
