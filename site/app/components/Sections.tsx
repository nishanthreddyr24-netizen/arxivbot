import Link from "next/link";

import { RECENT } from "@/lib/sample";

/** arXiv's listing, re-lit: id, title, category on one line; detail beneath. */
export function Listing() {
  return (
    <section className="mx-auto max-w-[1120px] px-4 py-16 sm:px-6">
      <div className="flex items-baseline gap-4 border-b border-hairline pb-4">
        <h2 className="text-[1.5rem] leading-none text-ink">Recent extractions</h2>
        <span className="font-mono text-[11px] text-faint">sample data</span>
      </div>

      <ul>
        {RECENT.map((paper) => (
          <li key={paper.id} className="border-b border-hairline">
            <Link
              href={`/app?paper=${paper.id}`}
              className="group block py-4 focus-visible:outline-offset-[-2px]"
            >
              <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <span className="font-mono text-[13px] text-red-ink">{paper.id}</span>
                <span className="min-w-0 flex-1 text-[17px] leading-snug text-ink underline-offset-4 group-hover:underline">
                  {paper.title}
                </span>
                <span className="font-mono text-[12px] text-faint">
                  {paper.category}
                </span>
              </div>
              <p className="mt-1 truncate font-mono text-[11.5px] text-faint">
                {paper.authors}
              </p>
              <p className="mt-0.5 font-mono text-[11.5px] text-muted">
                extracted: {paper.extracted.join(", ")}
              </p>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

const STEPS = [
  "Fetch the paper from arXiv.",
  "Locate the component in the text and equations.",
  "Generate runnable code with a reference to the section.",
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="mx-auto max-w-[1120px] px-4 pb-10 pt-16 sm:px-6">
      <h2 className="border-b border-hairline pb-4 text-[1.5rem] leading-none text-ink">
        How it works
      </h2>
      <ol className="max-w-[46rem]">
        {STEPS.map((step, index) => (
          <li
            key={step}
            className="flex gap-5 border-b border-hairline py-4 text-[17px] text-muted"
          >
            <span className="font-mono text-[13px] text-faint">{index + 1}</span>
            <span>{step}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function Limits() {
  return (
    <section className="mx-auto max-w-[1120px] px-4 py-16 sm:px-6">
      <h2 className="border-b border-hairline pb-4 text-[1.5rem] leading-none text-ink">
        Limits
      </h2>
      <div className="max-w-[46rem] pt-6 text-[17px] leading-[1.68] text-muted">
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
      <div className="mx-auto flex max-w-[1120px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-8 font-mono text-[12px] text-faint sm:px-6">
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
          Docs
        </Link>
        <span className="ml-auto">
          Not affiliated with arXiv or Cornell University.
        </span>
      </div>
    </footer>
  );
}
