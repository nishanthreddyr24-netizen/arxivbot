import Link from "next/link";

import { STEPS } from "./content";

/** One heading style for the whole page, so the sections read as a sequence. */
export function Rubric({
  children,
  note,
}: {
  children: React.ReactNode;
  note?: string;
}) {
  return (
    <div className="flex items-baseline gap-4 border-b border-hairline-strong pb-2">
      <h2 className="font-mono text-2xs uppercase tracking-[0.12em] text-muted">
        {children}
      </h2>
      {note ? <span className="font-mono text-2xs text-faint">{note}</span> : null}
    </div>
  );
}

/**
 * The four steps, across the page rather than down it.
 *
 * As a stacked list these ran to 62 characters in the left third and left the
 * other two thirds of every row empty; four parallel steps are four columns.
 */
export function HowItWorks() {
  return (
    <section
      id="how-it-works"
      className="mx-auto max-w-[1240px] px-4 pt-16 sm:px-6"
    >
      <Rubric>How it works</Rubric>
      <ol className="grid gap-x-8 gap-y-6 pt-6 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step, index) => (
          <li key={step.title} className="min-w-0 border-t border-hairline pt-4">
            <span className="font-mono text-2xs text-red-ink">
              {String(index + 1).padStart(2, "0")}
            </span>
            <h3 className="mt-2 text-base text-ink">{step.title}</h3>
            <p className="mt-2 text-sm text-muted">{step.body}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

/**
 * Kept prominent on purpose. A tool whose claim is that it admits what it does
 * not know cannot bury the part where it admits what it does not know.
 */
export function Limits() {
  return (
    <section className="mx-auto max-w-[1240px] px-4 pb-20 pt-16 sm:px-6">
      <Rubric note="read this one">Limits</Rubric>
      <div className="grid gap-x-10 gap-y-4 pt-6 text-base text-muted lg:grid-cols-2">
        <p>
          Generated code can be wrong. It is written from what the extraction
          recorded, and the extraction can miss a component, misread a value, or
          report something as unspecified that the paper states plainly. An
          audit of this tool&rsquo;s own gap reports found a third of them
          false; that was fixed, and it will happen again. Check every snippet
          against the section it links to before you rely on it.
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
      <div className="mx-auto flex max-w-[1240px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-6 font-mono text-2xs text-faint sm:px-6">
        <Link href="/app" className="-my-2 py-2 hover:text-red-ink">
          Open tool
        </Link>
        <Link
          href="https://github.com/nishanthreddyr24-netizen/arxivbot"
          className="-my-2 py-2 hover:text-red-ink"
        >
          GitHub
        </Link>
        <Link href="#how-it-works" className="-my-2 py-2 hover:text-red-ink">
          How it works
        </Link>
        <span className="sm:ml-auto">
          Not affiliated with arXiv or Cornell University.
        </span>
      </div>
    </footer>
  );
}
