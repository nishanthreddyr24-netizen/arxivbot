import Link from "next/link";

/**
 * `onPhone: false` links are anchors into the page you are already reading;
 * at 360px they pushed the bar 100px past the viewport and bought nothing.
 */
const LINKS = [
  { label: "How it works", href: "#how-it-works", onPhone: false },
  {
    label: "GitHub",
    href: "https://github.com/nishanthreddyr24-netizen/arxivbot",
    onPhone: true,
  },
];

export function TopBar() {
  return (
    // Sticky: the page is now one long argument, and the way back to the
    // input should not be a scroll to the top.
    <header className="sticky top-0 z-20 border-b border-hairline bg-ground/90 backdrop-blur-sm">
      <div className="mx-auto flex h-12 max-w-[1240px] items-center gap-6 px-4 sm:px-6">
        <Link
          href="/"
          className="-my-2 flex items-center gap-2 py-2 font-mono text-sm text-ink"
        >
          {/* The only red that is not interactive: a 6px register mark. */}
          <span aria-hidden className="size-1.5 bg-red-fill" />
          arxiv&rarr;code
        </Link>

        <nav className="ml-auto flex items-center gap-4 sm:gap-5">
          {LINKS.map((link) => (
            <Link
              key={link.label}
              href={link.href}
              className={[
                "py-2 font-mono text-xs text-muted underline-offset-4 transition-colors hover:text-red-ink hover:underline",
                link.onPhone ? "" : "hidden sm:inline",
              ].join(" ")}
            >
              {link.label}
            </Link>
          ))}
          <Link
            href="/app"
            className="border border-hairline-strong px-3 py-1.5 font-mono text-xs text-red-ink transition-colors hover:border-red-dim"
          >
            Open tool
          </Link>
        </nav>
      </div>
    </header>
  );
}
