import Link from "next/link";

/**
 * `onPhone: false` links are anchors into the page you are already reading;
 * at 360px they pushed the bar 100px past the viewport and bought nothing.
 */
const LINKS = [
  { label: "Examples", href: "#examples", onPhone: false },
  { label: "How it works", href: "#how-it-works", onPhone: false },
  {
    label: "GitHub",
    href: "https://github.com/nishanthreddyr24-netizen/arxivbot",
    onPhone: true,
  },
];

export function TopBar() {
  return (
    <header className="h-12 border-b border-hairline">
      <div className="mx-auto flex h-full max-w-[1120px] items-center gap-6 px-4 sm:px-6">
        <Link
          href="/"
          className="flex items-center gap-2 font-mono text-sm text-ink"
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
            className="py-2 font-mono text-xs text-red-ink underline-offset-4 hover:underline"
          >
            Open tool
          </Link>
        </nav>
      </div>
    </header>
  );
}
