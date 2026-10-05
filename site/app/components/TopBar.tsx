import Link from "next/link";

const LINKS = [
  { label: "Examples", href: "#examples" },
  { label: "Docs", href: "#how-it-works" },
  { label: "GitHub", href: "https://github.com/nishanthreddyr24-netizen/arxivbot" },
  { label: "Open tool", href: "/app" },
];

export function TopBar() {
  return (
    <header className="h-12 border-b border-hairline">
      <div className="mx-auto flex h-full max-w-[1120px] items-center gap-6 px-4 sm:px-6">
        <Link
          href="/"
          className="flex items-center gap-2 font-mono text-[13px] text-ink"
        >
          {/* The only red that is not interactive: a 6px register mark. */}
          <span aria-hidden className="size-1.5 bg-red-fill" />
          arxiv&rarr;code
        </Link>

        <nav className="ml-auto flex items-center gap-5">
          {LINKS.map((link) => (
            <Link
              key={link.label}
              href={link.href}
              className="font-mono text-[13px] text-muted underline-offset-4 transition-colors hover:text-red-ink hover:underline"
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
