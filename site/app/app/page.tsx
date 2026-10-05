import Link from "next/link";

import { Tool } from "./Tool";

export const dynamic = "force-dynamic";

/** `searchParams` is a Promise in Next 16. */
export default async function AppPage({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  const params = await searchParams;
  const read = (key: string) =>
    typeof params[key] === "string" ? (params[key] as string) : "";

  return (
    <>
      <header className="flex h-12 items-center gap-5 border-b border-hairline px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-mono text-[13px] text-ink">
          <span aria-hidden className="size-1.5 bg-red-fill" />
          arxiv&rarr;code
        </Link>
        <Link
          href="/"
          className="ml-auto font-mono text-[13px] text-muted underline-offset-4 hover:text-red-ink hover:underline"
        >
          New paper
        </Link>
      </header>
      <Tool paper={read("paper")} query={read("q")} />
    </>
  );
}
