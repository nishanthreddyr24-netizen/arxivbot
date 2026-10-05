import Link from "next/link";

/**
 * Placeholder for the tool itself, which is built separately. It exists so the
 * homepage's Go button lands somewhere that shows the request was carried,
 * rather than on a 404.
 *
 * `searchParams` is a Promise in Next 16.
 */
export default async function AppPage({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  const params = await searchParams;
  const read = (key: string) =>
    typeof params[key] === "string" ? (params[key] as string) : "";

  return (
    <main className="mx-auto max-w-[1120px] px-4 py-16 sm:px-6">
      <Link
        href="/"
        className="font-mono text-[13px] text-muted underline-offset-4 hover:text-red-ink hover:underline"
      >
        &larr; back
      </Link>
      <h1 className="mt-8 text-[clamp(1.6rem,4vw,2.2rem)] leading-tight text-ink">
        The tool is not built yet.
      </h1>
      <dl className="mt-8 max-w-[640px] border-t border-hairline font-mono text-[13px]">
        {(["paper", "q", "target"] as const).map((key) => (
          <div
            key={key}
            className="flex gap-6 border-b border-hairline py-3"
          >
            <dt className="w-[84px] shrink-0 text-muted">{key}</dt>
            <dd className="min-w-0 break-words text-ink">
              {read(key) || <span className="text-faint">—</span>}
            </dd>
          </div>
        ))}
      </dl>
    </main>
  );
}
