import { Footer } from "@/app/components/Sections";
import { TopBar } from "@/app/components/TopBar";

import { Merge } from "./Merge";

export const metadata = {
  title: "Compose — arxiv→code",
  description:
    "Put one paper's component into another's architecture, and see which assumptions collide.",
};

/** `searchParams` is a Promise in Next 16. */
export default async function MergePage({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  const params = await searchParams;
  const one = (key: string) =>
    (Array.isArray(params[key]) ? params[key][0] : params[key]) ?? "";

  return (
    <div className="min-h-dvh">
      <TopBar />
      <Merge initialHost={one("host")} initialDonor={one("donor")} />
      <Footer />
    </div>
  );
}
