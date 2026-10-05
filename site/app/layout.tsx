import type { Metadata } from "next";
import { JetBrains_Mono, Source_Serif_4 } from "next/font/google";

import "./globals.css";

const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  variable: "--font-source-serif",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "arxiv→code",
  description:
    "Paste an arXiv paper, ask for a component, and get code traced back to the section it came from.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sourceSerif.variable} ${jetbrainsMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
