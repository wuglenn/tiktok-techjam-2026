import type { Metadata } from "next";
import { Instrument_Serif } from "next/font/google";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";

import { AppHeader } from "@/components/app-header";
import { SiteFooter } from "@/components/essay";

import "./globals.css";

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
  variable: "--font-instrument-serif",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Seer — a detector for AI-generated images",
    template: "%s · Seer",
  },
  description:
    "A 302-million-parameter detector: DINOv3 ViT-L with dual global and patch heads. Image-level verdicts and pixel-level heatmaps. TikTok TechJam 2026 Track 5.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${GeistSans.variable} ${GeistMono.variable} ${instrumentSerif.variable}`}
    >
      <body className="paper-in min-h-screen font-sans antialiased">
        <div className="paper-grain" aria-hidden />
        <AppHeader />
        <main>{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
