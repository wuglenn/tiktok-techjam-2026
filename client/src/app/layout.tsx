import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";

import { Shell } from "@/components/shell";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Seer — AI-generated image detector",
    template: "%s · Seer",
  },
  description:
    "A sub-2B-parameter AI-generated image detector: DINOv3 ViT-L with dual global + patch heads. Image-level verdicts, pixel-level heatmaps. TikTok TechJam 2026 Track 5.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={GeistSans.variable}>
      <body className="min-h-screen font-sans antialiased">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
