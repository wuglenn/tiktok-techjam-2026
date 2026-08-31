"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { LogoMark } from "@/components/icons";
import type { StatusResponse } from "@/lib/types";

const NAV = [
  { href: "/", label: "Overview", exact: true },
  { href: "/analyze", label: "Analyze", exact: false },
  { href: "/robustness", label: "Robustness", exact: false },
  { href: "/errors", label: "Errors", exact: false },
];

/**
 * The only app chrome: a slim top bar with navigation on the left and the
 * live/demo mode on the right. Layout owns page structure; this owns nav.
 */
export function AppHeader() {
  const pathname = usePathname();
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    let alive = true;
    fetch("/api/status")
      .then((r) => r.json())
      .then((d: StatusResponse) => alive && setStatus(d))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  return (
    <header className="sticky top-0 z-40 border-b border-white/[0.06] bg-zinc-950/70 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex shrink-0 items-center gap-2.5" aria-label="Seer home">
          <LogoMark className="h-7 w-7" />
          <span className="hidden text-sm font-semibold tracking-[0.28em] text-white sm:block">
            SEER
          </span>
        </Link>

        <nav className="flex flex-1 items-center gap-1 overflow-x-auto">
          {NAV.map(({ href, label, exact }) => {
            const active = exact ? pathname === href : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`shrink-0 rounded-full px-3 py-1.5 text-sm transition-colors ${
                  active
                    ? "bg-white/[0.07] text-white"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>

      </div>
    </header>
  );
}
