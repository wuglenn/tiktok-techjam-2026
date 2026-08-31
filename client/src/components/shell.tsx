"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import {
  IconAlert,
  IconChart,
  IconGrid,
  IconScan,
  LogoMark,
} from "@/components/icons";
import type { StatusResponse } from "@/lib/types";

const NAV = [
  { href: "/", label: "Overview", Icon: IconGrid, exact: true },
  { href: "/analyze", label: "Analyze", Icon: IconScan, exact: false },
  { href: "/robustness", label: "Robustness", Icon: IconChart, exact: false },
  { href: "/errors", label: "Error analysis", Icon: IconAlert, exact: false },
];

function useStatus() {
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
  return status;
}

function StatusCard({ status }: { status: StatusResponse | null }) {
  if (!status) {
    return (
      <div className="panel p-3.5 text-xs text-zinc-500">
        <div className="h-2 w-20 rounded-full bg-zinc-800 animate-pulse-dot" />
      </div>
    );
  }
  const live = status.mode === "live";
  return (
    <div className="panel p-3.5">
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 rounded-full animate-pulse-dot ${
            live ? "bg-emerald-400" : "bg-amber-400"
          }`}
        />
        <span className="text-xs font-medium tracking-wide text-zinc-200">
          {live ? "Live model" : "Demo mode"}
        </span>
      </div>
      <p className="mt-1.5 truncate text-[11px] leading-4 text-zinc-500" title={status.checkpoint ?? undefined}>
        {live ? status.checkpoint : "no checkpoint — simulated verdicts"}
      </p>
    </div>
  );
}

function SidebarNav({ pathname }: { pathname: string }) {
  const status = useStatus();
  return (
    <aside className="fixed inset-y-0 left-0 z-40 hidden w-60 flex-col border-r border-white/[0.06] bg-zinc-950/60 backdrop-blur-xl lg:flex">
      <Link href="/" className="flex items-center gap-3 px-5 pb-6 pt-6">
        <LogoMark className="h-9 w-9" />
        <span>
          <span className="block text-lg font-semibold tracking-[0.28em] text-white">
            SEER
          </span>
          <span className="block text-[10px] font-medium uppercase tracking-[0.18em] text-zinc-500">
            AI image detector
          </span>
        </span>
      </Link>

      <nav className="flex flex-1 flex-col gap-1 px-3">
        {NAV.map(({ href, label, Icon, exact }) => {
          const active = exact ? pathname === href : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors ${
                active
                  ? "bg-white/[0.07] text-white"
                  : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200"
              }`}
            >
              {active && (
                <span className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-full bg-cyan-400" />
              )}
              <Icon className="h-[18px] w-[18px]" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="space-y-3 p-3">
        <StatusCard status={status} />
        <p className="px-1 text-[10px] leading-4 text-zinc-600">
          302M params · DINOv3 ViT-L/16
          <br />
          TikTok TechJam 2026 · Track 5
        </p>
      </div>
    </aside>
  );
}

function MobileHeader({ pathname }: { pathname: string }) {
  return (
    <header className="sticky top-0 z-40 border-b border-white/[0.06] bg-zinc-950/80 backdrop-blur-xl lg:hidden">
      <div className="flex items-center gap-3 px-4 py-3">
        <LogoMark className="h-8 w-8" />
        <span className="text-base font-semibold tracking-[0.28em] text-white">
          SEER
        </span>
      </div>
      <nav className="flex gap-1 overflow-x-auto px-3 pb-3">
        {NAV.map(({ href, label, Icon, exact }) => {
          const active = exact ? pathname === href : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex shrink-0 items-center gap-2 rounded-full px-3.5 py-1.5 text-[13px] transition-colors ${
                active
                  ? "bg-white/[0.08] text-white"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="min-h-screen">
      <SidebarNav pathname={pathname} />
      <MobileHeader pathname={pathname} />
      <main className="lg:pl-60">
        <div className="mx-auto w-full max-w-6xl px-4 pb-24 pt-8 sm:px-6 lg:px-10 lg:pt-12">
          {children}
        </div>
      </main>
    </div>
  );
}
