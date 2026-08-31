"use client";

import { IconCheck, IconSpark } from "@/components/icons";

export function VerdictPill({
  label,
  className,
}: {
  label: "AI" | "REAL" | string;
  className?: string;
}) {
  const ai = label === "AI";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] ${
        ai
          ? "border-rose-400/25 bg-rose-500/10 text-rose-300"
          : "border-emerald-400/25 bg-emerald-500/10 text-emerald-300"
      } ${className ?? ""}`}
    >
      {ai ? (
        <IconSpark className="h-3.5 w-3.5" />
      ) : (
        <IconCheck className="h-3.5 w-3.5" />
      )}
      {ai ? "AI generated" : "Real"}
    </span>
  );
}

/**
 * Probability bar on a fixed emerald->amber->rose gradient; the 0.5 decision
 * threshold is marked. The gradient is revealed with clip-path so it stays
 * anchored to the track as the value animates.
 */
export function ProbBar({ p, className }: { p: number; className?: string }) {
  const pct = Math.min(1, Math.max(0, p)) * 100;
  return (
    <div className={`relative h-2 overflow-hidden rounded-full bg-white/[0.06] ${className ?? ""}`}>
      <div
        className="absolute inset-0 rounded-full bg-[linear-gradient(90deg,#34d399_0%,#a3e635_30%,#fbbf24_50%,#fb923c_70%,#fb7185_100%)] transition-[clip-path] duration-700 ease-out"
        style={{ clipPath: `inset(0 ${100 - pct}% 0 0)` }}
      />
      <div className="absolute inset-y-0 left-1/2 w-px bg-white/50" />
    </div>
  );
}

/** Semicircle gauge for P(AI), 0 at left, 1 at right. */
export function ProbGauge({
  p,
  size = 132,
  label = "P(AI)",
}: {
  p: number;
  size?: number;
  label?: string;
}) {
  const v = Math.min(1, Math.max(0, p));
  const w = 10;
  const r = (size - w) / 2 - 6;
  const cx = size / 2;
  const cy = size / 2 + 2;
  const arc = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
  const theta = Math.PI * (1 - v);
  const mx = cx + r * Math.cos(theta);
  const my = cy - r * Math.sin(theta);
  return (
    <div className="relative" style={{ width: size, height: size * 0.62 }}>
      <svg width={size} height={size * 0.62} viewBox={`0 0 ${size} ${size * 0.62}`}>
        <defs>
          <linearGradient id="gauge-grad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="#34d399" />
            <stop offset="0.5" stopColor="#fbbf24" />
            <stop offset="1" stopColor="#fb7185" />
          </linearGradient>
        </defs>
        <path d={arc} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={w} strokeLinecap="round" />
        <path
          d={arc}
          fill="none"
          stroke="url(#gauge-grad)"
          strokeWidth={w}
          strokeLinecap="round"
          pathLength={1}
          strokeDasharray={`${v} 1`}
          className="transition-all duration-700 ease-out"
        />
        <circle cx={mx} cy={my} r={w / 2 + 2.5} fill="#0a0a0c" stroke="url(#gauge-grad)" strokeWidth={2.5} />
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center">
        <div className={`tabular text-2xl font-semibold ${v >= 0.5 ? "text-rose-300" : "text-emerald-300"}`}>
          {v.toFixed(3)}
        </div>
        <div className="text-[10px] font-medium tracking-[0.14em] text-zinc-500">{label}</div>
      </div>
    </div>
  );
}
