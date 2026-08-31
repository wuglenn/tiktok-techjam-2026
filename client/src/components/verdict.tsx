"use client";

export function VerdictPill({
  label,
  className,
}: {
  label: "AI" | "REAL" | string;
  className?: string;
}) {
  const ai = label === "AI";
  return (
    <span className={`chip ${className ?? ""}`}>
      {ai ? "AI generated" : "Real"}
    </span>
  );
}

export function ProbBar({ p, className }: { p: number; className?: string }) {
  const pct = Math.min(1, Math.max(0, p)) * 100;
  return (
    <div className={`relative h-[2px] bg-chip ${className ?? ""}`}>
      <div
        className="absolute inset-y-0 left-0 bg-ink-head"
        style={{ width: `${pct}%` }}
      />
      <div className="absolute inset-y-0 left-1/2 w-px bg-rule" />
    </div>
  );
}

export function ProbGauge({
  p,
  label = "P(AI)",
}: {
  p: number;
  size?: number;
  label?: string;
}) {
  const v = Math.min(1, Math.max(0, p));
  return (
    <div>
      <div className="tabular text-[28px] leading-none text-ink-head">
        {v.toFixed(3)}
      </div>
      <div className="caption mt-1">{label}</div>
    </div>
  );
}
