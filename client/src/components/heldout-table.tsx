"use client";

import { useEffect, useId, useRef, useState } from "react";

import { evalDisplayName, evalKey, finite, HELDOUT_NOTES, HELDOUT_ORDER } from "@/lib/eval-labels";
import { pct } from "@/lib/format";
import type { EvalDataset } from "@/lib/types";

interface HeldoutRow {
  set: string;
  key: string;
  note?: string;
  n: string;
  ratio: string;
  macro?: number;
  map?: number;
  auroc?: number;
  f1?: number;
  fpr?: number;
  fnr?: number;
}

const SCORE_KEYS = ["macro", "map", "auroc", "f1", "fpr", "fnr"] as const;
type ScoreKey = (typeof SCORE_KEYS)[number];
const LOWER_IS_BETTER = new Set<ScoreKey>(["fpr", "fnr"]);

function gcd(a: number, b: number): number {
  let x = Math.abs(Math.round(a));
  let y = Math.abs(Math.round(b));
  while (y) {
    const t = y;
    y = x % y;
    x = t;
  }
  return x || 1;
}

function trimZeros(s: string): string {
  return s.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

/** fake∶real, reduced when the parts stay small, otherwise scaled to 1. */
function fakeRealRatio(nFake: number, nReal: number): string {
  if (nFake === 0 && nReal === 0) return "—";
  if (nReal === 0) return "1∶0";
  if (nFake === 0) return "0∶1";
  const g = gcd(nFake, nReal);
  const a = nFake / g;
  const b = nReal / g;
  if (a <= 20 && b <= 20) return `${a}∶${b}`;
  const q = nFake / nReal;
  const digits = q >= 10 ? 1 : 2;
  return `${trimZeros(q.toFixed(digits))}∶1`;
}

function toRow(ds: EvalDataset): HeldoutRow {
  const m = ds.metrics;
  const key = evalKey(ds.name, ds.file);
  const nFake = m.n_fake ?? 0;
  const nReal = m.n_real ?? 0;
  const realsOnly = nFake === 0 && nReal > 0;
  const fakesOnly = nReal === 0 && nFake > 0;
  return {
    set: evalDisplayName(ds.name, ds.file),
    key,
    note: HELDOUT_NOTES[key],
    n: (m.n ?? 0).toLocaleString("en-US"),
    ratio: fakeRealRatio(nFake, nReal),
    macro: realsOnly || fakesOnly ? undefined : finite(m.macro_accuracy),
    map: finite(m.mAP as number | undefined),
    auroc: finite(m.auroc),
    f1: realsOnly ? undefined : finite(m.f1),
    fpr: fakesOnly ? undefined : finite(m.fpr),
    fnr: realsOnly ? undefined : finite(m.fnr),
  };
}

function bestOf(rows: HeldoutRow[], key: ScoreKey): number | undefined {
  const vals = rows.map((r) => r[key]).filter((v): v is number => v != null);
  if (!vals.length) return undefined;
  return LOWER_IS_BETTER.has(key) ? Math.min(...vals) : Math.max(...vals);
}

function Score({
  value,
  best,
}: {
  value: number | undefined;
  best: number | undefined;
}) {
  if (value == null) return <span className="text-ink-mute">—</span>;
  const win = best != null && value === best;
  return (
    <span className={`tabular${win ? " is-best" : ""}`}>{pct(value)}</span>
  );
}

function SetTip({ name, note }: { name: string; note?: string }) {
  const id = useId();
  const btnRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  function place() {
    const r = btnRef.current?.getBoundingClientRect();
    if (!r) return;
    const width = Math.min(320, window.innerWidth - 24);
    const left = Math.min(Math.max(12, r.left), window.innerWidth - width - 12);
    const below = r.bottom + 8;
    const above = r.top - 8;
    const top = below + 140 > window.innerHeight ? above : below;
    setPos({ top, left });
  }

  useEffect(() => {
    if (!open) return;
    place();
    const onScroll = () => place();
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onScroll);
    };
  }, [open]);

  if (!note) return <span className="text-ink-head">{name}</span>;

  const showAbove = pos.top < (btnRef.current?.getBoundingClientRect().top ?? 0);

  return (
    <span
      className="heldout-tip"
      onMouseEnter={() => {
        place();
        setOpen(true);
      }}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        ref={btnRef}
        type="button"
        className="heldout-tip-trigger"
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onFocus={() => {
          place();
          setOpen(true);
        }}
        onBlur={() => setOpen(false)}
      >
        {name}
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className={`heldout-tip-bubble${showAbove ? " is-above" : ""}`}
          style={{ top: pos.top, left: pos.left, width: Math.min(320, window.innerWidth - 24) }}
        >
          {note}
        </span>
      )}
    </span>
  );
}

export function HeldoutTable({ datasets }: { datasets: EvalDataset[] }) {
  const ranked = [...datasets].sort((a, b) => {
    const ia = HELDOUT_ORDER.indexOf(evalKey(a.name, a.file));
    const ib = HELDOUT_ORDER.indexOf(evalKey(b.name, b.file));
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });
  const rows = ranked.map(toRow);
  const best = Object.fromEntries(
    SCORE_KEYS.map((k) => [k, bestOf(rows, k)]),
  ) as Record<ScoreKey, number | undefined>;

  if (!rows.length) {
    return <p className="measure caption">No held-out eval JSONs found.</p>;
  }

  return (
    <div className="figure overflow-x-auto">
      <table className="paper-table min-w-[620px]">
        <thead>
          <tr>
            <th>Held-out set</th>
            <th>n (fake∶real)</th>
            <th>Macro acc</th>
            <th>mAP</th>
            <th>AUROC</th>
            <th>F1</th>
            <th>FPR</th>
            <th>FNR</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <td>
                <SetTip name={r.set} note={r.note} />
              </td>
              <td className="tabular">
                {r.n} <span className="text-ink-mute">({r.ratio})</span>
              </td>
              <td>
                <Score value={r.macro} best={best.macro} />
              </td>
              <td>
                <Score value={r.map} best={best.map} />
              </td>
              <td>
                <Score value={r.auroc} best={best.auroc} />
              </td>
              <td>
                <Score value={r.f1} best={best.f1} />
              </td>
              <td>
                <Score value={r.fpr} best={best.fpr} />
              </td>
              <td>
                <Score value={r.fnr} best={best.fnr} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
