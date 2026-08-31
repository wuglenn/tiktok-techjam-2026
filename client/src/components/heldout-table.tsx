"use client";

import { Fragment, useState } from "react";

import { MetricBar } from "@/components/charts";
import { IconInfo } from "@/components/icons";
import { pct } from "@/lib/format";

/**
 * The held-out results table from docs/deliverables/heldout-eval-step27500.md.
 * Rows can carry a note (shown via the info icon) — used for sets whose
 * collection protocol needs context to read the numbers honestly.
 */
interface HeldoutRow {
  set: string;
  note?: string;
  n: string;
  split: string;
  macro?: number;
  map?: number;
  auroc?: number;
  f1?: number;
  fpr?: number;
  fnr?: number;
}

const HELDOUT: HeldoutRow[] = [
  {
    set: "CommunityForensics-Eval",
    n: "51,836",
    split: "25,918 / 25,918",
    macro: 0.9565,
    map: 0.9962,
    auroc: 0.9954,
    f1: 0.9546,
    fpr: 0.0018,
    fnr: 0.0852,
  },
  {
    set: "OpenFake core/test",
    n: "89,225",
    split: "45,697 / 43,528",
    macro: 0.9719,
    map: 0.9984,
    auroc: 0.9981,
    f1: 0.9712,
    fpr: 0.0021,
    fnr: 0.0542,
  },
  {
    set: "OpenFake reddit/test",
    note:
      "In-the-wild test split only. Synthetic images are scraped from AI-generation subreddits, real images from photography subreddits — labels follow the subreddit, not the generator. Use this to evaluate how detectors trained on core transfer to naturally circulated content, with platform compression and unknown provenance.",
    n: "36,227",
    split: "29,116 / 7,111",
    macro: 0.8905,
    map: 0.9928,
    auroc: 0.9728,
    f1: 0.8874,
    fpr: 0.0205,
    fnr: 0.1984,
  },
  {
    set: "MIRAGE",
    n: "12,073",
    split: "10,682 / 1,391",
    macro: 0.8626,
    map: 0.9902,
    auroc: 0.9302,
    f1: 0.8732,
    fpr: 0.0554,
    fnr: 0.2194,
  },
  {
    set: "COCO val2017 (reals only)",
    n: "5,000",
    split: "0 / 5,000",
    fpr: 0.001,
  },
];

export function HeldoutTable() {
  const [open, setOpen] = useState<string | null>(null);

  return (
    <div className="panel overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px] text-sm">
          <thead>
            <tr className="border-b border-white/[0.06] text-left text-[11px] uppercase tracking-[0.12em] text-zinc-500">
              <th className="px-5 py-3.5 font-medium">Held-out set</th>
              <th className="px-4 py-3.5 font-medium">n (fake / real)</th>
              <th className="px-4 py-3.5 font-medium">Macro acc</th>
              <th className="px-4 py-3.5 font-medium">mAP</th>
              <th className="px-4 py-3.5 font-medium">AUROC</th>
              <th className="px-4 py-3.5 font-medium">F1</th>
              <th className="px-4 py-3.5 font-medium">FPR</th>
              <th className="px-5 py-3.5 font-medium">FNR</th>
            </tr>
          </thead>
          <tbody>
            {HELDOUT.map((r) => {
              const isOpen = open === r.set;
              return (
                <Fragment key={r.set}>
                  <tr
                    className={`transition-colors hover:bg-white/[0.03] ${
                      r !== HELDOUT[HELDOUT.length - 1]
                        ? "border-b border-white/[0.04]"
                        : ""
                    }`}
                  >
                    <td className="px-5 py-3.5 font-medium text-zinc-100">
                      <span className="flex items-center gap-1.5">
                        {r.set}
                        {r.note && (
                          <button
                            onClick={() => setOpen(isOpen ? null : r.set)}
                            aria-expanded={isOpen}
                            aria-label={`about ${r.set}`}
                            title="about this set"
                            className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full transition-colors ${
                              isOpen
                                ? "bg-cyan-500/20 text-cyan-300"
                                : "text-zinc-500 hover:bg-white/[0.06] hover:text-zinc-300"
                            }`}
                          >
                            <IconInfo className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </span>
                    </td>
                    <td className="tabular px-4 py-3.5 text-zinc-400">
                      <span className="text-zinc-300">{r.n}</span>
                      <span className="text-zinc-500"> ({r.split})</span>
                    </td>
                    <td className="px-4 py-3.5">
                      {r.macro != null ? (
                        <div className="flex items-center gap-2.5">
                          <span className="tabular w-12 font-medium text-white">
                            {pct(r.macro)}
                          </span>
                          <div className="w-20">
                            <MetricBar value={r.macro} />
                          </div>
                        </div>
                      ) : (
                        <span className="text-zinc-600">—</span>
                      )}
                    </td>
                    <td className="tabular px-4 py-3.5 text-zinc-300">
                      {r.map != null ? pct(r.map) : "—"}
                    </td>
                    <td className="tabular px-4 py-3.5 text-zinc-300">
                      {r.auroc != null ? pct(r.auroc) : "—"}
                    </td>
                    <td className="tabular px-4 py-3.5 text-zinc-300">
                      {r.f1 != null ? pct(r.f1) : "—"}
                    </td>
                    <td className="tabular px-4 py-3.5 text-emerald-300/90">
                      {r.fpr != null ? pct(r.fpr) : "—"}
                    </td>
                    <td className="tabular px-5 py-3.5 text-amber-300/90">
                      {r.fnr != null ? pct(r.fnr) : "—"}
                    </td>
                  </tr>
                  {isOpen && r.note && (
                    <tr className="bg-white/[0.02]">
                      <td colSpan={8} className="border-b border-white/[0.04] px-5 py-3.5">
                        <p className="max-w-3xl text-[11px] leading-relaxed text-zinc-400">
                          {r.note}
                        </p>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
