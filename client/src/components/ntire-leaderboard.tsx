"use client";

import {
  columnMarks,
  NTIRE_OPEN_TEST,
  SEER_NTIRE_FALLBACK,
  seerNtireFromEval,
  withSeer,
} from "@/lib/ntire-open-test";
import type { EvalDataset } from "@/lib/types";

function auc(v: number): string {
  return v.toFixed(4);
}

function Mark({
  value,
  best,
  second,
}: {
  value: number;
  best?: number;
  second?: number;
}) {
  const cls =
    value === best ? "is-best" : value === second ? "is-second" : "";
  return <span className={`tabular${cls ? ` ${cls}` : ""}`}>{auc(value)}</span>;
}

export function NtireLeaderboard({
  datasets,
}: {
  datasets?: EvalDataset[];
}) {
  const seer = seerNtireFromEval(datasets ?? []) ?? SEER_NTIRE_FALLBACK;
  const rows = withSeer(NTIRE_OPEN_TEST, seer);
  const auroc = columnMarks(rows, "auroc");
  const robust = columnMarks(rows, "robust");

  return (
    <div className="figure overflow-x-auto">
      <table className="paper-table min-w-[420px]">
        <thead>
          <tr>
            <th>Method</th>
            <th>Params</th>
            <th>ROC AUC</th>
            <th>Rob. ROC AUC</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.method} className={r.ours ? "is-ours" : undefined}>
              <td>
                <span className="text-ink-head">{r.method}</span>
                {r.ours && (
                  <span className="text-ink-mute"> (ours)</span>
                )}
              </td>
              <td className={`tabular${r.ours ? " is-best" : " text-ink-mute"}`}>
                {r.params}
              </td>
              <td>
                <Mark value={r.auroc} best={auroc.best} />
              </td>
              <td>
                <Mark
                  value={r.robust}
                  best={robust.best}
                  second={robust.second}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption">
        Published scores from Table 3 of{" "}
        <a
          href="https://arxiv.org/pdf/2604.11487"
          target="_blank"
          rel="noreferrer"
          className="ink-link"
        >
          Gushchin et al., NTIRE 2026 Challenge on Robust AI-Generated Image
          Detection in the Wild
        </a>
      </p>
    </div>
  );
}
