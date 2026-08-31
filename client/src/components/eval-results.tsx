"use client";

import { useEffect, useState } from "react";

import { EvalBreakdowns } from "@/components/eval-breakdowns";
import { Measure, Notice } from "@/components/essay";
import { HeldoutTable } from "@/components/heldout-table";
import { NtireLeaderboard } from "@/components/ntire-leaderboard";
import type { EvalResponse } from "@/lib/types";

export function EvalResults() {
  const [data, setData] = useState<EvalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/eval")
      .then((r) => r.json())
      .then((d: EvalResponse) => setData(d))
      .catch((e) => setError(String(e)));
  }, []);

  const datasets = data?.datasets ?? [];
  const step = datasets.find((d) => d.step != null)?.step;
  const ckpt = datasets.find((d) => d.checkpoint)?.checkpoint;

  return (
    <div className="space-y-6">
      {error && (
        <Measure>
          <Notice>{error}</Notice>
        </Measure>
      )}
      {data?.mode === "demo" && (
        <Measure>
          <Notice>
            Demo data. {data.note ?? "No live eval JSONs found — numbers shown are placeholders."}
          </Notice>
        </Measure>
      )}
      {!data && !error && (
        <div className="figure space-y-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="skeleton h-8 w-full" />
          ))}
        </div>
      )}
      {data && <HeldoutTable datasets={datasets} />}
      {data?.mode === "live" && (step != null || ckpt) && (
        <Measure>
          <p className="caption">
            source: eval/eval_step33500
            {step != null && ` · step ${step.toLocaleString("en-US")}`}
            {ckpt && ` · ${ckpt.split(/[\\/]/).pop()}`}
          </p>
        </Measure>
      )}
      <Measure>
        <h3 className="small-head">NTIRE 2026 open test</h3>
        <p className="mt-1 text-[16px] leading-[1.5] text-ink-body">
          The challenge leaderboard on ROC AUC and robust ROC AUC, from
          Table 3 of the{" "}
          <a
            href="https://arxiv.org/pdf/2604.11487"
            target="_blank"
            rel="noreferrer"
          >
            NTIRE 2026 report
          </a>
          . The published entries are 7B models; Seer is 302M and still third
          on robust AUROC — state of the art at this scale.
        </p>
      </Measure>
      <NtireLeaderboard datasets={datasets} />
      <Measure>
        <h3 className="small-head">Held-out breakdowns</h3>
        <p className="mt-1 text-[16px] leading-[1.5] text-ink-body">
          Where the misses live — by family, generator, and source.
        </p>
      </Measure>
      {data ? (
        <EvalBreakdowns datasets={datasets} />
      ) : (
        !error && (
          <div className="figure space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="skeleton h-8 w-full" />
            ))}
          </div>
        )
      )}
    </div>
  );
}
