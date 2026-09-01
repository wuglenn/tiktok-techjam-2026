"use client";

import { useEffect, useState } from "react";

import { Chip, Measure, Notice, Tabs } from "@/components/essay";
import { HeatLegend } from "@/components/heat-canvas";
import { ProbBar } from "@/components/verdict";
import { evalDisplayName } from "@/lib/eval-labels";
import type { ErrorEntry, EvalResponse } from "@/lib/types";

export default function ErrorsPage() {
  const [data, setData] = useState<EvalResponse | null>(null);
  const [active, setActive] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/eval")
      .then((r) => r.json())
      .then((d: EvalResponse) => setData(d))
      .catch((e) => setError(String(e)));
  }, []);

  const withErrors = (data?.datasets ?? []).filter(
    (d) => d.errors && d.errors.length > 0,
  );
  const ds = withErrors[Math.min(active, Math.max(0, withErrors.length - 1))];
  const fps = ds?.errors?.filter((e) => e.kind === "fp") ?? [];
  const fns = ds?.errors?.filter((e) => e.kind === "fn") ?? [];

  return (
    <div className="space-y-10">
      <Measure className="essay">
        <h1 className="essay-title">The mistakes that teach</h1>
        <p className="mt-4">
          Aggregate metrics say <em>that</em> a detector fails, never{" "}
          <em>why</em>. These are its most confident mistakes, ranked by
          confidence — a miss at P(AI)=0.51 is noise, one at 0.99 is a lesson.
        </p>
        <div className="mt-4">
          <HeatLegend />
        </div>
      </Measure>

      {error && (
        <Measure>
          <Notice>{error}</Notice>
        </Measure>
      )}

      {data?.mode === "demo" && (
        <Measure>
          <Notice>
            Demo data.{" "}
            {data.note ??
              "No real eval results found — panels below are illustrative placeholders."}
          </Notice>
        </Measure>
      )}

      {withErrors.length > 1 && (
        <Measure>
          <Tabs
            items={withErrors.map((d) => ({ label: evalDisplayName(d.name, d.file) }))}
            active={active}
            onChange={setActive}
          />
        </Measure>
      )}

      {data && !ds && (
        <Measure>
          <Notice>No error-analysis panels in this eval dump.</Notice>
        </Measure>
      )}

      {fps.length > 0 && (
        <ErrorSection
          title="False positives"
          blurb="Real photographs the live checkpoint called AI."
          entries={fps}
        />
      )}
      {fns.length > 0 && (
        <ErrorSection
          title="False negatives"
          blurb="Generated images the checkpoint called real."
          entries={fns}
        />
      )}

      {!data && !error && (
        <div className="figure space-y-6">
          {Array.from({ length: 2 }).map((_, i) => (
            <div key={i} className="skeleton aspect-video w-full" />
          ))}
        </div>
      )}

      <Measure className="essay">
        <h2 className="essay-title">Trade-offs in this approach</h2>
        <p className="mt-4">
          <em>Robustness costs clean accuracy.</em> Wild-simulation augmentation
          is applied symmetrically to both classes, which trades a fraction of a
          point of clean accuracy for holding up under JPEG q30 and heavy blur.
        </p>
        <p>
          <em>FPR control is data-bound.</em> False positives concentrate in real
          images with AI-like statistics — digital art, high-ISO noise, heavy
          bokeh. An eval set without those classes cannot reveal them.
        </p>
        <p>
          <em>These FNs are the generator, not the pipeline.</em> The three
          panels above are clean stills from DALL·E 3 Advanced, Midjourney 7,
          and Ideogram 2.0 that the global head under-scored. The heatmaps
          are the local head on those same forwards — a flat cool field
          means it missed the image everywhere, not a random overlay.
        </p>
      </Measure>
    </div>
  );
}

function ErrorSection({
  title,
  blurb,
  entries,
}: {
  title: string;
  blurb: string;
  entries: ErrorEntry[];
}) {
  return (
    <section className="space-y-6">
      <Measure>
        <h2 className="essay-title">{title}</h2>
        <p className="mt-3 text-[16px] leading-[1.5] text-ink-body">{blurb}</p>
      </Measure>
      <div className="space-y-8">
        {entries.map((e) => (
          <ErrorFigure key={`${e.kind}-${e.rank}`} e={e} />
        ))}
      </div>
    </section>
  );
}

const ERROR_SRC: Record<string, string> = {
  "fp-1": "/errors/fp_01_p0.986_docci.png",
  "fp-2": "/errors/fp_02_p0.728_imagenet.png",
  "fp-3": "/errors/fp_03_p0.647_docci.png",
  "fn-1": "/errors/fn_01_p0.032_1d4135eb385210a437a2c66e5a1487c1.png",
  "fn-2": "/errors/fn_02_p0.004_midjourney-7.png",
  "fn-3": "/errors/fn_03_p0.318_ideogram-2.0.png",
};

function errorImageSrc(e: ErrorEntry): string | null {
  return ERROR_SRC[`${e.kind}-${e.rank}`] ?? null;
}

function ErrorFigure({ e }: { e: ErrorEntry }) {
  const src = errorImageSrc(e);
  return (
    <figure className="figure">
      <div className="figure-frame relative aspect-video overflow-hidden">
        {src ? (
          <img
            src={src}
            alt=""
            className="absolute inset-0 h-full w-full object-contain"
          />
        ) : null}
      </div>
      <figcaption className="caption">
        {e.kind === "fp" ? "false positive" : "false negative"} · #{e.rank}
        {!src && " · demo panel, no image on disk"}
      </figcaption>
      <div className="mt-3 flex flex-wrap items-baseline justify-between gap-3">
        <span className="text-[16px] text-ink-head">{e.generator ?? "unknown source"}</span>
        <span className="caption">truth: {e.label === 1 ? "AI" : "real"}</span>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <span className="tabular w-12 font-mono text-[14px] text-ink-head">
          {e.prob_ai.toFixed(3)}
        </span>
        <ProbBar p={e.prob_ai} className="flex-1" />
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        {(e.distortions ?? []).length > 0 ? (
          (e.distortions ?? []).map((d) => <Chip key={d}>{d}</Chip>)
        ) : (
          <Chip>clean</Chip>
        )}
      </div>
    </figure>
  );
}
