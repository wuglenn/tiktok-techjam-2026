"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { HeatCanvas } from "@/components/heat-canvas";
import {
  IconDownload,
  IconEye,
  IconScan,
  IconUpload,
  IconX,
} from "@/components/icons";
import { ProbGauge, ProbBar, VerdictPill } from "@/components/verdict";
import { bytes as fmtBytes } from "@/lib/format";
import type { AnalyzeResponse, AnalyzeResult } from "@/lib/types";

const MAX_FILES = 12;

interface Card extends AnalyzeResult {
  /** object URL for preview */
  url: string;
}

export default function AnalyzePage() {
  const [cards, setCards] = useState<Card[]>([]);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<AnalyzeResponse | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const run = useCallback(async (fileList: FileList | File[]) => {
    const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    const batch = files.slice(0, MAX_FILES);
    setError(null);
    setBusy(true);
    setPending(batch.length);
    const urlFor = new Map<File, string>();
    batch.forEach((f) => urlFor.set(f, URL.createObjectURL(f)));
    try {
      const form = new FormData();
      batch.forEach((f) => form.append("files", f));
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      const data = (await res.json()) as AnalyzeResponse & { error?: string };
      if (!res.ok) {
        setError(data.error ?? `request failed (${res.status})`);
        return;
      }
      setMeta(data);
      setCards((prev) => {
        const next = [...prev];
        data.results.forEach((r, i) => {
          const f = batch[i];
          next.push({ ...r, url: urlFor.get(f) ?? "" });
        });
        return next;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "upload failed");
    } finally {
      setBusy(false);
      setPending(0);
    }
  }, []);

  // paste images straight from the clipboard
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const files = Array.from(e.clipboardData?.files ?? []).filter((f) =>
        f.type.startsWith("image/"),
      );
      if (files.length) void run(files);
    };
    document.addEventListener("paste", onPaste);
    return () => document.removeEventListener("paste", onPaste);
  }, [run]);

  const dropzone = (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (e.dataTransfer?.files?.length) void run(e.dataTransfer.files);
      }}
      className={`group relative flex min-h-[260px] cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed transition-all ${
        dragging
          ? "border-cyan-400/60 bg-cyan-500/[0.06]"
          : "border-white/[0.1] bg-white/[0.015] hover:border-white/[0.2] hover:bg-white/[0.03]"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => e.target.files && void run(e.target.files)}
      />
      <span
        className={`flex h-14 w-14 items-center justify-center rounded-2xl border transition-colors ${
          dragging
            ? "border-cyan-400/40 bg-cyan-500/10 text-cyan-300"
            : "border-white/[0.08] bg-white/[0.04] text-zinc-400 group-hover:text-zinc-200"
        }`}
      >
        <IconUpload className="h-6 w-6" />
      </span>
      <span className="text-center">
        <span className="block text-sm font-medium text-zinc-200">
          {dragging ? "Drop to analyze" : "Drop images, paste, or click to browse"}
        </span>
        <span className="mt-1 block text-xs text-zinc-500">
          up to {MAX_FILES} at once · JPEG, PNG, WebP, GIF · 40 MB each
        </span>
      </span>
    </label>
  );

  return (
    <div className="space-y-8">
      {/* header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-white">
            Analyze
          </h1>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-zinc-400">
            Run the detector on any image. The global head gives P(AI); the local
            head gives the per-patch heatmap — the overlay shows which regions
            pushed the verdict.
          </p>
        </div>
        <ModeBadge meta={meta} />
      </div>

      {/* mode note */}
      {meta?.mode === "simulated" && (
        <div className="rounded-xl border border-amber-400/20 bg-amber-500/[0.06] px-4 py-3 text-xs leading-relaxed text-amber-200/90">
          <strong className="font-semibold">Demo mode.</strong> {meta.note}
        </div>
      )}
      {meta?.mode === "live" && (
        <div className="rounded-xl border border-emerald-400/20 bg-emerald-500/[0.06] px-4 py-3 text-xs text-emerald-200/90">
          <strong className="font-semibold">Live model.</strong> Serving{" "}
          <span className="tabular">{meta.checkpoint}</span> through the repo&rsquo;s
          Python bridge.
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-rose-400/20 bg-rose-500/[0.06] px-4 py-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      {/* upload */}
      {cards.length === 0 ? (
        dropzone
      ) : (
        <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <IconScan className="h-4 w-4 text-cyan-400" />
            {cards.length} image{cards.length === 1 ? "" : "s"} analyzed
            {busy && ` · ${pending} in flight…`}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() =>
                download("seer_predictions.json", cards.map((c) => ({ image_path: c.name, pred: c.prob_ai })))
              }
              className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.1] bg-white/[0.03] px-3.5 py-2 text-xs font-medium text-zinc-200 transition-colors hover:bg-white/[0.07]"
              title="The deliverable format: image_path + pred per image"
            >
              <IconDownload className="h-3.5 w-3.5" />
              predictions.json
            </button>
            <button
              onClick={() => download("seer_report.json", cards)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.1] bg-white/[0.03] px-3.5 py-2 text-xs font-medium text-zinc-200 transition-colors hover:bg-white/[0.07]"
            >
              <IconDownload className="h-3.5 w-3.5" />
              full report
            </button>
            <button
              onClick={() => {
                cards.forEach((c) => URL.revokeObjectURL(c.url));
                setCards([]);
                setMeta(null);
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.1] bg-white/[0.03] px-3.5 py-2 text-xs font-medium text-zinc-400 transition-colors hover:bg-white/[0.07] hover:text-zinc-200"
            >
              <IconX className="h-3.5 w-3.5" />
              clear
            </button>
          </div>
        </div>
      )}

      {/* results */}
      <div className="grid gap-5 xl:grid-cols-2">
        {cards.map((c, i) => (
          <ResultCard key={`${c.name}-${i}`} c={c} />
        ))}
        {busy &&
          Array.from({ length: Math.min(pending, 3) }).map((_, i) => (
            <div key={`sk-${i}`} className="panel p-5">
              <div className="skeleton h-5 w-24 rounded-full" />
              <div className="skeleton mt-4 aspect-[4/3] w-full" />
              <div className="mt-4 flex justify-between">
                <div className="skeleton h-16 w-32 rounded-xl" />
                <div className="skeleton h-16 w-32 rounded-xl" />
              </div>
            </div>
          ))}
      </div>

      {cards.length > 0 && <div className="h-2" /> /* spacing before re-drop */}
      {cards.length > 0 && (
        <button
          onClick={() => inputRef.current?.click()}
          className="w-full rounded-xl border border-dashed border-white/[0.1] py-4 text-xs font-medium text-zinc-400 transition-colors hover:border-white/[0.2] hover:text-zinc-200"
        >
          + add more images
        </button>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- pieces */

function ModeBadge({ meta }: { meta: AnalyzeResponse | null }) {
  if (!meta) return null;
  const live = meta.mode === "live";
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-xs font-medium ${
        live
          ? "border-emerald-400/25 bg-emerald-500/10 text-emerald-300"
          : "border-amber-400/25 bg-amber-500/10 text-amber-300"
      }`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full animate-pulse-dot ${live ? "bg-emerald-400" : "bg-amber-400"}`}
      />
      {live ? "live inference" : "simulated"}
    </span>
  );
}

function ResultCard({ c }: { c: Card }) {
  const [showHeat, setShowHeat] = useState(true);

  return (
    <div className="panel animate-rise flex flex-col p-4 sm:p-5">
      <div className="flex items-center justify-between gap-3">
        <VerdictPill label={c.label} />
        <span className="truncate text-xs text-zinc-400" title={c.name}>
          {c.name}
        </span>
      </div>

      <div className="relative mt-4 aspect-[4/3] overflow-hidden rounded-xl ring-1 ring-white/10">
        <HeatCanvas grid={c.grid} src={c.url} showHeat={showHeat} />
        {c.width && c.height && (
          <span className="tabular absolute left-2.5 top-2.5 rounded-lg bg-black/55 px-2 py-1 text-[10px] font-medium text-zinc-300 backdrop-blur-sm">
            {c.width}×{c.height}
          </span>
        )}
        <button
          onClick={() => setShowHeat((v) => !v)}
          title="toggle heatmap"
          className={`absolute bottom-2.5 right-2.5 flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-[10px] font-medium backdrop-blur-md transition-colors ${
            showHeat
              ? "bg-cyan-500/85 text-zinc-950"
              : "bg-black/60 text-zinc-300 hover:text-white"
          }`}
        >
          <IconEye className="h-3.5 w-3.5" />
          heatmap
        </button>
      </div>

      <div className="mt-4 flex items-end justify-between gap-4">
        <ProbGauge p={c.prob_ai} size={118} />
        <dl className="grid flex-1 grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
          <Meta
            k="file"
            v={`${c.type?.replace("image/", "").toUpperCase() ?? "–"} · ${fmtBytes(c.bytes)}`}
          />
          <Meta
            k="patch grid"
            v={c.grid ? `${c.grid.length}×${c.grid[0].length}` : "page-only"}
          />
          <Meta k="latency" v={c.elapsedMs != null ? `${c.elapsedMs} ms` : "–"} />
        </dl>
      </div>

      <ProbBar p={c.prob_ai} className="mt-4" />
    </div>
  );
}

function Meta({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 border-b border-white/[0.04] pb-1">
      <dt className="text-zinc-500">{k}</dt>
      <dd className="tabular font-medium text-zinc-300">{v}</dd>
    </div>
  );
}

function download(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
