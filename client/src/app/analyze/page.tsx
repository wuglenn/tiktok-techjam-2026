"use client";

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";

import { Measure, Notice } from "@/components/essay";
import { HeatCanvas } from "@/components/heat-canvas";
import { ProbBar, ProbGauge, VerdictPill } from "@/components/verdict";
import { bytes as fmtBytes } from "@/lib/format";
import type { AnalyzeResponse, AnalyzeResult, StatusResponse } from "@/lib/types";

const MAX_FILES = 12;

interface Card extends AnalyzeResult {
  url: string;
}

export default function AnalyzePage() {
  const [cards, setCards] = useState<Card[]>([]);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<AnalyzeResponse | null>(null);
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [useModal, setUseModal] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let alive = true;
    const pull = () => {
      fetch("/api/status")
        .then((r) => r.json())
        .then((d: StatusResponse) => alive && setStatus(d))
        .catch(() => {});
    };
    pull();
    const id = setInterval(pull, 4000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // restore the persisted backend flag after mount (avoids an SSR/localStorage
  // hydration mismatch), and write it back only on user action
  useEffect(() => {
    try {
      setUseModal(window.localStorage.getItem("seer-use-modal") === "1");
    } catch {
      /* private mode etc. */
    }
  }, []);

  const toggleModal = useCallback((next: boolean) => {
    setUseModal(next);
    try {
      window.localStorage.setItem("seer-use-modal", next ? "1" : "0");
    } catch {
      /* private mode etc. */
    }
  }, []);

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
      if (useModal) form.append("backend", "modal");
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
  }, [useModal]);

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
      className={`dropzone ${dragging ? "is-dragging" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => e.target.files && void run(e.target.files)}
      />
      <span className="text-[16px] text-ink-head">
        {dragging ? "Drop to analyze." : "Drop images here, or paste, or browse."}
      </span>
      <span className="caption">
        up to {MAX_FILES} at once · JPEG, PNG, WebP, GIF · 40 MB each
      </span>
    </label>
  );

  return (
    <div className="space-y-10">
      <Measure className="essay">
        <h1 className="essay-title">Analyze an image</h1>
        <p className="mt-4">
          We run the detector on whatever you bring. The global head returns
          P(AI); the local head returns the per-patch heatmap — the overlay
          shows which regions pushed the verdict.
        </p>
        <div className="mt-3">
          <ModeBadge meta={meta} status={status} />
        </div>
        {status?.modal && (
          <label className="caption mt-3 flex w-fit cursor-pointer select-none items-center gap-2">
            <input
              type="checkbox"
              checked={useModal}
              onChange={(e) => toggleModal(e.target.checked)}
              className="size-3.5 accent-current"
            />
            <span>
              Score on Modal{" "}
              {status.modal.ok
                ? `— remote ${status.modal.device ?? "GPU"}, ready`
                : status.modal.ok === false
                  ? `— ${status.modal.error ?? "not ready"}`
                  : "— cold; the first request boots the GPU container"}
            </span>
          </label>
        )}
      </Measure>

      <Measure>
        {(meta?.mode === "simulated" || (!meta && status && status.mode !== "live")) && (
          <Notice>
            {status?.error ? "The checkpoint is still loading. " : "Demo mode. "}
            {meta?.note ??
              status?.error ??
              "Waiting for the live checkpoint — start client/scripts/seer_serve.py with best.pt."}
          </Notice>
        )}
        {(meta?.mode === "live" || (!meta && status?.mode === "live")) && (
          <Notice>
            {meta?.backend === "modal" ? "Live model on Modal." : "Live model."} Serving{" "}
            {meta?.checkpoint ?? status?.checkpoint}
            {meta?.backend === "modal"
              ? status?.modal?.device
                ? ` (${status.modal.device})`
                : ""
              : status?.device
                ? ` on ${status.device}`
                : " through the Python bridge"}
            .
          </Notice>
        )}
        {error && <Notice>{error}</Notice>}
      </Measure>

      {cards.length === 0 ? (
        <div className="figure">{dropzone}</div>
      ) : (
        <Measure>
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <p className="caption">
              {cards.length} image{cards.length === 1 ? "" : "s"} analyzed
              {busy && ` · ${pending} in flight`}
            </p>
            <div className="footer-links">
              <button
                type="button"
                className="ink-link"
                onClick={() =>
                  download(
                    "seer_predictions.json",
                    cards.map((c) => ({ image_path: c.name, pred: c.prob_ai })),
                  )
                }
              >
                predictions.json
              </button>
              <button type="button" className="ink-link" onClick={() => download("seer_report.json", cards)}>
                full report
              </button>
              <button
                type="button"
                className="ink-link"
                onClick={() => {
                  cards.forEach((c) => URL.revokeObjectURL(c.url));
                  setCards([]);
                  setMeta(null);
                }}
              >
                clear
              </button>
            </div>
          </div>
        </Measure>
      )}

      <div className="space-y-10">
        {cards.map((c, i) => (
          <ResultFigure key={`${c.name}-${i}`} c={c} />
        ))}
        {busy &&
          Array.from({ length: Math.min(pending, 3) }).map((_, i) => (
            <div key={`sk-${i}`} className="figure">
              <div className="skeleton h-5 w-24" />
              <div className="skeleton mt-4 aspect-[4/3] w-full" />
            </div>
          ))}
      </div>

      {cards.length > 0 && (
        <div className="figure">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="dropzone min-h-0 py-6"
          >
            Add more images
          </button>
        </div>
      )}
    </div>
  );
}

function ModeBadge({
  meta,
  status,
}: {
  meta: AnalyzeResponse | null;
  status: StatusResponse | null;
}) {
  const live =
    meta?.mode === "live" ||
    (!meta && (status?.mode === "live" || Boolean(status?.modal?.ok)));
  if (!meta && !status) return null;
  return (
    <span className="essay-kicker">
      {live ? "live inference" : status?.error ? "loading model" : "simulated"}
    </span>
  );
}

function previewStyle(width?: number, height?: number): CSSProperties {
  const ar = width && height && width > 0 && height > 0 ? width / height : 4 / 3;
  return {
    aspectRatio: `${ar}`,
    width: `min(100%, calc(min(72vh, 820px) * ${ar}))`,
  };
}

function ResultFigure({ c }: { c: Card }) {
  const [showHeat, setShowHeat] = useState(true);

  return (
    <figure className="figure">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <VerdictPill label={c.label} />
        <span className="caption truncate" title={c.name}>
          {c.name}
        </span>
      </div>

      <div
        className="figure-frame relative mt-3 max-w-full"
        style={previewStyle(c.width, c.height)}
      >
        <HeatCanvas grid={c.grid} src={c.url} showHeat={showHeat} />
        {c.width && c.height && (
          <span className="chip absolute left-2.5 top-2.5">
            {c.width}×{c.height}
          </span>
        )}
        <button
          type="button"
          onClick={() => setShowHeat((v) => !v)}
          className="chip absolute bottom-2.5 right-2.5"
        >
          {showHeat ? "hide heatmap" : "show heatmap"}
        </button>
      </div>

      <div className="mt-4 flex flex-wrap items-end justify-between gap-6">
        <ProbGauge p={c.prob_ai} />
        <dl className="min-w-[240px] flex-1">
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
    </figure>
  );
}

function Meta({ k, v }: { k: string; v: string }) {
  return (
    <div className="meta-pair">
      <dt>{k}</dt>
      <dd className="tabular">{v}</dd>
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
