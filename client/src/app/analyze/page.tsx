"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { HeatCanvas } from "@/components/heat-canvas";
import { IconImage, IconSpark, IconUpload, IconX } from "@/components/icons";
import { ProbBar, VerdictPill } from "@/components/verdict";
import { bytes as fmtBytes } from "@/lib/format";
import type { AnalyzeResponse, AnalyzeResult, StatusResponse } from "@/lib/types";

const MAX_FILES = 12;
const MAX_BYTES = 40 * 1024 * 1024;

type UploadState = "queued" | "processing";

interface UploadItem {
  id: string;
  file: File;
  url: string;
  state: UploadState;
}

interface Card extends AnalyzeResult {
  id: string;
  url: string;
}

let nextUploadId = 0;

export default function AnalyzePage() {
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [cards, setCards] = useState<Card[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const objectUrls = useRef(new Set<string>());

  const queued = uploads.filter((item) => item.state === "queued");
  const processing = uploads.filter((item) => item.state === "processing");
  const nextBatchCount = Math.min(queued.length, MAX_FILES);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    // Self-paced polling avoids stacking health probes while a remote model is waking up.
    const pull = async () => {
      try {
        const res = await fetch("/api/status");
        const data = (await res.json()) as StatusResponse;
        if (alive) setStatus(data);
      } catch {
        // Keep the last known state and try again after the gap.
      }
      if (alive) timer = setTimeout(pull, 4000);
    };

    void pull();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    const urls = objectUrls.current;
    return () => {
      urls.forEach((url) => URL.revokeObjectURL(url));
      urls.clear();
    };
  }, []);

  const releaseUrl = useCallback((url: string) => {
    URL.revokeObjectURL(url);
    objectUrls.current.delete(url);
  }, []);

  const addFiles = useCallback((fileList: FileList | File[]) => {
    const files = Array.from(fileList);
    const valid = files.filter(
      (file) => file.type.startsWith("image/") && file.size <= MAX_BYTES,
    );
    const tooLarge = files.filter(
      (file) => file.type.startsWith("image/") && file.size > MAX_BYTES,
    );
    const notImages = files.filter((file) => !file.type.startsWith("image/"));

    if (tooLarge.length || notImages.length) {
      const issues = [
        tooLarge.length
          ? `${tooLarge.length} image${tooLarge.length === 1 ? " is" : "s are"} over 40 MB`
          : null,
        notImages.length
          ? `${notImages.length} file${notImages.length === 1 ? " is not" : "s are not"} an image`
          : null,
      ].filter(Boolean);
      setError(`${issues.join("; ")}. Those files were not added.`);
    } else if (valid.length) {
      setError(null);
    }

    if (!valid.length) return;

    const additions = valid.map((file) => {
      const url = URL.createObjectURL(file);
      objectUrls.current.add(url);
      nextUploadId += 1;
      return {
        id: `upload-${nextUploadId}`,
        file,
        url,
        state: "queued" as const,
      };
    });
    setUploads((current) => [...current, ...additions]);
  }, []);

  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const files = Array.from(event.clipboardData?.files ?? []).filter((file) =>
        file.type.startsWith("image/"),
      );
      if (files.length) addFiles(files);
    };
    document.addEventListener("paste", onPaste);
    return () => document.removeEventListener("paste", onPaste);
  }, [addFiles]);

  const removeQueued = useCallback(
    (id: string) => {
      const item = uploads.find((candidate) => candidate.id === id);
      if (!item || item.state !== "queued") return;
      releaseUrl(item.url);
      setUploads((current) => current.filter((candidate) => candidate.id !== id));
    },
    [releaseUrl, uploads],
  );

  const predict = useCallback(async () => {
    const batch = uploads.filter((item) => item.state === "queued").slice(0, MAX_FILES);
    if (!batch.length) return;

    const ids = new Set(batch.map((item) => item.id));
    setError(null);
    setUploads((current) =>
      current.map((item) => (ids.has(item.id) ? { ...item, state: "processing" } : item)),
    );

    try {
      const form = new FormData();
      batch.forEach((item) => form.append("files", item.file));
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      const data = (await res.json()) as AnalyzeResponse & { error?: string };
      if (!res.ok) throw new Error(data.error ?? `Request failed (${res.status})`);

      const resultCount = Math.min(data.results.length, batch.length);
      const completed = data.results.slice(0, resultCount).map((result, index) => ({
        ...result,
        id: batch[index].id,
        url: batch[index].url,
      }));
      const completedIds = new Set(completed.map((item) => item.id));

      setCards((current) => [...completed, ...current]);
      setUploads((current) =>
        current.flatMap((item) => {
          if (completedIds.has(item.id)) return [];
          if (ids.has(item.id)) return [{ ...item, state: "queued" as const }];
          return [item];
        }),
      );

      if (resultCount !== batch.length) {
        setError(
          `The model returned ${resultCount} of ${batch.length} predictions. The remaining images are ready to retry.`,
        );
      }
    } catch (reason) {
      setUploads((current) =>
        current.map((item) =>
          ids.has(item.id) ? { ...item, state: "queued" as const } : item,
        ),
      );
      setError(reason instanceof Error ? reason.message : "Prediction failed. Please try again.");
    }
  }, [uploads]);

  const clearResults = useCallback(() => {
    cards.forEach((card) => releaseUrl(card.url));
    setCards([]);
  }, [cards, releaseUrl]);

  const removeResult = useCallback(
    (id: string) => {
      const card = cards.find((candidate) => candidate.id === id);
      if (!card) return;
      releaseUrl(card.url);
      setCards((current) => current.filter((candidate) => candidate.id !== id));
    },
    [cards, releaseUrl],
  );

  return (
    <div className="analyze-page">
      <header className="analyze-hero">
        <div>
          <p className="essay-kicker">Image forensics</p>
          <h1 className="analyze-title">See what the model sees.</h1>
          <p className="analyze-intro">
            Add an image to estimate whether it was AI-generated and inspect the regions
            behind the prediction.
          </p>
        </div>
        <ModelState status={status} />
      </header>

      <div className="analyze-live-region" aria-live="polite">
        {status?.mode === "unavailable" && (
          <p className="analyze-alert">
            The inference model is currently unavailable. You can still prepare images and
            retry when it comes online.
          </p>
        )}
        {error && (
          <p className="analyze-alert is-error" role="alert">
            {error}
          </p>
        )}
      </div>

      <section className="upload-composer" aria-labelledby="upload-heading">
        <label
          className={`composer-drop ${dragging ? "is-dragging" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={(event) => {
            const nextTarget = event.relatedTarget;
            if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) {
              setDragging(false);
            }
          }}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            if (event.dataTransfer.files.length) addFiles(event.dataTransfer.files);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(event) => {
              if (event.currentTarget.files?.length) addFiles(event.currentTarget.files);
              event.currentTarget.value = "";
            }}
          />
          <span className="composer-icon" aria-hidden>
            {dragging ? <IconImage /> : <IconUpload />}
          </span>
          <span className="composer-copy">
            <strong id="upload-heading">
              {dragging ? "Drop them here" : "Drop or paste images"}
            </strong>
            <span>JPEG, PNG, WebP or GIF · up to 40 MB each</span>
          </span>
          <span className="composer-browse">Choose images</span>
        </label>

        {queued.length > 0 && (
          <div className="queued-area">
            <div className="section-heading compact">
              <div>
                <h2>Ready to predict</h2>
                <p>{queued.length} image{queued.length === 1 ? "" : "s"} selected</p>
              </div>
              {queued.length > MAX_FILES && (
                <span className="caption">Next {MAX_FILES} will run first</span>
              )}
            </div>
            <div className="upload-grid">
              {queued.map((item) => (
                <UploadPreview
                  key={item.id}
                  item={item}
                  onRemove={() => removeQueued(item.id)}
                />
              ))}
            </div>
          </div>
        )}

        <div className="composer-actions">
          <p className="composer-hint">
            {queued.length
              ? `${nextBatchCount} ready${processing.length ? ` · ${processing.length} already processing` : ""}`
              : processing.length
                ? "Add another image while the current prediction runs."
                : "Tip: paste a screenshot anywhere on this page."}
          </p>
          <button
            type="button"
            className="predict-button"
            disabled={!nextBatchCount}
            onClick={() => void predict()}
          >
            <IconSpark aria-hidden />
            {nextBatchCount
              ? `Predict ${nextBatchCount} image${nextBatchCount === 1 ? "" : "s"}`
              : "Predict"}
          </button>
        </div>
      </section>

      {processing.length > 0 && (
        <section className="activity-section" aria-labelledby="activity-heading" aria-live="polite">
          <div className="section-heading">
            <div>
              <h2 id="activity-heading">Analyzing</h2>
              <p>The model is reading global and patch-level signals.</p>
            </div>
            <span className="activity-count">
              <span />
              {processing.length} in progress
            </span>
          </div>
          <div className="processing-grid">
            {processing.map((item) => (
              <ProcessingPreview key={item.id} item={item} />
            ))}
          </div>
        </section>
      )}

      {cards.length > 0 && (
        <section className="results-section" aria-labelledby="results-heading">
          <div className="results-toolbar">
            <div>
              <p className="essay-kicker">Results</p>
              <h2 id="results-heading">
                {cards.length} prediction{cards.length === 1 ? "" : "s"}
              </h2>
            </div>
            <div className="results-actions">
              <button
                type="button"
                onClick={() =>
                  download(
                    "seer_predictions.json",
                    cards.map((card) => ({ image_path: card.name, pred: card.prob_ai })),
                  )
                }
              >
                Predictions
              </button>
              <button
                type="button"
                onClick={() => download("seer_report.json", cards.map(toReportResult))}
              >
                Full report
              </button>
              <button type="button" onClick={clearResults}>
                Clear
              </button>
            </div>
          </div>

          <div className="results-grid">
            {cards.map((card) => (
              <ResultCard key={card.id} card={card} onRemove={() => removeResult(card.id)} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function ModelState({ status }: { status: StatusResponse | null }) {
  let label = "Checking model";
  let tone = "is-checking";

  if (status?.mode === "live") {
    label = "Model online";
    tone = "is-online";
  } else if (status?.modal?.ok == null && status?.modal) {
    label = "Model warming up";
    tone = "is-warming";
  } else if (status?.mode === "unavailable") {
    label = "Model offline";
    tone = "is-offline";
  }

  return (
    <div className={`model-state ${tone}`} role="status">
      <span className="model-dot" />
      {label}
    </div>
  );
}

function UploadPreview({ item, onRemove }: { item: UploadItem; onRemove: () => void }) {
  return (
    <article className="upload-preview">
      <div className="upload-thumb">
        {/* Object URLs are local previews and cannot be handled by next/image. */}
        <img src={item.url} alt="" />
        <button type="button" onClick={onRemove} aria-label={`Remove ${item.file.name}`}>
          <IconX aria-hidden />
        </button>
      </div>
      <div className="upload-meta">
        <strong title={item.file.name}>{item.file.name}</strong>
        <span>{fmtBytes(item.file.size)}</span>
      </div>
    </article>
  );
}

function ProcessingPreview({ item }: { item: UploadItem }) {
  return (
    <article className="processing-preview" aria-busy="true">
      <div className="processing-image">
        {/* Object URLs are local previews and cannot be handled by next/image. */}
        <img src={item.url} alt="" />
        <span className="scan-line" />
      </div>
      <div>
        <strong title={item.file.name}>{item.file.name}</strong>
        <span>Finding model traces…</span>
      </div>
    </article>
  );
}

function ResultCard({ card, onRemove }: { card: Card; onRemove: () => void }) {
  const [showHeat, setShowHeat] = useState(true);
  const aspectRatio =
    card.width && card.height && card.width > 0 && card.height > 0
      ? card.width / card.height
      : 4 / 3;

  return (
    <article className="result-card">
      <div className="result-card-head">
        <div className="result-name">
          <VerdictPill label={card.label} />
          <span title={card.name}>{card.name}</span>
        </div>
        <button type="button" onClick={onRemove} aria-label={`Remove result for ${card.name}`}>
          <IconX aria-hidden />
        </button>
      </div>

      <div className="result-image" style={{ aspectRatio }}>
        <HeatCanvas grid={card.grid} src={card.url} showHeat={showHeat} />
        {card.width && card.height && (
          <span className="chip result-dimensions">
            {card.width}×{card.height}
          </span>
        )}
        {card.grid && (
          <button
            type="button"
            className="heat-toggle"
            onClick={() => setShowHeat((current) => !current)}
          >
            {showHeat ? "Original" : "Show heatmap"}
          </button>
        )}
      </div>

      <div className="result-summary">
        <div>
          <span className="result-probability">{card.prob_ai.toFixed(3)}</span>
          <span className="caption">P(AI)</span>
        </div>
        <dl>
          <Meta
            k="File"
            v={`${card.type?.replace("image/", "").toUpperCase() ?? "–"} · ${fmtBytes(card.bytes)}`}
          />
          <Meta
            k="Patch grid"
            v={card.grid ? `${card.grid.length}×${card.grid[0]?.length ?? 0}` : "Page only"}
          />
          <Meta k="Latency" v={card.elapsedMs != null ? `${card.elapsedMs} ms` : "–"} />
        </dl>
      </div>
      <ProbBar p={card.prob_ai} className="result-probability-bar" />
    </article>
  );
}

function Meta({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  );
}

function toReportResult(card: Card): AnalyzeResult {
  return {
    name: card.name,
    prob_ai: card.prob_ai,
    label: card.label,
    grid: card.grid,
    width: card.width,
    height: card.height,
    bytes: card.bytes,
    type: card.type,
    elapsedMs: card.elapsedMs,
  };
}

function download(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
