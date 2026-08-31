"use client";

import { useEffect, useRef, useState } from "react";

import { turbo, turboGradientCss } from "@/lib/heat";
import type { PatchGrid } from "@/lib/types";

/**
 * Renders an image with the model's per-patch probability grid overlaid in
 * the same `turbo` colormap `seer/heatmap.py` uses for its matplotlib panels.
 * The G x G grid is upsampled bilinearly by canvas scaling.
 */
export function HeatCanvas({
  grid,
  src,
  opacity = 0.55,
  showHeat = true,
  className,
}: {
  grid: PatchGrid | null;
  src?: string | null;
  opacity?: number;
  showHeat?: boolean;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!src) {
      setImg(null);
      return;
    }
    const el = new Image();
    el.onload = () => setImg(el);
    el.src = src;
    return () => {
      el.onload = null;
    };
  }, [src]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    const ro = new ResizeObserver(() => setTick((t) => t + 1));
    ro.observe(parent);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.max(1, Math.round(parent.clientWidth * dpr));
    const H = Math.max(1, Math.round(parent.clientHeight * dpr));
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, W, H);

    // Letterbox / fill: image and heatmap must share one dest rect so the
    // 32×32 local head (computed on a square 512 resize) maps back onto the
    // original photo instead of the card's crop box.
    let dx = 0;
    let dy = 0;
    let dw = W;
    let dh = H;
    if (img) {
      const iw = img.naturalWidth || 1;
      const ih = img.naturalHeight || 1;
      const s = Math.min(W / iw, H / ih);
      dw = Math.max(1, iw * s);
      dh = Math.max(1, ih * s);
      dx = (W - dw) / 2;
      dy = (H - dh) / 2;
    }

    if (img) {
      ctx.fillStyle = "#fcfaf5";
      ctx.fillRect(0, 0, W, H);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, dx, dy, dw, dh);
    } else {
      ctx.fillStyle = "#fff8e8";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "#a3a3a3";
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1;
      const step = Math.max(24, W / 12);
      for (let x = step; x < W; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
      }
      for (let y = step; y < H; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    if (!showHeat || !grid || !grid.length) return;

    // heatmap layer: G x G ImageData colored through turbo, bilinear-scaled
    // to the same rectangle as the photo (inverse of eval_transform's square resize)
    const Gh = grid.length;
    const Gw = grid[0].length;
    const off = document.createElement("canvas");
    off.width = Gw;
    off.height = Gh;
    const octx = off.getContext("2d");
    if (!octx) return;
    const data = octx.createImageData(Gw, Gh);
    for (let y = 0; y < Gh; y++) {
      for (let x = 0; x < Gw; x++) {
        const [r, g, b] = turbo(grid[y][x] ?? 0);
        const i = (y * Gw + x) * 4;
        data.data[i] = r;
        data.data[i + 1] = g;
        data.data[i + 2] = b;
        data.data[i + 3] = 255;
      }
    }
    octx.putImageData(data, 0, 0);

    ctx.save();
    ctx.globalAlpha = opacity;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, dx, dy, dw, dh);
    ctx.restore();
  }, [img, grid, opacity, showHeat, tick]);

  return <canvas ref={canvasRef} className={`absolute inset-0 h-full w-full ${className ?? ""}`} />;
}

export function HeatLegend({ className }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className ?? ""}`}>
      <span className="font-mono text-[14px] text-ink-mute">real</span>
      <div
        className="h-1 w-28"
        style={{ background: turboGradientCss() }}
      />
      <span className="font-mono text-[14px] text-ink-mute">AI</span>
    </div>
  );
}
