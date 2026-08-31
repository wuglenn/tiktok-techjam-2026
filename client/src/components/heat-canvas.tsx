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

    // base layer: the image (cover fit) or a neutral procedural placeholder
    if (img) {
      const iw = img.naturalWidth || 1;
      const ih = img.naturalHeight || 1;
      const s = Math.max(W / iw, H / ih);
      const dw = iw * s;
      const dh = ih * s;
      ctx.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
    } else {
      const g = ctx.createLinearGradient(0, 0, W, H);
      g.addColorStop(0, "#141417");
      g.addColorStop(0.5, "#101013");
      g.addColorStop(1, "#17151c");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
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
    }

    if (!showHeat || !grid || !grid.length) return;

    // heatmap layer: G x G ImageData colored through turbo, scaled up smoothly
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
    ctx.drawImage(off, 0, 0, W, H);
    ctx.restore();
  }, [img, grid, opacity, showHeat, tick]);

  return <canvas ref={canvasRef} className={`absolute inset-0 h-full w-full ${className ?? ""}`} />;
}

export function HeatLegend({ className }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className ?? ""}`}>
      <span className="text-[10px] font-medium tracking-wide text-zinc-500">real</span>
      <div
        className="h-2 w-28 rounded-full ring-1 ring-white/10"
        style={{ background: turboGradientCss() }}
      />
      <span className="text-[10px] font-medium tracking-wide text-zinc-500">AI</span>
    </div>
  );
}
