"use client";

import { useCallback, useRef, useState, type PointerEvent, type KeyboardEvent } from "react";

import { Figure } from "@/components/essay";

const PROB = 0.96;
const CLEAN = "/hero-demo-clean.png";
const HEAT = "/hero-demo.png";

export function HeroDemo() {
  return (
    <Figure
      className="hero-figure"
      caption={
        <>
          P(AI) <span className="hero-prob">{PROB.toFixed(3)}</span>
          {" · 32×32 patch grid · threshold 0.5 · drag to compare"}
        </>
      }
    >
      <CompareSlider
        before={CLEAN}
        after={HEAT}
        beforeAlt="Original diner specials board, no overlay"
        afterAlt="Same board with the 32 by 32 patch heatmap, scored P(AI) 0.960"
        beforeLabel="photo"
        afterLabel="heatmap"
      />
    </Figure>
  );
}

function CompareSlider({
  before,
  after,
  beforeAlt,
  afterAlt,
  beforeLabel,
  afterLabel,
}: {
  before: string;
  after: string;
  beforeAlt: string;
  afterAlt: string;
  beforeLabel: string;
  afterLabel: string;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const [split, setSplit] = useState(50);

  const setFromClientX = useCallback((clientX: number) => {
    const frame = frameRef.current;
    if (!frame) return;
    const rect = frame.getBoundingClientRect();
    if (rect.width <= 0) return;
    const next = ((clientX - rect.left) / rect.width) * 100;
    setSplit(Math.min(100, Math.max(0, next)));
  }, []);

  function onPointerDown(e: PointerEvent<HTMLDivElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    setFromClientX(e.clientX);
  }

  function onPointerMove(e: PointerEvent<HTMLDivElement>) {
    if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
    setFromClientX(e.clientX);
  }

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    const step = e.shiftKey ? 10 : 2;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault();
      setSplit((v) => Math.max(0, v - step));
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault();
      setSplit((v) => Math.min(100, v + step));
    } else if (e.key === "Home") {
      e.preventDefault();
      setSplit(0);
    } else if (e.key === "End") {
      e.preventDefault();
      setSplit(100);
    }
  }

  return (
    <div
      ref={frameRef}
      className="compare"
      role="slider"
      tabIndex={0}
      aria-label="Compare the original photo with the heatmap"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(split)}
      aria-valuetext={`${Math.round(split)} percent photo, ${Math.round(100 - split)} percent heatmap`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onKeyDown={onKeyDown}
    >
      <img className="compare-img" src={after} alt={afterAlt} draggable={false} />
      <div className="compare-clip" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
        <img className="compare-img" src={before} alt={beforeAlt} draggable={false} />
      </div>
      <span className="compare-label is-before">{beforeLabel}</span>
      <span className="compare-label is-after">{afterLabel}</span>
      <div className="compare-rule" style={{ left: `${split}%` }} aria-hidden>
        <span className="compare-knob" />
      </div>
    </div>
  );
}
