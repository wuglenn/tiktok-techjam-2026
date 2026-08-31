/** Minimal inline icon set (stroke, 24x24) — no icon dependency. */
import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement>;

function base(props: P) {
  return {
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.6,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    ...props,
  };
}

export function IconGrid(p: P) {
  return (
    <svg {...base(p)}>
      <rect x="3" y="3" width="7.5" height="7.5" rx="1.5" />
      <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5" />
      <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5" />
      <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5" />
    </svg>
  );
}

export function IconScan(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M3 8V5.5A2.5 2.5 0 0 1 5.5 3H8" />
      <path d="M16 3h2.5A2.5 2.5 0 0 1 21 5.5V8" />
      <path d="M21 16v2.5a2.5 2.5 0 0 1-2.5 2.5H16" />
      <path d="M8 21H5.5A2.5 2.5 0 0 1 3 18.5V16" />
      <path d="M7 12h10" />
    </svg>
  );
}

export function IconChart(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M3 3v18h18" />
      <path d="M7 15v3" />
      <path d="M12 9.5V18" />
      <path d="M17 5.5V18" />
    </svg>
  );
}

export function IconAlert(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M10.3 3.9 2.5 17.5A2 2 0 0 0 4.2 20.5h15.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
      <path d="M12 9v4.5" />
      <path d="M12 17h.01" />
    </svg>
  );
}

export function IconUpload(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M12 16V4" />
      <path d="m6.5 9.5 5.5-5.5 5.5 5.5" />
      <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
    </svg>
  );
}

export function IconDownload(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M12 4v12" />
      <path d="m6.5 10.5 5.5 5.5 5.5-5.5" />
      <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
    </svg>
  );
}

export function IconSpark(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9Z" />
      <path d="M19 15.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8Z" />
    </svg>
  );
}

export function IconArrowRight(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M4 12h16" />
      <path d="m13 5 7 7-7 7" />
    </svg>
  );
}

export function IconImage(p: P) {
  return (
    <svg {...base(p)}>
      <rect x="3" y="4" width="18" height="16" rx="2.5" />
      <circle cx="8.75" cy="9.75" r="1.75" />
      <path d="m3.5 17.5 4.8-4.8a2 2 0 0 1 2.8 0l2.4 2.4" />
      <path d="m13.5 15.5 1.7-1.7a2 2 0 0 1 2.8 0l2.5 2.5" />
    </svg>
  );
}

export function IconEye(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

export function IconShield(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M12 2.5 4.5 5.5v6c0 4.6 3.2 8.1 7.5 10 4.3-1.9 7.5-5.4 7.5-10v-6Z" />
      <path d="m9 11.8 2.2 2.2 4-4.2" />
    </svg>
  );
}

export function IconCpu(p: P) {
  return (
    <svg {...base(p)}>
      <rect x="5.5" y="5.5" width="13" height="13" rx="2" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
      <path d="M9.5 2.5v3M14.5 2.5v3M9.5 18.5v3M14.5 18.5v3" />
      <path d="M2.5 9.5h3M2.5 14.5h3M18.5 9.5h3M18.5 14.5h3" />
    </svg>
  );
}

export function IconLayers(p: P) {
  return (
    <svg {...base(p)}>
      <path d="m12 3 9 4.5-9 4.5-9-4.5Z" />
      <path d="m3 12 9 4.5 9-4.5" />
      <path d="m3 16.5 9 4.5 9-4.5" />
    </svg>
  );
}

export function IconZap(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M13 2.5 4.5 13.5H11l-1 8 8.5-11H12Z" />
    </svg>
  );
}

export function IconFlame(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M12 22c4.1 0 6.8-2.7 6.8-6.5 0-3-1.7-5-3.2-6.9C14 6.6 13 5 13 2.5c-3 1.8-4.6 4.2-4.9 6.6-.2 1.5.1 2.6.4 3.4-.9-.2-1.9-1-2.5-2.2-1 1.3-1.8 3-1.8 4.9C4.2 19.4 7.4 22 12 22Z" />
    </svg>
  );
}

export function IconCheck(p: P) {
  return (
    <svg {...base(p)}>
      <path d="m4.5 12.5 5 5 10-11" />
    </svg>
  );
}

export function IconX(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}

export function IconRefresh(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M21 12a9 9 0 1 1-2.6-6.3" />
      <path d="M21 3v5h-5" />
    </svg>
  );
}

export function IconFlask(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M9.5 3h5M10 3v6l-5.3 8.8A2 2 0 0 0 6.4 21h11.2a2 2 0 0 0 1.7-3.2L14 9V3" />
      <path d="M7.5 15h9" />
    </svg>
  );
}

export function IconInfo(p: P) {
  return (
    <svg {...base(p)}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5" />
      <path d="M12 7.5h.01" />
    </svg>
  );
}

export function IconChevron(p: P) {
  return (
    <svg {...base(p)}>
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

export function IconExternal(p: P) {
  return (
    <svg {...base(p)}>
      <path d="M14 4h6v6" />
      <path d="M20 4 10 14" />
      <path d="M18 13.5V18a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4.5" />
    </svg>
  );
}

/** Brand mark: a gradient eye. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <defs>
        <linearGradient id="seer-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#22d3ee" />
          <stop offset="1" stopColor="#38bdf8" />
        </linearGradient>
      </defs>
      <path
        d="M16 5.5C9.4 5.5 4.7 11 2.5 16c2.2 5 6.9 10.5 13.5 10.5S27.8 21 30 16c-2.2-5-6.9-10.5-13.5-10.5Z"
        fill="none"
        stroke="url(#seer-g)"
        strokeWidth="2.2"
        strokeLinejoin="round"
      />
      <circle cx="16" cy="16" r="5.6" fill="url(#seer-g)" />
      <circle cx="16" cy="16" r="2.1" fill="#08080a" />
    </svg>
  );
}
