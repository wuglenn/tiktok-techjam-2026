import Link from "next/link";
import type { ReactNode } from "react";

export function Measure({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`measure ${className ?? ""}`}>{children}</div>;
}

export function Figure({
  children,
  caption,
  wash,
  className,
}: {
  children: ReactNode;
  caption?: ReactNode;
  wash?: boolean;
  className?: string;
}) {
  return (
    <figure className={`figure ${className ?? ""}`}>
      <div className={wash ? "figure-wash" : undefined}>{children}</div>
      {caption != null && caption !== "" && (
        <figcaption className="caption">{caption}</figcaption>
      )}
    </figure>
  );
}

export function Chip({ children }: { children: ReactNode }) {
  return <span className="chip">{children}</span>;
}

export function MetaPairs({
  rows,
}: {
  rows: { label: string; value: ReactNode; detail?: ReactNode }[];
}) {
  return (
    <dl className="meta-grid">
      {rows.map((row) => (
        <div key={row.label} className="meta-pair">
          <dt>{row.label}</dt>
          <dd>
            <div className="tabular">{row.value}</div>
            {row.detail != null && <div className="meta-sub">{row.detail}</div>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function Tabs({
  items,
  active,
  onChange,
}: {
  items: { label: string; hint?: string }[];
  active: number;
  onChange: (index: number) => void;
}) {
  return (
    <div className="tab-row" role="tablist">
      {items.map((item, i) => (
        <button
          key={item.label}
          type="button"
          role="tab"
          title={item.hint}
          aria-pressed={i === active}
          onClick={() => onChange(i)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

export function Notice({ children }: { children: ReactNode }) {
  return <p className="notice">{children}</p>;
}

export function SiteFooter() {
  return (
    <footer className="measure site-footer">
      <p>
        We trained this detector on public data for TikTok TechJam 2026, Track 5.
      </p>
      <p className="footer-links">
        <Link href="/analyze">Analyze</Link>
        <Link href="/robustness">Robustness</Link>
        <Link href="/errors">Errors</Link>
      </p>
      <p className="footer-copy">© 2026 Seer</p>
    </footer>
  );
}
