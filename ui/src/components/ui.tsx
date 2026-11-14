import type { ReactNode } from "react";

export function Panel({ title, aside, children, className = "" }: { title?: string; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`bg-surface border border-line ${className}`}>
      {title && (
        <header className="flex flex-wrap items-baseline justify-between gap-2 border-b border-line px-4 py-2.5">
          <h2 className="font-semibold">{title}</h2>
          {aside && <div className="text-sm text-muted">{aside}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function PageHeader({ title, lead }: { title: string; lead: string }) {
  return (
    <header className="mb-5">
      <h1 className="font-serif text-2xl md:text-3xl">{title}</h1>
      <p className="mt-1 max-w-[70ch] text-muted">{lead}</p>
    </header>
  );
}

export function ErrorNote({ message, hint }: { message: string; hint?: string }) {
  return (
    <div role="alert" className="border-l-4 border-decline bg-decline-tint px-3 py-2 text-decline-ink">
      <p className="font-semibold">{message}</p>
      {hint && <p className="text-sm">{hint}</p>}
    </div>
  );
}

export function Loading({ what }: { what: string }) {
  return <p role="status" className="text-muted">Loading {what}…</p>;
}

/** An assumption or limit that the reader must see next to the number it qualifies. */
export function Caveat({ children }: { children: ReactNode }) {
  return (
    <p className="flex gap-2 border-l-4 border-ink bg-paper px-3 py-2 text-sm">
      <span aria-hidden="true" className="font-semibold">!</span>
      <span>{children}</span>
    </p>
  );
}

export function Stat({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div>
      <dt className="text-sm text-muted">{label}</dt>
      <dd className="num font-serif text-2xl">{value}</dd>
      {note && <dd className="text-sm text-muted">{note}</dd>}
    </div>
  );
}
