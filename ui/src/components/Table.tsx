import type { ReactNode } from "react";

export interface Column<R> { head: string; cell: (row: R) => ReactNode; numeric?: boolean; note?: string }

export function Table<R>({ caption, columns, rows, rowKey, highlight }: {
  caption: string; columns: Column<R>[]; rows: R[]; rowKey: (row: R) => string; highlight?: (row: R) => boolean;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b-2 border-ink text-left">
            {columns.map((c, i) => (
              <th key={i} scope="col" title={c.note} className={`px-2 py-1.5 align-bottom font-semibold ${c.numeric ? "text-right" : ""}`}>{c.head}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={rowKey(r)} className={`border-b border-line ${highlight?.(r) ? "bg-paper font-semibold" : ""}`}>
              {columns.map((c, i) => i === 0
                ? <th key={i} scope="row" className="px-2 py-1.5 text-left font-semibold">{c.cell(r)}</th>
                  : <td key={i} className={`px-2 py-1.5 ${c.numeric ? "num text-right" : ""}`}>{c.cell(r)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
