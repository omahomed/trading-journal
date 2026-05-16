"use client";

// In-app reference for the 13-rule sell-rule taxonomy. Mounted only
// on the Log Sell page (full-width, below the two-column form grid).
// Collapsed by default; user can toggle and the state persists via
// SectionExpander's localStorage key.
//
// Content lives at lib/trade-rules.ts (canonical source). This file
// is pure presentation: a list of RuleCards plus the structured
// Rule Interaction Hierarchy table. Mechanics is rendered as GFM
// markdown (the same renderer used by the Daily Recap section in
// daily-report-card.tsx) so tables and bold work without ceremony.

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  SELL_RULES,
  RULE_HIERARCHY,
  type SellRule,
  type RuleHierarchyEntry,
} from "@/lib/trade-rules";
import { SectionExpander } from "./section-expander";

function RuleCard({ rule }: { rule: SellRule }) {
  return (
    <div
      className="rounded-[12px] px-4 py-3.5"
      style={{
        background: "var(--bg)",
        border: "1px solid var(--border)",
      }}
    >
      <div className="flex items-baseline gap-2 mb-1.5">
        <span
          className="text-[12px] font-semibold"
          style={{
            fontFamily: "var(--font-jetbrains), monospace",
            color: "var(--ink-3)",
          }}
        >
          {rule.code}
        </span>
        <span className="text-[14px] font-semibold" style={{ color: "var(--ink)" }}>
          {rule.description}
        </span>
      </div>
      <div className="text-[13px] leading-[1.55]" style={{ color: "var(--ink-2)" }}>
        {rule.oneLiner}
      </div>
      {rule.mechanics && (
        <div className="prose-custom mt-2" style={{ color: "var(--ink-2)" }}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{rule.mechanics}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}

function HierarchyTable({ entries }: { entries: readonly RuleHierarchyEntry[] }) {
  // Canonical app data-table pattern (matches analytics.tsx:589 and
  // trade-journal.tsx:1399): 9px uppercase tracked muted headers on
  // surface-2, 11px body cells, row-level border-bottom only (no per-
  // td borders and no zebra stripes), Inter sans inherited from body.
  return (
    <div>
      <div
        className="text-[11px] uppercase tracking-[0.08em] font-semibold mb-2"
        style={{ color: "var(--ink-4)" }}
      >
        Rule Interaction Hierarchy
      </div>
      <div
        className="overflow-x-auto rounded-[8px]"
        style={{ border: "1px solid var(--border)" }}
      >
        <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr>
              {["Conflict", "Winner", "Reasoning"].map((h) => (
                <th
                  key={h}
                  className="text-left px-3 py-2 text-[9px] uppercase tracking-[0.06em] font-semibold whitespace-nowrap"
                  style={{
                    color: "var(--ink-4)",
                    background: "var(--surface-2)",
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map((e, i) => (
              <tr
                key={e.conflict}
                style={{
                  borderBottom:
                    i < entries.length - 1 ? "1px solid var(--border)" : "none",
                }}
              >
                <td className="px-3 py-2 align-top font-semibold" style={{ color: "var(--ink-2)" }}>
                  {e.conflict}
                </td>
                <td className="px-3 py-2 align-top" style={{ color: "var(--ink-2)" }}>
                  {e.winner}
                </td>
                <td className="px-3 py-2 align-top" style={{ color: "var(--ink-4)" }}>
                  {e.reasoning}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function SellRuleGlossary() {
  return (
    <div className="mt-6">
      <SectionExpander
        title="Sell rule reference"
        defaultExpanded={false}
        localStorageKey="mo-log-sell-glossary-expanded"
        headerCaption={(open) => (open ? "Hide" : "Show 13 rules + hierarchy")}
      >
        <div className="p-5 flex flex-col gap-3">
          {SELL_RULES.map((rule) => (
            <RuleCard key={rule.code} rule={rule} />
          ))}
          <HierarchyTable entries={RULE_HIERARCHY} />
        </div>
      </SectionExpander>
    </div>
  );
}
