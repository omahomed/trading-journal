"use client";

// Phase 3 Weekly Thoughts editor. Path A from the audit:
// contentEditable + document.execCommand + HTML storage. execCommand is
// formally deprecated in MDN but enjoys universal browser support and has
// no standardized replacement; this is the pragmatic v1 choice for a
// single-surface rich text editor that doesn't justify pulling in a 150KB
// editor library. When Phase 7 redesigns the daily editor with the same
// chrome, this component should extract to a shared <ThoughtsEditor>.
//
// The component owns:
//   - The collapsible card chrome (header + body, matches the Phase 2
//     Per-Ticker expander idiom)
//   - localStorage-persisted collapse state
//   - The 12-button toolbar (10 functional + 2 inert select buttons + 1
//     inert mic for v1)
//   - The contentEditable area, its placeholder, and paste sanitization
//   - Cursor-preserving hydration: external value updates do NOT overwrite
//     innerHTML while the editor is focused (mirrors the Phase 0 stale-
//     list-race lesson)
//
// The parent (weekly-retro.tsx) owns the value state, the dirtyRef flip
// on edit, the debounced PUT, and the API roundtrip.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DOMPurify from "dompurify";
import { Icons } from "./icons";

interface WeeklyThoughtsProps {
  value: string;
  onChange: (html: string) => void;
}

const EXPANDED_KEY = "mo-weekly-retro-thoughts-expanded";

// Closed allow-list of inline HTML tags accepted from paste sources.
// Matches what document.execCommand emits natively, plus a few synonyms
// (em/strong as well as i/b; strike as well as s) so a page that uses one
// flavor doesn't look bare after sanitization.
const PASTE_ALLOWED_TAGS = [
  "b", "i", "u", "s", "strike", "em", "strong", "a", "br", "p", "div", "span",
];
// Only href on anchors. No styles, no classes, no data-*, no event
// handlers. DOMPurify's defaults already strip <script>; this whitelist
// further reduces what survives.
const PASTE_ALLOWED_ATTR = ["href"];

// Fixed colors per the Phase 3 brief. Highlight is the design's amber
// (#f59f00 → matches the orange dot on the section header). Text color is
// the design's emerald. Both can be made user-pickable later by attaching
// a swatch popover to each color button.
const HIGHLIGHT_COLOR = "#ffeaa7";
const TEXT_COLOR = "#08a86b";

export function WeeklyThoughts({ value, onChange }: WeeklyThoughtsProps) {
  // Default EXPANDED — different from Per-Ticker (which defaults collapsed).
  // Thoughts is the active reflection workspace, not a reference panel.
  const [expanded, setExpanded] = useState<boolean>(() => {
    try {
      const stored = localStorage.getItem(EXPANDED_KEY);
      return stored == null ? true : stored === "true";
    } catch {
      return true;
    }
  });

  const editorRef = useRef<HTMLDivElement | null>(null);

  // Tracks the last HTML we wrote to the DOM. Used by the cursor-preserving
  // hydration effect: if the incoming `value` matches what's already in the
  // DOM, skip the write (would destroy the cursor). If it's different but
  // the editor is focused, also skip — assume the user is mid-edit and
  // their local state is authoritative.
  const lastWrittenHtmlRef = useRef<string>(value);

  const toggle = useCallback(() => {
    setExpanded(prev => {
      const next = !prev;
      try { localStorage.setItem(EXPANDED_KEY, String(next)); }
      catch { /* incognito quota — UI still works */ }
      return next;
    });
  }, []);

  // Word count for the collapsed-header caption. Strips HTML tags, splits
  // on whitespace, filters empties.
  const wordCount = useMemo(() => {
    if (!value) return 0;
    const text = value.replace(/<[^>]*>/g, " ").replace(/&nbsp;/g, " ");
    return text.split(/\s+/).filter(Boolean).length;
  }, [value]);

  // Cursor-preserving hydration. Mirrors the Phase 0 lesson: external
  // state changes must not stomp local in-flight edits.
  useEffect(() => {
    const el = editorRef.current;
    if (!el) return;
    if (value === lastWrittenHtmlRef.current) return;
    if (document.activeElement === el) return; // user mid-edit; skip
    el.innerHTML = value || "";
    lastWrittenHtmlRef.current = value || "";
  }, [value]);

  // Set initial content on mount (the useEffect above only catches changes
  // to value; we still need a first render baseline).
  useEffect(() => {
    const el = editorRef.current;
    if (!el) return;
    if (el.innerHTML !== value) {
      el.innerHTML = value || "";
      lastWrittenHtmlRef.current = value || "";
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const emitChange = useCallback(() => {
    const el = editorRef.current;
    if (!el) return;
    const html = el.innerHTML;
    lastWrittenHtmlRef.current = html;
    onChange(html);
  }, [onChange]);

  // Toolbar command dispatcher. Re-focuses the editor before each command
  // so toolbar clicks don't blur away the selection (execCommand only
  // operates on the active document selection).
  const exec = useCallback((command: string, arg?: string) => {
    const el = editorRef.current;
    if (!el) return;
    el.focus();
    document.execCommand(command, false, arg);
    emitChange();
  }, [emitChange]);

  const handleLink = useCallback(() => {
    const url = window.prompt("Enter URL");
    if (!url) return;
    exec("createLink", url);
  }, [exec]);

  // Paste handler: intercept, sanitize via DOMPurify, then insert as HTML.
  // Falls back to text/plain if the clipboard has no HTML payload.
  const handlePaste = useCallback((e: React.ClipboardEvent<HTMLDivElement>) => {
    e.preventDefault();
    const html = e.clipboardData.getData("text/html");
    const text = e.clipboardData.getData("text/plain");
    let clean: string;
    if (html) {
      clean = DOMPurify.sanitize(html, {
        ALLOWED_TAGS: PASTE_ALLOWED_TAGS,
        ALLOWED_ATTR: PASTE_ALLOWED_ATTR,
      });
    } else {
      // Escape plain-text HTML chars then preserve newlines as <br>.
      const esc = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\n/g, "<br>");
      clean = esc;
    }
    document.execCommand("insertHTML", false, clean);
    emitChange();
  }, [emitChange]);

  const handleInput = useCallback(() => {
    emitChange();
  }, [emitChange]);

  const placeholderVisible = !value || value === "" || value === "<br>";

  return (
    <div
      style={{
        borderRadius: 14,
        overflow: "hidden",
        marginBottom: 24,
        background: "var(--surface)",
        border: "1px solid var(--border)",
        boxShadow: "var(--card-shadow)",
      }}
    >
      {/* Header — always visible. Click toggles open/close. */}
      <button
        type="button"
        onClick={toggle}
        aria-expanded={expanded}
        aria-controls="weekly-thoughts-body"
        className="w-full flex items-center text-left"
        style={{
          gap: 10,
          padding: "12px 18px",
          background: "transparent",
          border: "none",
          borderBottom: expanded ? "1px solid var(--border)" : "none",
          cursor: "pointer",
        }}
      >
        <span
          aria-hidden
          style={{
            display: "inline-flex",
            transition: "transform 150ms",
            transform: expanded ? "rotate(90deg)" : "none",
            color: "var(--ink-4)",
          }}
        >
          <Icons.chevronRight />
        </span>
        <span
          aria-hidden
          style={{
            width: 6,
            height: 6,
            borderRadius: 999,
            background: "#f59f00",
          }}
        />
        <span style={{ fontSize: 13, fontWeight: 600 }}>Weekly Thoughts</span>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "var(--ink-4)" }}>
          {expanded
            ? "Synthesize what you learned this week"
            : wordCount > 0
              ? `${wordCount} words`
              : ""}
        </span>
      </button>

      {expanded && (
        <div id="weekly-thoughts-body">
          {/* Toolbar */}
          <div
            role="toolbar"
            aria-label="Weekly Thoughts formatting"
            className="flex items-center"
            style={{
              gap: 4,
              padding: "6px 10px",
              borderBottom: "1px solid var(--border)",
              background: "var(--surface-2)",
              flexWrap: "wrap",
            }}
          >
            {/* Inert font family / size selects (v1 — render only) */}
            <ToolbarBtn label="Font family (coming soon)" inert>
              <span style={{ fontSize: 11, fontWeight: 500 }}>Default</span>
              <span style={{ marginLeft: 4 }}><Icons.chevronDown /></span>
            </ToolbarBtn>
            <ToolbarBtn label="Font size (coming soon)" inert>
              <span style={{ fontSize: 11, fontWeight: 500 }}>13</span>
              <span style={{ marginLeft: 4 }}><Icons.chevronDown /></span>
            </ToolbarBtn>

            <Divider />

            <ToolbarBtn label="Bold" onClick={() => exec("bold")}><Icons.bold /></ToolbarBtn>
            <ToolbarBtn label="Italic" onClick={() => exec("italic")}><Icons.italic /></ToolbarBtn>
            <ToolbarBtn label="Underline" onClick={() => exec("underline")}><Icons.underline /></ToolbarBtn>
            <ToolbarBtn label="Strikethrough" onClick={() => exec("strikeThrough")}><Icons.strike /></ToolbarBtn>

            <Divider />

            <ToolbarBtn label="Insert link" onClick={handleLink}><Icons.link /></ToolbarBtn>
            <ToolbarBtn label="Clear formatting" onClick={() => exec("removeFormat")}><Icons.eraser /></ToolbarBtn>

            <Divider />

            <ToolbarColorBtn
              label="Highlight color"
              swatch={HIGHLIGHT_COLOR}
              onClick={() => exec("hiliteColor", HIGHLIGHT_COLOR)}
              glyph={<Icons.highlight />}
            />
            <ToolbarColorBtn
              label="Text color"
              swatch={TEXT_COLOR}
              onClick={() => exec("foreColor", TEXT_COLOR)}
              glyph={<Icons.textColor />}
            />

            <Divider />

            <ToolbarBtn label="Align left" onClick={() => exec("justifyLeft")}><Icons.alignLeft /></ToolbarBtn>
            <ToolbarBtn label="Align center" onClick={() => exec("justifyCenter")}><Icons.alignCenter /></ToolbarBtn>
            <ToolbarBtn label="Align right" onClick={() => exec("justifyRight")}><Icons.alignRight /></ToolbarBtn>

            <span style={{ flex: 1 }} />

            {/* Inert mic (v1 — Web Speech API integration deferred) */}
            <ToolbarBtn label="Voice dictation (coming soon)" inert><Icons.mic /></ToolbarBtn>
          </div>

          {/* Editor + placeholder. Placeholder is a positioned overlay
              with pointer-events:none so it doesn't intercept clicks. */}
          <div style={{ padding: 16, position: "relative" }}>
            {placeholderVisible && (
              <div
                aria-hidden
                style={{
                  position: "absolute",
                  top: 28, // 16 padding + 12 editor padding
                  left: 30, // 16 padding + 14 editor padding
                  color: "var(--ink-4)",
                  fontStyle: "italic",
                  fontSize: 13,
                  fontFamily: "var(--font-num), monospace",
                  lineHeight: 1.6,
                  pointerEvents: "none",
                  userSelect: "none",
                }}
              >
                What did you learn this week? Patterns, mistakes, rule refinements&hellip;
              </div>
            )}
            <div
              ref={editorRef}
              contentEditable
              suppressContentEditableWarning
              role="textbox"
              aria-multiline="true"
              aria-label="Weekly Thoughts"
              onInput={handleInput}
              onPaste={handlePaste}
              style={{
                width: "100%",
                minHeight: 240,
                padding: "12px 14px",
                borderRadius: 10,
                background: "var(--bg)",
                border: "1px solid var(--border)",
                color: "var(--ink)",
                fontFamily: "var(--font-num), monospace",
                fontSize: 13,
                lineHeight: 1.6,
                outline: "none",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Internal toolbar primitives ─────────────────────────────────────────

interface ToolbarBtnProps {
  label: string;
  onClick?: () => void;
  inert?: boolean;
  children: React.ReactNode;
}

function ToolbarBtn({ label, onClick, inert, children }: ToolbarBtnProps) {
  return (
    <button
      type="button"
      onMouseDown={e => e.preventDefault() /* keep editor focus on click */}
      onClick={inert ? undefined : onClick}
      aria-label={label}
      title={label}
      disabled={inert && !onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 2,
        height: 26,
        padding: "0 6px",
        background: "transparent",
        border: "none",
        borderRadius: 6,
        color: inert ? "var(--ink-4)" : "var(--ink-3)",
        cursor: inert ? "not-allowed" : "pointer",
        opacity: inert ? 0.7 : 1,
      }}
    >
      {children}
    </button>
  );
}

interface ToolbarColorBtnProps {
  label: string;
  swatch: string;
  glyph: React.ReactNode;
  onClick: () => void;
}

function ToolbarColorBtn({ label, swatch, glyph, onClick }: ToolbarColorBtnProps) {
  return (
    <button
      type="button"
      onMouseDown={e => e.preventDefault()}
      onClick={onClick}
      aria-label={label}
      title={label}
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 1,
        height: 26,
        padding: "0 6px",
        background: "transparent",
        border: "none",
        borderRadius: 6,
        color: "var(--ink-3)",
        cursor: "pointer",
      }}
    >
      <span style={{ display: "inline-flex" }}>{glyph}</span>
      <span style={{
        display: "block",
        width: 14,
        height: 3,
        borderRadius: 1,
        background: swatch,
      }} />
    </button>
  );
}

function Divider() {
  return (
    <span
      aria-hidden
      style={{
        display: "inline-block",
        width: 1,
        height: 18,
        margin: "0 4px",
        background: "var(--border)",
      }}
    />
  );
}
