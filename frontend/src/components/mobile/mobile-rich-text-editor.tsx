"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import DOMPurify from "dompurify";
import {
  Bold,
  Code,
  Heading,
  IndentDecrease,
  IndentIncrease,
  Italic,
  Link as LinkIcon,
  List,
  ListOrdered,
  Palette,
  Quote,
  Redo,
  RemoveFormatting,
  Underline,
  Undo,
} from "lucide-react";

/**
 * Mobile rich text editor — Phase 2 T2-4b.
 *
 * contentEditable + execCommand + 15-surface toolbar + class-based
 * color picker + visualViewport-aware sticky-above-keyboard toolbar.
 * Used by Daily Recap and Daily Thoughts editors on mobile.
 *
 * Differences from the desktop ThoughtsEditor:
 *   - 15 toolbar surfaces (trimmed from desktop's 28)
 *   - No image paste (Captures handles attachment per audit A3)
 *   - No table / task list / video / voice (deferred)
 *   - Color via .text-color-X classes (mirrors .indent-N pattern;
 *     execCommand foreColor emits style="color:..." which DOMPurify
 *     strips, so we wrap selection in <span class="text-color-X">
 *     via custom insertHTML)
 *   - Sticky-above-keyboard toolbar via visualViewport observer
 *
 * DOMPurify config inherited verbatim from desktop ThoughtsEditor's
 * PASTE_ALLOWED_TAGS + PASTE_ALLOWED_ATTR (minus tags this editor
 * doesn't emit: input, iframe, img, table — kept in case content
 * was authored on desktop and surfaces here, but stripped to keep
 * the mobile surface tight).
 *
 * iOS native BIU selection menu works automatically with
 * contentEditable; no custom code needed.
 */

// ── DOMPurify config ────────────────────────────────────────────────

const PASTE_ALLOWED_TAGS = [
  "b", "i", "u", "s", "strike", "em", "strong", "a", "br", "p", "div", "span",
  "ul", "ol", "li",
  "h1", "h2", "h3",
  "blockquote", "pre", "code", "hr",
];

const PASTE_ALLOWED_ATTR = [
  "href",
  "class", // .text-color-X / .indent-N
];

const TEXT_COLORS = ["rose", "amber", "emerald", "sky", "violet"] as const;
type TextColor = (typeof TEXT_COLORS)[number];

// ── Helpers (mirrored from desktop ThoughtsEditor where applicable) ──

/** Convert inline `margin-left: Npx` (execCommand indent output) to
 *  `.indent-N` classes so they survive the style-disallowed sanitizer.
 *  Mirrors thoughts-editor.tsx:226-248. */
function normalizeIndentation(root: HTMLElement): void {
  const blocks = root.querySelectorAll<HTMLElement>(
    "p, div, h1, h2, h3, blockquote, ul, ol, pre, li",
  );
  blocks.forEach((el) => {
    const ml = el.style.marginLeft;
    if (!ml) return;
    const px = parseInt(ml, 10);
    if (isNaN(px) || px <= 0) {
      el.style.removeProperty("margin-left");
      if (!el.getAttribute("style")?.trim()) el.removeAttribute("style");
      return;
    }
    const level = Math.min(6, Math.max(1, Math.round(px / 40)));
    el.style.removeProperty("margin-left");
    if (!el.getAttribute("style")?.trim()) el.removeAttribute("style");
    Array.from(el.classList).forEach((c) => {
      if (c.startsWith("indent-")) el.classList.remove(c);
    });
    el.classList.add(`indent-${level}`);
  });
}

/** Sanitize HTML via DOMPurify with the same allow-list as desktop. */
function sanitize(html: string): string {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: PASTE_ALLOWED_TAGS,
    ALLOWED_ATTR: PASTE_ALLOWED_ATTR,
  });
}

/** Wrap current selection in <span class="text-color-X">. */
function applyTextColor(color: TextColor): void {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return;
  const range = selection.getRangeAt(0);
  const span = document.createElement("span");
  span.className = `text-color-${color}`;
  // Wrap the selected fragment. If the selection spans multiple
  // existing color spans, extractContents preserves their inner text
  // but the new wrap replaces the outer color anyway.
  try {
    const contents = range.extractContents();
    span.appendChild(contents);
    range.insertNode(span);
    // Restore selection over the new span so subsequent taps work.
    selection.removeAllRanges();
    const fresh = document.createRange();
    fresh.selectNodeContents(span);
    selection.addRange(fresh);
  } catch {
    // Cross-block selections can throw; silent fail keeps the editor
    // responsive.
  }
}

/** Remove any .text-color-X wrapping covering the current selection. */
function clearTextColor(): void {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return;
  const range = selection.getRangeAt(0);
  // Walk up to find the nearest ancestor span with a text-color- class.
  let node: Node | null = range.commonAncestorContainer;
  while (node && node !== document.body) {
    if (node.nodeType === Node.ELEMENT_NODE) {
      const el = node as HTMLElement;
      if (el.tagName === "SPAN") {
        const colorClass = Array.from(el.classList).find((c) =>
          c.startsWith("text-color-"),
        );
        if (colorClass) {
          // Unwrap: replace span with its children, preserve selection.
          const parent = el.parentNode;
          if (!parent) return;
          while (el.firstChild) parent.insertBefore(el.firstChild, el);
          parent.removeChild(el);
          return;
        }
      }
    }
    node = node.parentNode;
  }
}

// ── Component ───────────────────────────────────────────────────────

export type MobileRichTextEditorProps = {
  /** Initial HTML value seeded into the editor body on mount. */
  initialValue: string;
  /** Fires on every input with the sanitized HTML. */
  onChange: (html: string) => void;
  /** Optional dirty change reporter — fires when current value
   *  flips relative to `initialValue`. */
  onDirtyChange?: (dirty: boolean) => void;
  /** Placeholder shown when the editor body is empty. */
  placeholder?: string;
};

export function MobileRichTextEditor({
  initialValue,
  onChange,
  onDirtyChange,
  placeholder = "Start writing…",
}: MobileRichTextEditorProps) {
  const editorRef = useRef<HTMLDivElement | null>(null);
  const initialRef = useRef(initialValue);
  const lastDirtyRef = useRef(false);
  // Saved selection range — captured when the heading or color trigger
  // is tapped, restored before the chosen swatch/option fires its
  // command. iOS Safari can blur the editor when the popover opens,
  // and the `onMouseDown`/`onPointerDown` preventDefault on toolbar
  // buttons doesn't always protect against this for popovers that
  // re-render with a new DOM subtree. Save+restore makes the
  // dropdown-then-action flow reliable.
  const savedRangeRef = useRef<Range | null>(null);

  const [headingOpen, setHeadingOpen] = useState(false);
  const [colorOpen, setColorOpen] = useState(false);
  const [keyboardHeight, setKeyboardHeight] = useState(0);
  const [isEmpty, setIsEmpty] = useState(initialValue.trim().length === 0);

  /** Snapshot the current selection range so the chosen popover
   *  action can restore it before firing a command. */
  const saveSelection = useCallback(() => {
    const sel = window.getSelection();
    if (sel && sel.rangeCount > 0) {
      savedRangeRef.current = sel.getRangeAt(0).cloneRange();
    }
  }, []);

  /** Restore the previously-saved selection range, if any, and focus
   *  the editor. No-ops when nothing was saved. */
  const restoreSelection = useCallback(() => {
    editorRef.current?.focus();
    const sel = window.getSelection();
    if (savedRangeRef.current && sel) {
      sel.removeAllRanges();
      sel.addRange(savedRangeRef.current);
    }
  }, []);

  // ── Seed contentEditable with initial HTML on mount ───────────────
  useEffect(() => {
    const el = editorRef.current;
    if (!el) return;
    el.innerHTML = initialValue || "";
    initialRef.current = initialValue;
    setIsEmpty(initialValue.trim().length === 0);
    // Focus the editor after the sheet animation lands.
    const t = window.setTimeout(() => {
      el.focus();
      const sel = window.getSelection();
      if (sel) {
        const range = document.createRange();
        range.selectNodeContents(el);
        range.collapse(false);
        sel.removeAllRanges();
        sel.addRange(range);
      }
    }, 320);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── visualViewport observer for sticky-above-keyboard toolbar ─────
  useEffect(() => {
    if (typeof window === "undefined") return;
    const vv = window.visualViewport;
    if (!vv) return;
    const update = () => {
      const offset = window.innerHeight - vv.height - vv.offsetTop;
      setKeyboardHeight(Math.max(0, offset));
    };
    update();
    vv.addEventListener("resize", update);
    vv.addEventListener("scroll", update);
    return () => {
      vv.removeEventListener("resize", update);
      vv.removeEventListener("scroll", update);
    };
  }, []);

  // ── Emit change + dirty signal ────────────────────────────────────
  const emitChange = useCallback(() => {
    const el = editorRef.current;
    if (!el) return;
    normalizeIndentation(el);
    const raw = el.innerHTML;
    const clean = sanitize(raw);
    if (clean !== raw) {
      // Re-applying clean would clobber the caret. Skip in-place
      // rewrite during typing; sanitization runs again on Save when
      // the consumer reads the value via onChange.
    }
    onChange(clean);
    setIsEmpty(el.textContent?.trim().length === 0);
    const dirty = clean !== initialRef.current;
    if (dirty !== lastDirtyRef.current) {
      lastDirtyRef.current = dirty;
      onDirtyChange?.(dirty);
    }
  }, [onChange, onDirtyChange]);

  // ── Toolbar actions ───────────────────────────────────────────────
  const runCommand = useCallback(
    (cmd: string, arg?: string) => {
      const el = editorRef.current;
      el?.focus();
      document.execCommand(cmd, false, arg);
      emitChange();
    },
    [emitChange],
  );

  const insertCodeInline = useCallback(() => {
    const el = editorRef.current;
    el?.focus();
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    if (range.collapsed) return;
    const code = document.createElement("code");
    try {
      code.appendChild(range.extractContents());
      range.insertNode(code);
      sel.removeAllRanges();
      const fresh = document.createRange();
      fresh.selectNodeContents(code);
      sel.addRange(fresh);
      emitChange();
    } catch {
      // ignore cross-block selections
    }
  }, [emitChange]);

  const insertLink = useCallback(() => {
    const el = editorRef.current;
    el?.focus();
    const url = window.prompt("Enter URL");
    if (!url) return;
    document.execCommand("createLink", false, url);
    emitChange();
  }, [emitChange]);

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLDivElement>) => {
      // Block image paste entirely (T2-4b A3 decision: Captures
      // handles image attachment; no inline images in this editor).
      const files = Array.from(e.clipboardData.files ?? []);
      if (files.some((f) => f.type.startsWith("image/"))) {
        e.preventDefault();
        return;
      }
      // Sanitize text/HTML paste through DOMPurify before insert.
      e.preventDefault();
      const html = e.clipboardData.getData("text/html");
      const text = e.clipboardData.getData("text/plain");
      const incoming = html || text;
      if (!incoming) return;
      const clean = sanitize(incoming);
      document.execCommand("insertHTML", false, clean);
      emitChange();
    },
    [emitChange],
  );

  const applyHeading = useCallback(
    (tag: "h1" | "h2" | "h3" | "p") => {
      restoreSelection();
      document.execCommand("formatBlock", false, `<${tag}>`);
      emitChange();
      setHeadingOpen(false);
    },
    [restoreSelection, emitChange],
  );

  const applyColor = useCallback(
    (color: TextColor | "default") => {
      restoreSelection();
      if (color === "default") clearTextColor();
      else applyTextColor(color);
      emitChange();
      setColorOpen(false);
    },
    [restoreSelection, emitChange],
  );

  const openHeadingMenu = useCallback(() => {
    saveSelection();
    setHeadingOpen((v) => !v);
  }, [saveSelection]);

  const openColorMenu = useCallback(() => {
    saveSelection();
    setColorOpen((v) => !v);
  }, [saveSelection]);

  // ── Render ────────────────────────────────────────────────────────
  const toolbarStyle: CSSProperties = {
    bottom: keyboardHeight > 0 ? `${keyboardHeight}px` : undefined,
  };

  return (
    <div
      data-testid="mobile-rich-text-editor"
      className="relative flex h-full flex-col"
    >
      {/* contentEditable body — fills available space above toolbar. */}
      <div
        ref={editorRef}
        contentEditable
        suppressContentEditableWarning
        onInput={emitChange}
        onPaste={handlePaste}
        data-testid="mobile-rich-text-editor-body"
        aria-label="Editor"
        className="mobile-rte-body flex-1 overflow-y-auto px-4 py-3 text-[15px] leading-relaxed text-m-text focus:outline-none"
        style={{
          paddingBottom: `${56 + keyboardHeight + 16}px`,
          WebkitUserSelect: "text",
        }}
      />
      {isEmpty && (
        <div
          data-testid="mobile-rich-text-editor-placeholder"
          aria-hidden="true"
          className="pointer-events-none absolute left-4 top-3 text-[15px] text-m-text-faint"
        >
          {placeholder}
        </div>
      )}

      {/* Sticky-above-keyboard toolbar. */}
      <div
        data-testid="mobile-rich-text-editor-toolbar"
        className="sticky z-10 flex shrink-0 items-center overflow-x-auto whitespace-nowrap border-t-[0.5px] border-m-border bg-m-surface"
        style={{
          ...toolbarStyle,
          position: "sticky",
          bottom: keyboardHeight > 0 ? `${keyboardHeight}px` : 0,
          paddingLeft: 8,
          paddingRight: 8,
          paddingTop: 5,
          paddingBottom: 5,
          gap: 2,
          WebkitOverflowScrolling: "touch",
        }}
      >
        <ToolbarBtn label="Bold" icon={<Bold size={16} />} onAction={() => runCommand("bold")} />
        <ToolbarBtn label="Italic" icon={<Italic size={16} />} onAction={() => runCommand("italic")} />
        <ToolbarBtn label="Underline" icon={<Underline size={16} />} onAction={() => runCommand("underline")} />
        <Divider />

        <HeadingDropdown
          open={headingOpen}
          onOpen={openHeadingMenu}
          onSelect={applyHeading}
        />
        <Divider />

        <ToolbarBtn label="Bulleted list" icon={<List size={16} />} onAction={() => runCommand("insertUnorderedList")} />
        <ToolbarBtn label="Numbered list" icon={<ListOrdered size={16} />} onAction={() => runCommand("insertOrderedList")} />
        <ToolbarBtn label="Indent" icon={<IndentIncrease size={16} />} onAction={() => runCommand("indent")} />
        <ToolbarBtn label="Outdent" icon={<IndentDecrease size={16} />} onAction={() => runCommand("outdent")} />
        <Divider />

        <ColorPicker
          open={colorOpen}
          onOpen={openColorMenu}
          onSelect={applyColor}
        />
        <ToolbarBtn label="Insert link" icon={<LinkIcon size={16} />} onAction={insertLink} />
        <Divider />

        <ToolbarBtn label="Quote" icon={<Quote size={16} />} onAction={() => runCommand("formatBlock", "<blockquote>")} />
        <ToolbarBtn label="Inline code" icon={<Code size={16} />} onAction={insertCodeInline} />
        <Divider />

        <ToolbarBtn
          label="Clear formatting"
          icon={<RemoveFormatting size={16} />}
          onAction={() => runCommand("removeFormat")}
        />
        <Divider />

        <ToolbarBtn label="Undo" icon={<Undo size={16} />} onAction={() => runCommand("undo")} />
        <ToolbarBtn label="Redo" icon={<Redo size={16} />} onAction={() => runCommand("redo")} />
      </div>
    </div>
  );
}

// ── Toolbar sub-components ─────────────────────────────────────────

function ToolbarBtn({
  label,
  icon,
  onAction,
}: {
  label: string;
  icon: React.ReactNode;
  onAction: () => void;
}) {
  // Execute the action IN the pointerdown handler — NOT in onClick.
  //
  // Why: iOS Safari, given preventDefault on pointerdown to preserve
  // contentEditable selection, suppresses the synthetic click event
  // that would otherwise follow. If the action is wired to onClick,
  // it never fires on iOS. Production contentEditable toolbars
  // (TipTap mobile, Lexical mobile, etc.) fire from pointerdown.
  //
  // onClick is NOT a fallback here: on desktop, pointerdown fires
  // for mouse interactions too, so wiring both would double-fire.
  // Mobile editor toolbars don't need keyboard-activation support
  // (Tab + Enter on an icon-only button) — that's a desktop pattern,
  // and the desktop ThoughtsEditor handles that case separately.
  // mousedown handler stays for any non-pointer-supporting browsers.
  return (
    <button
      type="button"
      onMouseDown={(e) => e.preventDefault()}
      onPointerDown={(e) => {
        e.preventDefault();
        onAction();
      }}
      aria-label={label}
      data-testid={`mobile-rte-toolbar-${slug(label)}`}
      className="flex h-9 w-9 shrink-0 items-center justify-center rounded-m-sm text-m-text active:bg-m-surface-2"
    >
      {icon}
    </button>
  );
}

function Divider() {
  return (
    <span
      aria-hidden="true"
      className="h-5 w-px shrink-0 bg-m-border"
      style={{ margin: "0 4px" }}
    />
  );
}

function HeadingDropdown({
  open,
  onOpen,
  onSelect,
}: {
  open: boolean;
  onOpen: () => void;
  onSelect: (tag: "h1" | "h2" | "h3" | "p") => void;
}) {
  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onMouseDown={(e) => e.preventDefault()}
        onPointerDown={(e) => {
          e.preventDefault();
          onOpen();
        }}
        aria-label="Headings"
        aria-expanded={open}
        data-testid="mobile-rte-toolbar-headings"
        className="flex h-9 w-9 items-center justify-center rounded-m-sm text-m-text active:bg-m-surface-2"
      >
        <Heading size={16} aria-hidden="true" />
      </button>
      {open && (
        <div
          role="menu"
          data-testid="mobile-rte-heading-menu"
          className="absolute bottom-[44px] left-0 z-20 min-w-[120px] overflow-hidden rounded-m-md border-[0.5px] border-m-border bg-m-surface shadow-lg"
        >
          {(
            [
              { tag: "p", label: "Normal" },
              { tag: "h1", label: "Heading 1" },
              { tag: "h2", label: "Heading 2" },
              { tag: "h3", label: "Heading 3" },
            ] as const
          ).map((opt) => (
            <button
              key={opt.tag}
              type="button"
              role="menuitem"
              onMouseDown={(e) => e.preventDefault()}
              onPointerDown={(e) => {
                e.preventDefault();
                onSelect(opt.tag);
              }}
              data-testid={`mobile-rte-heading-${opt.tag}`}
              className="block w-full border-b-[0.5px] border-m-border px-3 py-2 text-left text-[13px] text-m-text last:border-b-0 active:bg-m-surface-2"
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// Saturated palette colors for inline text styling on dark surface.
// Mirrors the dot color from TAG_PALETTE (more legible against
// --m-surface than the muted .text values used in tag chips).
const COLOR_SWATCH_HEX: Record<TextColor, string> = {
  rose: "#f43f5e",
  amber: "#f59f00",
  emerald: "#08a86b",
  sky: "#0d6efd",
  violet: "#8b5cf6",
};

function ColorPicker({
  open,
  onOpen,
  onSelect,
}: {
  open: boolean;
  onOpen: () => void;
  onSelect: (color: TextColor | "default") => void;
}) {
  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onMouseDown={(e) => e.preventDefault()}
        onPointerDown={(e) => {
          e.preventDefault();
          onOpen();
        }}
        aria-label="Text color"
        aria-expanded={open}
        data-testid="mobile-rte-toolbar-color"
        className="flex h-9 w-9 items-center justify-center rounded-m-sm text-m-text active:bg-m-surface-2"
      >
        <Palette size={16} aria-hidden="true" />
      </button>
      {open && (
        <div
          role="menu"
          data-testid="mobile-rte-color-menu"
          className="absolute bottom-[44px] left-0 z-20 flex items-center gap-1.5 rounded-m-md border-[0.5px] border-m-border bg-m-surface px-2 py-2 shadow-lg"
        >
          {TEXT_COLORS.map((c) => (
            <button
              key={c}
              type="button"
              role="menuitem"
              onMouseDown={(e) => e.preventDefault()}
              onPointerDown={(e) => {
                e.preventDefault();
                onSelect(c);
              }}
              aria-label={`${c} text`}
              data-testid={`mobile-rte-color-${c}`}
              className="h-7 w-7 rounded-full border-[0.5px] border-m-border"
              style={{ background: COLOR_SWATCH_HEX[c] }}
            />
          ))}
          <button
            type="button"
            role="menuitem"
            onMouseDown={(e) => e.preventDefault()}
            onPointerDown={(e) => {
              e.preventDefault();
              onSelect("default");
            }}
            aria-label="Default (reset color)"
            data-testid="mobile-rte-color-default"
            className="flex h-7 w-7 items-center justify-center rounded-full border border-dashed border-m-border-strong text-m-text-dim"
          >
            <span aria-hidden="true">⊘</span>
          </button>
        </div>
      )}
    </div>
  );
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}
