"use client";

// Phase 3.5 Weekly Thoughts editor. Builds on the Phase 3 contentEditable
// + document.execCommand foundation, adding lists, indent/outdent,
// headings, blockquote/code/HR, an inline link popover (replacing the
// old window.prompt), functional font family/size dropdowns, video
// embed (YouTube/Vimeo), tables with a hover toolbar, and task lists
// with clickable checkboxes.
//
// execCommand is formally deprecated in MDN but enjoys universal
// browser support and has no standardized replacement; this is still
// the pragmatic v1 choice for a single-surface rich text editor that
// doesn't justify pulling in a 150KB editor library (TipTap,
// ProseMirror). When Phase 7 redesigns the daily editor and we need
// richer features (collab, comments, blocks), this component is the
// migration trigger.
//
// The component owns:
//   - The collapsible card chrome (header + body, matches the Phase 2
//     Per-Ticker expander idiom)
//   - localStorage-persisted collapse state
//   - The expanded toolbar — three custom dropdowns (Style/Font/Size),
//     ~14 inline buttons, and 2 inline popover triggers (link + video)
//   - The contentEditable area, its placeholder, paste sanitization,
//     and a delegated click handler for task-list checkboxes
//   - Cursor-preserving hydration: external value updates do NOT
//     overwrite innerHTML while the editor is focused OR while either
//     popover is open (mirrors the Phase 0 stale-list-race lesson)
//   - Selection-change tracking for the headings-dropdown current-
//     value label AND the table hover toolbar
//   - Range save/restore around popover lifecycle so insertHTML lands
//     where the user clicked
//
// The parent (weekly-retro.tsx) owns the value state, dirtyRef flip
// on edit, debounced PUT, and API roundtrip — unchanged from Phase 3.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DOMPurify from "dompurify";
import { api } from "@/lib/api";
import { Icons } from "./icons";
import { SectionExpander } from "./section-expander";
import { ImageLightbox, type LightboxImage } from "./image-lightbox";
import { ToolbarDropdown, type ToolbarDropdownOption } from "./weekly-thoughts/toolbar-dropdown";
import { ColorPicker } from "./weekly-thoughts/color-picker";
import { UrlPopover } from "./weekly-thoughts/url-popover";

interface WeeklyThoughtsProps {
  value: string;
  onChange: (html: string) => void;
  /** Phase 4.1: required for inline image paste. When null, image-paste
   *  attempts surface a friendly "save the retro first" inline error
   *  (matches TagPicker / WeeklySnapshot's disabled-state idiom). When
   *  undefined, image paste is silently disabled — same UX as null. */
  retroId?: number | null;
  /** Phase 4.1: portfolio name forwarded to the upload endpoint. */
  portfolio?: string;
}

// Persisted under this key by <SectionExpander>. The constant is held here
// (not inlined at the call site) so a future test or migration can find
// the canonical name without reading the JSX prop tree.
const EXPANDED_KEY = "mo-weekly-retro-thoughts-expanded";

// =============================================================================
// DOMPurify configuration
// =============================================================================
//
// HOOK NOTE: DOMPurify hooks are GLOBAL on the singleton — they apply
// to every DOMPurify.sanitize() call in the app, not just ours. We
// register once at module-load via the _hooksRegistered guard. Both
// hooks below are *strict tightenings* (iframe-src whitelist,
// input-type filter), so even if a future consumer of DOMPurify
// appears, this side effect can only make their pass *safer*, never
// less safe. If we ever need per-call config, switch to
// `createDOMPurify(window)` factory instances.

// Closed allow-list of inline HTML tags accepted from paste sources.
// Matches what document.execCommand emits natively across browsers,
// plus a few synonyms (em/strong as well as i/b; strike as well as s),
// plus the Phase 3.5 block-level surface (lists, headings, blockquote,
// pre/code, hr, tables, task-list input, video iframe).
const PASTE_ALLOWED_TAGS = [
  // Phase 3 inline
  "b", "i", "u", "s", "strike", "em", "strong", "a", "br", "p", "div", "span",
  // Phase 3.5 lists
  "ul", "ol", "li",
  // Phase 3.5 headings
  "h1", "h2", "h3",
  // Phase 3.5 block content
  "blockquote", "pre", "code", "hr",
  // Phase 3.5 tables
  "table", "thead", "tbody", "tr", "td", "th", "colgroup", "col",
  // Phase 3.5 task lists — input element, type-gated by hook below
  "input",
  // Phase 3.5 video embed — src-gated by hook below
  "iframe",
  // Phase 4.1 inline images — src-gated by hook below (R2 prefix only)
  "img",
];

// Attributes allowed on the above tags. `style` is INTENTIONALLY NOT
// here — indent normalization (§9 of the audit) converts execCommand's
// inline `margin-left` to class-based `.indent-N` instead, sidestepping
// the style-attribute XSS surface entirely.
const PASTE_ALLOWED_ATTR = [
  "href",
  "class",                                            // .task-list / .wt-* / .indent-N / .wt-uploading
  "rowspan", "colspan",                               // tables
  "type", "checked", "disabled",                      // input (type filtered by hook)
  "src", "width", "height",                           // iframe + img (src filtered by hook)
  "frameborder", "allowfullscreen", "allow",          // iframe
  // Phase 4.1 img attrs
  "alt", "loading",
];

// Strict YouTube + Vimeo embed-URL whitelist. The query-string portion
// is allowed any subset of [\w=&,.-] so YouTube's typical `?start=…` or
// `?rel=0` survives without inviting open-ended attribute values.
const IFRAME_SRC_PATTERN =
  /^https:\/\/(www\.youtube\.com\/embed\/[\w-]+(\?[\w=&,.-]*)?|player\.vimeo\.com\/video\/\d+(\?[\w=&,.-]*)?)$/;

// Phase 4.1 — img src whitelist for inline images. Only allow URLs from
// our R2 public bucket under the weekly_retros/ prefix. If
// NEXT_PUBLIC_R2_PUBLIC_URL is set at build time, use that exact prefix.
// Otherwise fall back to a permissive HTTPS-pattern that still requires
// the /weekly_retros/ path + an image extension (defense in depth).
const _R2_PUBLIC_URL = (process.env.NEXT_PUBLIC_R2_PUBLIC_URL || "")
  .replace(/\/$/, "");
function _escapeRegex(s: string): string {
  return s.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&");
}
const IMG_SRC_PATTERN = _R2_PUBLIC_URL
  ? new RegExp(
      "^" + _escapeRegex(_R2_PUBLIC_URL) +
      "/weekly_retros/[\\w/.-]+\\.(png|jpe?g|gif|webp)(\\?[\\w=&,.-]*)?$",
    )
  : /^https:\/\/[\w.-]+\/weekly_retros\/[\w/.-]+\.(png|jpe?g|gif|webp)(\?[\w=&,.-]*)?$/i;

let _hooksRegistered = false;
function ensureSanitizerHooks() {
  if (_hooksRegistered) return;
  _hooksRegistered = true;

  // Strip iframe src that doesn't match the whitelist; strip input type
  // values other than "checkbox"; strip img src that doesn't match the
  // R2-prefix whitelist. These hooks run during sanitize().
  DOMPurify.addHook("uponSanitizeAttribute", (node, data) => {
    if (node.nodeName === "IFRAME" && data.attrName === "src") {
      if (!IFRAME_SRC_PATTERN.test(data.attrValue)) data.keepAttr = false;
    }
    if (node.nodeName === "INPUT" && data.attrName === "type") {
      if (data.attrValue !== "checkbox") data.keepAttr = false;
    }
    if (node.nodeName === "IMG" && data.attrName === "src") {
      // Allow blob: URLs during the in-flight upload window. The
      // editor's emitChange is suppressed while uploads pend, so a
      // blob URL never reaches the server. Sanitize() also runs on
      // paste — but blob URLs are document-scoped and won't appear
      // in pasted HTML from external sources.
      if (!data.attrValue.startsWith("blob:") && !IMG_SRC_PATTERN.test(data.attrValue)) {
        data.keepAttr = false;
      }
    }
  });

  // Remove iframes whose src was stripped (otherwise we'd keep an
  // empty iframe element). Same for inputs whose type was stripped —
  // anything that's not a checkbox-input is removed entirely. Same
  // for imgs whose src was stripped.
  // NOTE: must use afterSanitizeAttributes (per-element, runs *after*
  // that element's attribute pass), NOT afterSanitizeElements (whole-
  // document, runs once at the end with the root fragment).
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.nodeName === "IFRAME" && !(node as Element).getAttribute("src")) {
      (node as Element).parentNode?.removeChild(node);
      return;
    }
    if (node.nodeName === "INPUT") {
      const t = (node as HTMLInputElement).getAttribute("type");
      if (t !== "checkbox") (node as Element).parentNode?.removeChild(node);
      return;
    }
    if (node.nodeName === "IMG" && !(node as Element).getAttribute("src")) {
      (node as Element).parentNode?.removeChild(node);
    }
  });
}

// =============================================================================
// Helper functions
// =============================================================================

// Convert inline `margin-left: Npx` (Chrome/WebKit indent output) to a
// `.indent-N` class so it survives the style-disallowed DOMPurify pass.
// Firefox emits nested <blockquote> for indent — CSS handles that
// naturally; the normalizer ignores blockquotes.
export function normalizeIndentation(root: HTMLElement) {
  const blocks = root.querySelectorAll<HTMLElement>(
    "p, div, h1, h2, h3, blockquote, ul, ol, pre, li",
  );
  blocks.forEach(el => {
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
    // Strip any pre-existing indent class then apply the new one.
    Array.from(el.classList).forEach(c => {
      if (c.startsWith("indent-")) el.classList.remove(c);
    });
    el.classList.add(`indent-${level}`);
  });
}

// Parse a YouTube or Vimeo URL into a normalized embed src. Returns
// null for anything else. Exported for unit testing.
export function parseVideoUrl(url: string): { src: string; provider: "youtube" | "vimeo" } | null {
  // youtu.be/{id}
  let m = url.match(/^https?:\/\/youtu\.be\/([\w-]+)/);
  if (m) return { src: `https://www.youtube.com/embed/${m[1]}`, provider: "youtube" };
  // youtube.com/watch?v={id}
  m = url.match(/^https?:\/\/(?:www\.|m\.)?youtube\.com\/watch\?[^#]*?v=([\w-]+)/);
  if (m) return { src: `https://www.youtube.com/embed/${m[1]}`, provider: "youtube" };
  // youtube.com/embed/{id} — passthrough
  m = url.match(/^https?:\/\/(?:www\.)?youtube\.com\/embed\/([\w-]+)/);
  if (m) return { src: `https://www.youtube.com/embed/${m[1]}`, provider: "youtube" };
  // vimeo.com/{numericId}
  m = url.match(/^https?:\/\/(?:www\.)?vimeo\.com\/(\d+)/);
  if (m) return { src: `https://player.vimeo.com/video/${m[1]}`, provider: "vimeo" };
  return null;
}

// Minimal HTML-character escape for user-supplied URLs/text being
// injected into an attribute or text node. Not a sanitizer — the
// DOMPurify pass downstream is the real boundary.
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Walk the selection up to find the nearest <table>/<td>/<th>/etc.
function closestEl<T extends Element>(node: Node | null, selector: string): T | null {
  let cur: Node | null = node;
  while (cur && cur.nodeType !== Node.ELEMENT_NODE) cur = cur.parentNode;
  return cur ? ((cur as Element).closest(selector) as T | null) : null;
}

// =============================================================================
// Component
// =============================================================================

// Fixed swatch colors per Phase 3. Highlight is the design's amber
// (#f59f00 → matches the orange dot on the section header). Text
// color is the design's emerald.
// Phase 3 legacy single-color constants. Retained only as the
// "default first swatch" of the Phase 4.2 palettes. The ToolbarColorBtn
// behavior was replaced with ColorPicker popovers.
const HIGHLIGHT_COLOR = "#ffeaa7";
const TEXT_COLOR = "#08a86b";

// Phase 4.2 — color picker palettes. The trailing entry of each
// palette is the "reset" sentinel: "transparent" for highlight clears
// the background; "inherit" for text color resets to the editor's
// default ink. ColorPicker renders the reset swatch with a diagonal-
// slash visual instead of a solid fill.
const HIGHLIGHT_PALETTE = [
  HIGHLIGHT_COLOR,    // amber (default — matches Phase 3 fixed color)
  "#fef3c7",          // amber-100
  "#dbeafe",          // blue-100
  "#dcfce7",          // green-100
  "#fee2e2",          // red-100
  "#e0e7ff",          // indigo-100
  "#fce7f3",          // pink-100
  "transparent",      // reset
];

const TEXT_COLOR_PALETTE = [
  TEXT_COLOR,         // emerald (default — matches Phase 3 fixed color)
  "#0f172a",          // slate-900
  "#475569",          // slate-600
  "#dc2626",          // red-600
  "#ea580c",          // orange-600
  "#2563eb",          // blue-600
  "#7c3aed",          // violet-600
  "inherit",          // reset
];

const HEADING_OPTIONS: ToolbarDropdownOption[] = [
  { value: "<p>",  label: "Normal" },
  { value: "<h1>", label: "Heading 1" },
  { value: "<h2>", label: "Heading 2" },
  { value: "<h3>", label: "Heading 3" },
];

const FONT_OPTIONS: ToolbarDropdownOption[] = [
  { value: "",                                         label: "Default" },
  { value: '"Inter", system-ui, sans-serif',           label: "Sans-serif" },
  { value: '"Fraunces", Georgia, serif',               label: "Serif" },
  { value: '"JetBrains Mono", ui-monospace, monospace', label: "Mono" },
];

// execCommand("fontSize", N) — N is 1-7 (legacy). Coarse but stable.
const SIZE_OPTIONS: ToolbarDropdownOption[] = [
  { value: "1", label: "10" },
  { value: "2", label: "12" },
  { value: "3", label: "13" },
  { value: "4", label: "16" },
  { value: "5", label: "18" },
  { value: "6", label: "24" },
  { value: "7", label: "32" },
];

// Map `formatBlock` query result back to a visible label for the
// Style dropdown trigger.
const BLOCK_LABELS: Record<string, string> = {
  h1: "Heading 1",
  h2: "Heading 2",
  h3: "Heading 3",
  p: "Normal",
  div: "Normal",
  blockquote: "Quote",
  pre: "Code",
};

export function WeeklyThoughts({
  value,
  onChange,
  retroId = null,
  portfolio = "",
}: WeeklyThoughtsProps) {
  // Register DOMPurify hooks on first component construction. Idempotent.
  ensureSanitizerHooks();

  // Phase 3.5 popovers + selection tracking state.
  const [linkPopoverOpen, setLinkPopoverOpen] = useState(false);
  const [videoPopoverOpen, setVideoPopoverOpen] = useState(false);
  const [linkInitialUrl, setLinkInitialUrl] = useState("");
  const [linkHasTextInput, setLinkHasTextInput] = useState(false);
  const [currentBlockLabel, setCurrentBlockLabel] = useState("Normal");
  // Track the focused table for the hover toolbar. Re-derived on every
  // selectionchange so it stays in sync as the cursor moves.
  const [focusedTable, setFocusedTable] = useState<HTMLTableElement | null>(null);
  const [focusedCell, setFocusedCell] = useState<HTMLTableCellElement | null>(null);
  const [tableToolbarPos, setTableToolbarPos] = useState<{ top: number; left: number } | null>(null);
  // Phase 4.1: inline error for paste rejections (oversize, bad MIME,
  // retroId missing). Auto-dismisses after 3s.
  const [inlineError, setInlineError] = useState<string | null>(null);
  // Phase 4.2: lightbox for inline editor images. Click on an <img>
  // in the editor body opens a single-image lightbox (no prev/next —
  // there's no ordered list of "all images in the editor"; treat each
  // click as a discrete preview).
  const [lightboxImage, setLightboxImage] = useState<LightboxImage | null>(null);

  const editorRef = useRef<HTMLDivElement | null>(null);

  // Tracks the last HTML we wrote to the DOM. Used by the cursor-
  // preserving hydration effect: if incoming `value` matches what's
  // already in the DOM, skip the write (would destroy the cursor).
  const lastWrittenHtmlRef = useRef<string>(value);

  // Saved Range for popover round-trips. The popover focuses an input
  // (link URL field, video URL field), which clears the editor's
  // selection. We restore on submit so insertHTML lands in the right
  // place.
  const savedRangeRef = useRef<Range | null>(null);

  // Phase 4.1: in-flight image uploads. Save (onChange) is SUPPRESSED
  // while this count is > 0 — the editor body contains blob: URLs
  // which would be meaningless across reload and must never reach
  // the DB. When the counter returns to 0, any queued emit fires
  // with the now-real URLs.
  const pendingUploadsRef = useRef(0);
  const pendingEmitRef = useRef(false);

  // Word count for the collapsed-header caption. Strips HTML tags,
  // splits on whitespace, filters empties.
  const wordCount = useMemo(() => {
    if (!value) return 0;
    const text = value.replace(/<[^>]*>/g, " ").replace(/&nbsp;/g, " ");
    return text.split(/\s+/).filter(Boolean).length;
  }, [value]);

  // Cursor-preserving hydration. Extended in Phase 3.5: skip when
  // either popover is open (focus is on a popover input, not the
  // editor, so the activeElement check would otherwise fail and the
  // effect would stomp the user's in-flight edit).
  useEffect(() => {
    const el = editorRef.current;
    if (!el) return;
    if (value === lastWrittenHtmlRef.current) return;
    if (
      document.activeElement === el ||
      linkPopoverOpen ||
      videoPopoverOpen
    ) return;
    el.innerHTML = value || "";
    lastWrittenHtmlRef.current = value || "";
  }, [value, linkPopoverOpen, videoPopoverOpen]);

  // Set initial content on mount (the effect above only catches
  // changes to value; we still need a first render baseline).
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
    // Phase 4.1: suppress emit while image uploads are in flight. The
    // editor body contains blob: URLs at this point — useless across
    // reload. Mark pending so we fire the queued emit when the last
    // upload completes.
    if (pendingUploadsRef.current > 0) {
      pendingEmitRef.current = true;
      return;
    }
    pendingEmitRef.current = false;
    const html = el.innerHTML;
    lastWrittenHtmlRef.current = html;
    onChange(html);
  }, [onChange]);

  // Phase 4.1: inline error toast helper. Auto-dismisses after 3s.
  const showInlineError = useCallback((msg: string) => {
    setInlineError(msg);
    window.setTimeout(() => setInlineError(null), 3000);
  }, []);

  // Phase 4.1: upload a single image file and swap the optimistic blob
  // URL <img> for the real R2 URL when the upload returns. The img
  // element is identified via a data-upload-id matching the blob URL
  // we generated. On failure the img is removed and an inline error
  // shown.
  const uploadAndInsertImage = useCallback(async (file: File) => {
    if (retroId == null) {
      showInlineError("Save the retro first to embed images.");
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    const uploadId = `wt-img-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    // Insert the optimistic <img> at cursor. data-upload-id lets us
    // find this exact element later to swap src or remove on failure.
    // class="wt-uploading" picks up the CSS opacity/outline styling.
    const html =
      `<img src="${previewUrl}" data-upload-id="${uploadId}" ` +
      `class="wt-uploading" alt="${file.name.replace(/"/g, "&quot;")}" />`;
    const el = editorRef.current;
    if (!el) { URL.revokeObjectURL(previewUrl); return; }
    el.focus();
    document.execCommand("insertHTML", false, html);
    pendingUploadsRef.current += 1;

    try {
      const res = await api.uploadWeeklyThoughtsImage(retroId, file, portfolio);
      if (!res || typeof res !== "object" || !("view_url" in res)) {
        throw new Error("upload_failed");
      }
      // Swap the placeholder src for the real R2 URL.
      const target = el.querySelector<HTMLImageElement>(`img[data-upload-id="${uploadId}"]`);
      if (target) {
        target.src = res.view_url;
        target.removeAttribute("data-upload-id");
        target.classList.remove("wt-uploading");
      }
    } catch {
      // Remove the placeholder; surface an inline error.
      const target = el.querySelector(`img[data-upload-id="${uploadId}"]`);
      target?.remove();
      showInlineError(`Failed to upload ${file.name}`);
    } finally {
      URL.revokeObjectURL(previewUrl);
      pendingUploadsRef.current = Math.max(0, pendingUploadsRef.current - 1);
      // If a save was queued during the upload window, fire it now.
      if (pendingUploadsRef.current === 0 && pendingEmitRef.current) {
        emitChange();
      } else if (pendingUploadsRef.current === 0) {
        // No queued emit, but the DOM changed (src swap) — emit so the
        // parent's value matches the editor.
        emitChange();
      }
    }
  }, [retroId, portfolio, showInlineError, emitChange]);

  // Toolbar command dispatcher. Re-focuses the editor before each
  // command so toolbar clicks don't blur away the selection.
  const exec = useCallback((command: string, arg?: string) => {
    const el = editorRef.current;
    if (!el) return;
    el.focus();
    document.execCommand(command, false, arg);
    // Indent/outdent on Chrome emits inline `margin-left` styles;
    // normalize them to .indent-N classes so DOMPurify doesn't strip
    // them on round-trip.
    if (command === "indent" || command === "outdent") {
      normalizeIndentation(el);
    }
    emitChange();
  }, [emitChange]);

  // ---------------------------------------------------------------------------
  // Link popover
  // ---------------------------------------------------------------------------

  const openLinkPopover = useCallback(() => {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) {
      // No selection yet (editor never focused) — open with an empty
      // text-input form. We synthesize a Range at the end of the editor
      // so subsequent insertHTML has somewhere to go.
      savedRangeRef.current = null;
      setLinkInitialUrl("");
      setLinkHasTextInput(true);
      setLinkPopoverOpen(true);
      return;
    }
    const range = sel.getRangeAt(0).cloneRange();
    savedRangeRef.current = range;

    // Prefill from an existing <a> if the cursor is inside one.
    const anchor = closestEl<HTMLAnchorElement>(range.startContainer, "a");
    if (anchor) {
      setLinkInitialUrl(anchor.getAttribute("href") || "");
      setLinkHasTextInput(false);  // edit-in-place uses createLink on text
    } else {
      setLinkInitialUrl("");
      setLinkHasTextInput(range.collapsed);
    }
    setLinkPopoverOpen(true);
  }, []);

  const closeLinkPopover = useCallback(() => {
    setLinkPopoverOpen(false);
    setLinkInitialUrl("");
  }, []);

  const restoreSavedRange = useCallback(() => {
    const range = savedRangeRef.current;
    const editor = editorRef.current;
    if (!editor) return;
    editor.focus();
    if (!range) return;
    const sel = window.getSelection();
    if (!sel) return;
    sel.removeAllRanges();
    sel.addRange(range);
  }, []);

  const confirmLink = useCallback((url: string, text: string): string | null => {
    if (!url) return "Please enter a URL";
    restoreSavedRange();
    const sel = window.getSelection();
    const collapsed = !sel || sel.rangeCount === 0 || sel.getRangeAt(0).collapsed;

    if (collapsed) {
      // No selected text — insert the link text (or fallback to the URL itself).
      const visible = text || url;
      const html = `<a href="${escapeHtml(url)}">${escapeHtml(visible)}</a>`;
      document.execCommand("insertHTML", false, html);
    } else {
      document.execCommand("createLink", false, url);
    }
    emitChange();
    closeLinkPopover();
    return null;
  }, [restoreSavedRange, emitChange, closeLinkPopover]);

  // ---------------------------------------------------------------------------
  // Video embed popover
  // ---------------------------------------------------------------------------

  const openVideoPopover = useCallback(() => {
    const sel = window.getSelection();
    savedRangeRef.current = sel && sel.rangeCount > 0
      ? sel.getRangeAt(0).cloneRange()
      : null;
    setVideoPopoverOpen(true);
  }, []);

  const closeVideoPopover = useCallback(() => {
    setVideoPopoverOpen(false);
  }, []);

  const confirmVideo = useCallback((url: string): string | null => {
    const parsed = parseVideoUrl(url);
    if (!parsed) return "Only YouTube and Vimeo URLs are supported";
    restoreSavedRange();
    // Build the iframe HTML. Defense in depth: DOMPurify pass before
    // injection — even if parseVideoUrl had a regex bug, the iframe-src
    // hook would catch a non-whitelisted src here.
    const dirty =
      `<div class="wt-video-embed"><iframe src="${escapeHtml(parsed.src)}" frameborder="0" ` +
      `allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" ` +
      `allowfullscreen></iframe></div>`;
    const clean = DOMPurify.sanitize(dirty, {
      ALLOWED_TAGS: PASTE_ALLOWED_TAGS,
      ALLOWED_ATTR: PASTE_ALLOWED_ATTR,
    });
    document.execCommand("insertHTML", false, clean);
    emitChange();
    closeVideoPopover();
    return null;
  }, [restoreSavedRange, emitChange, closeVideoPopover]);

  // ---------------------------------------------------------------------------
  // Tables
  // ---------------------------------------------------------------------------

  const insertTable = useCallback(() => {
    const html =
      `<table class="wt-table">` +
      `<thead><tr><th>&nbsp;</th><th>&nbsp;</th><th>&nbsp;</th></tr></thead>` +
      `<tbody>` +
        `<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>` +
        `<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>` +
      `</tbody>` +
      `</table><p>&nbsp;</p>`;
    const editor = editorRef.current;
    if (!editor) return;
    editor.focus();
    document.execCommand("insertHTML", false, html);
    emitChange();
  }, [emitChange]);

  const tableAddRow = useCallback((position: "above" | "below") => {
    const cell = focusedCell;
    const table = focusedTable;
    if (!cell || !table) return;
    const tr = cell.parentElement as HTMLTableRowElement | null;
    if (!tr) return;
    const cellCount = tr.cells.length;
    const newRow = document.createElement("tr");
    for (let i = 0; i < cellCount; i++) {
      const td = document.createElement("td");
      td.innerHTML = "&nbsp;";
      newRow.appendChild(td);
    }
    if (position === "above") tr.parentElement?.insertBefore(newRow, tr);
    else tr.parentElement?.insertBefore(newRow, tr.nextSibling);
    emitChange();
  }, [focusedCell, focusedTable, emitChange]);

  const tableAddCol = useCallback((position: "left" | "right") => {
    const cell = focusedCell;
    const table = focusedTable;
    if (!cell || !table) return;
    const tr = cell.parentElement as HTMLTableRowElement | null;
    if (!tr) return;
    const colIndex = Array.from(tr.cells).indexOf(cell);
    if (colIndex < 0) return;
    const allRows = table.querySelectorAll<HTMLTableRowElement>("tr");
    allRows.forEach(row => {
      const refCell = row.cells[colIndex];
      const newCell = document.createElement(refCell?.tagName === "TH" ? "th" : "td");
      newCell.innerHTML = "&nbsp;";
      if (refCell) {
        if (position === "left") row.insertBefore(newCell, refCell);
        else row.insertBefore(newCell, refCell.nextSibling);
      } else {
        row.appendChild(newCell);
      }
    });
    emitChange();
  }, [focusedCell, focusedTable, emitChange]);

  const tableDeleteRow = useCallback(() => {
    const cell = focusedCell;
    const table = focusedTable;
    if (!cell || !table) return;
    const tr = cell.parentElement as HTMLTableRowElement | null;
    if (!tr) return;
    // Prevent deleting the header row if it's the only header. Allow body deletion always.
    const parent = tr.parentElement;
    if (parent && parent.children.length === 1 && parent.tagName === "THEAD") {
      // Don't delete the sole header row.
      return;
    }
    tr.remove();
    setFocusedCell(null);
    emitChange();
  }, [focusedCell, focusedTable, emitChange]);

  const tableDeleteCol = useCallback(() => {
    const cell = focusedCell;
    const table = focusedTable;
    if (!cell || !table) return;
    const tr = cell.parentElement as HTMLTableRowElement | null;
    if (!tr) return;
    const colIndex = Array.from(tr.cells).indexOf(cell);
    if (colIndex < 0) return;
    // Don't allow deleting the last column — leaves a row-only fragment.
    if (tr.cells.length <= 1) return;
    const allRows = table.querySelectorAll<HTMLTableRowElement>("tr");
    allRows.forEach(row => {
      const c = row.cells[colIndex];
      if (c) c.remove();
    });
    setFocusedCell(null);
    emitChange();
  }, [focusedCell, focusedTable, emitChange]);

  const tableDelete = useCallback(() => {
    const table = focusedTable;
    if (!table) return;
    table.remove();
    setFocusedTable(null);
    setFocusedCell(null);
    setTableToolbarPos(null);
    emitChange();
  }, [focusedTable, emitChange]);

  // ---------------------------------------------------------------------------
  // Task lists
  // ---------------------------------------------------------------------------

  const insertTaskList = useCallback(() => {
    const html =
      `<ul class="contains-task-list">` +
      `<li class="task-list-item"><input type="checkbox" /> </li>` +
      `</ul>`;
    const editor = editorRef.current;
    if (!editor) return;
    editor.focus();
    document.execCommand("insertHTML", false, html);
    emitChange();
  }, [emitChange]);

  // Editor-level delegated click for checkboxes AND inline images.
  // contentEditable swallows the native click behavior of <input>, so
  // we manually toggle the attribute (not the property — innerHTML
  // serializes the attribute) and emit a change so the value
  // round-trips through save. Same delegation pattern picks up clicks
  // on inline images (Phase 4.2) and opens the lightbox.
  const handleEditorClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    if (!target) return;
    if (target.tagName === "INPUT") {
      const input = target as HTMLInputElement;
      if (input.type !== "checkbox") return;
      e.preventDefault();
      input.toggleAttribute("checked");
      emitChange();
      return;
    }
    if (target.tagName === "IMG") {
      const img = target as HTMLImageElement;
      // Skip the in-flight upload placeholder — clicking a blob URL
      // would just open a useless local preview.
      if (img.classList.contains("wt-uploading")) return;
      const src = img.getAttribute("src") || "";
      if (!src) return;
      e.preventDefault();
      setLightboxImage({ url: src, alt: img.getAttribute("alt") || "" });
    }
  }, [emitChange]);

  // ---------------------------------------------------------------------------
  // Selection-change: update Style dropdown label + table hover toolbar
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const handler = () => {
      const editor = editorRef.current;
      if (!editor) return;
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return;
      const range = sel.getRangeAt(0);
      // Only respond to selections inside our editor.
      if (!editor.contains(range.startContainer)) return;

      // Style dropdown label.
      try {
        const block = (document.queryCommandValue("formatBlock") || "").toLowerCase();
        setCurrentBlockLabel(BLOCK_LABELS[block] || "Normal");
      } catch { /* jsdom or older browser — leave as is */ }

      // Table hover toolbar — find the nearest <table> ancestor.
      const cell = closestEl<HTMLTableCellElement>(range.startContainer, "td, th");
      const table = closestEl<HTMLTableElement>(range.startContainer, "table");
      if (table && cell && editor.contains(table)) {
        setFocusedTable(table);
        setFocusedCell(cell);
        // Position toolbar above the table, relative to the editor wrapper.
        const tableRect = table.getBoundingClientRect();
        const editorRect = editor.getBoundingClientRect();
        setTableToolbarPos({
          top: tableRect.top - editorRect.top - 38,
          left: tableRect.left - editorRect.left,
        });
      } else {
        setFocusedTable(null);
        setFocusedCell(null);
        setTableToolbarPos(null);
      }
    };
    document.addEventListener("selectionchange", handler);
    return () => document.removeEventListener("selectionchange", handler);
  }, []);

  // ---------------------------------------------------------------------------
  // Paste handler: sanitize via DOMPurify, then insert. Indent normalization
  // catches any inline margin-left coming from external sources.
  // ---------------------------------------------------------------------------

  const handlePaste = useCallback((e: React.ClipboardEvent<HTMLDivElement>) => {
    // Phase 4.1: detect image clipboard items FIRST. Items survive image
    // pastes from screenshots / file drags; `files` doesn't always
    // populate for clipboard sources. If any image item is present we
    // take the upload path and prevent the default HTML/text paste.
    const items = e.clipboardData?.items;
    if (items) {
      const imageFiles: File[] = [];
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        if (it.kind === "file" && it.type.startsWith("image/")) {
          const f = it.getAsFile();
          if (f) imageFiles.push(f);
        }
      }
      if (imageFiles.length > 0) {
        e.preventDefault();
        if (retroId == null) {
          showInlineError("Save the retro first to embed images.");
          return;
        }
        // Client-side size + MIME guard (mirrors the server's 5MB +
        // PNG/JPEG/GIF/WEBP allow-list). Fail-fast before paying the
        // upload latency.
        const allowedMimes = new Set(["image/png", "image/jpeg", "image/gif", "image/webp"]);
        const MAX_BYTES = 5 * 1024 * 1024;
        for (const file of imageFiles) {
          if (!allowedMimes.has(file.type)) {
            showInlineError(`${file.name}: only PNG / JPEG / GIF / WEBP allowed`);
            continue;
          }
          if (file.size > MAX_BYTES) {
            showInlineError(`${file.name}: exceeds 5MB limit`);
            continue;
          }
          void uploadAndInsertImage(file);
        }
        return;
      }
    }

    // Existing text/HTML paste path (unchanged from Phase 3.5).
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
      const esc = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\n/g, "<br>");
      clean = esc;
    }
    document.execCommand("insertHTML", false, clean);
    // Catch any inline margin-left that snuck in via pasted content.
    if (editorRef.current) normalizeIndentation(editorRef.current);
    emitChange();
  }, [emitChange, retroId, showInlineError, uploadAndInsertImage]);

  const handleInput = useCallback(() => {
    emitChange();
  }, [emitChange]);

  const placeholderVisible = !value || value === "" || value === "<br>";

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <SectionExpander
      title="Weekly Thoughts"
      showDot
      defaultExpanded={true}
      localStorageKey={EXPANDED_KEY}
      bodyId="weekly-thoughts-body"
      headerCaption={(open) =>
        open
          ? "Synthesize what you learned this week"
          : wordCount > 0
            ? `${wordCount} words`
            : ""
      }
    >
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
            {/* Phase 4.1: Undo / Redo. Conventionally at the leading
                edge of the toolbar. Keyboard shortcuts (Cmd+Z / Cmd+
                Shift+Z) already work natively via contentEditable;
                these buttons just expose the affordance visually. */}
            <ToolbarBtn label="Undo" onClick={() => exec("undo")}><Icons.undo /></ToolbarBtn>
            <ToolbarBtn label="Redo" onClick={() => exec("redo")}><Icons.redo /></ToolbarBtn>

            <Divider />

            {/* Block style (Phase 3.5) */}
            <ToolbarDropdown
              ariaLabel="Block style"
              triggerLabel={currentBlockLabel}
              options={HEADING_OPTIONS}
              onSelect={(v) => exec("formatBlock", v)}
              editorRef={editorRef}
              width={130}
            />
            {/* Functional font family (Phase 3.5) */}
            <ToolbarDropdown
              ariaLabel="Font family"
              triggerLabel="Font"
              options={FONT_OPTIONS}
              onSelect={(v) => {
                if (v === "") exec("removeFormat");
                else exec("fontName", v);
              }}
              editorRef={editorRef}
              width={140}
            />
            {/* Functional font size (Phase 3.5) */}
            <ToolbarDropdown
              ariaLabel="Font size"
              triggerLabel="Size"
              options={SIZE_OPTIONS}
              onSelect={(v) => exec("fontSize", v)}
              editorRef={editorRef}
              width={80}
            />

            <Divider />

            <ToolbarBtn label="Bold" onClick={() => exec("bold")}><Icons.bold /></ToolbarBtn>
            <ToolbarBtn label="Italic" onClick={() => exec("italic")}><Icons.italic /></ToolbarBtn>
            <ToolbarBtn label="Underline" onClick={() => exec("underline")}><Icons.underline /></ToolbarBtn>
            <ToolbarBtn label="Strikethrough" onClick={() => exec("strikeThrough")}><Icons.strike /></ToolbarBtn>

            <Divider />

            {/* Insert link — inline popover (Phase 3.5, replaces window.prompt) */}
            <div style={{ position: "relative", display: "inline-flex" }}>
              <ToolbarBtn label="Insert link" onClick={openLinkPopover}><Icons.link /></ToolbarBtn>
              {linkPopoverOpen && (
                <UrlPopover
                  title="Insert link"
                  urlPlaceholder="https://..."
                  initialUrl={linkInitialUrl}
                  showLinkTextInput={linkHasTextInput}
                  onConfirm={confirmLink}
                  onClose={closeLinkPopover}
                />
              )}
            </div>
            <ToolbarBtn label="Clear formatting" onClick={() => exec("removeFormat")}><Icons.eraser /></ToolbarBtn>

            <Divider />

            {/* Phase 4.2: Highlight + Text color now open popover
                pickers instead of applying a single fixed color. */}
            <ColorPicker
              ariaLabel="Highlight color"
              triggerGlyph={<Icons.highlight />}
              triggerSwatch={HIGHLIGHT_COLOR}
              palette={HIGHLIGHT_PALETTE}
              resetLabel="None"
              onPick={(color) => exec("hiliteColor", color)}
              editorRef={editorRef}
            />
            <ColorPicker
              ariaLabel="Text color"
              triggerGlyph={<Icons.textColor />}
              triggerSwatch={TEXT_COLOR}
              palette={TEXT_COLOR_PALETTE}
              resetLabel="Default"
              onPick={(color) => exec("foreColor", color)}
              editorRef={editorRef}
            />

            <Divider />

            <ToolbarBtn label="Align left" onClick={() => exec("justifyLeft")}><Icons.alignLeft /></ToolbarBtn>
            <ToolbarBtn label="Align center" onClick={() => exec("justifyCenter")}><Icons.alignCenter /></ToolbarBtn>
            <ToolbarBtn label="Align right" onClick={() => exec("justifyRight")}><Icons.alignRight /></ToolbarBtn>

            <Divider />

            {/* Lists + indent (Phase 3.5) */}
            <ToolbarBtn label="Bulleted list" onClick={() => exec("insertUnorderedList")}><Icons.listBullet /></ToolbarBtn>
            <ToolbarBtn label="Numbered list" onClick={() => exec("insertOrderedList")}><Icons.listOrdered /></ToolbarBtn>
            <ToolbarBtn label="Outdent" onClick={() => exec("outdent")}><Icons.indentLeft /></ToolbarBtn>
            <ToolbarBtn label="Indent" onClick={() => exec("indent")}><Icons.indentRight /></ToolbarBtn>

            <Divider />

            {/* Block content (Phase 3.5) */}
            <ToolbarBtn label="Quote" onClick={() => exec("formatBlock", "<blockquote>")}><Icons.quote /></ToolbarBtn>
            <ToolbarBtn label="Code block" onClick={() => exec("formatBlock", "<pre>")}><Icons.code /></ToolbarBtn>
            <ToolbarBtn label="Horizontal rule" onClick={() => exec("insertHorizontalRule")}><Icons.horizontalRule /></ToolbarBtn>

            <Divider />

            {/* Tables, task lists, video (Phase 3.5) */}
            <ToolbarBtn label="Insert table" onClick={insertTable}><Icons.table /></ToolbarBtn>
            <ToolbarBtn label="Task list" onClick={insertTaskList}><Icons.checkSquare /></ToolbarBtn>
            <div style={{ position: "relative", display: "inline-flex" }}>
              <ToolbarBtn label="Embed video" onClick={openVideoPopover}><Icons.video /></ToolbarBtn>
              {videoPopoverOpen && (
                <UrlPopover
                  title="Embed video"
                  urlPlaceholder="YouTube or Vimeo URL"
                  showLinkTextInput={false}
                  onConfirm={confirmVideo}
                  onClose={closeVideoPopover}
                  okLabel="Embed"
                />
              )}
            </div>

            <span style={{ flex: 1 }} />

            {/* Inert mic (v1 — Web Speech API integration deferred) */}
            <ToolbarBtn label="Voice dictation (coming soon)" inert><Icons.mic /></ToolbarBtn>
          </div>

          {/* Editor + placeholder + table hover toolbar. The wrapper is
              position: relative so absolutely-positioned children
              (placeholder, table hover toolbar) anchor to it. */}
          <div style={{ padding: 16, position: "relative" }}>
            {/* Phase 4.1: paste rejection inline toast (oversize / bad
                MIME / retroId missing). Auto-dismisses after 3s. */}
            {inlineError && (
              <div
                role="alert"
                style={{
                  marginBottom: 8,
                  padding: "6px 10px",
                  fontSize: 12,
                  borderRadius: 6,
                  background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
                  color: "#dc2626",
                  border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))",
                }}
              >
                {inlineError}
              </div>
            )}
            {placeholderVisible && (
              <div
                aria-hidden
                style={{
                  position: "absolute",
                  top: 28,
                  left: 30,
                  color: "var(--ink-4)",
                  fontStyle: "italic",
                  fontSize: 13,
                  fontFamily: "var(--font-inter), system-ui, sans-serif",
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
              onClick={handleEditorClick}
              className="weekly-thoughts-editor"
              style={{
                width: "100%",
                minHeight: 240,
                padding: "12px 14px",
                borderRadius: 10,
                background: "var(--bg)",
                border: "1px solid var(--border)",
                color: "var(--ink)",
                // Phase 3.5: switched from monospace (--font-num) to
                // sans-serif. Monospace looked bad for headings/prose;
                // monospace is now scoped to <pre>/<code> via the
                // .weekly-thoughts-editor CSS block in globals.css.
                fontFamily: "var(--font-inter), system-ui, sans-serif",
                fontSize: 13,
                lineHeight: 1.6,
                outline: "none",
              }}
            />
            {focusedTable && tableToolbarPos && (
              <TableHoverToolbar
                pos={tableToolbarPos}
                onAddRowAbove={() => tableAddRow("above")}
                onAddRowBelow={() => tableAddRow("below")}
                onAddColLeft={() => tableAddCol("left")}
                onAddColRight={() => tableAddCol("right")}
                onDeleteRow={tableDeleteRow}
                onDeleteCol={tableDeleteCol}
                onDeleteTable={tableDelete}
              />
            )}
          </div>
          {/* Phase 4.2 — inline image lightbox. Single-image mode
              (no prev/next); each click on an inline <img> opens
              its own preview. */}
          <ImageLightbox
            images={lightboxImage ? [lightboxImage] : []}
            activeIndex={lightboxImage ? 0 : null}
            onClose={() => setLightboxImage(null)}
          />
    </SectionExpander>
  );
}

// ─── Internal toolbar primitives ─────────────────────────────────────────────

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

// ─── Table hover toolbar ────────────────────────────────────────────────────

interface TableHoverToolbarProps {
  pos: { top: number; left: number };
  onAddRowAbove: () => void;
  onAddRowBelow: () => void;
  onAddColLeft: () => void;
  onAddColRight: () => void;
  onDeleteRow: () => void;
  onDeleteCol: () => void;
  onDeleteTable: () => void;
}

function TableHoverToolbar({
  pos,
  onAddRowAbove,
  onAddRowBelow,
  onAddColLeft,
  onAddColRight,
  onDeleteRow,
  onDeleteCol,
  onDeleteTable,
}: TableHoverToolbarProps) {
  const btn = (label: string, onClick: () => void, danger?: boolean) => (
    <button
      type="button"
      onMouseDown={e => e.preventDefault()}
      onClick={onClick}
      aria-label={label}
      title={label}
      style={{
        height: 24,
        padding: "0 8px",
        fontSize: 11,
        fontWeight: 500,
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 6,
        color: danger ? "#dc2626" : "var(--ink-3)",
        cursor: "pointer",
      }}
    >
      {label}
    </button>
  );
  return (
    <div
      role="toolbar"
      aria-label="Table actions"
      style={{
        position: "absolute",
        top: pos.top,
        left: pos.left,
        display: "flex",
        gap: 4,
        padding: 4,
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        boxShadow: "var(--card-shadow)",
        zIndex: 15,
        flexWrap: "wrap",
      }}
    >
      {btn("+ Row above", onAddRowAbove)}
      {btn("+ Row below", onAddRowBelow)}
      {btn("+ Col left", onAddColLeft)}
      {btn("+ Col right", onAddColRight)}
      {btn("− Row", onDeleteRow)}
      {btn("− Col", onDeleteCol)}
      {btn("× Table", onDeleteTable, true)}
    </div>
  );
}
