"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

/**
 * Mobile textarea editor — Phase 2 T2-4b architectural replacement.
 *
 * After 5 iterations (3 contentEditable / execCommand + 2 Lexical)
 * failed on iOS Safari, pivoted to native textarea + live preview.
 * The actual user workflow is "paste AI-generated content,
 * occasionally edit" — not "compose with a toolbar." A textarea
 * is bulletproof on every device; live preview uses the same
 * ReactMarkdown + remarkGfm + rehypeRaw chain the Daily Report
 * itself uses, so what the user sees in preview matches what
 * renders after save.
 *
 * External API is identical to the deleted MobileRichTextEditor.
 * All three consumers (Recap, Thoughts, Market Notes via the
 * useFieldEditor hook in mobile-daily-report.tsx) keep working
 * unchanged.
 */

// DOMPurify allow-list — mirrors thoughts-editor.tsx's PASTE_ALLOWED_*
// values for tags this editor's preview is expected to render. Iframe /
// input / img omitted: the textarea editor isn't a paste-from-anywhere
// surface, and inline images go through the Captures section, not the
// content body. (Pasted images in the textarea would be raw URLs / HTML
// from AI tooling, not rich-paste embeds.)
const ALLOWED_TAGS = [
  "b", "i", "u", "s", "strike", "em", "strong", "a", "br", "p", "div", "span",
  "ul", "ol", "li",
  "h1", "h2", "h3",
  "blockquote", "pre", "code", "hr",
  "table", "thead", "tbody", "tr", "td", "th", "colgroup", "col",
];

const ALLOWED_ATTR = [
  "href",
  "class",
  "rowspan", "colspan",
];

function sanitize(html: string): string {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS,
    ALLOWED_ATTR,
  });
}

export type MobileTextareaEditorProps = {
  /** Initial HTML content. Hydrates the textarea on mount. */
  initialValue: string;
  /** Fires with the current textarea value on every change. */
  onChange: (html: string) => void;
  /** Fires when dirty state toggles (current value !== initialValue).
   *  Only fires on transitions, not every keystroke. */
  onDirtyChange?: (dirty: boolean) => void;
  /** Textarea placeholder when empty. */
  placeholder?: string;
};

export function MobileTextareaEditor({
  initialValue,
  onChange,
  onDirtyChange,
  placeholder = "Type or paste content here…",
}: MobileTextareaEditorProps) {
  const [value, setValue] = useState(initialValue);
  const lastDirtyRef = useRef<boolean>(false);
  const initialRef = useRef(initialValue);

  // Sync internal state if the consumer remounts the editor with a
  // different initialValue (e.g., sheet reopens after server save).
  useEffect(() => {
    initialRef.current = initialValue;
    setValue(initialValue);
    lastDirtyRef.current = false;
  }, [initialValue]);

  // Dirty-change emission. Only fires when the boolean transitions.
  useEffect(() => {
    if (!onDirtyChange) return;
    const dirty = value !== initialRef.current;
    if (dirty !== lastDirtyRef.current) {
      lastDirtyRef.current = dirty;
      onDirtyChange(dirty);
    }
  }, [value, onDirtyChange]);

  const cleanHtml = useMemo(() => sanitize(value), [value]);
  const hasContent = value.trim().length > 0;

  return (
    <div className="flex h-full flex-col">
      <textarea
        data-testid="mobile-textarea-input"
        className="mobile-textarea-input min-h-[200px] flex-1 resize-none border-0 bg-transparent p-3 text-[14px] leading-[1.5] text-m-text outline-none placeholder:text-m-text-faint"
        style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}
        value={value}
        placeholder={placeholder}
        onChange={(e) => {
          const next = e.target.value;
          setValue(next);
          onChange(next);
        }}
      />
      <div className="mt-3 border-t border-m-border px-3 pt-2">
        <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-m-text-faint">
          Preview
        </span>
      </div>
      {hasContent ? (
        <div
          data-testid="mobile-textarea-preview"
          className="mobile-textarea-preview flex-1 overflow-y-auto p-3 text-[13px] leading-snug text-m-text"
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
            {cleanHtml}
          </ReactMarkdown>
        </div>
      ) : (
        <div
          data-testid="mobile-textarea-preview-empty"
          className="flex-1 p-3 text-[12px] italic text-m-text-faint"
        >
          Preview appears here
        </div>
      )}
    </div>
  );
}

export default MobileTextareaEditor;
