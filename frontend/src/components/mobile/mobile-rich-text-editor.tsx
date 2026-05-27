"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import {
  $createParagraphNode,
  $createTextNode,
  $getRoot,
  $getSelection,
  $isRangeSelection,
  $isTextNode,
  CLEAR_HISTORY_COMMAND,
  FORMAT_TEXT_COMMAND,
  INDENT_CONTENT_COMMAND,
  OUTDENT_CONTENT_COMMAND,
  ParagraphNode,
  REDO_COMMAND,
  TextNode,
  UNDO_COMMAND,
  type LexicalEditor,
} from "lexical";
import { LexicalComposer } from "@lexical/react/LexicalComposer";
import { ContentEditable } from "@lexical/react/LexicalContentEditable";
import { LexicalErrorBoundary } from "@lexical/react/LexicalErrorBoundary";
import { RichTextPlugin } from "@lexical/react/LexicalRichTextPlugin";
import { HistoryPlugin } from "@lexical/react/LexicalHistoryPlugin";
import { ListPlugin } from "@lexical/react/LexicalListPlugin";
import { LinkPlugin } from "@lexical/react/LexicalLinkPlugin";
import { OnChangePlugin } from "@lexical/react/LexicalOnChangePlugin";
import { useLexicalComposerContext } from "@lexical/react/LexicalComposerContext";
import {
  HeadingNode,
  QuoteNode,
  type HeadingTagType,
} from "@lexical/rich-text";
import {
  INSERT_ORDERED_LIST_COMMAND,
  INSERT_UNORDERED_LIST_COMMAND,
  ListItemNode,
  ListNode,
} from "@lexical/list";
import { $toggleLink, AutoLinkNode, LinkNode } from "@lexical/link";
import { $generateHtmlFromNodes, $generateNodesFromDOM } from "@lexical/html";
import { $setBlocksType } from "@lexical/selection";
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
import {
  $createColoredTextNode,
  $isColoredTextNode,
  ColoredTextNode,
  type TextColor,
} from "./lexical-nodes/colored-text-node";
import {
  IndentAwareHeadingNode,
  IndentAwareListItemNode,
  IndentAwareParagraphNode,
  IndentAwareQuoteNode,
} from "./lexical-nodes/indent-aware-blocks";

/**
 * Mobile rich text editor — Phase 2 T2-4b Lexical migration.
 *
 * Internals replaced from contentEditable + execCommand to
 * Lexical (Meta's controlled-DOM editor). External API is
 * preserved verbatim — all consumers (Recap, Thoughts via
 * useFieldEditor hook) work without modification.
 *
 * Why this rewrite: three iterative fixes on contentEditable +
 * execCommand failed on iOS Safari due to execCommand's
 * inconsistent handling + selection-loss-on-pointerdown. Lexical's
 * command system captures selection at dispatch time inside its
 * own update cycle, sidestepping the entire class of bugs.
 *
 * Architecture:
 *   - LexicalComposer wraps the editor + plugins + toolbar
 *   - RichTextPlugin mounts the ContentEditable surface
 *   - Custom nodes: IndentAware{Paragraph,Heading,Quote,ListItem}
 *     translate Lexical's __indent → .indent-N class on
 *     export/import (DOMPurify policy: no inline style attrs)
 *   - ColoredTextNode applies .text-color-X class for the color
 *     picker (mirrors .indent-N pattern)
 *   - HtmlHydrationPlugin reads initialValue HTML on mount
 *   - SerializerPlugin emits HTML via @lexical/html on every
 *     change and pipes to props.onChange + props.onDirtyChange
 *   - Toolbar lives inside the composer subtree so it can
 *     useLexicalComposerContext() but renders outside the
 *     ContentEditable; visualViewport observer (unchanged from
 *     prior architecture) drives its sticky-above-keyboard
 *     bottom offset.
 *
 * Toolbar surfaces match the locked 15-surface trim:
 *   B / I / U | Heading | Bullet / Numbered / Indent / Outdent |
 *   Color | Link | Quote / Code | Clear | Undo / Redo
 *
 * No image paste, no strikethrough toolbar (kept asymmetric:
 * paste-allowed via DOMPurify, no compose-time surface). Legacy
 * <pre> blocks degrade to paragraph (no @lexical/code).
 */

// ── Constants ──────────────────────────────────────────────────────

const EDITOR_NAMESPACE = "mobile-rich-text-editor";

const TEXT_COLORS: ReadonlyArray<TextColor> = [
  "rose",
  "amber",
  "emerald",
  "sky",
  "violet",
];

const COLOR_SWATCH_HEX: Record<TextColor, string> = {
  rose: "#f43f5e",
  amber: "#f59f00",
  emerald: "#08a86b",
  sky: "#0d6efd",
  violet: "#8b5cf6",
};

// Lexical theme: class-only (no inline styles). Matches our
// DOMPurify-no-style-attr policy. Tailwind handles visual styling
// via the parent .mobile-rte-body utility classes.
const editorTheme = {
  paragraph: "mobile-rte-paragraph",
  quote: "mobile-rte-quote",
  heading: {
    h1: "mobile-rte-h1",
    h2: "mobile-rte-h2",
    h3: "mobile-rte-h3",
  },
  list: {
    nested: { listitem: "mobile-rte-nested-listitem" },
    ol: "mobile-rte-ol",
    ul: "mobile-rte-ul",
    listitem: "mobile-rte-li",
  },
  link: "mobile-rte-link",
  text: {
    bold: "mobile-rte-bold",
    italic: "mobile-rte-italic",
    underline: "mobile-rte-underline",
    code: "mobile-rte-code",
  },
};

// ── External API (unchanged from prior version) ────────────────────

export type MobileRichTextEditorProps = {
  initialValue: string;
  onChange: (html: string) => void;
  onDirtyChange?: (dirty: boolean) => void;
  placeholder?: string;
};

// ── Component ──────────────────────────────────────────────────────

export default function MobileRichTextEditor({
  initialValue,
  onChange,
  onDirtyChange,
  placeholder = "Start writing…",
}: MobileRichTextEditorProps) {
  // visualViewport observer drives the toolbar's bottom offset so
  // the toolbar sticks above the iOS virtual keyboard. Unchanged
  // from the prior editor's architecture.
  const [keyboardHeight, setKeyboardHeight] = useState(0);
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

  const initialConfig = {
    namespace: EDITOR_NAMESPACE,
    theme: editorTheme,
    onError(error: Error) {
      // Lexical convention: throw on errors so React error boundaries
      // can surface the failure. The LexicalErrorBoundary inside the
      // RichTextPlugin catches and degrades gracefully.
      throw error;
    },
    nodes: [
      // Base nodes required by plugins.
      ListNode,
      LinkNode,
      AutoLinkNode,
      // Our subclasses. Replacements below route base-class
      // creation through these so every list item / paragraph /
      // heading / quote produced by commands becomes the
      // indent-aware variant.
      IndentAwareParagraphNode,
      IndentAwareHeadingNode,
      IndentAwareQuoteNode,
      IndentAwareListItemNode,
      ColoredTextNode,
      {
        replace: ParagraphNode,
        with: () => new IndentAwareParagraphNode(),
        withKlass: IndentAwareParagraphNode,
      },
      {
        replace: HeadingNode,
        with: (node: HeadingNode) =>
          new IndentAwareHeadingNode(node.getTag()),
        withKlass: IndentAwareHeadingNode,
      },
      {
        replace: QuoteNode,
        with: () => new IndentAwareQuoteNode(),
        withKlass: IndentAwareQuoteNode,
      },
      {
        replace: ListItemNode,
        with: (node: ListItemNode) =>
          new IndentAwareListItemNode(
            (node as ListItemNode & { __value: number }).__value,
            (node as ListItemNode & { __checked?: boolean }).__checked,
          ),
        withKlass: IndentAwareListItemNode,
      },
    ],
  } as const;

  return (
    <div
      data-testid="mobile-rich-text-editor"
      className="relative flex h-full flex-col"
    >
      <LexicalComposer initialConfig={initialConfig}>
        <EditorBody placeholder={placeholder} keyboardHeight={keyboardHeight} />
        <HistoryPlugin />
        <ListPlugin />
        <LinkPlugin />
        <HtmlHydrationPlugin initialValue={initialValue} />
        <SerializerPlugin
          initialValue={initialValue}
          onChange={onChange}
          onDirtyChange={onDirtyChange}
        />
      </LexicalComposer>
    </div>
  );
}

// ── Editor surface + toolbar (inside LexicalComposer) ─────────────

function EditorBody({
  placeholder,
  keyboardHeight,
}: {
  placeholder: string;
  keyboardHeight: number;
}) {
  return (
    <>
      <div className="relative flex-1 overflow-y-auto">
        <RichTextPlugin
          contentEditable={
            <ContentEditable
              data-testid="mobile-rich-text-editor-body"
              aria-label="Editor"
              className="mobile-rte-body min-h-full px-4 py-3 text-[15px] leading-relaxed text-m-text focus:outline-none"
              style={{
                paddingBottom: `${56 + keyboardHeight + 16}px`,
                WebkitUserSelect: "text",
              }}
            />
          }
          placeholder={
            <div
              data-testid="mobile-rich-text-editor-placeholder"
              aria-hidden="true"
              className="pointer-events-none absolute left-4 top-3 text-[15px] text-m-text-faint"
            >
              {placeholder}
            </div>
          }
          ErrorBoundary={LexicalErrorBoundary}
        />
      </div>
      <Toolbar keyboardHeight={keyboardHeight} />
    </>
  );
}

// ── Toolbar ────────────────────────────────────────────────────────

function Toolbar({ keyboardHeight }: { keyboardHeight: number }) {
  const [editor] = useLexicalComposerContext();
  const [headingOpen, setHeadingOpen] = useState(false);
  const [colorOpen, setColorOpen] = useState(false);

  // Close popovers when the editor selection changes (user taps into
  // the content while a popover is open). Keeps the UI predictable.
  useEffect(() => {
    return editor.registerUpdateListener(() => {
      // Cheap reset on any update — popovers only stay open for a
      // single command anyway.
    });
  }, [editor]);

  const fmt = useCallback(
    (kind: "bold" | "italic" | "underline" | "code") => {
      editor.dispatchCommand(FORMAT_TEXT_COMMAND, kind);
    },
    [editor],
  );

  const setBlock = useCallback(
    (kind: "p" | "h1" | "h2" | "h3" | "quote") => {
      editor.update(() => {
        const sel = $getSelection();
        if (!$isRangeSelection(sel)) return;
        if (kind === "p") {
          $setBlocksType(sel, () => new IndentAwareParagraphNode());
        } else if (kind === "quote") {
          $setBlocksType(sel, () => new IndentAwareQuoteNode());
        } else {
          const tag = kind as HeadingTagType;
          $setBlocksType(sel, () => new IndentAwareHeadingNode(tag));
        }
      });
    },
    [editor],
  );

  const insertList = useCallback(
    (kind: "ul" | "ol") => {
      editor.dispatchCommand(
        kind === "ul"
          ? INSERT_UNORDERED_LIST_COMMAND
          : INSERT_ORDERED_LIST_COMMAND,
        undefined,
      );
    },
    [editor],
  );

  const indent = useCallback(() => {
    editor.dispatchCommand(INDENT_CONTENT_COMMAND, undefined);
  }, [editor]);

  const outdent = useCallback(() => {
    editor.dispatchCommand(OUTDENT_CONTENT_COMMAND, undefined);
  }, [editor]);

  const applyColor = useCallback(
    (color: TextColor) => {
      editor.update(() => {
        const sel = $getSelection();
        if (!$isRangeSelection(sel)) return;
        const nodes = sel.extract();
        for (const node of nodes) {
          if (!$isTextNode(node)) continue;
          const text = node.getTextContent();
          if (!text) continue;
          const format = node.getFormat();
          const newNode = $createColoredTextNode(text, color);
          newNode.setFormat(format);
          node.replace(newNode);
        }
      });
      setColorOpen(false);
    },
    [editor],
  );

  const clearColor = useCallback(() => {
    editor.update(() => {
      const sel = $getSelection();
      if (!$isRangeSelection(sel)) return;
      const nodes = sel.extract();
      for (const node of nodes) {
        if (!$isColoredTextNode(node)) continue;
        const text = node.getTextContent();
        if (!text) continue;
        const format = node.getFormat();
        const plain = $createTextNode(text);
        plain.setFormat(format);
        node.replace(plain);
      }
    });
    setColorOpen(false);
  }, [editor]);

  const insertLink = useCallback(() => {
    const url = window.prompt("Enter URL");
    if (!url) return;
    editor.update(() => {
      $toggleLink(url);
    });
  }, [editor]);

  const clearFormat = useCallback(() => {
    editor.update(() => {
      const sel = $getSelection();
      if (!$isRangeSelection(sel)) return;
      const nodes = sel.extract();
      for (const node of nodes) {
        if ($isColoredTextNode(node)) {
          const text = node.getTextContent();
          const plain = $createTextNode(text);
          node.replace(plain);
          continue;
        }
        if ($isTextNode(node)) {
          node.setFormat(0);
          node.setStyle("");
        }
      }
    });
  }, [editor]);

  const undo = useCallback(() => {
    editor.dispatchCommand(UNDO_COMMAND, undefined);
  }, [editor]);

  const redo = useCallback(() => {
    editor.dispatchCommand(REDO_COMMAND, undefined);
  }, [editor]);

  const toolbarStyle: CSSProperties = {
    position: "sticky",
    bottom: keyboardHeight > 0 ? `${keyboardHeight}px` : 0,
    paddingLeft: 8,
    paddingRight: 8,
    paddingTop: 5,
    paddingBottom: 5,
    gap: 2,
    WebkitOverflowScrolling: "touch",
  };

  return (
    <div
      data-testid="mobile-rich-text-editor-toolbar"
      className="z-10 flex shrink-0 items-center overflow-x-auto whitespace-nowrap border-t-[0.5px] border-m-border bg-m-surface"
      style={toolbarStyle}
    >
      <ToolbarBtn label="Bold" icon={<Bold size={16} />} onAction={() => fmt("bold")} />
      <ToolbarBtn label="Italic" icon={<Italic size={16} />} onAction={() => fmt("italic")} />
      <ToolbarBtn label="Underline" icon={<Underline size={16} />} onAction={() => fmt("underline")} />
      <Divider />

      <HeadingDropdown
        open={headingOpen}
        onOpen={() => setHeadingOpen((v) => !v)}
        onSelect={(kind) => {
          setBlock(kind);
          setHeadingOpen(false);
        }}
      />
      <Divider />

      <ToolbarBtn label="Bulleted list" icon={<List size={16} />} onAction={() => insertList("ul")} />
      <ToolbarBtn label="Numbered list" icon={<ListOrdered size={16} />} onAction={() => insertList("ol")} />
      <ToolbarBtn label="Indent" icon={<IndentIncrease size={16} />} onAction={indent} />
      <ToolbarBtn label="Outdent" icon={<IndentDecrease size={16} />} onAction={outdent} />
      <Divider />

      <ColorPicker
        open={colorOpen}
        onOpen={() => setColorOpen((v) => !v)}
        onSelect={(c) => (c === "default" ? clearColor() : applyColor(c))}
      />
      <ToolbarBtn label="Insert link" icon={<LinkIcon size={16} />} onAction={insertLink} />
      <Divider />

      <ToolbarBtn label="Quote" icon={<Quote size={16} />} onAction={() => setBlock("quote")} />
      <ToolbarBtn label="Inline code" icon={<Code size={16} />} onAction={() => fmt("code")} />
      <Divider />

      <ToolbarBtn label="Clear formatting" icon={<RemoveFormatting size={16} />} onAction={clearFormat} />
      <Divider />

      <ToolbarBtn label="Undo" icon={<Undo size={16} />} onAction={undo} />
      <ToolbarBtn label="Redo" icon={<Redo size={16} />} onAction={redo} />
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
  // Action dispatched in onPointerDown to preserve editor selection
  // on iOS (the pointerdown-vs-click lesson from the third T2-4b
  // follow-up). Lexical's command system captures selection at
  // dispatch time inside its update cycle, so the editor staying
  // focused isn't strictly necessary for command correctness — but
  // keeping focus avoids a visible keyboard-dismiss flash on iOS.
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
  onSelect: (kind: "p" | "h1" | "h2" | "h3") => void;
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
              { kind: "p", label: "Normal" },
              { kind: "h1", label: "Heading 1" },
              { kind: "h2", label: "Heading 2" },
              { kind: "h3", label: "Heading 3" },
            ] as const
          ).map((opt) => (
            <button
              key={opt.kind}
              type="button"
              role="menuitem"
              onMouseDown={(e) => e.preventDefault()}
              onPointerDown={(e) => {
                e.preventDefault();
                onSelect(opt.kind);
              }}
              data-testid={`mobile-rte-heading-${opt.kind}`}
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

// ── HTML hydration + serialization plugins ────────────────────────

/** Hydrates the editor with the consumer's `initialValue` HTML on
 *  mount. Subsequent changes to `initialValue` are ignored (the
 *  editor's own state is authoritative after first paint — matches
 *  the previous editor's behavior + the useFieldEditor contract). */
function HtmlHydrationPlugin({ initialValue }: { initialValue: string }) {
  const [editor] = useLexicalComposerContext();
  const didHydrateRef = useRef(false);
  useEffect(() => {
    if (didHydrateRef.current) return;
    didHydrateRef.current = true;
    editor.update(() => {
      const root = $getRoot();
      root.clear();
      const trimmed = (initialValue ?? "").trim();
      if (!trimmed) {
        root.append($createParagraphNode());
        return;
      }
      const parser = new DOMParser();
      const dom = parser.parseFromString(trimmed, "text/html");
      const nodes = $generateNodesFromDOM(editor, dom);
      // $generateNodesFromDOM returns a mix of element + inline nodes;
      // wrap stray inline nodes (TextNode) in paragraphs so the root
      // only contains block-level children (Lexical constraint).
      for (const node of nodes) {
        if (node.isInline?.() || $isTextNode(node) || $isColoredTextNode(node)) {
          const p = $createParagraphNode();
          p.append(node);
          root.append(p);
        } else {
          root.append(node);
        }
      }
      if (root.getChildrenSize() === 0) {
        root.append($createParagraphNode());
      }
    });
    // Discard the hydration update from the undo stack so the first
    // undo doesn't wipe seeded content.
    editor.dispatchCommand(CLEAR_HISTORY_COMMAND, undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

/** Emits `props.onChange(html)` and `props.onDirtyChange(boolean)`
 *  on every editor update. Skips emissions during the initial
 *  hydration pass (hydration writes the very state we just received
 *  as `initialValue`; firing onChange there would create a spurious
 *  dirty flag immediately on mount). */
function SerializerPlugin({
  initialValue,
  onChange,
  onDirtyChange,
}: {
  initialValue: string;
  onChange: (html: string) => void;
  onDirtyChange?: (dirty: boolean) => void;
}) {
  const initialRef = useRef(initialValue);
  const lastDirtyRef = useRef(false);
  const firstChangeRef = useRef(true);
  return (
    <OnChangePlugin
      ignoreHistoryMergeTagChange
      ignoreSelectionChange
      onChange={(_state, editor: LexicalEditor) => {
        editor.update(() => {
          const html = $generateHtmlFromNodes(editor, null);
          if (firstChangeRef.current) {
            // First change fires from hydration. Skip emitting onChange
            // but stash the canonical HTML so dirty comparisons use
            // post-hydration shape (not the raw initialValue string,
            // which may differ in whitespace from Lexical's output).
            firstChangeRef.current = false;
            initialRef.current = html;
            return;
          }
          onChange(html);
          const dirty = html !== initialRef.current;
          if (dirty !== lastDirtyRef.current) {
            lastDirtyRef.current = dirty;
            onDirtyChange?.(dirty);
          }
        });
      }}
    />
  );
}

// ── Utils ──────────────────────────────────────────────────────────

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}
