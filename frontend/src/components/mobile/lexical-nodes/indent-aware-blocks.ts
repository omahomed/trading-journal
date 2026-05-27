/**
 * Indent-aware block node subclasses — Phase 2 T2-4b Lexical migration.
 *
 * Lexical tracks indent as `__indent: number` on every ElementNode and
 * exports it as inline `style="padding-inline-start: ..."`. Our
 * DOMPurify pass strips `style` attributes (only `class` and `href`
 * are allow-listed), so the default Lexical indent markup wouldn't
 * survive a round-trip.
 *
 * These four subclasses override `exportDOM` to emit
 * `class="indent-{N}"` instead, and `importDOM` to recognize that
 * class on incoming HTML and re-establish `__indent` on the resulting
 * node. This matches the existing `.indent-N` CSS in globals.css
 * (lines 410-418 for desktop, 420+ for mobile per T2-4b CSS addition).
 *
 * Scope: 4 subclasses (Paragraph / Heading / Quote / ListItem) cover
 * every block type the toolbar can produce. Each delegates to the
 * super exportDOM, then mutates the returned element to swap inline
 * padding for the indent class.
 */

import {
  $applyNodeReplacement,
  ParagraphNode,
  type DOMConversion,
  type DOMConversionMap,
  type DOMExportOutput,
  type LexicalEditor,
  type LexicalNode,
  type LexicalUpdateJSON,
  type SerializedElementNode,
  type SerializedParagraphNode,
} from "lexical";
import {
  HeadingNode,
  QuoteNode,
  type HeadingTagType,
  type SerializedHeadingNode,
} from "@lexical/rich-text";
import { ListItemNode, type SerializedListItemNode } from "@lexical/list";

const INDENT_CLASS_RE = /^indent-([1-5])$/;
const MAX_INDENT = 5;

function readIndentClass(el: HTMLElement): number | null {
  for (const c of Array.from(el.classList)) {
    const m = c.match(INDENT_CLASS_RE);
    if (m) return parseInt(m[1], 10);
  }
  return null;
}

/** Adds `class="indent-N"` to an exportDOM output element (mutating it
 *  in place) and strips Lexical's default `padding-inline-start` style.
 *  Returns the same output for chaining. */
function applyIndentClassToExport<T extends DOMExportOutput>(
  out: T,
  indent: number,
): T {
  const el = out.element;
  if (!(el instanceof HTMLElement)) return out;
  if (indent > 0) {
    const level = Math.min(MAX_INDENT, indent);
    el.classList.add(`indent-${level}`);
  }
  el.style.removeProperty("padding-inline-start");
  if (!el.getAttribute("style")?.trim()) {
    el.removeAttribute("style");
  }
  return out;
}

/** Wraps a parent class's importDOM map so the resulting node picks
 *  up `__indent` from any `class="indent-{N}"` it carries on the DOM. */
function wrapImportDOMForIndent(
  parentMap: DOMConversionMap | null,
): DOMConversionMap | null {
  if (!parentMap) return null;
  const out: DOMConversionMap = {};
  for (const [tagName, propFn] of Object.entries(parentMap)) {
    out[tagName] = (el: HTMLElement) => {
      const baseConv = propFn(el);
      if (!baseConv) return null;
      const wrapped: DOMConversion = {
        ...baseConv,
        conversion: (domNode: HTMLElement) => {
          const inner = baseConv.conversion(domNode);
          if (!inner || !inner.node) return inner;
          const node = inner.node as LexicalNode;
          if (typeof (node as { setIndent?: unknown }).setIndent === "function") {
            const indent = readIndentClass(domNode);
            if (indent != null) {
              (node as ParagraphNode).setIndent(indent);
            }
          }
          return inner;
        },
      };
      return wrapped;
    };
  }
  return out;
}

// ── Paragraph ──────────────────────────────────────────────────────

export class IndentAwareParagraphNode extends ParagraphNode {
  static getType(): string {
    return "indent-aware-paragraph";
  }

  static clone(node: IndentAwareParagraphNode): IndentAwareParagraphNode {
    return new IndentAwareParagraphNode(node.__key);
  }

  static importDOM(): DOMConversionMap | null {
    return wrapImportDOMForIndent(ParagraphNode.importDOM?.() ?? null);
  }

  static importJSON(
    json: SerializedParagraphNode,
  ): IndentAwareParagraphNode {
    return $applyNodeReplacement(
      new IndentAwareParagraphNode(),
    ).updateFromJSON(json);
  }

  exportJSON(): SerializedParagraphNode {
    return {
      ...super.exportJSON(),
      type: IndentAwareParagraphNode.getType(),
    };
  }

  exportDOM(editor: LexicalEditor): DOMExportOutput {
    return applyIndentClassToExport(super.exportDOM(editor), this.__indent);
  }
}

// ── Heading ────────────────────────────────────────────────────────

export class IndentAwareHeadingNode extends HeadingNode {
  static getType(): string {
    return "indent-aware-heading";
  }

  static clone(node: IndentAwareHeadingNode): IndentAwareHeadingNode {
    return new IndentAwareHeadingNode(node.__tag, node.__key);
  }

  static importDOM(): DOMConversionMap | null {
    return wrapImportDOMForIndent(HeadingNode.importDOM?.() ?? null);
  }

  static importJSON(json: SerializedHeadingNode): IndentAwareHeadingNode {
    return $applyNodeReplacement(
      new IndentAwareHeadingNode(json.tag as HeadingTagType),
    ).updateFromJSON(json);
  }

  updateFromJSON(
    serializedNode: LexicalUpdateJSON<SerializedHeadingNode>,
  ): this {
    return super.updateFromJSON(serializedNode);
  }

  exportJSON(): SerializedHeadingNode {
    return {
      ...super.exportJSON(),
      type: IndentAwareHeadingNode.getType(),
    };
  }

  exportDOM(editor: LexicalEditor): DOMExportOutput {
    return applyIndentClassToExport(super.exportDOM(editor), this.__indent);
  }
}

// ── Quote ──────────────────────────────────────────────────────────

export class IndentAwareQuoteNode extends QuoteNode {
  static getType(): string {
    return "indent-aware-quote";
  }

  static clone(node: IndentAwareQuoteNode): IndentAwareQuoteNode {
    return new IndentAwareQuoteNode(node.__key);
  }

  static importDOM(): DOMConversionMap | null {
    return wrapImportDOMForIndent(QuoteNode.importDOM?.() ?? null);
  }

  static importJSON(json: SerializedElementNode): IndentAwareQuoteNode {
    return $applyNodeReplacement(
      new IndentAwareQuoteNode(),
    ).updateFromJSON(json);
  }

  exportJSON(): SerializedElementNode {
    return {
      ...super.exportJSON(),
      type: IndentAwareQuoteNode.getType(),
    };
  }

  exportDOM(editor: LexicalEditor): DOMExportOutput {
    return applyIndentClassToExport(super.exportDOM(editor), this.__indent);
  }
}

// ── List item ──────────────────────────────────────────────────────

export class IndentAwareListItemNode extends ListItemNode {
  static getType(): string {
    return "indent-aware-listitem";
  }

  static clone(node: IndentAwareListItemNode): IndentAwareListItemNode {
    return new IndentAwareListItemNode(node.__value, node.__checked, node.__key);
  }

  static importDOM(): DOMConversionMap | null {
    // ListItemNode declares its importDOM via $config() rather than as
    // a static method, so the static surface is optional-or-absent.
    // Defer to the $config-derived static config if Lexical attached
    // one at runtime; otherwise fall back to no import map (HTML
    // imports of <li> still work through the parent <ul>/<ol> matcher).
    type MaybeImportDOM = (() => DOMConversionMap | null) | undefined;
    const fn = (ListItemNode as unknown as { importDOM?: MaybeImportDOM })
      .importDOM;
    return wrapImportDOMForIndent(typeof fn === "function" ? fn() : null);
  }

  static importJSON(json: SerializedListItemNode): IndentAwareListItemNode {
    return $applyNodeReplacement(
      new IndentAwareListItemNode(json.value, json.checked),
    ).updateFromJSON(json);
  }

  exportJSON(): SerializedListItemNode {
    return {
      ...super.exportJSON(),
      type: IndentAwareListItemNode.getType(),
    };
  }

  exportDOM(editor: LexicalEditor): DOMExportOutput {
    return applyIndentClassToExport(super.exportDOM(editor), this.__indent);
  }
}
