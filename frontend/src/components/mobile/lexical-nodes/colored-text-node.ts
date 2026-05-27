/**
 * ColoredTextNode — Phase 2 T2-4b Lexical migration.
 *
 * Custom TextNode subclass that applies a `text-color-X` class to the
 * rendered DOM span (where X is one of rose / amber / emerald / sky /
 * violet — the locked TAG_PALETTE keys from tag-palette.ts).
 *
 * Why a subclass instead of inline styles:
 *   - DOMPurify in the editor's sanitize pass strips the `style`
 *     attribute (only `href` and `class` are allow-listed). Lexical's
 *     built-in $patchStyleText writes inline styles → would be stripped
 *     on save. Subclassing TextNode and applying via className survives
 *     the sanitize pass and round-trips correctly through the existing
 *     ReactMarkdown + rehypeRaw chain on desktop display.
 *   - Matches the established `.indent-N` pattern in indent-aware-blocks.ts
 *     (class-based formatting, no inline style attribute).
 *
 * The corresponding CSS lives in globals.css:
 *   .text-color-rose / -amber / -emerald / -sky / -violet
 *
 * Default-reset is handled by the consumer (mobile-rich-text-editor.tsx):
 * walk the selection, for each ColoredTextNode call `replace($createTextNode(text))`
 * preserving format flags. ColoredTextNode itself doesn't carry a
 * "default" state — absence of the class is the default.
 */

import {
  $applyNodeReplacement,
  TextNode,
  type DOMConversionMap,
  type DOMConversionOutput,
  type DOMExportOutput,
  type EditorConfig,
  type LexicalUpdateJSON,
  type NodeKey,
  type SerializedTextNode,
  type Spread,
} from "lexical";

export type TextColor = "rose" | "amber" | "emerald" | "sky" | "violet";

const TEXT_COLORS: ReadonlyArray<TextColor> = [
  "rose",
  "amber",
  "emerald",
  "sky",
  "violet",
] as const;

export function isTextColor(value: string): value is TextColor {
  return (TEXT_COLORS as ReadonlyArray<string>).includes(value);
}

export type SerializedColoredTextNode = Spread<
  { color: TextColor },
  SerializedTextNode
>;

export class ColoredTextNode extends TextNode {
  __color: TextColor;

  static getType(): string {
    return "colored-text";
  }

  static clone(node: ColoredTextNode): ColoredTextNode {
    return new ColoredTextNode(node.__text, node.__color, node.__key);
  }

  constructor(text: string, color: TextColor, key?: NodeKey) {
    super(text, key);
    this.__color = color;
  }

  getColor(): TextColor {
    return this.getLatest().__color;
  }

  setColor(color: TextColor): this {
    const writable = this.getWritable();
    writable.__color = color;
    return writable;
  }

  createDOM(config: EditorConfig): HTMLElement {
    const el = super.createDOM(config);
    el.classList.add(`text-color-${this.__color}`);
    return el;
  }

  updateDOM(
    prevNode: this,
    dom: HTMLElement,
    config: EditorConfig,
  ): boolean {
    const updated = super.updateDOM(prevNode, dom, config);
    if (prevNode.__color !== this.__color) {
      dom.classList.remove(`text-color-${prevNode.__color}`);
      dom.classList.add(`text-color-${this.__color}`);
    }
    return updated;
  }

  // ── HTML export ────────────────────────────────────────────────
  // TextNode's default exportDOM emits format-flag wrappers (<strong>,
  // <em>, <u>, <code>) around the text. We let super handle that and
  // wrap the entire result in our colored span.
  exportDOM(): DOMExportOutput {
    const span = document.createElement("span");
    span.className = `text-color-${this.__color}`;
    span.textContent = this.__text;
    return { element: span };
  }

  // ── HTML import ────────────────────────────────────────────────
  // Match <span class="text-color-X"> on paste / hydration. Priority 1
  // beats Lexical's default span handler (priority 0) so colored spans
  // become ColoredTextNodes instead of plain TextNodes.
  static importDOM(): DOMConversionMap | null {
    return {
      span: (el: HTMLElement) => {
        const colorClass = Array.from(el.classList).find((c) =>
          c.startsWith("text-color-"),
        );
        if (!colorClass) return null;
        const color = colorClass.slice("text-color-".length);
        if (!isTextColor(color)) return null;
        return {
          conversion: (domNode: HTMLElement): DOMConversionOutput => {
            const text = domNode.textContent ?? "";
            return { node: $createColoredTextNode(text, color) };
          },
          priority: 1,
        };
      },
    };
  }

  // ── JSON serialization (for editor state / undo history) ──────
  static importJSON(json: SerializedColoredTextNode): ColoredTextNode {
    return $createColoredTextNode(json.text, json.color).updateFromJSON(json);
  }

  updateFromJSON(
    serializedNode: LexicalUpdateJSON<SerializedColoredTextNode>,
  ): this {
    const node = super.updateFromJSON(serializedNode);
    if (serializedNode.color && isTextColor(serializedNode.color)) {
      node.setColor(serializedNode.color);
    }
    return node;
  }

  exportJSON(): SerializedColoredTextNode {
    return {
      ...super.exportJSON(),
      color: this.__color,
      type: ColoredTextNode.getType(),
    };
  }
}

export function $createColoredTextNode(
  text: string,
  color: TextColor,
): ColoredTextNode {
  return $applyNodeReplacement(new ColoredTextNode(text, color));
}

export function $isColoredTextNode(
  node: unknown,
): node is ColoredTextNode {
  return node instanceof ColoredTextNode;
}
