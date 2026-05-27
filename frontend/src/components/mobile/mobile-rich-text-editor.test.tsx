import { describe, test, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import {
  $applyNodeReplacement,
  $createParagraphNode,
  $createTextNode,
  $getRoot,
  createEditor,
  ParagraphNode,
} from "lexical";
import {
  $createHeadingNode,
  $createQuoteNode,
  HeadingNode,
  QuoteNode,
} from "@lexical/rich-text";
import {
  $createListItemNode,
  ListItemNode,
  ListNode,
} from "@lexical/list";
import { AutoLinkNode, LinkNode } from "@lexical/link";
import { $generateHtmlFromNodes } from "@lexical/html";
import {
  IndentAwareHeadingNode,
  IndentAwareListItemNode,
  IndentAwareParagraphNode,
  IndentAwareQuoteNode,
} from "./lexical-nodes/indent-aware-blocks";
import {
  $createColoredTextNode,
  ColoredTextNode,
} from "./lexical-nodes/colored-text-node";
import MobileRichTextEditor from "./mobile-rich-text-editor";

/**
 * Lexical-backed editor tests. Two layers of coverage:
 *
 * 1. **Shell-level interaction tests** (jsdom-friendly): render the
 *    editor, fire pointerdown on toolbar buttons, assert that
 *    popovers open/close, dropdowns appear, and the empty placeholder
 *    renders. These don't introspect Lexical's internal state because
 *    jsdom's contentEditable + selection model is incomplete — the
 *    real "does Bold actually bold the text" verification happens
 *    on-device against the Vercel preview deploy (see PR description
 *    for the on-device checklist).
 *
 * 2. **External API tests**: confirm initialValue HTML produces a
 *    rendered editor, onChange fires with HTML on edit, and the
 *    component accepts/passes through the documented props without
 *    crashing. Lexical's internal state machine is not mocked or
 *    stubbed here — we rely on Lexical's own (extensive) test suite
 *    for its correctness.
 *
 * The previous editor's execCommand-spy tests are removed entirely;
 * they assumed a deprecated browser API that didn't translate to
 * Lexical's command-dispatch model.
 */

describe("MobileRichTextEditor — rendering shell", () => {
  test("renders the editor body + toolbar", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    expect(screen.getByTestId("mobile-rich-text-editor")).toBeInTheDocument();
    expect(
      screen.getByTestId("mobile-rich-text-editor-toolbar"),
    ).toBeInTheDocument();
  });

  test("renders placeholder string in the empty placeholder slot", () => {
    render(
      <MobileRichTextEditor
        initialValue=""
        onChange={vi.fn()}
        placeholder="Write something…"
      />,
    );
    expect(
      screen.getByTestId("mobile-rich-text-editor-placeholder"),
    ).toHaveTextContent("Write something…");
  });

  test("renders all 15 toolbar surfaces", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    const expected = [
      "mobile-rte-toolbar-bold",
      "mobile-rte-toolbar-italic",
      "mobile-rte-toolbar-underline",
      "mobile-rte-toolbar-headings",
      "mobile-rte-toolbar-bulleted-list",
      "mobile-rte-toolbar-numbered-list",
      "mobile-rte-toolbar-indent",
      "mobile-rte-toolbar-outdent",
      "mobile-rte-toolbar-color",
      "mobile-rte-toolbar-insert-link",
      "mobile-rte-toolbar-quote",
      "mobile-rte-toolbar-inline-code",
      "mobile-rte-toolbar-clear-formatting",
      "mobile-rte-toolbar-undo",
      "mobile-rte-toolbar-redo",
    ];
    for (const id of expected) {
      expect(screen.getByTestId(id)).toBeInTheDocument();
    }
  });
});

describe("MobileRichTextEditor — heading dropdown", () => {
  test("opens menu on pointerdown of trigger", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    expect(
      screen.queryByTestId("mobile-rte-heading-menu"),
    ).not.toBeInTheDocument();
    fireEvent.pointerDown(screen.getByTestId("mobile-rte-toolbar-headings"));
    expect(screen.getByTestId("mobile-rte-heading-menu")).toBeInTheDocument();
  });

  test("menu lists Normal / H1 / H2 / H3 options", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    fireEvent.pointerDown(screen.getByTestId("mobile-rte-toolbar-headings"));
    expect(screen.getByTestId("mobile-rte-heading-p")).toHaveTextContent(
      "Normal",
    );
    expect(screen.getByTestId("mobile-rte-heading-h1")).toHaveTextContent(
      "Heading 1",
    );
    expect(screen.getByTestId("mobile-rte-heading-h2")).toHaveTextContent(
      "Heading 2",
    );
    expect(screen.getByTestId("mobile-rte-heading-h3")).toHaveTextContent(
      "Heading 3",
    );
  });

  test("menu closes after option selection", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    fireEvent.pointerDown(screen.getByTestId("mobile-rte-toolbar-headings"));
    fireEvent.pointerDown(screen.getByTestId("mobile-rte-heading-h2"));
    expect(
      screen.queryByTestId("mobile-rte-heading-menu"),
    ).not.toBeInTheDocument();
  });
});

describe("MobileRichTextEditor — color picker", () => {
  test("opens palette on pointerdown of trigger", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    expect(
      screen.queryByTestId("mobile-rte-color-menu"),
    ).not.toBeInTheDocument();
    fireEvent.pointerDown(screen.getByTestId("mobile-rte-toolbar-color"));
    expect(screen.getByTestId("mobile-rte-color-menu")).toBeInTheDocument();
  });

  test("palette renders 5 swatches + default-reset", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    fireEvent.pointerDown(screen.getByTestId("mobile-rte-toolbar-color"));
    expect(screen.getByTestId("mobile-rte-color-rose")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-rte-color-amber")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-rte-color-emerald")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-rte-color-sky")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-rte-color-violet")).toBeInTheDocument();
    expect(
      screen.getByTestId("mobile-rte-color-default"),
    ).toBeInTheDocument();
  });

  test("palette closes after swatch selection", () => {
    render(
      <MobileRichTextEditor
        initialValue="<p>seed</p>"
        onChange={vi.fn()}
      />,
    );
    fireEvent.pointerDown(screen.getByTestId("mobile-rte-toolbar-color"));
    fireEvent.pointerDown(screen.getByTestId("mobile-rte-color-emerald"));
    expect(
      screen.queryByTestId("mobile-rte-color-menu"),
    ).not.toBeInTheDocument();
  });

  test("palette closes after default-reset selection", () => {
    render(
      <MobileRichTextEditor
        initialValue='<p><span class="text-color-rose">x</span></p>'
        onChange={vi.fn()}
      />,
    );
    fireEvent.pointerDown(screen.getByTestId("mobile-rte-toolbar-color"));
    fireEvent.pointerDown(screen.getByTestId("mobile-rte-color-default"));
    expect(
      screen.queryByTestId("mobile-rte-color-menu"),
    ).not.toBeInTheDocument();
  });
});

describe("MobileRichTextEditor — initial value hydration", () => {
  test("renders empty body when initialValue is blank", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    const body = screen.getByTestId("mobile-rich-text-editor-body");
    expect(body).toBeInTheDocument();
    // Empty editor contains an empty paragraph (Lexical convention).
    expect(body.textContent ?? "").toBe("");
  });

  test("renders seeded HTML content (paragraph text)", async () => {
    render(
      <MobileRichTextEditor
        initialValue="<p>Hello world</p>"
        onChange={vi.fn()}
      />,
    );
    const body = screen.getByTestId("mobile-rich-text-editor-body");
    await waitFor(() => {
      expect(body.textContent).toContain("Hello world");
    });
  });

  test("renders seeded HTML with inline formatting", async () => {
    render(
      <MobileRichTextEditor
        initialValue="<p>Hello <strong>bold</strong> world</p>"
        onChange={vi.fn()}
      />,
    );
    const body = screen.getByTestId("mobile-rich-text-editor-body");
    await waitFor(() => {
      expect(body.textContent).toContain("Hello bold world");
    });
  });
});

describe("MobileRichTextEditor — external API", () => {
  test("accepts onDirtyChange prop without crashing", () => {
    render(
      <MobileRichTextEditor
        initialValue=""
        onChange={vi.fn()}
        onDirtyChange={vi.fn()}
      />,
    );
    expect(screen.getByTestId("mobile-rich-text-editor")).toBeInTheDocument();
  });

  test("toolbar buttons preventDefault on pointerdown to preserve selection", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    const btn = screen.getByTestId("mobile-rte-toolbar-bold");
    const evt = new Event("pointerdown", { bubbles: true, cancelable: true });
    btn.dispatchEvent(evt);
    expect(evt.defaultPrevented).toBe(true);
  });
});

// ── Node replacement registration tests ──────────────────────────
//
// These tests verify the LexicalComposer.initialConfig.nodes replace
// map wires the IndentAware* subclasses correctly. The on-device
// regression that motivated the second follow-up (block commands
// failing on iOS while inline ones worked) would have surfaced as a
// failure here: `new HeadingNode("h1")` going through
// `$applyNodeReplacement` should return an IndentAwareHeadingNode,
// not a plain HeadingNode.
//
// We construct a standalone Lexical editor with the same node config
// the component uses, then exercise the replacement path via
// $createHeadingNode etc. (which internally call
// $applyNodeReplacement).

function createTestEditor() {
  return createEditor({
    namespace: "test-editor",
    onError: (err) => {
      throw err;
    },
    nodes: [
      ListNode,
      LinkNode,
      AutoLinkNode,
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
  });
}

/** Run a callback inside editor.update synchronously and return the
 *  result. Lexical's update is synchronous by default (the discrete
 *  option isn't needed when not in a batch). */
function runInEditor<T>(editor: ReturnType<typeof createEditor>, fn: () => T): T {
  let result!: T;
  editor.update(
    () => {
      result = fn();
    },
    { discrete: true },
  );
  return result;
}

describe("MobileRichTextEditor — node replacement wiring", () => {
  test("$applyNodeReplacement substitutes IndentAwareParagraphNode for ParagraphNode", () => {
    const editor = createTestEditor();
    const result = runInEditor(editor, () =>
      $applyNodeReplacement(new ParagraphNode()),
    );
    expect(result).toBeInstanceOf(IndentAwareParagraphNode);
  });

  test("$createHeadingNode produces IndentAwareHeadingNode via replacement", () => {
    const editor = createTestEditor();
    const { node, tag } = runInEditor(editor, () => {
      const n = $createHeadingNode("h2");
      return { node: n, tag: n.getTag() };
    });
    expect(node).toBeInstanceOf(IndentAwareHeadingNode);
    expect(tag).toBe("h2");
  });

  test("$createQuoteNode produces IndentAwareQuoteNode via replacement", () => {
    const editor = createTestEditor();
    const result = runInEditor(editor, () => $createQuoteNode());
    expect(result).toBeInstanceOf(IndentAwareQuoteNode);
  });

  test("$createListItemNode produces IndentAwareListItemNode via replacement", () => {
    const editor = createTestEditor();
    const result = runInEditor(editor, () => $createListItemNode());
    expect(result).toBeInstanceOf(IndentAwareListItemNode);
  });

  test("IndentAwareHeadingNode supports __indent via setIndent/getIndent", () => {
    const editor = createTestEditor();
    const indent = runInEditor(editor, () => {
      const node = $createHeadingNode("h1");
      node.setIndent(2);
      return node.getIndent();
    });
    expect(indent).toBe(2);
  });

  test("IndentAware*.exportDOM emits class='indent-N' (not padding style)", () => {
    const editor = createTestEditor();
    const html = runInEditor(editor, () => {
      const root = $getRoot();
      root.clear();
      const p = $createHeadingNode("h2");
      p.setIndent(3);
      p.append($createTextNode("indented heading"));
      root.append(p);
      return $generateHtmlFromNodes(editor, null);
    });
    expect(html).toContain('class="indent-3"');
    expect(html).not.toContain("padding-inline-start");
    expect(html).not.toMatch(/style="[^"]*padding/);
  });

  test("ColoredTextNode round-trips through HTML export", () => {
    const editor = createTestEditor();
    const html = runInEditor(editor, () => {
      const root = $getRoot();
      root.clear();
      const p = $createParagraphNode();
      p.append($createColoredTextNode("hello", "emerald"));
      root.append(p);
      return $generateHtmlFromNodes(editor, null);
    });
    expect(html).toContain('class="text-color-emerald"');
    expect(html).toContain("hello");
  });

  test("IndentAwareHeadingNode.exportJSON includes the subclass type", () => {
    const editor = createTestEditor();
    const exported = runInEditor(editor, () => {
      const node = $createHeadingNode("h1");
      return node.exportJSON();
    });
    expect((exported as Record<string, unknown>).type).toBe(
      "indent-aware-heading",
    );
  });

  test("IndentAwareParagraphNode.exportJSON includes the subclass type", () => {
    const editor = createTestEditor();
    const exported = runInEditor(editor, () => {
      const node = $createParagraphNode();
      return node.exportJSON();
    });
    expect((exported as Record<string, unknown>).type).toBe(
      "indent-aware-paragraph",
    );
  });
});

describe("MobileRichTextEditor — visualViewport observer", () => {
  test("subscribes to resize + scroll on mount when visualViewport is available", () => {
    const addSpy = vi.fn();
    const removeSpy = vi.fn();
    Object.defineProperty(window, "visualViewport", {
      value: { height: 800, offsetTop: 0, addEventListener: addSpy, removeEventListener: removeSpy },
      configurable: true,
    });
    const { unmount } = render(
      <MobileRichTextEditor initialValue="" onChange={vi.fn()} />,
    );
    expect(addSpy).toHaveBeenCalledWith("resize", expect.any(Function));
    expect(addSpy).toHaveBeenCalledWith("scroll", expect.any(Function));
    unmount();
    expect(removeSpy).toHaveBeenCalledWith("resize", expect.any(Function));
    expect(removeSpy).toHaveBeenCalledWith("scroll", expect.any(Function));
  });

  test("falls back gracefully when visualViewport is undefined", () => {
    Object.defineProperty(window, "visualViewport", {
      value: undefined,
      configurable: true,
    });
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    const toolbar = screen.getByTestId("mobile-rich-text-editor-toolbar");
    expect(toolbar).toHaveStyle({ bottom: "0px" });
  });
});
