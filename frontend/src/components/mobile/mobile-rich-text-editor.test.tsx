import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, act } from "@testing-library/react";
import { MobileRichTextEditor } from "./mobile-rich-text-editor";

// execCommand is deprecated but works in jsdom + all relevant
// browsers. Stub the document.execCommand surface for assertions.
let execSpy: ReturnType<typeof vi.fn>;

beforeEach(() => {
  execSpy = vi.fn().mockReturnValue(true);
  // @ts-expect-error overriding for test
  document.execCommand = execSpy;
});

afterEach(() => {
  vi.restoreAllMocks();
});

function setupVisualViewport(
  height: number,
  offsetTop: number,
  innerHeight: number,
): {
  listeners: Map<string, EventListener>;
  trigger: (event: string) => void;
} {
  const listeners = new Map<string, EventListener>();
  const stub = {
    height,
    offsetTop,
    addEventListener: vi.fn((event: string, handler: EventListener) => {
      listeners.set(event, handler);
    }),
    removeEventListener: vi.fn((event: string) => {
      listeners.delete(event);
    }),
  };
  Object.defineProperty(window, "visualViewport", {
    value: stub,
    configurable: true,
  });
  Object.defineProperty(window, "innerHeight", {
    value: innerHeight,
    configurable: true,
  });
  return {
    listeners,
    trigger: (event: string) => listeners.get(event)?.(new Event(event)),
  };
}

function clearVisualViewport() {
  Object.defineProperty(window, "visualViewport", {
    value: undefined,
    configurable: true,
  });
}

// ── Rendering ─────────────────────────────────────────────────────

describe("MobileRichTextEditor — rendering", () => {
  test("renders contentEditable body + toolbar", () => {
    render(
      <MobileRichTextEditor initialValue="<p>seed</p>" onChange={vi.fn()} />,
    );
    expect(screen.getByTestId("mobile-rich-text-editor-body")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-rich-text-editor-toolbar")).toBeInTheDocument();
  });

  test("seeds editor with initialValue HTML on mount", () => {
    render(
      <MobileRichTextEditor
        initialValue="<p>Hello <strong>world</strong></p>"
        onChange={vi.fn()}
      />,
    );
    const body = screen.getByTestId("mobile-rich-text-editor-body");
    expect(body.innerHTML).toContain("<strong>world</strong>");
  });

  test("renders placeholder when empty", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} placeholder="Write…" />);
    expect(screen.getByTestId("mobile-rich-text-editor-placeholder")).toHaveTextContent("Write…");
  });

  test("does NOT render placeholder when seeded with content", () => {
    render(<MobileRichTextEditor initialValue="<p>seed</p>" onChange={vi.fn()} />);
    expect(screen.queryByTestId("mobile-rich-text-editor-placeholder")).not.toBeInTheDocument();
  });
});

// ── execCommand routing ───────────────────────────────────────────

describe("MobileRichTextEditor — toolbar execCommand wiring", () => {
  test.each([
    ["bold", "Bold"],
    ["italic", "Italic"],
    ["underline", "Underline"],
    ["insertUnorderedList", "Bulleted list"],
    ["insertOrderedList", "Numbered list"],
    ["indent", "Indent"],
    ["outdent", "Outdent"],
    ["removeFormat", "Clear formatting"],
    ["undo", "Undo"],
    ["redo", "Redo"],
  ])("toolbar fires execCommand('%s')", (cmd, label) => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    const slug = label.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    fireEvent.click(screen.getByTestId(`mobile-rte-toolbar-${slug}`));
    expect(execSpy).toHaveBeenCalledWith(cmd, false, undefined);
  });

  test("Quote button fires formatBlock(<blockquote>)", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    fireEvent.click(screen.getByTestId("mobile-rte-toolbar-quote"));
    expect(execSpy).toHaveBeenCalledWith("formatBlock", false, "<blockquote>");
  });
});

// ── Heading dropdown ──────────────────────────────────────────────

describe("MobileRichTextEditor — heading dropdown", () => {
  test("opens menu on tap", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    expect(screen.queryByTestId("mobile-rte-heading-menu")).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId("mobile-rte-toolbar-headings"));
    expect(screen.getByTestId("mobile-rte-heading-menu")).toBeInTheDocument();
  });

  test.each([
    ["mobile-rte-heading-h1", "<h1>"],
    ["mobile-rte-heading-h2", "<h2>"],
    ["mobile-rte-heading-h3", "<h3>"],
    ["mobile-rte-heading-p", "<p>"],
  ])("selecting %s fires formatBlock(%s)", (testid, tag) => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    fireEvent.click(screen.getByTestId("mobile-rte-toolbar-headings"));
    fireEvent.click(screen.getByTestId(testid));
    expect(execSpy).toHaveBeenCalledWith("formatBlock", false, tag);
    // Menu closes after selection.
    expect(screen.queryByTestId("mobile-rte-heading-menu")).not.toBeInTheDocument();
  });
});

// ── Color picker ──────────────────────────────────────────────────

describe("MobileRichTextEditor — color picker", () => {
  test("opens palette on tap", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    expect(screen.queryByTestId("mobile-rte-color-menu")).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId("mobile-rte-toolbar-color"));
    expect(screen.getByTestId("mobile-rte-color-menu")).toBeInTheDocument();
  });

  test("renders 5 color swatches + default-reset", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    fireEvent.click(screen.getByTestId("mobile-rte-toolbar-color"));
    expect(screen.getByTestId("mobile-rte-color-rose")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-rte-color-amber")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-rte-color-emerald")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-rte-color-sky")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-rte-color-violet")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-rte-color-default")).toBeInTheDocument();
  });

  test("color swatch click closes the menu and fires onChange", () => {
    // Note: jsdom's Range/Selection support is partial; the DOM
    // mutation that wraps the selection in <span class="text-color-X">
    // depends on Range.extractContents()/insertNode() which jsdom
    // doesn't fully simulate. We verify the click reaches the handler
    // (menu closes + onChange fires) here; the DOM transform is
    // covered by on-device manual testing.
    const onChange = vi.fn();
    render(<MobileRichTextEditor initialValue="<p>hello</p>" onChange={onChange} />);
    fireEvent.click(screen.getByTestId("mobile-rte-toolbar-color"));
    expect(screen.getByTestId("mobile-rte-color-menu")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("mobile-rte-color-emerald"));
    expect(screen.queryByTestId("mobile-rte-color-menu")).not.toBeInTheDocument();
    expect(onChange).toHaveBeenCalled();
  });

  test("default-reset closes the menu and fires onChange", () => {
    const onChange = vi.fn();
    render(
      <MobileRichTextEditor
        initialValue='<p><span class="text-color-rose">colored</span></p>'
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByTestId("mobile-rte-toolbar-color"));
    fireEvent.click(screen.getByTestId("mobile-rte-color-default"));
    expect(screen.queryByTestId("mobile-rte-color-menu")).not.toBeInTheDocument();
    expect(onChange).toHaveBeenCalled();
  });
});

// ── visualViewport observer ───────────────────────────────────────

describe("MobileRichTextEditor — visualViewport observer", () => {
  afterEach(() => clearVisualViewport());

  test("subscribes to resize + scroll on mount", () => {
    const { listeners } = setupVisualViewport(800, 0, 800);
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    expect(listeners.has("resize")).toBe(true);
    expect(listeners.has("scroll")).toBe(true);
  });

  test("toolbar bottom offset reflects computed keyboard height", () => {
    // innerHeight=800, vv.height=500, offsetTop=0 → keyboardHeight=300.
    const { trigger } = setupVisualViewport(500, 0, 800);
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    act(() => trigger("resize"));
    const toolbar = screen.getByTestId("mobile-rich-text-editor-toolbar");
    expect(toolbar).toHaveStyle({ bottom: "300px" });
  });

  test("falls back to bottom: 0 when visualViewport is unavailable", () => {
    clearVisualViewport();
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    const toolbar = screen.getByTestId("mobile-rich-text-editor-toolbar");
    // No vv → keyboardHeight stays 0 → bottom: 0 inline style.
    expect(toolbar).toHaveStyle({ bottom: "0px" });
  });

  test("cleans up listeners on unmount", () => {
    const { listeners } = setupVisualViewport(800, 0, 800);
    const { unmount } = render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    expect(listeners.size).toBeGreaterThan(0);
    unmount();
    expect(listeners.size).toBe(0);
  });
});

// ── onChange wiring ───────────────────────────────────────────────

describe("MobileRichTextEditor — onChange wiring", () => {
  test("fires onChange with sanitized HTML on input", () => {
    const onChange = vi.fn();
    render(<MobileRichTextEditor initialValue="<p>seed</p>" onChange={onChange} />);
    const body = screen.getByTestId("mobile-rich-text-editor-body");
    body.innerHTML = "<p>updated</p>";
    fireEvent.input(body);
    expect(onChange).toHaveBeenCalledWith(expect.stringContaining("updated"));
  });

  test("onDirtyChange fires when value diverges from initialValue", () => {
    const onDirtyChange = vi.fn();
    render(
      <MobileRichTextEditor
        initialValue="<p>seed</p>"
        onChange={vi.fn()}
        onDirtyChange={onDirtyChange}
      />,
    );
    const body = screen.getByTestId("mobile-rich-text-editor-body");
    body.innerHTML = "<p>modified</p>";
    fireEvent.input(body);
    expect(onDirtyChange).toHaveBeenCalledWith(true);
  });
});

// ── Paste handling ────────────────────────────────────────────────

describe("MobileRichTextEditor — paste handling", () => {
  test("blocks image paste (clipboard files include image/*)", () => {
    const onChange = vi.fn();
    render(<MobileRichTextEditor initialValue="" onChange={onChange} />);
    const file = new File([new Uint8Array(8)], "x.png", { type: "image/png" });
    // jsdom lacks a real DataTransfer; stub clipboardData with a
    // minimal shape that satisfies our handler's reads.
    const clipboardData = {
      files: [file],
      getData: (_t: string) => "",
    };
    fireEvent.paste(screen.getByTestId("mobile-rich-text-editor-body"), {
      clipboardData,
    });
    // execCommand insertHTML should NOT have been called for the image.
    expect(execSpy).not.toHaveBeenCalledWith(
      "insertHTML",
      false,
      expect.any(String),
    );
  });

  test("text/html paste sanitizes before insert", () => {
    render(<MobileRichTextEditor initialValue="" onChange={vi.fn()} />);
    const clipboardData = {
      files: [] as File[],
      getData: (t: string) =>
        t === "text/html" ? "<p>safe <strong>bold</strong></p>" : "",
    };
    fireEvent.paste(screen.getByTestId("mobile-rich-text-editor-body"), {
      clipboardData,
    });
    expect(execSpy).toHaveBeenCalledWith(
      "insertHTML",
      false,
      expect.stringContaining("<strong>bold</strong>"),
    );
  });
});

// ── Inline link insertion ─────────────────────────────────────────

describe("MobileRichTextEditor — link insertion", () => {
  test("prompts for URL and fires createLink", () => {
    const promptSpy = vi.spyOn(window, "prompt").mockReturnValue("https://example.com");
    render(<MobileRichTextEditor initialValue="<p>x</p>" onChange={vi.fn()} />);
    fireEvent.click(screen.getByTestId("mobile-rte-toolbar-insert-link"));
    expect(promptSpy).toHaveBeenCalled();
    expect(execSpy).toHaveBeenCalledWith("createLink", false, "https://example.com");
  });

  test("cancelling the URL prompt skips createLink", () => {
    vi.spyOn(window, "prompt").mockReturnValue(null);
    render(<MobileRichTextEditor initialValue="<p>x</p>" onChange={vi.fn()} />);
    fireEvent.click(screen.getByTestId("mobile-rte-toolbar-insert-link"));
    expect(execSpy).not.toHaveBeenCalledWith("createLink", expect.anything(), expect.anything());
  });
});
