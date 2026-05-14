import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";

// jsdom localStorage shim — matches the active-campaign / weekly-retro tests.
if (typeof window !== "undefined" && !(window as any).localStorage?.getItem) {
  const _store = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (k: string) => _store.get(k) ?? null,
      setItem: (k: string, v: string) => { _store.set(k, String(v)); },
      removeItem: (k: string) => { _store.delete(k); },
      clear: () => { _store.clear(); },
      key: (i: number) => Array.from(_store.keys())[i] ?? null,
      get length() { return _store.size; },
    },
    writable: true,
  });
}

import { WeeklyThoughts } from "./weekly-thoughts";

const EXPANDED_KEY = "mo-weekly-retro-thoughts-expanded";

// jsdom does not implement document.execCommand. Install a stub on the
// prototype before any test runs so vi.spyOn can hook it. Same for
// document.queryCommandState (not used today but adjacent surface).
if (typeof document !== "undefined" && typeof document.execCommand !== "function") {
  Object.defineProperty(document, "execCommand", {
    configurable: true,
    writable: true,
    value: () => true,
  });
}

describe("WeeklyThoughts — Phase 3", () => {
  beforeEach(() => {
    try { localStorage.clear(); } catch { /* shim */ }
    vi.spyOn(document, "execCommand").mockReturnValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("renders the placeholder when value is empty", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    expect(
      await screen.findByText(/What did you learn this week/i),
    ).toBeInTheDocument();
  });

  test("hides the placeholder once content is provided", async () => {
    render(<WeeklyThoughts value="<p>existing thoughts</p>" onChange={() => {}} />);
    expect(screen.queryByText(/What did you learn this week/i)).not.toBeInTheDocument();
  });

  test("editor renders with the value as innerHTML", async () => {
    render(<WeeklyThoughts value="<p>hello world</p>" onChange={() => {}} />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    // Wait for the mount-effect to write innerHTML.
    await waitFor(() => expect(editor.innerHTML).toContain("hello world"));
  });

  test("onChange fires when the editor receives input", async () => {
    const onChange = vi.fn();
    render(<WeeklyThoughts value="" onChange={onChange} />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await act(async () => {
      (editor as HTMLDivElement).innerHTML = "<p>typed</p>";
      fireEvent.input(editor);
    });
    expect(onChange).toHaveBeenCalledWith("<p>typed</p>");
  });

  test("Bold button fires execCommand('bold')", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const btn = screen.getByRole("button", { name: /^Bold$/i });
    await act(async () => { fireEvent.click(btn); });
    expect(document.execCommand).toHaveBeenCalledWith("bold", false, undefined);
  });

  test("Italic / Underline / Strikethrough dispatch the right commands", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Italic$/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("italic", false, undefined);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Underline$/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("underline", false, undefined);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Strikethrough$/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("strikeThrough", false, undefined);
  });

  test("Insert link prompts for URL and fires createLink", async () => {
    const promptSpy = vi.spyOn(window, "prompt").mockReturnValue("https://example.com");
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Insert link/i }));
    });
    expect(promptSpy).toHaveBeenCalled();
    expect(document.execCommand).toHaveBeenCalledWith("createLink", false, "https://example.com");
  });

  test("Insert link with no URL → no execCommand call", async () => {
    vi.spyOn(window, "prompt").mockReturnValue(null);
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Insert link/i }));
    });
    // execCommand was NOT called for a link command — but the editor's
    // mount may have scheduled other calls; assert by looking at recorded args.
    const linkCalls = (document.execCommand as any).mock.calls.filter(
      ([cmd]: [string]) => cmd === "createLink",
    );
    expect(linkCalls).toHaveLength(0);
  });

  test("Eraser fires removeFormat", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Clear formatting/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("removeFormat", false, undefined);
  });

  test("Inert buttons (font / size / mic) do not trigger execCommand", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    // Each inert button has an aria-label ending in "(coming soon)".
    const fontFamily = screen.getByRole("button", { name: /Font family/i });
    const fontSize = screen.getByRole("button", { name: /Font size/i });
    const mic = screen.getByRole("button", { name: /Voice dictation/i });
    // disabled prevents click, but verify nothing happens regardless.
    await act(async () => {
      fireEvent.click(fontFamily);
      fireEvent.click(fontSize);
      fireEvent.click(mic);
    });
    expect(document.execCommand).not.toHaveBeenCalled();
  });

  test("Paste sanitization strips <script> and unknown tags via DOMPurify", async () => {
    const onChange = vi.fn();
    render(<WeeklyThoughts value="" onChange={onChange} />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });

    // Build a synthetic ClipboardEvent. JSDOM's ClipboardEvent doesn't
    // accept a clipboardData payload directly, so we craft a minimal one.
    const dirty = "<p>safe</p><script>alert(1)</script><iframe src='x'></iframe><b>kept</b>";
    const fakeEvent = {
      preventDefault: vi.fn(),
      clipboardData: {
        getData: (type: string) => (type === "text/html" ? dirty : ""),
      },
    } as unknown as React.ClipboardEvent<HTMLDivElement>;

    await act(async () => {
      // Manually invoke the handler since fireEvent.paste in jsdom doesn't
      // populate clipboardData reliably. Find the editor's onPaste prop
      // via the rendered element's React fiber is overkill; instead we
      // dispatch a paste event with a mocked clipboardData by extending
      // the standard Event — the React handler runs and reads clipboardData.
      // Simpler approach: call the React-synthesized handler by emitting
      // a paste event that has the data attached.
      const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
      ev.clipboardData = fakeEvent.clipboardData;
      editor.dispatchEvent(ev);
    });

    // execCommand was called with insertHTML; the inserted string must
    // contain the safe tags but not <script> or <iframe>.
    const insertCalls = (document.execCommand as any).mock.calls.filter(
      ([cmd]: [string]) => cmd === "insertHTML",
    );
    expect(insertCalls.length).toBeGreaterThanOrEqual(1);
    const cleaned = insertCalls[0][2] as string;
    expect(cleaned).toContain("<p>safe</p>");
    expect(cleaned).toContain("<b>kept</b>");
    expect(cleaned).not.toContain("<script>");
    expect(cleaned).not.toContain("<iframe");
  });

  test("Section expander defaults to expanded; click collapses; click reopens", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const header = screen.getByRole("button", { name: /Weekly Thoughts/i });
    expect(header).toHaveAttribute("aria-expanded", "true");
    expect(document.getElementById("weekly-thoughts-body")).not.toBeNull();

    await act(async () => { fireEvent.click(header); });
    expect(header).toHaveAttribute("aria-expanded", "false");
    expect(document.getElementById("weekly-thoughts-body")).toBeNull();

    await act(async () => { fireEvent.click(header); });
    expect(header).toHaveAttribute("aria-expanded", "true");
  });

  test("Collapse state persists in localStorage", async () => {
    const { unmount } = render(<WeeklyThoughts value="" onChange={() => {}} />);
    const header = screen.getByRole("button", { name: /Weekly Thoughts/i });
    await act(async () => { fireEvent.click(header); });
    expect(localStorage.getItem(EXPANDED_KEY)).toBe("false");
    unmount();

    // Remount — should hydrate as collapsed from localStorage.
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const header2 = screen.getByRole("button", { name: /Weekly Thoughts/i });
    expect(header2).toHaveAttribute("aria-expanded", "false");
  });

  test("Collapsed header shows word count when value has content; hides when empty", async () => {
    const { rerender } = render(
      <WeeklyThoughts value="<p>three short words here</p>" onChange={() => {}} />,
    );
    const header = screen.getByRole("button", { name: /Weekly Thoughts/i });
    // Collapse first.
    await act(async () => { fireEvent.click(header); });
    expect(header).toHaveTextContent(/4 words/);

    // Empty value → no caption (just "Weekly Thoughts" remains).
    rerender(<WeeklyThoughts value="" onChange={() => {}} />);
    expect(header).not.toHaveTextContent(/words/);
  });

  test("Cursor-preserving hydration: value change while editor is focused does NOT overwrite innerHTML", async () => {
    const { rerender } = render(
      <WeeklyThoughts value="<p>initial</p>" onChange={() => {}} />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    // Wait for mount-effect write.
    await waitFor(() => expect(editor.innerHTML).toContain("initial"));

    // Focus the editor and type new content into it directly (simulating
    // an in-flight user edit).
    await act(async () => {
      (editor as HTMLDivElement).focus();
      (editor as HTMLDivElement).innerHTML = "<p>user typing</p>";
    });
    expect(editor.innerHTML).toContain("user typing");

    // Now an external value prop change arrives (e.g., post-save merge).
    // Because the editor is still focused, the hydration effect must SKIP
    // overwriting innerHTML — the user's local edit is authoritative.
    rerender(<WeeklyThoughts value="<p>server response</p>" onChange={() => {}} />);
    expect(editor.innerHTML).toContain("user typing");
    expect(editor.innerHTML).not.toContain("server response");
  });

  test("When editor is NOT focused, value prop changes DO update innerHTML", async () => {
    const { rerender } = render(
      <WeeklyThoughts value="<p>initial</p>" onChange={() => {}} />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.innerHTML).toContain("initial"));

    // Editor is unfocused. External update should propagate.
    rerender(<WeeklyThoughts value="<p>updated externally</p>" onChange={() => {}} />);
    await waitFor(() => expect(editor.innerHTML).toContain("updated externally"));
  });
});
