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

import { WeeklyThoughts, normalizeIndentation, parseVideoUrl } from "./weekly-thoughts";

const EXPANDED_KEY = "mo-weekly-retro-thoughts-expanded";

// jsdom does not implement document.execCommand. Install a stub on the
// prototype before any test runs so vi.spyOn can hook it. Same for
// document.queryCommandValue (used by the Phase 3.5 Style-dropdown
// label tracker, wrapped in try/catch but cleaner to stub).
if (typeof document !== "undefined" && typeof document.execCommand !== "function") {
  Object.defineProperty(document, "execCommand", {
    configurable: true,
    writable: true,
    value: () => true,
  });
}
if (typeof document !== "undefined" && typeof document.queryCommandValue !== "function") {
  Object.defineProperty(document, "queryCommandValue", {
    configurable: true,
    writable: true,
    value: () => "",
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

  test("Eraser fires removeFormat", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Clear formatting/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("removeFormat", false, undefined);
  });

  test("Voice-dictation mic button is inert (Phase 3.5: font/size are NOT)", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const mic = screen.getByRole("button", { name: /Voice dictation/i });
    await act(async () => { fireEvent.click(mic); });
    expect(document.execCommand).not.toHaveBeenCalled();
  });

  test("Paste sanitization strips <script> and unknown tags via DOMPurify", async () => {
    const onChange = vi.fn();
    render(<WeeklyThoughts value="" onChange={onChange} />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });

    const dirty = "<p>safe</p><script>alert(1)</script><object data='x'></object><b>kept</b>";
    const fakeEvent = {
      preventDefault: vi.fn(),
      clipboardData: {
        getData: (type: string) => (type === "text/html" ? dirty : ""),
      },
    } as unknown as React.ClipboardEvent<HTMLDivElement>;

    await act(async () => {
      const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
      ev.clipboardData = fakeEvent.clipboardData;
      editor.dispatchEvent(ev);
    });

    const insertCalls = (document.execCommand as any).mock.calls.filter(
      ([cmd]: [string]) => cmd === "insertHTML",
    );
    expect(insertCalls.length).toBeGreaterThanOrEqual(1);
    const cleaned = insertCalls[0][2] as string;
    expect(cleaned).toContain("<p>safe</p>");
    expect(cleaned).toContain("<b>kept</b>");
    expect(cleaned).not.toContain("<script>");
    expect(cleaned).not.toContain("<object");
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

    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const header2 = screen.getByRole("button", { name: /Weekly Thoughts/i });
    expect(header2).toHaveAttribute("aria-expanded", "false");
  });

  test("Collapsed header shows word count when value has content; hides when empty", async () => {
    const { rerender } = render(
      <WeeklyThoughts value="<p>three short words here</p>" onChange={() => {}} />,
    );
    const header = screen.getByRole("button", { name: /Weekly Thoughts/i });
    await act(async () => { fireEvent.click(header); });
    expect(header).toHaveTextContent(/4 words/);

    rerender(<WeeklyThoughts value="" onChange={() => {}} />);
    expect(header).not.toHaveTextContent(/words/);
  });

  test("Cursor-preserving hydration: focused editor + value change → no overwrite", async () => {
    const { rerender } = render(
      <WeeklyThoughts value="<p>initial</p>" onChange={() => {}} />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.innerHTML).toContain("initial"));

    await act(async () => {
      (editor as HTMLDivElement).focus();
      (editor as HTMLDivElement).innerHTML = "<p>user typing</p>";
    });
    expect(editor.innerHTML).toContain("user typing");

    rerender(<WeeklyThoughts value="<p>server response</p>" onChange={() => {}} />);
    expect(editor.innerHTML).toContain("user typing");
    expect(editor.innerHTML).not.toContain("server response");
  });

  test("Unfocused editor: value prop changes DO update innerHTML", async () => {
    const { rerender } = render(
      <WeeklyThoughts value="<p>initial</p>" onChange={() => {}} />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.innerHTML).toContain("initial"));

    rerender(<WeeklyThoughts value="<p>updated externally</p>" onChange={() => {}} />);
    await waitFor(() => expect(editor.innerHTML).toContain("updated externally"));
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3.5 expansion — lists, headings, blockquote/code/hr, link popover,
// video embed, font dropdowns, tables, task lists, DOMPurify config,
// indent normalizer, popover-survives hydration.
// ─────────────────────────────────────────────────────────────────────────────

describe("WeeklyThoughts — Phase 3.5 expansion", () => {
  beforeEach(() => {
    try { localStorage.clear(); } catch { /* shim */ }
    vi.spyOn(document, "execCommand").mockReturnValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ─── Lists ─────────────────────────────────────────────────────────────

  test("Bulleted list button → insertUnorderedList", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Bulleted list/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("insertUnorderedList", false, undefined);
  });

  test("Numbered list button → insertOrderedList", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Numbered list/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("insertOrderedList", false, undefined);
  });

  // ─── Indent / outdent ──────────────────────────────────────────────────

  test("Indent button → exec('indent') and triggers normalizer", async () => {
    render(<WeeklyThoughts value="<p>x</p>" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Indent$/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("indent", false, undefined);
  });

  test("Outdent button → exec('outdent')", async () => {
    render(<WeeklyThoughts value="<p>x</p>" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Outdent$/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("outdent", false, undefined);
  });

  // ─── Headings dropdown ─────────────────────────────────────────────────

  test("Style dropdown 'Heading 1' option → formatBlock('<h1>')", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Block style/i }));
    });
    const opt = await screen.findByRole("option", { name: /Heading 1/i });
    await act(async () => { fireEvent.click(opt); });
    expect(document.execCommand).toHaveBeenCalledWith("formatBlock", false, "<h1>");
  });

  test("Style dropdown 'Normal' option → formatBlock('<p>')", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Block style/i }));
    });
    const opt = await screen.findByRole("option", { name: /Normal/i });
    await act(async () => { fireEvent.click(opt); });
    expect(document.execCommand).toHaveBeenCalledWith("formatBlock", false, "<p>");
  });

  // ─── Blockquote / code / HR ────────────────────────────────────────────

  test("Quote button → formatBlock('<blockquote>')", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Quote$/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("formatBlock", false, "<blockquote>");
  });

  test("Code-block button → formatBlock('<pre>')", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Code block/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("formatBlock", false, "<pre>");
  });

  test("Horizontal-rule button → insertHorizontalRule", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Horizontal rule/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("insertHorizontalRule", false, undefined);
  });

  // ─── Link popover ──────────────────────────────────────────────────────

  test("Insert link button opens popover (no window.prompt)", async () => {
    const promptSpy = vi.spyOn(window, "prompt");
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Insert link/i }));
    });
    expect(promptSpy).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog", { name: /Insert link/i })).toBeInTheDocument();
  });

  test("Link popover Cancel button closes the popover", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Insert link/i }));
    });
    expect(screen.getByRole("dialog", { name: /Insert link/i })).toBeInTheDocument();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Cancel/i }));
    });
    expect(screen.queryByRole("dialog", { name: /Insert link/i })).not.toBeInTheDocument();
  });

  test("Link popover OK button fires an execCommand call containing the URL", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Insert link/i }));
    });
    const urlInput = screen.getByPlaceholderText(/https/i);
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "https://example.com" } });
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^OK$/i }));
    });
    // Either createLink (with selection) or insertHTML containing the URL.
    const calls = (document.execCommand as any).mock.calls.filter(
      (c: any[]) =>
        c[0] === "createLink" ||
        (c[0] === "insertHTML" && typeof c[2] === "string" && c[2].includes("example.com")),
    );
    expect(calls.length).toBeGreaterThan(0);
  });

  test("Link popover OK with empty URL → inline error, no execCommand call", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Insert link/i }));
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^OK$/i }));
    });
    // Error is rendered as role="alert"
    expect(screen.getByRole("alert")).toHaveTextContent(/URL/i);
    const linkCalls = (document.execCommand as any).mock.calls.filter(
      ([cmd]: [string]) => cmd === "createLink",
    );
    expect(linkCalls).toHaveLength(0);
  });

  test("Link popover prefills href when cursor is inside an existing <a>", async () => {
    render(
      <WeeklyThoughts
        value='<p>Click <a href="https://prefill.example/path">here</a></p>'
        onChange={() => {}}
      />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("a")).not.toBeNull());

    const anchor = editor.querySelector("a")!;
    // Synthesize a Range positioned inside the <a>'s text node.
    const range = document.createRange();
    range.setStart(anchor.firstChild!, 0);
    range.setEnd(anchor.firstChild!, anchor.firstChild!.textContent!.length);
    const sel = window.getSelection()!;
    sel.removeAllRanges();
    sel.addRange(range);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Insert link/i }));
    });
    const urlInput = screen.getByPlaceholderText(/https/i) as HTMLInputElement;
    expect(urlInput.value).toBe("https://prefill.example/path");
  });

  // ─── Video embed ───────────────────────────────────────────────────────

  test("Video popover with valid YouTube URL → insertHTML containing youtube embed iframe", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Embed video/i }));
    });
    const urlInput = screen.getByPlaceholderText(/YouTube or Vimeo URL/i);
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "https://youtu.be/dQw4w9WgXcQ" } });
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Embed$/i }));
    });
    const calls = (document.execCommand as any).mock.calls.filter(
      (c: any[]) => c[0] === "insertHTML" &&
        typeof c[2] === "string" &&
        c[2].includes("www.youtube.com/embed/"),
    );
    expect(calls.length).toBeGreaterThan(0);
    expect(calls[0][2]).toContain("wt-video-embed");
  });

  test("Video popover with invalid URL → inline error, no insertHTML", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Embed video/i }));
    });
    const urlInput = screen.getByPlaceholderText(/YouTube or Vimeo URL/i);
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "https://evil.example/foo" } });
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Embed$/i }));
    });
    expect(screen.getByRole("alert")).toHaveTextContent(/YouTube and Vimeo/i);
    const calls = (document.execCommand as any).mock.calls.filter(
      (c: any[]) => c[0] === "insertHTML" &&
        typeof c[2] === "string" &&
        c[2].includes("iframe"),
    );
    expect(calls).toHaveLength(0);
  });

  // ─── parseVideoUrl unit tests ──────────────────────────────────────────

  test("parseVideoUrl: youtu.be → embed", () => {
    expect(parseVideoUrl("https://youtu.be/dQw4w9WgXcQ")).toEqual({
      src: "https://www.youtube.com/embed/dQw4w9WgXcQ",
      provider: "youtube",
    });
  });

  test("parseVideoUrl: youtube.com/watch?v= → embed", () => {
    expect(parseVideoUrl("https://www.youtube.com/watch?v=abc123&t=10")).toEqual({
      src: "https://www.youtube.com/embed/abc123",
      provider: "youtube",
    });
  });

  test("parseVideoUrl: youtube.com/embed/ → passthrough normalized", () => {
    expect(parseVideoUrl("https://www.youtube.com/embed/xyz789")).toEqual({
      src: "https://www.youtube.com/embed/xyz789",
      provider: "youtube",
    });
  });

  test("parseVideoUrl: vimeo.com/{id} → embed", () => {
    expect(parseVideoUrl("https://vimeo.com/123456789")).toEqual({
      src: "https://player.vimeo.com/video/123456789",
      provider: "vimeo",
    });
  });

  test("parseVideoUrl: junk URL → null", () => {
    expect(parseVideoUrl("https://evil.example/foo")).toBeNull();
  });

  test("parseVideoUrl: empty string → null", () => {
    expect(parseVideoUrl("")).toBeNull();
  });

  // ─── Font family / size dropdowns ──────────────────────────────────────

  test("Font dropdown 'Sans-serif' option → exec('fontName', expected family)", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Font family/i }));
    });
    const opt = await screen.findByRole("option", { name: /Sans-serif/i });
    await act(async () => { fireEvent.click(opt); });
    expect(document.execCommand).toHaveBeenCalledWith(
      "fontName",
      false,
      '"Inter", system-ui, sans-serif',
    );
  });

  test("Font dropdown 'Default' option → exec('removeFormat')", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Font family/i }));
    });
    const opt = await screen.findByRole("option", { name: /Default/i });
    await act(async () => { fireEvent.click(opt); });
    expect(document.execCommand).toHaveBeenCalledWith("removeFormat", false, undefined);
  });

  test("Size dropdown '16' option → exec('fontSize', '4')", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Font size/i }));
    });
    const opt = await screen.findByRole("option", { name: /^16$/ });
    await act(async () => { fireEvent.click(opt); });
    expect(document.execCommand).toHaveBeenCalledWith("fontSize", false, "4");
  });

  // ─── Tables ────────────────────────────────────────────────────────────

  test("Insert table button → insertHTML containing a 3x3 .wt-table", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Insert table/i }));
    });
    const calls = (document.execCommand as any).mock.calls.filter(
      (c: any[]) => c[0] === "insertHTML",
    );
    expect(calls.length).toBeGreaterThan(0);
    const html = calls[0][2] as string;
    expect(html).toContain("wt-table");
    expect(html).toContain("<thead>");
    expect(html).toContain("<th>");
    // 3 ths + 6 tds = 9 cells; sanity check on column count.
    const thCount = (html.match(/<th>/g) || []).length;
    const tdCount = (html.match(/<td>/g) || []).length;
    expect(thCount).toBe(3);
    expect(tdCount).toBe(6);
  });

  test("Paste with <table> markup is preserved through DOMPurify", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });

    const dirty = "<table><tr><td>cell-a</td><td>cell-b</td></tr></table>";
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = { getData: (t: string) => (t === "text/html" ? dirty : "") };
    await act(async () => { editor.dispatchEvent(ev); });

    const calls = (document.execCommand as any).mock.calls.filter(
      ([c]: [string]) => c === "insertHTML",
    );
    const cleaned = calls.at(-1)![2] as string;
    expect(cleaned).toContain("<table");
    expect(cleaned).toContain("cell-a");
    expect(cleaned).toContain("cell-b");
  });

  // ─── Task lists ────────────────────────────────────────────────────────

  test("Insert task list button → insertHTML containing ul.contains-task-list", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Task list/i }));
    });
    const calls = (document.execCommand as any).mock.calls.filter(
      (c: any[]) => c[0] === "insertHTML",
    );
    expect(calls.length).toBeGreaterThan(0);
    const html = calls[0][2] as string;
    expect(html).toContain('class="contains-task-list"');
    expect(html).toContain('type="checkbox"');
  });

  test("Checkbox click toggles the 'checked' ATTRIBUTE (not just property)", async () => {
    render(
      <WeeklyThoughts
        value='<ul class="contains-task-list"><li class="task-list-item"><input type="checkbox" /> task</li></ul>'
        onChange={() => {}}
      />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("input[type='checkbox']")).not.toBeNull());

    const checkbox = editor.querySelector<HTMLInputElement>("input[type='checkbox']")!;
    expect(checkbox.hasAttribute("checked")).toBe(false);

    await act(async () => { fireEvent.click(checkbox); });
    // Attribute (not property) — innerHTML serializes the attribute.
    expect(checkbox.hasAttribute("checked")).toBe(true);

    await act(async () => { fireEvent.click(checkbox); });
    expect(checkbox.hasAttribute("checked")).toBe(false);
  });

  // ─── DOMPurify hooks ───────────────────────────────────────────────────

  test("Paste with whitelisted YouTube iframe → preserved", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });

    const dirty = '<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>';
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = { getData: (t: string) => (t === "text/html" ? dirty : "") };
    await act(async () => { editor.dispatchEvent(ev); });

    const calls = (document.execCommand as any).mock.calls.filter(
      ([c]: [string]) => c === "insertHTML",
    );
    const cleaned = calls.at(-1)![2] as string;
    expect(cleaned).toContain("<iframe");
    expect(cleaned).toContain("www.youtube.com/embed/dQw4w9WgXcQ");
  });

  test("Paste with non-whitelisted iframe src → iframe entirely removed", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });

    const dirty = '<p>before</p><iframe src="https://evil.example/foo"></iframe><p>after</p>';
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = { getData: (t: string) => (t === "text/html" ? dirty : "") };
    await act(async () => { editor.dispatchEvent(ev); });

    const calls = (document.execCommand as any).mock.calls.filter(
      ([c]: [string]) => c === "insertHTML",
    );
    const cleaned = calls.at(-1)![2] as string;
    expect(cleaned).toContain("before");
    expect(cleaned).toContain("after");
    expect(cleaned).not.toContain("<iframe");
    expect(cleaned).not.toContain("evil.example");
  });

  test("Paste with input[type=text] → input entirely removed", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });

    const dirty = '<p>keep</p><input type="text" name="bad" />';
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = { getData: (t: string) => (t === "text/html" ? dirty : "") };
    await act(async () => { editor.dispatchEvent(ev); });

    const calls = (document.execCommand as any).mock.calls.filter(
      ([c]: [string]) => c === "insertHTML",
    );
    const cleaned = calls.at(-1)![2] as string;
    expect(cleaned).toContain("keep");
    expect(cleaned).not.toContain("<input");
  });

  test("Paste with input[type=checkbox][checked] → preserved", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });

    const dirty = '<ul><li><input type="checkbox" checked> done</li></ul>';
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = { getData: (t: string) => (t === "text/html" ? dirty : "") };
    await act(async () => { editor.dispatchEvent(ev); });

    const calls = (document.execCommand as any).mock.calls.filter(
      ([c]: [string]) => c === "insertHTML",
    );
    const cleaned = calls.at(-1)![2] as string;
    expect(cleaned).toContain('type="checkbox"');
    // DOMPurify normalizes boolean attrs to checked="" or just checked
    expect(cleaned).toMatch(/checked/);
  });

  test("Paste with inline style='margin-left' → style attribute stripped", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });

    const dirty = '<p style="margin-left: 40px; color: red;">indented</p>';
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = { getData: (t: string) => (t === "text/html" ? dirty : "") };
    await act(async () => { editor.dispatchEvent(ev); });

    const calls = (document.execCommand as any).mock.calls.filter(
      ([c]: [string]) => c === "insertHTML",
    );
    const cleaned = calls.at(-1)![2] as string;
    expect(cleaned).toContain("indented");
    expect(cleaned).not.toContain("margin-left");
    expect(cleaned).not.toContain("style=");
  });

  // ─── Indent normalizer ─────────────────────────────────────────────────

  test("normalizeIndentation: inline margin-left:80px → .indent-2 class", () => {
    const root = document.createElement("div");
    root.innerHTML = '<p style="margin-left: 80px">x</p>';
    normalizeIndentation(root);
    const p = root.querySelector("p")!;
    expect(p.classList.contains("indent-2")).toBe(true);
    expect(p.getAttribute("style") ?? "").not.toContain("margin-left");
  });

  test("normalizeIndentation: margin-left:200px → .indent-5", () => {
    const root = document.createElement("div");
    root.innerHTML = '<p style="margin-left: 200px">x</p>';
    normalizeIndentation(root);
    expect(root.querySelector("p")!.classList.contains("indent-5")).toBe(true);
  });

  test("normalizeIndentation: huge margin caps at .indent-6", () => {
    const root = document.createElement("div");
    root.innerHTML = '<p style="margin-left: 9999px">x</p>';
    normalizeIndentation(root);
    expect(root.querySelector("p")!.classList.contains("indent-6")).toBe(true);
  });

  test("normalizeIndentation: replaces an existing .indent-N rather than stacking", () => {
    const root = document.createElement("div");
    root.innerHTML = '<p class="indent-1" style="margin-left: 80px">x</p>';
    normalizeIndentation(root);
    const p = root.querySelector("p")!;
    expect(p.classList.contains("indent-1")).toBe(false);
    expect(p.classList.contains("indent-2")).toBe(true);
  });

  test("normalizeIndentation: no margin-left → no class change", () => {
    const root = document.createElement("div");
    root.innerHTML = "<p>x</p>";
    normalizeIndentation(root);
    const p = root.querySelector("p")!;
    expect(Array.from(p.classList).some(c => c.startsWith("indent-"))).toBe(false);
  });

  // ─── Popover open + value-change preservation (Phase 3.5 decision #2) ──

  test("link popover open + external value change → editor content survives", async () => {
    const { rerender } = render(
      <WeeklyThoughts value="<p>initial</p>" onChange={() => {}} />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.innerHTML).toContain("initial"));

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Insert link/i }));
    });
    // Sanity: popover is open
    expect(screen.getByRole("dialog", { name: /Insert link/i })).toBeInTheDocument();

    // External value change while popover is open
    rerender(<WeeklyThoughts value="<p>server update</p>" onChange={() => {}} />);
    expect(editor.innerHTML).toContain("initial");
    expect(editor.innerHTML).not.toContain("server update");
  });

  test("video popover open + external value change → editor content survives", async () => {
    const { rerender } = render(
      <WeeklyThoughts value="<p>initial</p>" onChange={() => {}} />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.innerHTML).toContain("initial"));

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Embed video/i }));
    });
    expect(screen.getByRole("dialog", { name: /Embed video/i })).toBeInTheDocument();

    rerender(<WeeklyThoughts value="<p>server update</p>" onChange={() => {}} />);
    expect(editor.innerHTML).toContain("initial");
    expect(editor.innerHTML).not.toContain("server update");
  });
});
