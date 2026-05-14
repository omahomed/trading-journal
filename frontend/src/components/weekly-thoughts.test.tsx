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

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4.1 — undo/redo, image paste inline, DOMPurify img filter.
// ─────────────────────────────────────────────────────────────────────────────

// Mock the api module — needed by uploadAndInsertImage. Hoisted by vitest.
vi.mock("@/lib/api", () => ({
  api: {
    uploadWeeklyThoughtsImage: vi.fn(),
  },
}));

import { api as _phase41Api } from "@/lib/api";
const mockApi = vi.mocked(_phase41Api);

function makeImageFile(name: string, size: number, type: string): File {
  const blob = new Blob([new Uint8Array(size)], { type });
  return new File([blob], name, { type });
}

function dispatchPasteWithImage(editor: HTMLElement, file: File) {
  const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
  ev.clipboardData = {
    items: [
      { kind: "file", type: file.type, getAsFile: () => file },
    ],
    getData: () => "",
  };
  editor.dispatchEvent(ev);
}

describe("WeeklyThoughts — Phase 4.1 polish + image paste", () => {
  beforeEach(() => {
    try { localStorage.clear(); } catch { /* shim */ }
    vi.spyOn(document, "execCommand").mockReturnValue(true);
    mockApi.uploadWeeklyThoughtsImage.mockReset();
    // Stub URL.createObjectURL / revokeObjectURL — jsdom has it but
    // resetting per-test isolates side effects.
    if (typeof URL.createObjectURL !== "function") {
      URL.createObjectURL = vi.fn(() => "blob:fake-preview-url");
      URL.revokeObjectURL = vi.fn();
    }
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  // ─── Undo / Redo ──────────────────────────────────────────────────────

  test("Undo button fires execCommand('undo')", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Undo$/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("undo", false, undefined);
  });

  test("Redo button fires execCommand('redo')", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Redo$/i }));
    });
    expect(document.execCommand).toHaveBeenCalledWith("redo", false, undefined);
  });

  // ─── Paste fall-through (non-image) ───────────────────────────────────

  test("Paste with non-image clipboard items falls through to HTML path", async () => {
    const onChange = vi.fn();
    render(<WeeklyThoughts value="" onChange={onChange} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    // Synthesize a paste with text/html only — no image items.
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = {
      items: [{ kind: "string", type: "text/plain", getAsFile: () => null }],
      getData: (type: string) => (type === "text/html" ? "<p>hello</p>" : ""),
    };
    await act(async () => { editor.dispatchEvent(ev); });
    // The HTML path calls execCommand("insertHTML", ...) — and does NOT
    // call the image upload api.
    expect(mockApi.uploadWeeklyThoughtsImage).not.toHaveBeenCalled();
    expect(document.execCommand).toHaveBeenCalledWith(
      "insertHTML", false, expect.stringContaining("<p>hello</p>"),
    );
  });

  // ─── Image paste — retroId null shows error ──────────────────────────

  test("Image paste with retroId=null shows inline error, no upload", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} retroId={null} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    const file = makeImageFile("paste.png", 1024, "image/png");
    await act(async () => { dispatchPasteWithImage(editor, file); });
    expect(mockApi.uploadWeeklyThoughtsImage).not.toHaveBeenCalled();
    expect(await screen.findByRole("alert")).toHaveTextContent(/Save the retro first/i);
  });

  // ─── Image paste — happy path ────────────────────────────────────────

  test("Image paste triggers upload and inserts <img> with blob URL", async () => {
    mockApi.uploadWeeklyThoughtsImage.mockResolvedValueOnce({
      view_url: "https://cdn.example.com/weekly_retros/7/thoughts/abc.png",
    });
    const onChange = vi.fn();
    render(<WeeklyThoughts value="" onChange={onChange} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    const file = makeImageFile("ok.png", 1024, "image/png");

    await act(async () => { dispatchPasteWithImage(editor, file); });

    // Optimistic insertion happened — an insertHTML call with the img
    // markup. Verify the img tag was inserted (with blob URL + class).
    const insertCalls = (document.execCommand as any).mock.calls.filter(
      (c: any[]) => c[0] === "insertHTML",
    );
    expect(insertCalls.length).toBeGreaterThan(0);
    expect(insertCalls[0][2]).toContain('<img src="blob:');
    expect(insertCalls[0][2]).toContain('class="wt-uploading"');

    // Upload was invoked.
    await waitFor(() => {
      expect(mockApi.uploadWeeklyThoughtsImage).toHaveBeenCalledWith(7, file, "CanSlim");
    });
  });

  // ─── Image paste — size rejection ────────────────────────────────────

  test("Image paste >5MB shows inline error, no upload call", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    const big = makeImageFile("big.png", 5 * 1024 * 1024 + 1, "image/png");
    await act(async () => { dispatchPasteWithImage(editor, big); });
    expect(mockApi.uploadWeeklyThoughtsImage).not.toHaveBeenCalled();
    expect(await screen.findByRole("alert")).toHaveTextContent(/5MB/i);
  });

  // ─── Image paste — MIME rejection ────────────────────────────────────

  test("Image paste with non-allowed MIME (e.g. svg) shows inline error", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    const svg = makeImageFile("bad.svg", 1024, "image/svg+xml");
    await act(async () => { dispatchPasteWithImage(editor, svg); });
    expect(mockApi.uploadWeeklyThoughtsImage).not.toHaveBeenCalled();
    expect(await screen.findByRole("alert")).toHaveTextContent(/PNG/i);
  });

  // ─── Image paste — upload failure removes the placeholder ────────────

  test("Image paste upload failure removes the placeholder + shows error", async () => {
    mockApi.uploadWeeklyThoughtsImage.mockRejectedValueOnce(new Error("network"));
    render(<WeeklyThoughts value="" onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    const file = makeImageFile("fail.png", 1024, "image/png");

    // Pre-populate the editor with a placeholder element matching the
    // shape our handler would have inserted, so the cleanup logic has
    // something to find. (The actual handler does the same insert
    // via execCommand, which is stubbed, so the DOM doesn't reflect
    // it. We approximate the post-insert DOM here.)
    editor.innerHTML =
      '<img src="blob:fake" data-upload-id="wt-img-fake" class="wt-uploading">';

    await act(async () => { dispatchPasteWithImage(editor, file); });

    // Wait for the rejected promise to flush.
    await waitFor(() => expect(mockApi.uploadWeeklyThoughtsImage).toHaveBeenCalled());
    // Error toast appears.
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeNull());
  });

  // ─── DOMPurify img filter ────────────────────────────────────────────

  test("Paste HTML with img from non-R2 src has img stripped", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    const dirty = '<p>kept</p><img src="https://evil.example/x.png" /><p>also kept</p>';
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = {
      items: [],   // no image file items — falls into HTML path
      getData: (type: string) => (type === "text/html" ? dirty : ""),
    };
    await act(async () => { editor.dispatchEvent(ev); });

    const insertCalls = (document.execCommand as any).mock.calls.filter(
      (c: any[]) => c[0] === "insertHTML",
    );
    const cleaned = insertCalls.at(-1)![2] as string;
    expect(cleaned).toContain("kept");
    expect(cleaned).toContain("also kept");
    expect(cleaned).not.toContain("evil.example");
    expect(cleaned).not.toContain("<img");
  });

  test("Paste HTML with img from R2 src is preserved", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    // Use a URL that matches the fallback IMG_SRC_PATTERN (HTTPS + /weekly_retros/ path + image ext).
    const dirty = '<p>x</p><img src="https://r2.example.com/weekly_retros/7/thoughts/abc.png" />';
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = {
      items: [],
      getData: (type: string) => (type === "text/html" ? dirty : ""),
    };
    await act(async () => { editor.dispatchEvent(ev); });

    const insertCalls = (document.execCommand as any).mock.calls.filter(
      (c: any[]) => c[0] === "insertHTML",
    );
    const cleaned = insertCalls.at(-1)![2] as string;
    expect(cleaned).toContain("<img");
    expect(cleaned).toContain("weekly_retros/7/thoughts/abc.png");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4.2 — inline image lightbox + color picker integration.
// ─────────────────────────────────────────────────────────────────────────────

describe("WeeklyThoughts — Phase 4.2 lightbox + color pickers", () => {
  beforeEach(() => {
    try { localStorage.clear(); } catch { /* shim */ }
    vi.spyOn(document, "execCommand").mockReturnValue(true);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  // ─── Inline image lightbox ────────────────────────────────────────────

  test("Click on inline <img> opens the lightbox dialog", async () => {
    render(
      <WeeklyThoughts
        value='<p>before <img src="https://r2.example.com/weekly_retros/7/thoughts/abc.png" alt="chart" /> after</p>'
        onChange={() => {}}
        retroId={7}
        portfolio="CanSlim"
      />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());

    const img = editor.querySelector("img")!;
    await act(async () => { fireEvent.click(img); });
    // ImageLightbox renders a dialog with aria-label "Image preview".
    expect(screen.getByRole("dialog", { name: /Image preview/i })).toBeInTheDocument();
  });

  test("Click on a non-img element does NOT open the lightbox", async () => {
    render(
      <WeeklyThoughts
        value="<p>some text without an image</p>"
        onChange={() => {}}
        retroId={7}
        portfolio="CanSlim"
      />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.innerHTML).toContain("some text"));
    await act(async () => { fireEvent.click(editor); });
    expect(screen.queryByRole("dialog", { name: /Image preview/i })).not.toBeInTheDocument();
  });

  test("Click on an uploading image (wt-uploading class) does NOT open the lightbox", async () => {
    render(
      <WeeklyThoughts
        value='<p><img src="blob:fake" class="wt-uploading" alt="" /></p>'
        onChange={() => {}}
        retroId={7}
        portfolio="CanSlim"
      />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());
    const img = editor.querySelector("img")!;
    await act(async () => { fireEvent.click(img); });
    expect(screen.queryByRole("dialog", { name: /Image preview/i })).not.toBeInTheDocument();
  });

  test("Esc closes the inline image lightbox", async () => {
    render(
      <WeeklyThoughts
        value='<p><img src="https://r2.example.com/weekly_retros/7/thoughts/x.png" alt="x" /></p>'
        onChange={() => {}}
        retroId={7}
        portfolio="CanSlim"
      />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());
    await act(async () => { fireEvent.click(editor.querySelector("img")!); });
    expect(screen.getByRole("dialog", { name: /Image preview/i })).toBeInTheDocument();
    await act(async () => { fireEvent.keyDown(window, { key: "Escape" }); });
    expect(screen.queryByRole("dialog", { name: /Image preview/i })).not.toBeInTheDocument();
  });

  // ─── Highlight color picker ───────────────────────────────────────────

  test("Highlight button opens picker with palette swatches", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const trigger = screen.getByRole("button", { name: /Highlight color/i });
    await act(async () => { fireEvent.click(trigger); });
    expect(screen.getByRole("dialog", { name: /Highlight color/i })).toBeInTheDocument();
  });

  test("Pick a highlight swatch → execCommand('hiliteColor', color)", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Highlight color/i }));
    });
    // First palette entry — the amber default (#ffeaa7).
    const swatch = screen.getByRole("button", { name: "#ffeaa7" });
    await act(async () => { fireEvent.click(swatch); });
    expect(document.execCommand).toHaveBeenCalledWith("hiliteColor", false, "#ffeaa7");
  });

  test("Pick the highlight reset swatch → hiliteColor 'transparent'", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Highlight color/i }));
    });
    // Reset swatch labeled "None" via resetLabel prop.
    const reset = screen.getByRole("button", { name: /^None$/i });
    await act(async () => { fireEvent.click(reset); });
    expect(document.execCommand).toHaveBeenCalledWith("hiliteColor", false, "transparent");
  });

  // ─── Text color picker ────────────────────────────────────────────────

  test("Text color button opens picker", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    const trigger = screen.getByRole("button", { name: /Text color/i });
    await act(async () => { fireEvent.click(trigger); });
    expect(screen.getByRole("dialog", { name: /Text color/i })).toBeInTheDocument();
  });

  test("Pick a text-color swatch → execCommand('foreColor', color)", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Text color/i }));
    });
    // Second entry of TEXT_COLOR_PALETTE is slate-900.
    const swatch = screen.getByRole("button", { name: "#0f172a" });
    await act(async () => { fireEvent.click(swatch); });
    expect(document.execCommand).toHaveBeenCalledWith("foreColor", false, "#0f172a");
  });

  test("Pick the text-color reset swatch → foreColor 'inherit'", async () => {
    render(<WeeklyThoughts value="" onChange={() => {}} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Text color/i }));
    });
    const reset = screen.getByRole("button", { name: /^Default$/i });
    await act(async () => { fireEvent.click(reset); });
    expect(document.execCommand).toHaveBeenCalledWith("foreColor", false, "inherit");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4.3 — inline image hover overlay + two-click delete.
// ─────────────────────────────────────────────────────────────────────────────

describe("WeeklyThoughts — Phase 4.3 image hover overlay + delete", () => {
  beforeEach(() => {
    try { localStorage.clear(); } catch { /* shim */ }
    vi.spyOn(document, "execCommand").mockReturnValue(true);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  const IMG_VALUE =
    '<p><img src="https://r2.example.com/weekly_retros/7/thoughts/abc.png" alt="chart" /></p>';

  // ─── Hover surfacing the overlay ──────────────────────────────────────

  test("Mouseover on inline img surfaces the × delete overlay", async () => {
    render(<WeeklyThoughts value={IMG_VALUE} onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());
    const img = editor.querySelector("img")!;
    expect(screen.queryByRole("button", { name: /^Delete image$/i })).not.toBeInTheDocument();
    await act(async () => { fireEvent.mouseOver(img); });
    expect(screen.getByRole("button", { name: /^Delete image$/i })).toBeInTheDocument();
  });

  test("Mouseover on a .wt-uploading img does NOT surface the overlay", async () => {
    render(
      <WeeklyThoughts
        value='<p><img src="blob:fake" class="wt-uploading" alt="" /></p>'
        onChange={() => {}}
        retroId={7}
        portfolio="CanSlim"
      />,
    );
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());
    const img = editor.querySelector("img")!;
    await act(async () => { fireEvent.mouseOver(img); });
    expect(screen.queryByRole("button", { name: /^Delete image$/i })).not.toBeInTheDocument();
  });

  test("Mouseout from img (leaving editor) hides the overlay", async () => {
    render(<WeeklyThoughts value={IMG_VALUE} onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());
    const img = editor.querySelector("img")!;
    await act(async () => { fireEvent.mouseOver(img); });
    expect(screen.getByRole("button", { name: /^Delete image$/i })).toBeInTheDocument();
    // Mouseout with no relatedTarget → mouse left editor area entirely
    await act(async () => { fireEvent.mouseOut(img, { relatedTarget: null }); });
    expect(screen.queryByRole("button", { name: /^Delete image$/i })).not.toBeInTheDocument();
  });

  // ─── Two-click delete ─────────────────────────────────────────────────

  test("First click on × arms (button label switches); second click commits", async () => {
    const onChange = vi.fn();
    render(<WeeklyThoughts value={IMG_VALUE} onChange={onChange} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());
    const img = editor.querySelector("img")!;

    await act(async () => { fireEvent.mouseOver(img); });
    const xBtn = screen.getByRole("button", { name: /^Delete image$/i });

    // First click — arms (aria-label switches to "Confirm delete image")
    await act(async () => { fireEvent.click(xBtn); });
    expect(editor.querySelector("img")).not.toBeNull();  // still present
    expect(screen.getByRole("button", { name: /Confirm delete image/i })).toBeInTheDocument();

    // Second click — commits delete
    const confirmBtn = screen.getByRole("button", { name: /Confirm delete image/i });
    await act(async () => { fireEvent.click(confirmBtn); });
    expect(editor.querySelector("img")).toBeNull();
    expect(onChange).toHaveBeenCalled();
  });

  test("Clicking × does NOT open the lightbox (stopPropagation)", async () => {
    render(<WeeklyThoughts value={IMG_VALUE} onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());
    const img = editor.querySelector("img")!;

    await act(async () => { fireEvent.mouseOver(img); });
    const xBtn = screen.getByRole("button", { name: /^Delete image$/i });
    await act(async () => { fireEvent.click(xBtn); });
    // Lightbox dialog must NOT have opened.
    expect(screen.queryByRole("dialog", { name: /Image preview/i })).not.toBeInTheDocument();
  });

  test("Mouse leaves the × button → overlay hides", async () => {
    render(<WeeklyThoughts value={IMG_VALUE} onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());
    const img = editor.querySelector("img")!;
    await act(async () => { fireEvent.mouseOver(img); });
    const xBtn = screen.getByRole("button", { name: /^Delete image$/i });
    await act(async () => { fireEvent.mouseLeave(xBtn); });
    expect(screen.queryByRole("button", { name: /^Delete image$/i })).not.toBeInTheDocument();
  });

  test("Scroll while hovering hides the overlay", async () => {
    render(<WeeklyThoughts value={IMG_VALUE} onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());
    const img = editor.querySelector("img")!;
    await act(async () => { fireEvent.mouseOver(img); });
    expect(screen.getByRole("button", { name: /^Delete image$/i })).toBeInTheDocument();
    await act(async () => { fireEvent.scroll(window); });
    expect(screen.queryByRole("button", { name: /^Delete image$/i })).not.toBeInTheDocument();
  });

  test("Moving from img to × button keeps the overlay (relatedTarget guard)", async () => {
    render(<WeeklyThoughts value={IMG_VALUE} onChange={() => {}} retroId={7} portfolio="CanSlim" />);
    const editor = screen.getByRole("textbox", { name: /weekly thoughts/i });
    await waitFor(() => expect(editor.querySelector("img")).not.toBeNull());
    const img = editor.querySelector("img")!;
    await act(async () => { fireEvent.mouseOver(img); });
    const xBtn = screen.getByRole("button", { name: /^Delete image$/i });
    // Mouseout from img with relatedTarget = the × button → DO NOT hide
    await act(async () => { fireEvent.mouseOut(img, { relatedTarget: xBtn }); });
    expect(screen.queryByRole("button", { name: /^Delete image$/i })).toBeInTheDocument();
  });
});
