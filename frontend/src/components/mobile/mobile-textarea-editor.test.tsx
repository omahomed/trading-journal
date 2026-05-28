import { describe, test, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MobileTextareaEditor } from "./mobile-textarea-editor";

describe("MobileTextareaEditor — mount + hydration", () => {
  test("initialValue populates the textarea on mount", () => {
    render(
      <MobileTextareaEditor
        initialValue="<p>seed content</p>"
        onChange={() => {}}
      />,
    );
    const ta = screen.getByTestId("mobile-textarea-input") as HTMLTextAreaElement;
    expect(ta.value).toBe("<p>seed content</p>");
  });

  test("placeholder shows when textarea is empty", () => {
    render(
      <MobileTextareaEditor
        initialValue=""
        onChange={() => {}}
        placeholder="Type something…"
      />,
    );
    const ta = screen.getByTestId("mobile-textarea-input") as HTMLTextAreaElement;
    expect(ta.placeholder).toBe("Type something…");
  });

  test("empty value shows preview empty-state copy", () => {
    render(<MobileTextareaEditor initialValue="" onChange={() => {}} />);
    expect(screen.getByTestId("mobile-textarea-preview-empty")).toHaveTextContent(
      "Preview appears here",
    );
    expect(screen.queryByTestId("mobile-textarea-preview")).not.toBeInTheDocument();
  });

  test("re-hydrates when initialValue prop changes", () => {
    const { rerender } = render(
      <MobileTextareaEditor initialValue="<p>first</p>" onChange={() => {}} />,
    );
    const ta = screen.getByTestId("mobile-textarea-input") as HTMLTextAreaElement;
    expect(ta.value).toBe("<p>first</p>");
    rerender(
      <MobileTextareaEditor initialValue="<p>second</p>" onChange={() => {}} />,
    );
    expect(ta.value).toBe("<p>second</p>");
  });
});

describe("MobileTextareaEditor — onChange + onDirtyChange", () => {
  test("onChange fires with current value on textarea input", () => {
    const onChange = vi.fn();
    render(
      <MobileTextareaEditor initialValue="" onChange={onChange} />,
    );
    const ta = screen.getByTestId("mobile-textarea-input") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "hello" } });
    expect(onChange).toHaveBeenCalledWith("hello");
  });

  test("onDirtyChange fires true once when value diverges from initial", () => {
    const onDirtyChange = vi.fn();
    render(
      <MobileTextareaEditor
        initialValue="seed"
        onChange={() => {}}
        onDirtyChange={onDirtyChange}
      />,
    );
    const ta = screen.getByTestId("mobile-textarea-input") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "seedX" } });
    expect(onDirtyChange).toHaveBeenLastCalledWith(true);
  });

  test("onDirtyChange does not re-fire on every keystroke after dirty", () => {
    const onDirtyChange = vi.fn();
    render(
      <MobileTextareaEditor
        initialValue="seed"
        onChange={() => {}}
        onDirtyChange={onDirtyChange}
      />,
    );
    const ta = screen.getByTestId("mobile-textarea-input") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "a" } });
    fireEvent.change(ta, { target: { value: "ab" } });
    fireEvent.change(ta, { target: { value: "abc" } });
    const dirtyCalls = onDirtyChange.mock.calls.filter(([d]) => d === true);
    expect(dirtyCalls).toHaveLength(1);
  });

  test("onDirtyChange fires false when value returns to initial", () => {
    const onDirtyChange = vi.fn();
    render(
      <MobileTextareaEditor
        initialValue="seed"
        onChange={() => {}}
        onDirtyChange={onDirtyChange}
      />,
    );
    const ta = screen.getByTestId("mobile-textarea-input") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "different" } });
    fireEvent.change(ta, { target: { value: "seed" } });
    expect(onDirtyChange).toHaveBeenLastCalledWith(false);
  });
});

describe("MobileTextareaEditor — preview rendering", () => {
  test("plain text renders as text in preview", () => {
    render(
      <MobileTextareaEditor
        initialValue="just plain words"
        onChange={() => {}}
      />,
    );
    expect(screen.getByTestId("mobile-textarea-preview")).toHaveTextContent(
      "just plain words",
    );
  });

  test("renders <p>, <strong>, <em> HTML markup", () => {
    render(
      <MobileTextareaEditor
        initialValue="<p><strong>bold</strong> and <em>italic</em></p>"
        onChange={() => {}}
      />,
    );
    const preview = screen.getByTestId("mobile-textarea-preview");
    expect(preview.querySelector("strong")).toHaveTextContent("bold");
    expect(preview.querySelector("em")).toHaveTextContent("italic");
  });

  test("renders <ul> and <li> lists", () => {
    render(
      <MobileTextareaEditor
        initialValue="<ul><li>one</li><li>two</li></ul>"
        onChange={() => {}}
      />,
    );
    const preview = screen.getByTestId("mobile-textarea-preview");
    const items = preview.querySelectorAll("li");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("one");
    expect(items[1]).toHaveTextContent("two");
  });

  test("renders <h2> heading", () => {
    render(
      <MobileTextareaEditor
        initialValue="<h2>Section</h2>"
        onChange={() => {}}
      />,
    );
    const preview = screen.getByTestId("mobile-textarea-preview");
    expect(preview.querySelector("h2")).toHaveTextContent("Section");
  });

  test("renders <blockquote>", () => {
    render(
      <MobileTextareaEditor
        initialValue="<blockquote>thought</blockquote>"
        onChange={() => {}}
      />,
    );
    expect(
      screen.getByTestId("mobile-textarea-preview").querySelector("blockquote"),
    ).toHaveTextContent("thought");
  });

  test("renders <a> link with href", () => {
    render(
      <MobileTextareaEditor
        initialValue={`<p>see <a href="https://example.com">site</a></p>`}
        onChange={() => {}}
      />,
    );
    const a = screen
      .getByTestId("mobile-textarea-preview")
      .querySelector("a");
    expect(a).not.toBeNull();
    expect(a?.getAttribute("href")).toBe("https://example.com");
  });

  test("preserves class=\"text-color-X\" spans", () => {
    render(
      <MobileTextareaEditor
        initialValue={`<p><span class="text-color-emerald">green</span></p>`}
        onChange={() => {}}
      />,
    );
    const span = screen
      .getByTestId("mobile-textarea-preview")
      .querySelector("span.text-color-emerald");
    expect(span).not.toBeNull();
    expect(span).toHaveTextContent("green");
  });

  test("preserves class=\"indent-N\" on block elements", () => {
    render(
      <MobileTextareaEditor
        initialValue={`<p class="indent-2">indented</p>`}
        onChange={() => {}}
      />,
    );
    const p = screen
      .getByTestId("mobile-textarea-preview")
      .querySelector("p.indent-2");
    expect(p).not.toBeNull();
  });

  test("preview reflects textarea changes immediately", () => {
    render(<MobileTextareaEditor initialValue="" onChange={() => {}} />);
    const ta = screen.getByTestId("mobile-textarea-input") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "<p>typed</p>" } });
    expect(screen.getByTestId("mobile-textarea-preview")).toHaveTextContent(
      "typed",
    );
  });
});

describe("MobileTextareaEditor — sanitization", () => {
  test("strips <script> tags from preview", () => {
    render(
      <MobileTextareaEditor
        initialValue={`<p>ok</p><script>alert(1)</script>`}
        onChange={() => {}}
      />,
    );
    const preview = screen.getByTestId("mobile-textarea-preview");
    expect(preview.querySelector("script")).toBeNull();
    expect(preview).toHaveTextContent("ok");
  });

  test("strips onclick attributes", () => {
    render(
      <MobileTextareaEditor
        initialValue={`<p onclick="alert(1)">click</p>`}
        onChange={() => {}}
      />,
    );
    const p = screen
      .getByTestId("mobile-textarea-preview")
      .querySelector("p");
    expect(p?.hasAttribute("onclick")).toBe(false);
  });

  test("strips javascript: URLs from <a href>", () => {
    render(
      <MobileTextareaEditor
        initialValue={`<a href="javascript:alert(1)">x</a>`}
        onChange={() => {}}
      />,
    );
    const a = screen
      .getByTestId("mobile-textarea-preview")
      .querySelector("a");
    // DOMPurify either strips the href entirely or rewrites; in both
    // cases it must not be a javascript: URL.
    const href = a?.getAttribute("href") ?? "";
    expect(href.toLowerCase().startsWith("javascript:")).toBe(false);
  });

  test("allows inline elements that match the allow-list", () => {
    render(
      <MobileTextareaEditor
        initialValue={`<p><strong>b</strong><em>i</em><u>u</u></p>`}
        onChange={() => {}}
      />,
    );
    const preview = screen.getByTestId("mobile-textarea-preview");
    expect(preview.querySelector("strong")).not.toBeNull();
    expect(preview.querySelector("em")).not.toBeNull();
    expect(preview.querySelector("u")).not.toBeNull();
  });
});

describe("MobileTextareaEditor — external API parity", () => {
  // Surface-level type check that the props shape matches the deleted
  // MobileRichTextEditor's signature, so consumers of the form
  // <MobileTextareaEditor initialValue=... onChange=... onDirtyChange=...
  // placeholder=... /> compile and behave the same way.
  test("accepts the same props as the deleted MobileRichTextEditor", () => {
    const props = {
      initialValue: "<p>seed</p>",
      onChange: vi.fn(),
      onDirtyChange: vi.fn(),
      placeholder: "ph",
    };
    render(<MobileTextareaEditor {...props} />);
    const ta = screen.getByTestId("mobile-textarea-input") as HTMLTextAreaElement;
    expect(ta.value).toBe("<p>seed</p>");
    expect(ta.placeholder).toBe("ph");
  });
});
