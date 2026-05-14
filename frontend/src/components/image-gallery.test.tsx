import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, afterEach } from "vitest";

import { ImageGallery, type ImageGalleryItem } from "./image-gallery";

const IMAGE_ITEMS: ImageGalleryItem[] = [
  { id: 1, url: "https://cdn.example.com/a.png", filename: "a.png" },
  { id: 2, url: "https://cdn.example.com/b.png", filename: "b.png" },
];

describe("ImageGallery", () => {
  afterEach(() => { vi.restoreAllMocks(); });

  // ─── Empty state ──────────────────────────────────────────────────────

  test("renders nothing when items is empty and no emptyState provided", () => {
    const { container } = render(<ImageGallery items={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  test("renders emptyState when items is empty and prop provided", () => {
    render(<ImageGallery items={[]} emptyState={<div>No charts yet</div>} />);
    expect(screen.getByText(/No charts yet/i)).toBeInTheDocument();
  });

  // ─── Grid rendering ───────────────────────────────────────────────────

  test("renders one card per item with filename caption", () => {
    render(<ImageGallery items={IMAGE_ITEMS} />);
    expect(screen.getByAltText("a.png")).toBeInTheDocument();
    expect(screen.getByAltText("b.png")).toBeInTheDocument();
    // Caption rows
    expect(screen.getByText("a.png", { selector: "span" })).toBeInTheDocument();
    expect(screen.getByText("b.png", { selector: "span" })).toBeInTheDocument();
  });

  test("single-image grid uses 1-column layout", () => {
    render(<ImageGallery items={[IMAGE_ITEMS[0]]} />);
    const grid = screen.getByRole("list");
    expect(grid).toHaveStyle({ gridTemplateColumns: "1fr" });
  });

  test("multi-image grid uses 2-column layout", () => {
    render(<ImageGallery items={IMAGE_ITEMS} />);
    const grid = screen.getByRole("list");
    expect(grid).toHaveStyle({ gridTemplateColumns: "1fr 1fr" });
  });

  test("forceColumns prop overrides the default layout heuristic", () => {
    render(<ImageGallery items={[IMAGE_ITEMS[0]]} forceColumns={2} />);
    const grid = screen.getByRole("list");
    expect(grid).toHaveStyle({ gridTemplateColumns: "1fr 1fr" });
  });

  // ─── Image click ──────────────────────────────────────────────────────

  test("clicking image fires onImageClick with the index", () => {
    const onImageClick = vi.fn();
    render(<ImageGallery items={IMAGE_ITEMS} onImageClick={onImageClick} />);
    act(() => { fireEvent.click(screen.getByAltText("b.png")); });
    expect(onImageClick).toHaveBeenCalledWith(1);
  });

  test("when onImageClick is omitted, the image is not interactive", () => {
    render(<ImageGallery items={IMAGE_ITEMS} />);
    // The clickable wrapper has cursor-pointer only when onImageClick is set.
    const img = screen.getByAltText("a.png");
    const wrapper = img.parentElement!;
    expect(wrapper.className).not.toContain("cursor-pointer");
  });

  // ─── Delete affordance — Phase 4.5 inline two-click confirm ─────────

  test("First click on Delete arms the button — text swaps to Confirm?", () => {
    const onDelete = vi.fn();
    render(<ImageGallery items={IMAGE_ITEMS} onDelete={onDelete} />);
    const delBtn = screen.getByRole("button", { name: /^Delete a.png$/i });
    expect(delBtn).toHaveTextContent("Delete");
    act(() => { fireEvent.click(delBtn); });
    // onDelete must NOT fire on the first click.
    expect(onDelete).not.toHaveBeenCalled();
    // Armed state — aria-label swaps to "Confirm delete a.png".
    const armedBtn = screen.getByRole("button", { name: /Confirm delete a.png/i });
    expect(armedBtn).toHaveTextContent(/Confirm/);
  });

  test("Second click on Confirm? fires onDelete with the item id", () => {
    const onDelete = vi.fn();
    render(<ImageGallery items={IMAGE_ITEMS} onDelete={onDelete} />);
    const delBtn = screen.getByRole("button", { name: /^Delete a.png$/i });
    act(() => { fireEvent.click(delBtn); });
    const confirmBtn = screen.getByRole("button", { name: /Confirm delete a.png/i });
    act(() => { fireEvent.click(confirmBtn); });
    expect(onDelete).toHaveBeenCalledWith(1);
    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  test("Click Delete then wait 2s → button resets to Delete; onDelete NOT fired", () => {
    vi.useFakeTimers();
    const onDelete = vi.fn();
    render(<ImageGallery items={IMAGE_ITEMS} onDelete={onDelete} />);
    act(() => { fireEvent.click(screen.getByRole("button", { name: /^Delete a.png$/i })); });
    expect(screen.getByRole("button", { name: /Confirm delete a.png/i })).toBeInTheDocument();
    // Advance past the 2s timeout.
    act(() => { vi.advanceTimersByTime(2100); });
    expect(screen.queryByRole("button", { name: /Confirm delete a.png/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Delete a.png$/i })).toBeInTheDocument();
    expect(onDelete).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  test("Click Delete on item B while item A is armed → only B is armed", () => {
    render(<ImageGallery items={IMAGE_ITEMS} onDelete={() => {}} />);
    // Arm A
    act(() => { fireEvent.click(screen.getByRole("button", { name: /^Delete a.png$/i })); });
    expect(screen.getByRole("button", { name: /Confirm delete a.png/i })).toBeInTheDocument();
    // Click B — A should un-arm, B should arm
    act(() => { fireEvent.click(screen.getByRole("button", { name: /^Delete b.png$/i })); });
    expect(screen.queryByRole("button", { name: /Confirm delete a.png/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Confirm delete b.png/i })).toBeInTheDocument();
    // A should be back to Delete state
    expect(screen.getByRole("button", { name: /^Delete a.png$/i })).toBeInTheDocument();
  });

  test("Component unmounts while armed → no error from pending timer", () => {
    const { unmount } = render(<ImageGallery items={IMAGE_ITEMS} onDelete={() => {}} />);
    act(() => { fireEvent.click(screen.getByRole("button", { name: /^Delete a.png$/i })); });
    // No assertion needed beyond "unmount doesn't throw". Vitest will
    // fail the test if cleanup leaks an unhandled exception.
    expect(() => unmount()).not.toThrow();
  });

  test("Delete button absent when onDelete is omitted", () => {
    render(<ImageGallery items={IMAGE_ITEMS} />);
    expect(screen.queryByRole("button", { name: /Delete a.png/i })).not.toBeInTheDocument();
  });

  // ─── PDF auto-detection ───────────────────────────────────────────────

  test("PDF items render as a clickable link instead of an img", () => {
    render(
      <ImageGallery
        items={[
          { id: 1, url: "https://cdn.example.com/doc.pdf", filename: "doc.pdf" },
        ]}
      />,
    );
    // No img for PDF
    expect(screen.queryByAltText("doc.pdf")).not.toBeInTheDocument();
    // Link is present with "Open PDF" text
    const link = screen.getByRole("link", { name: /Open PDF/i });
    expect(link).toHaveAttribute("href", "https://cdn.example.com/doc.pdf");
    expect(link).toHaveAttribute("target", "_blank");
  });

  test("PDF detection matches url with query string", () => {
    render(
      <ImageGallery
        items={[
          { id: 1, url: "https://cdn.example.com/doc.pdf?token=abc", filename: "doc" },
        ]}
      />,
    );
    expect(screen.getByRole("link", { name: /Open PDF/i })).toBeInTheDocument();
  });

  // ─── Uploading state ──────────────────────────────────────────────────

  test("uploading items render with the placeholder visual + no delete button", () => {
    render(
      <ImageGallery
        items={[
          ...IMAGE_ITEMS,
          { id: -1, url: "", filename: "new.png", uploading: true },
        ]}
        onDelete={() => {}}
      />,
    );
    // The placeholder card uses data-testid="upload-placeholder"
    const placeholder = screen.getByTestId("upload-placeholder");
    expect(placeholder).toBeInTheDocument();
    expect(placeholder).toHaveTextContent(/Uploading new\.png/i);
    // No Delete button for the uploading item (only the real items)
    expect(screen.queryByRole("button", { name: /Delete new\.png/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Delete a.png/i })).toBeInTheDocument();
  });
});
