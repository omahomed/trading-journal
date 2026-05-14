import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, afterEach } from "vitest";

import { ImageLightbox, type LightboxImage } from "./image-lightbox";

const IMAGES: LightboxImage[] = [
  { url: "https://cdn.example.com/a.png", alt: "a" },
  { url: "https://cdn.example.com/b.png", alt: "b" },
  { url: "https://cdn.example.com/c.png", alt: "c" },
];

describe("ImageLightbox", () => {
  afterEach(() => { vi.restoreAllMocks(); });

  // ─── Render state ─────────────────────────────────────────────────────

  test("renders nothing when activeIndex is null", () => {
    const { container } = render(
      <ImageLightbox images={IMAGES} activeIndex={null} onClose={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  test("renders dialog with current image when activeIndex is set", () => {
    render(<ImageLightbox images={IMAGES} activeIndex={1} onClose={() => {}} />);
    const dialog = screen.getByRole("dialog", { name: /Image preview/i });
    expect(dialog).toBeInTheDocument();
    expect(dialog.querySelector("img")?.getAttribute("src"))
      .toBe("https://cdn.example.com/b.png");
  });

  test("custom ariaLabel propagates to the dialog", () => {
    render(
      <ImageLightbox
        images={IMAGES}
        activeIndex={0}
        onClose={() => {}}
        ariaLabel="Custom preview"
      />,
    );
    expect(screen.getByRole("dialog", { name: /Custom preview/i })).toBeInTheDocument();
  });

  // ─── Closing ──────────────────────────────────────────────────────────

  test("Esc fires onClose", () => {
    const onClose = vi.fn();
    render(<ImageLightbox images={IMAGES} activeIndex={0} onClose={onClose} />);
    act(() => { fireEvent.keyDown(window, { key: "Escape" }); });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("Click on backdrop fires onClose", () => {
    const onClose = vi.fn();
    render(<ImageLightbox images={IMAGES} activeIndex={0} onClose={onClose} />);
    const dialog = screen.getByRole("dialog");
    act(() => { fireEvent.click(dialog); });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("Click on the inner image does NOT close (stopPropagation)", () => {
    const onClose = vi.fn();
    render(<ImageLightbox images={IMAGES} activeIndex={0} onClose={onClose} />);
    const img = screen.getByRole("dialog").querySelector("img")!;
    act(() => { fireEvent.click(img); });
    expect(onClose).not.toHaveBeenCalled();
  });

  // ─── Navigation ───────────────────────────────────────────────────────

  test("ArrowRight calls onNavigate with the next index", () => {
    const onNavigate = vi.fn();
    render(
      <ImageLightbox
        images={IMAGES}
        activeIndex={0}
        onClose={() => {}}
        onNavigate={onNavigate}
      />,
    );
    act(() => { fireEvent.keyDown(window, { key: "ArrowRight" }); });
    expect(onNavigate).toHaveBeenCalledWith(1);
  });

  test("ArrowLeft calls onNavigate with the previous index", () => {
    const onNavigate = vi.fn();
    render(
      <ImageLightbox
        images={IMAGES}
        activeIndex={1}
        onClose={() => {}}
        onNavigate={onNavigate}
      />,
    );
    act(() => { fireEvent.keyDown(window, { key: "ArrowLeft" }); });
    expect(onNavigate).toHaveBeenCalledWith(0);
  });

  test("ArrowRight wraps around from the last index to 0", () => {
    const onNavigate = vi.fn();
    render(
      <ImageLightbox
        images={IMAGES}
        activeIndex={2}
        onClose={() => {}}
        onNavigate={onNavigate}
      />,
    );
    act(() => { fireEvent.keyDown(window, { key: "ArrowRight" }); });
    expect(onNavigate).toHaveBeenCalledWith(0);
  });

  test("ArrowLeft wraps around from index 0 to the last", () => {
    const onNavigate = vi.fn();
    render(
      <ImageLightbox
        images={IMAGES}
        activeIndex={0}
        onClose={() => {}}
        onNavigate={onNavigate}
      />,
    );
    act(() => { fireEvent.keyDown(window, { key: "ArrowLeft" }); });
    expect(onNavigate).toHaveBeenCalledWith(2);
  });

  test("Arrow keys are no-ops when onNavigate is omitted (single-image mode)", () => {
    // No onNavigate prop → arrows must NOT throw or do anything.
    const onClose = vi.fn();
    render(
      <ImageLightbox
        images={[IMAGES[0]]}
        activeIndex={0}
        onClose={onClose}
      />,
    );
    act(() => {
      fireEvent.keyDown(window, { key: "ArrowRight" });
      fireEvent.keyDown(window, { key: "ArrowLeft" });
    });
    // onClose still works.
    expect(onClose).not.toHaveBeenCalled();
    act(() => { fireEvent.keyDown(window, { key: "Escape" }); });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test("Arrow keys are no-ops when images.length is 1 even with onNavigate", () => {
    // No navigation makes sense with one image — guard rejects arrows.
    const onNavigate = vi.fn();
    render(
      <ImageLightbox
        images={[IMAGES[0]]}
        activeIndex={0}
        onClose={() => {}}
        onNavigate={onNavigate}
      />,
    );
    act(() => {
      fireEvent.keyDown(window, { key: "ArrowRight" });
      fireEvent.keyDown(window, { key: "ArrowLeft" });
    });
    expect(onNavigate).not.toHaveBeenCalled();
  });

  // ─── Defensive ────────────────────────────────────────────────────────

  test("renders null when activeIndex is out of bounds", () => {
    const { container } = render(
      <ImageLightbox images={IMAGES} activeIndex={99} onClose={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
