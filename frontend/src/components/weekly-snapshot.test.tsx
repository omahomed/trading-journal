import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";

// Mock the api module before importing the component — match the Phase 1
// tag-picker test pattern.
vi.mock("@/lib/api", () => ({
  api: {
    listWeeklyRetroSnapshots: vi.fn(),
    uploadWeeklyRetroSnapshot: vi.fn(),
    deleteWeeklyRetroSnapshot: vi.fn(),
  },
}));

import { api } from "@/lib/api";
import { WeeklySnapshot } from "./weekly-snapshot";

// Test helpers ---------------------------------------------------------------

function makeRow(overrides: Partial<{
  id: number;
  view_url: string;
  file_name: string;
  storage_ref: string;
}> = {}) {
  return {
    id: 1,
    weekly_retro_id: 7,
    storage_ref: "weekly_retros/7/a.png",
    view_url: "https://cdn.example.com/weekly_retros/7/a.png",
    file_name: "a.png",
    mime_type: "image/png",
    file_size_bytes: 1000,
    width: null,
    height: null,
    sort_order: 0,
    caption: "",
    created_at: "2026-05-13T12:00:00",
    ...overrides,
  };
}

function makeFile(name: string, size: number, type: string): File {
  const blob = new Blob([new Uint8Array(size)], { type });
  return new File([blob], name, { type });
}

const mockApi = vi.mocked(api);

describe("WeeklySnapshot — Phase 4", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockApi.listWeeklyRetroSnapshots.mockResolvedValue([]);
  });

  afterEach(() => {
    vi.useRealTimers();          // belt-and-suspenders: ensure no test leaves fake timers active
    vi.restoreAllMocks();
  });

  // ─── Disabled (retroId null) ───────────────────────────────────────────

  test("Drop zone disabled when retroId is null", () => {
    render(<WeeklySnapshot retroId={null} portfolio="CanSlim" />);
    const zone = screen.getByRole("button", { name: /Upload snapshot/i });
    expect(zone).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText(/Save the retro first/i)).toBeInTheDocument();
    // listWeeklyRetroSnapshots NOT called when retroId is null.
    expect(mockApi.listWeeklyRetroSnapshots).not.toHaveBeenCalled();
  });

  test("Click on disabled drop zone is a no-op", async () => {
    render(<WeeklySnapshot retroId={null} portfolio="CanSlim" />);
    const zone = screen.getByRole("button", { name: /Upload snapshot/i });
    await act(async () => { fireEvent.click(zone); });
    expect(mockApi.uploadWeeklyRetroSnapshot).not.toHaveBeenCalled();
  });

  // ─── Initial fetch ─────────────────────────────────────────────────────

  test("Fetches snapshots on mount when retroId is set", async () => {
    mockApi.listWeeklyRetroSnapshots.mockResolvedValueOnce([
      makeRow({ id: 1, file_name: "a.png" }),
      makeRow({ id: 2, file_name: "b.png", view_url: "https://cdn.example.com/b.png" }),
    ]);
    render(<WeeklySnapshot retroId={7} portfolio="CanSlim" />);
    await waitFor(() => {
      expect(mockApi.listWeeklyRetroSnapshots).toHaveBeenCalledWith(7, "CanSlim");
    });
    await waitFor(() => {
      expect(screen.getByAltText("a.png")).toBeInTheDocument();
      expect(screen.getByAltText("b.png")).toBeInTheDocument();
    });
  });

  test("Re-fetches when retroId changes", async () => {
    const { rerender } = render(<WeeklySnapshot retroId={1} portfolio="CanSlim" />);
    await waitFor(() => expect(mockApi.listWeeklyRetroSnapshots).toHaveBeenCalledWith(1, "CanSlim"));
    rerender(<WeeklySnapshot retroId={2} portfolio="CanSlim" />);
    await waitFor(() => expect(mockApi.listWeeklyRetroSnapshots).toHaveBeenCalledWith(2, "CanSlim"));
  });

  // ─── onCountChange ─────────────────────────────────────────────────────

  test("onCountChange fires with the visible-count on fetch", async () => {
    mockApi.listWeeklyRetroSnapshots.mockResolvedValueOnce([
      makeRow({ id: 1 }), makeRow({ id: 2 }), makeRow({ id: 3 }),
    ]);
    const onCountChange = vi.fn();
    render(
      <WeeklySnapshot retroId={7} portfolio="CanSlim" onCountChange={onCountChange} />,
    );
    await waitFor(() => expect(onCountChange).toHaveBeenCalledWith(3));
  });

  // ─── Upload (client-side validation) ───────────────────────────────────

  test("Rejects file >5MB client-side; no upload call", async () => {
    render(<WeeklySnapshot retroId={7} portfolio="CanSlim" />);
    await waitFor(() => expect(mockApi.listWeeklyRetroSnapshots).toHaveBeenCalled());

    const big = makeFile("big.png", 5 * 1024 * 1024 + 1, "image/png");
    // Inject via the hidden file input — testing-library uses
    // change events on input[type=file].
    const input = document.querySelector<HTMLInputElement>('input[type="file"]')!;
    await act(async () => {
      Object.defineProperty(input, "files", { value: [big], configurable: true });
      fireEvent.change(input);
    });
    expect(mockApi.uploadWeeklyRetroSnapshot).not.toHaveBeenCalled();
    expect(await screen.findByRole("alert")).toHaveTextContent(/5MB/i);
  });

  test("Rejects non-image file client-side; no upload call", async () => {
    render(<WeeklySnapshot retroId={7} portfolio="CanSlim" />);
    await waitFor(() => expect(mockApi.listWeeklyRetroSnapshots).toHaveBeenCalled());

    const txt = makeFile("notes.txt", 100, "text/plain");
    const input = document.querySelector<HTMLInputElement>('input[type="file"]')!;
    await act(async () => {
      Object.defineProperty(input, "files", { value: [txt], configurable: true });
      fireEvent.change(input);
    });
    expect(mockApi.uploadWeeklyRetroSnapshot).not.toHaveBeenCalled();
    expect(await screen.findByRole("alert")).toHaveTextContent(/PNG/i);
  });

  test("Valid image triggers upload with optimistic placeholder", async () => {
    mockApi.uploadWeeklyRetroSnapshot.mockResolvedValueOnce(
      makeRow({ id: 99, file_name: "ok.png" }),
    );
    render(<WeeklySnapshot retroId={7} portfolio="CanSlim" />);
    await waitFor(() => expect(mockApi.listWeeklyRetroSnapshots).toHaveBeenCalled());

    const file = makeFile("ok.png", 1024, "image/png");
    const input = document.querySelector<HTMLInputElement>('input[type="file"]')!;
    await act(async () => {
      Object.defineProperty(input, "files", { value: [file], configurable: true });
      fireEvent.change(input);
    });

    await waitFor(() => {
      expect(mockApi.uploadWeeklyRetroSnapshot).toHaveBeenCalledWith(7, file, "CanSlim");
    });
    // Real row eventually appears in the grid.
    await waitFor(() => {
      expect(screen.queryByTestId("upload-placeholder")).not.toBeInTheDocument();
      expect(screen.getByAltText("ok.png")).toBeInTheDocument();
    });
  });

  // ─── Two-click delete ──────────────────────────────────────────────────

  test("Delete: first click arms, second click commits", async () => {
    mockApi.listWeeklyRetroSnapshots.mockResolvedValueOnce([
      makeRow({ id: 1, file_name: "a.png" }),
    ]);
    mockApi.deleteWeeklyRetroSnapshot.mockResolvedValueOnce({ deleted: true, id: 1 });
    // Phase 4.4: ImageGallery's Delete button uses window.confirm() —
    // stub it to auto-accept so the test exercises the deletion path.
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<WeeklySnapshot retroId={7} portfolio="CanSlim" />);
    await waitFor(() => expect(screen.getByAltText("a.png")).toBeInTheDocument());

    const delBtn = screen.getByRole("button", { name: /Delete a.png/i });
    await act(async () => { fireEvent.click(delBtn); });

    expect(confirmSpy).toHaveBeenCalled();
    await waitFor(() => expect(mockApi.deleteWeeklyRetroSnapshot).toHaveBeenCalledWith(1));
    await waitFor(() => expect(screen.queryByAltText("a.png")).not.toBeInTheDocument());
  });

  test("Delete cancelled at confirm() does NOT delete the snapshot", async () => {
    mockApi.listWeeklyRetroSnapshots.mockResolvedValueOnce([
      makeRow({ id: 1, file_name: "a.png" }),
    ]);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<WeeklySnapshot retroId={7} portfolio="CanSlim" />);
    await waitFor(() => expect(screen.getByAltText("a.png")).toBeInTheDocument());

    const delBtn = screen.getByRole("button", { name: /Delete a.png/i });
    await act(async () => { fireEvent.click(delBtn); });

    expect(confirmSpy).toHaveBeenCalled();
    expect(mockApi.deleteWeeklyRetroSnapshot).not.toHaveBeenCalled();
    // Snapshot still in the gallery.
    expect(screen.getByAltText("a.png")).toBeInTheDocument();
  });

  // ─── Lightbox ──────────────────────────────────────────────────────────
  // Phase 4.4: clickable area moved from a labeled <button> wrapping the
  // img to a div with onClick (matches the Trade Journal pattern via
  // <ImageGallery>). Tests now click the img directly via getByAltText.

  test("Click thumbnail opens lightbox; Esc closes it", async () => {
    mockApi.listWeeklyRetroSnapshots.mockResolvedValueOnce([
      makeRow({ id: 1, file_name: "a.png" }),
    ]);
    render(<WeeklySnapshot retroId={7} portfolio="CanSlim" />);
    await waitFor(() => expect(screen.getByAltText("a.png")).toBeInTheDocument());

    await act(async () => {
      fireEvent.click(screen.getByAltText("a.png"));
    });
    expect(screen.getByRole("dialog", { name: /Snapshot preview/i })).toBeInTheDocument();

    await act(async () => {
      fireEvent.keyDown(window, { key: "Escape" });
    });
    expect(screen.queryByRole("dialog", { name: /Snapshot preview/i })).not.toBeInTheDocument();
  });

  test("Lightbox arrow keys cycle through snapshots", async () => {
    mockApi.listWeeklyRetroSnapshots.mockResolvedValueOnce([
      makeRow({ id: 1, file_name: "a.png", view_url: "u-a" }),
      makeRow({ id: 2, file_name: "b.png", view_url: "u-b" }),
      makeRow({ id: 3, file_name: "c.png", view_url: "u-c" }),
    ]);
    render(<WeeklySnapshot retroId={7} portfolio="CanSlim" />);
    await waitFor(() => expect(screen.getByAltText("a.png")).toBeInTheDocument());

    await act(async () => {
      fireEvent.click(screen.getByAltText("a.png"));
    });
    const dialog = screen.getByRole("dialog", { name: /Snapshot preview/i });
    expect(dialog.querySelector("img")?.getAttribute("src")).toBe("u-a");

    await act(async () => { fireEvent.keyDown(window, { key: "ArrowRight" }); });
    expect(dialog.querySelector("img")?.getAttribute("src")).toBe("u-b");

    await act(async () => { fireEvent.keyDown(window, { key: "ArrowRight" }); });
    expect(dialog.querySelector("img")?.getAttribute("src")).toBe("u-c");

    // Wrap around forward
    await act(async () => { fireEvent.keyDown(window, { key: "ArrowRight" }); });
    expect(dialog.querySelector("img")?.getAttribute("src")).toBe("u-a");

    // Wrap around backward
    await act(async () => { fireEvent.keyDown(window, { key: "ArrowLeft" }); });
    expect(dialog.querySelector("img")?.getAttribute("src")).toBe("u-c");
  });

  // ─── Paste handler ─────────────────────────────────────────────────────

  test("Paste handler ignores non-image clipboard items", async () => {
    render(<WeeklySnapshot retroId={7} portfolio="CanSlim" />);
    await waitFor(() => expect(mockApi.listWeeklyRetroSnapshots).toHaveBeenCalled());

    // Synthesize a paste event with a text-plain item.
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = {
      items: [
        { kind: "string", type: "text/plain", getAsFile: () => null },
      ],
    };
    await act(async () => { window.dispatchEvent(ev); });
    expect(mockApi.uploadWeeklyRetroSnapshot).not.toHaveBeenCalled();
  });

  test("Paste handler not attached when retroId is null", async () => {
    render(<WeeklySnapshot retroId={null} portfolio="CanSlim" />);
    // Synthesize a paste with an image item — should NOT trigger upload
    // because the listener isn't attached.
    const file = makeFile("p.png", 100, "image/png");
    const ev = new Event("paste", { bubbles: true, cancelable: true }) as any;
    ev.clipboardData = {
      items: [
        { kind: "file", type: "image/png", getAsFile: () => file },
      ],
    };
    await act(async () => { window.dispatchEvent(ev); });
    expect(mockApi.uploadWeeklyRetroSnapshot).not.toHaveBeenCalled();
  });
});
