import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";

import {
  MobileImageUpload,
  type ImageUploadRow,
  type ImageUploadResult,
  type ImageDeleteResult,
} from "./mobile-image-upload";

// ── File mock helper ─────────────────────────────────────────────
// Inlined per audit §5 — pattern matches weekly-snapshot.test.tsx:42.
function makeFile(name: string, size: number, type: string): File {
  const blob = new Blob([new ArrayBuffer(size)], { type });
  return new File([blob], name, { type });
}

// ── Fixture row + harness ────────────────────────────────────────

type FakeRow = ImageUploadRow & { storage_ref: string };

function rowFixture(opts: Partial<FakeRow> & { id: number }): FakeRow {
  return {
    id: opts.id,
    view_url: opts.view_url ?? `https://r2/img-${opts.id}.png`,
    file_name: opts.file_name ?? `img-${opts.id}.png`,
    storage_ref: opts.storage_ref ?? `r2/path-${opts.id}`,
  };
}

/**
 * Controlled harness that consumes the primitive and tracks rows in
 * local state. Mirrors how T2-4 / T2-5 will wire it: consumer owns
 * `rows`, primitive fires `onUpload` / `onDelete` callbacks, consumer
 * applies the result back to state.
 */
function Harness(props: {
  initialRows?: FakeRow[];
  onUpload: (file: File) => Promise<ImageUploadResult<FakeRow>>;
  onDelete: (id: number) => Promise<ImageDeleteResult>;
  disabled?: boolean;
  maxFileBytes?: number;
  acceptedMimes?: Set<string>;
  emptyStateLabel?: string;
  onThumbnailTap?: (row: FakeRow, idx: number) => void;
}) {
  const [rows, setRows] = useState<FakeRow[]>(props.initialRows ?? []);
  return (
    <MobileImageUpload<FakeRow>
      rows={rows}
      onUpload={async (file) => {
        const res = await props.onUpload(file);
        if (res && typeof res === "object" && !("error" in res)) {
          setRows((prev) => [...prev, res as FakeRow]);
        }
        return res;
      }}
      onDelete={async (id) => {
        const res = await props.onDelete(id);
        if (res && typeof res === "object" && !("error" in res)) {
          setRows((prev) => prev.filter((r) => r.id !== id));
        }
        return res;
      }}
      disabled={props.disabled}
      maxFileBytes={props.maxFileBytes}
      acceptedMimes={props.acceptedMimes}
      emptyStateLabel={props.emptyStateLabel}
      onThumbnailTap={props.onThumbnailTap}
    />
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("MobileImageUpload — state rendering", () => {
  test("empty state renders dashed CTA with format hint", () => {
    render(<Harness onUpload={vi.fn()} onDelete={vi.fn()} />);
    const cta = screen.getByTestId("image-upload-empty-cta");
    expect(cta).toHaveTextContent(/Add images/);
    expect(cta).toHaveTextContent(/15 MB max/);
    expect(cta).toHaveTextContent(/PNG|JPG/);
  });

  test("populated state renders thumbnail strip + add button", () => {
    render(
      <Harness
        initialRows={[rowFixture({ id: 1 }), rowFixture({ id: 2 })]}
        onUpload={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    expect(screen.getByTestId("image-upload-strip")).toBeInTheDocument();
    expect(screen.getByTestId("image-upload-thumb-1")).toBeInTheDocument();
    expect(screen.getByTestId("image-upload-thumb-2")).toBeInTheDocument();
    expect(screen.getByTestId("image-upload-add-more")).toBeInTheDocument();
  });

  test("disabled state renders muted CTA with disabledMessage + sub-line", () => {
    render(
      <Harness
        onUpload={vi.fn()}
        onDelete={vi.fn()}
        disabled
      />,
    );
    const node = screen.getByTestId("image-upload-disabled");
    expect(node).toHaveTextContent(/Save entry first/);
    expect(node).toHaveTextContent(/unlock after first save/);
  });

  test("custom emptyStateLabel renders verbatim", () => {
    render(
      <Harness
        onUpload={vi.fn()}
        onDelete={vi.fn()}
        emptyStateLabel="Attach screenshots"
      />,
    );
    expect(screen.getByTestId("image-upload-empty-cta")).toHaveTextContent(/Attach screenshots/);
  });

  test("custom maxFileBytes reflected in format hint", () => {
    // Pick a value distinct from the 15 MB default so the assertion
    // actually exercises the override path.
    render(
      <Harness
        onUpload={vi.fn()}
        onDelete={vi.fn()}
        maxFileBytes={25 * 1024 * 1024}
      />,
    );
    expect(screen.getByTestId("image-upload-empty-cta")).toHaveTextContent(/25 MB max/);
  });
});

describe("MobileImageUpload — file picker", () => {
  test("tapping empty-state CTA triggers the hidden file input", () => {
    render(<Harness onUpload={vi.fn()} onDelete={vi.fn()} />);
    const input = screen.getByTestId("image-upload-input") as HTMLInputElement;
    const clickSpy = vi.spyOn(input, "click");
    fireEvent.click(screen.getByTestId("image-upload-empty-cta"));
    expect(clickSpy).toHaveBeenCalledTimes(1);
  });

  test("tapping the add-more thumbnail also triggers the picker", () => {
    render(
      <Harness
        initialRows={[rowFixture({ id: 1 })]}
        onUpload={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    const input = screen.getByTestId("image-upload-input") as HTMLInputElement;
    const clickSpy = vi.spyOn(input, "click");
    fireEvent.click(screen.getByTestId("image-upload-add-more"));
    expect(clickSpy).toHaveBeenCalledTimes(1);
  });
});

describe("MobileImageUpload — upload flow", () => {
  test("upload success: onUpload called with file; row materializes in strip", async () => {
    let nextId = 100;
    const onUpload = vi.fn(async (file: File): Promise<ImageUploadResult<FakeRow>> => {
      const id = nextId++;
      return rowFixture({ id, file_name: file.name });
    });
    render(<Harness onUpload={onUpload} onDelete={vi.fn()} />);
    const input = screen.getByTestId("image-upload-input") as HTMLInputElement;
    const file = makeFile("photo.png", 1024, "image/png");
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
    });
    await waitFor(() => expect(onUpload).toHaveBeenCalledWith(file));
    expect(screen.getByTestId("image-upload-thumb-100")).toBeInTheDocument();
  });

  test("upload failure (rejected promise) renders error banner", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const onUpload = vi.fn().mockRejectedValue(new Error("server down"));
    render(<Harness onUpload={onUpload} onDelete={vi.fn()} />);
    const input = screen.getByTestId("image-upload-input") as HTMLInputElement;
    const file = makeFile("photo.png", 1024, "image/png");
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
    });
    const banner = await screen.findByTestId("image-upload-error-banner");
    expect(banner).toHaveTextContent(/photo\.png/);
    expect(banner).toHaveTextContent(/server down/);
  });

  test("upload failure (error envelope response) also renders banner", async () => {
    const onUpload = vi.fn(async (): Promise<ImageUploadResult<FakeRow>> => ({
      error: "R2 quota exceeded",
    }));
    render(<Harness onUpload={onUpload} onDelete={vi.fn()} />);
    const input = screen.getByTestId("image-upload-input") as HTMLInputElement;
    const file = makeFile("photo.png", 1024, "image/png");
    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
    });
    const banner = await screen.findByTestId("image-upload-error-banner");
    expect(banner).toHaveTextContent(/R2 quota exceeded/);
  });

  test("multi-file pick fires onUpload once per valid file in parallel", async () => {
    let nextId = 200;
    const onUpload = vi.fn(async (file: File): Promise<ImageUploadResult<FakeRow>> =>
      rowFixture({ id: nextId++, file_name: file.name }),
    );
    render(<Harness onUpload={onUpload} onDelete={vi.fn()} />);
    const input = screen.getByTestId("image-upload-input") as HTMLInputElement;
    const f1 = makeFile("one.png", 1024, "image/png");
    const f2 = makeFile("two.jpg", 1024, "image/jpeg");
    const f3 = makeFile("three.webp", 1024, "image/webp");
    await act(async () => {
      fireEvent.change(input, { target: { files: [f1, f2, f3] } });
    });
    await waitFor(() => expect(onUpload).toHaveBeenCalledTimes(3));
    expect(onUpload).toHaveBeenCalledWith(f1);
    expect(onUpload).toHaveBeenCalledWith(f2);
    expect(onUpload).toHaveBeenCalledWith(f3);
  });
});

describe("MobileImageUpload — client-side validation", () => {
  test("MIME rejection: HEIC file triggers banner WITHOUT calling onUpload", async () => {
    const onUpload = vi.fn();
    render(<Harness onUpload={onUpload} onDelete={vi.fn()} />);
    const input = screen.getByTestId("image-upload-input") as HTMLInputElement;
    const heic = makeFile("photo.heic", 1024, "image/heic");
    await act(async () => {
      fireEvent.change(input, { target: { files: [heic] } });
    });
    const banner = await screen.findByTestId("image-upload-error-banner");
    expect(banner).toHaveTextContent(/photo\.heic/);
    expect(banner).toHaveTextContent(/not supported/);
    expect(onUpload).not.toHaveBeenCalled();
  });

  test("size rejection: oversized file triggers banner WITHOUT calling onUpload", async () => {
    const onUpload = vi.fn();
    render(
      <Harness
        onUpload={onUpload}
        onDelete={vi.fn()}
        maxFileBytes={1024}
      />,
    );
    const input = screen.getByTestId("image-upload-input") as HTMLInputElement;
    const big = makeFile("big.png", 2048, "image/png");
    await act(async () => {
      fireEvent.change(input, { target: { files: [big] } });
    });
    const banner = await screen.findByTestId("image-upload-error-banner");
    expect(banner).toHaveTextContent(/big\.png/);
    expect(banner).toHaveTextContent(/exceeds/);
    expect(onUpload).not.toHaveBeenCalled();
  });

  test("mixed batch: valid files upload, invalid files rejected aggregately", async () => {
    let nextId = 300;
    const onUpload = vi.fn(async (file: File): Promise<ImageUploadResult<FakeRow>> =>
      rowFixture({ id: nextId++, file_name: file.name }),
    );
    render(<Harness onUpload={onUpload} onDelete={vi.fn()} />);
    const input = screen.getByTestId("image-upload-input") as HTMLInputElement;
    const ok = makeFile("ok.png", 1024, "image/png");
    const heic = makeFile("photo.heic", 1024, "image/heic");
    await act(async () => {
      fireEvent.change(input, { target: { files: [ok, heic] } });
    });
    await waitFor(() => expect(onUpload).toHaveBeenCalledTimes(1));
    expect(onUpload).toHaveBeenCalledWith(ok);
    expect(screen.getByTestId("image-upload-error-banner")).toHaveTextContent(/photo\.heic/);
  });
});

describe("MobileImageUpload — delete flow", () => {
  test("tapping X enters delete-confirm state; sibling thumbs dim; confirm bar appears", () => {
    render(
      <Harness
        initialRows={[rowFixture({ id: 1 }), rowFixture({ id: 2 })]}
        onUpload={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("image-upload-remove-1"));
    expect(screen.getByTestId("image-upload-delete-target-1")).toBeInTheDocument();
    expect(screen.getByTestId("image-upload-delete-confirm-bar")).toBeInTheDocument();
    // Sibling dimmed via inline opacity.
    const sibling = screen.getByTestId("image-upload-thumb-2");
    expect(sibling.getAttribute("style") ?? "").toMatch(/opacity:\s*0\.4/);
  });

  test("Cancel button exits confirm without calling onDelete", () => {
    const onDelete = vi.fn();
    render(
      <Harness
        initialRows={[rowFixture({ id: 1 })]}
        onUpload={vi.fn()}
        onDelete={onDelete}
      />,
    );
    fireEvent.click(screen.getByTestId("image-upload-remove-1"));
    fireEvent.click(screen.getByTestId("image-upload-delete-cancel"));
    expect(onDelete).not.toHaveBeenCalled();
    expect(screen.queryByTestId("image-upload-delete-confirm-bar")).not.toBeInTheDocument();
  });

  test("Delete button calls onDelete with row.id; row disappears from strip", async () => {
    const onDelete = vi.fn(async (): Promise<ImageDeleteResult> => ({ deleted: true }));
    render(
      <Harness
        initialRows={[rowFixture({ id: 1 }), rowFixture({ id: 2 })]}
        onUpload={vi.fn()}
        onDelete={onDelete}
      />,
    );
    fireEvent.click(screen.getByTestId("image-upload-remove-1"));
    await act(async () => {
      fireEvent.click(screen.getByTestId("image-upload-delete-confirm"));
    });
    await waitFor(() => expect(onDelete).toHaveBeenCalledWith(1));
    expect(screen.queryByTestId("image-upload-thumb-1")).not.toBeInTheDocument();
    expect(screen.getByTestId("image-upload-thumb-2")).toBeInTheDocument();
  });
});

describe("MobileImageUpload — thumbnail tap", () => {
  test("onThumbnailTap fires with row + index on tap", () => {
    const onThumbnailTap = vi.fn();
    render(
      <Harness
        initialRows={[rowFixture({ id: 1 }), rowFixture({ id: 2 }), rowFixture({ id: 3 })]}
        onUpload={vi.fn()}
        onDelete={vi.fn()}
        onThumbnailTap={onThumbnailTap}
      />,
    );
    const thumb = screen.getByTestId("image-upload-thumb-2");
    // Tap the image area (the button wrapping <img>), not the X.
    const tapTarget = within(thumb).getByRole("button", { name: /View img-2\.png/ });
    fireEvent.click(tapTarget);
    expect(onThumbnailTap).toHaveBeenCalledTimes(1);
    const callArg = onThumbnailTap.mock.calls[0];
    expect(callArg[0].id).toBe(2);
    expect(callArg[1]).toBe(1); // index
  });
});
