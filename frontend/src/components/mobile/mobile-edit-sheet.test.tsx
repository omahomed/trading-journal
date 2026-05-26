import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { MobileEditSheet } from "./mobile-edit-sheet";

function Harness({
  initialOpen = true,
  initialDirty = false,
  onCloseSpy,
  onSaveSpy,
}: {
  initialOpen?: boolean;
  initialDirty?: boolean;
  onCloseSpy?: () => void;
  onSaveSpy?: () => void;
}) {
  const [open, setOpen] = useState(initialOpen);
  const [isDirty] = useState(initialDirty);
  return (
    <MobileEditSheet
      open={open}
      onClose={() => {
        setOpen(false);
        onCloseSpy?.();
      }}
      title="Test Sheet"
      isDirty={isDirty}
      rightAction={{
        label: "Save",
        onClick: () => {
          onSaveSpy?.();
        },
      }}
    >
      <div data-testid="sheet-body-content">Body content</div>
    </MobileEditSheet>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  document.body.style.overflow = "";
});

describe("MobileEditSheet — rendering", () => {
  test("open=false → returns null (no sheet in DOM)", () => {
    render(<Harness initialOpen={false} />);
    expect(screen.queryByTestId("mobile-edit-sheet")).not.toBeInTheDocument();
  });

  test("open=true → renders header + body + backdrop", () => {
    render(<Harness />);
    expect(screen.getByTestId("mobile-edit-sheet")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-edit-sheet-title")).toHaveTextContent("Test Sheet");
    expect(screen.getByTestId("mobile-edit-sheet-body")).toBeInTheDocument();
    expect(screen.getByTestId("sheet-body-content")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-edit-sheet-backdrop")).toBeInTheDocument();
  });

  test("renders rightAction Save button when provided", () => {
    render(<Harness />);
    const save = screen.getByTestId("mobile-edit-sheet-save");
    expect(save).toHaveTextContent("Save");
  });

  test("does NOT render confirm sheet initially", () => {
    render(<Harness />);
    expect(screen.queryByTestId("mobile-edit-sheet-confirm")).not.toBeInTheDocument();
  });
});

describe("MobileEditSheet — clean dismiss (isDirty=false)", () => {
  test("X button fires onClose immediately", () => {
    const onCloseSpy = vi.fn();
    render(<Harness onCloseSpy={onCloseSpy} />);
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-close"));
    expect(onCloseSpy).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId("mobile-edit-sheet-confirm")).not.toBeInTheDocument();
  });

  test("backdrop tap fires onClose immediately", () => {
    const onCloseSpy = vi.fn();
    render(<Harness onCloseSpy={onCloseSpy} />);
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-backdrop"));
    expect(onCloseSpy).toHaveBeenCalledTimes(1);
  });

  test("Escape key fires onClose immediately", () => {
    const onCloseSpy = vi.fn();
    render(<Harness onCloseSpy={onCloseSpy} />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onCloseSpy).toHaveBeenCalledTimes(1);
  });
});

describe("MobileEditSheet — dirty dismiss (isDirty=true)", () => {
  test("X button shows confirm sheet instead of firing onClose", () => {
    const onCloseSpy = vi.fn();
    render(<Harness initialDirty onCloseSpy={onCloseSpy} />);
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-close"));
    expect(screen.getByTestId("mobile-edit-sheet-confirm")).toBeInTheDocument();
    expect(onCloseSpy).not.toHaveBeenCalled();
  });

  test("confirm Discard fires onClose", () => {
    const onCloseSpy = vi.fn();
    render(<Harness initialDirty onCloseSpy={onCloseSpy} />);
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-close"));
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-confirm-discard"));
    expect(onCloseSpy).toHaveBeenCalledTimes(1);
  });

  test("confirm Keep Editing dismisses the confirm and keeps the sheet open", () => {
    const onCloseSpy = vi.fn();
    render(<Harness initialDirty onCloseSpy={onCloseSpy} />);
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-close"));
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-confirm-keep"));
    expect(screen.queryByTestId("mobile-edit-sheet-confirm")).not.toBeInTheDocument();
    expect(screen.getByTestId("mobile-edit-sheet")).toBeInTheDocument();
    expect(onCloseSpy).not.toHaveBeenCalled();
  });

  test("Escape during confirm dismisses confirm only (sheet stays open)", () => {
    const onCloseSpy = vi.fn();
    render(<Harness initialDirty onCloseSpy={onCloseSpy} />);
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-close"));
    expect(screen.getByTestId("mobile-edit-sheet-confirm")).toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByTestId("mobile-edit-sheet-confirm")).not.toBeInTheDocument();
    expect(screen.getByTestId("mobile-edit-sheet")).toBeInTheDocument();
    expect(onCloseSpy).not.toHaveBeenCalled();
  });

  test("backdrop tap on dirty sheet shows confirm (not immediate close)", () => {
    const onCloseSpy = vi.fn();
    render(<Harness initialDirty onCloseSpy={onCloseSpy} />);
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-backdrop"));
    expect(screen.getByTestId("mobile-edit-sheet-confirm")).toBeInTheDocument();
    expect(onCloseSpy).not.toHaveBeenCalled();
  });

  test("confirm backdrop tap is equivalent to Keep Editing", () => {
    const onCloseSpy = vi.fn();
    render(<Harness initialDirty onCloseSpy={onCloseSpy} />);
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-close"));
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-confirm-backdrop"));
    expect(screen.queryByTestId("mobile-edit-sheet-confirm")).not.toBeInTheDocument();
    expect(onCloseSpy).not.toHaveBeenCalled();
  });
});

describe("MobileEditSheet — Save action", () => {
  test("Save button fires rightAction.onClick", () => {
    const onSaveSpy = vi.fn();
    render(<Harness onSaveSpy={onSaveSpy} />);
    fireEvent.click(screen.getByTestId("mobile-edit-sheet-save"));
    expect(onSaveSpy).toHaveBeenCalledTimes(1);
  });

  test("rightAction.disabled honored", () => {
    function H() {
      return (
        <MobileEditSheet
          open
          onClose={vi.fn()}
          title="x"
          isDirty={false}
          rightAction={{ label: "Save", onClick: vi.fn(), disabled: true }}
        >
          <div />
        </MobileEditSheet>
      );
    }
    render(<H />);
    const save = screen.getByTestId("mobile-edit-sheet-save") as HTMLButtonElement;
    expect(save.disabled).toBe(true);
  });
});

describe("MobileEditSheet — body scroll lock", () => {
  test("locks body overflow while open; restores on close", () => {
    document.body.style.overflow = "auto";
    function Wrap({ open }: { open: boolean }) {
      return (
        <MobileEditSheet
          open={open}
          onClose={vi.fn()}
          title="x"
          isDirty={false}
        >
          <div />
        </MobileEditSheet>
      );
    }
    const { rerender } = render(<Wrap open />);
    expect(document.body.style.overflow).toBe("hidden");
    rerender(<Wrap open={false} />);
    expect(document.body.style.overflow).toBe("auto");
  });
});
