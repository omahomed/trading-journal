import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

// jsdom localStorage shim — same as weekly-retro.test.tsx.
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

vi.mock("@/lib/api", () => ({
  api: {
    listTags: vi.fn(),
    listTagAssignments: vi.fn(),
    createTag: vi.fn(),
    createTagAssignment: vi.fn(),
    deleteTagAssignment: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api, type Tag, type TagAssignment } from "@/lib/api";
import { TagPicker } from "./tag-picker";

const mListTags        = vi.mocked(api.listTags);
const mListAssignments = vi.mocked(api.listTagAssignments);
const mCreateTag       = vi.mocked(api.createTag);
const mCreateAssign    = vi.mocked(api.createTagAssignment);
const mDeleteAssign    = vi.mocked(api.deleteTagAssignment);

function tag(id: number, name: string, color = "rose"): Tag {
  return {
    id, portfolio: "CanSlim", name, color,
    created_at: "2026-05-01T00:00:00Z", updated_at: "2026-05-01T00:00:00Z",
  };
}

function assignment(id: number, tagId: number, name: string, color = "rose"): TagAssignment {
  return {
    id, tag_id: tagId, tag_name: name, tag_color: color,
    entity_type: "weekly_retro", entity_id: 42,
    created_at: "2026-05-13T00:00:00Z",
  };
}

function setupDefaults() {
  mListTags.mockResolvedValue([]);
  mListAssignments.mockResolvedValue([]);
  mCreateTag.mockResolvedValue(tag(99, "x", "sky"));
  mCreateAssign.mockResolvedValue(assignment(999, 99, "x", "sky"));
  mDeleteAssign.mockResolvedValue({ status: "ok", id: 1 });
}

describe("TagPicker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("disabled state when entityId is null — no fetch fires", async () => {
    render(<TagPicker entityType="weekly_retro" entityId={null} portfolio="CanSlim" />);
    // No "+ Add tag" interactive button — render the disabled placeholder.
    expect(screen.queryByRole("button", { name: /add tag/i })).not.toBeInTheDocument();
    // The disabled placeholder still says "Add tag" as plain text.
    expect(screen.getByText(/add tag/i)).toBeInTheDocument();
    // No fetches.
    expect(mListTags).not.toHaveBeenCalled();
    expect(mListAssignments).not.toHaveBeenCalled();
  });

  test("fetches tags and assignments on mount when entityId is set", async () => {
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    await waitFor(() => {
      expect(mListTags).toHaveBeenCalledWith("CanSlim");
      expect(mListAssignments).toHaveBeenCalledWith({ entity_type: "weekly_retro", entity_id: 42 });
    });
  });

  test("renders existing assignment pills", async () => {
    mListAssignments.mockResolvedValue([
      assignment(1, 7, "drawdown", "rose"),
      assignment(2, 8, "FOMC",     "sky"),
    ]);
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    expect(await screen.findByText("drawdown")).toBeInTheDocument();
    expect(await screen.findByText("FOMC")).toBeInTheDocument();
  });

  test("'+ Add tag' click opens the dropdown", async () => {
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    const btn = await screen.findByRole("button", { name: /add tag/i });
    await act(async () => { fireEvent.click(btn); });
    expect(await screen.findByPlaceholderText(/search or create tag/i)).toBeInTheDocument();
  });

  test("dropdown filters out already-assigned tags", async () => {
    mListTags.mockResolvedValue([tag(1, "drawdown"), tag(2, "FOMC")]);
    mListAssignments.mockResolvedValue([assignment(99, 1, "drawdown", "rose")]);
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    await waitFor(() => expect(mListTags).toHaveBeenCalled());

    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /add tag/i }));
    });
    // Only FOMC appears in the unassigned list (drawdown is already pinned).
    const listbox = await screen.findByRole("listbox");
    expect(listbox).toHaveTextContent("FOMC");
    // drawdown still rendered above as a pill, but inside the listbox itself
    // it must not appear as a clickable option.
    const options = listbox.querySelectorAll("button");
    const optionTexts = Array.from(options).map(o => o.textContent ?? "");
    expect(optionTexts.some(t => t.includes("drawdown") && !t.includes("Create"))).toBe(false);
  });

  test("typing filters the unassigned list", async () => {
    mListTags.mockResolvedValue([tag(1, "drawdown"), tag(2, "FOMC"), tag(3, "earnings week")]);
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    await waitFor(() => expect(mListTags).toHaveBeenCalled());
    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /add tag/i }));
    });
    const input = await screen.findByPlaceholderText(/search or create tag/i);
    await act(async () => { fireEvent.change(input, { target: { value: "earn" } }); });
    expect(screen.queryByText("drawdown")).not.toBeInTheDocument();
    expect(screen.queryByText("FOMC")).not.toBeInTheDocument();
    expect(screen.getByText("earnings week")).toBeInTheDocument();
  });

  test("non-matching query shows '+ Create' affordance", async () => {
    mListTags.mockResolvedValue([tag(1, "drawdown")]);
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    await waitFor(() => expect(mListTags).toHaveBeenCalled());
    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /add tag/i }));
    });
    const input = await screen.findByPlaceholderText(/search or create tag/i);
    await act(async () => { fireEvent.change(input, { target: { value: "newTag" } }); });
    // The create-affordance button's accessible name is "Create "newTag""
    // (smart-quoted in the JSX). Match by role + name to avoid colliding
    // with the "Create new" section header.
    expect(
      await screen.findByRole("button", { name: /create.*newTag/i }),
    ).toBeInTheDocument();
  });

  test("clicking an existing tag optimistically adds the pill and POSTs", async () => {
    mListTags.mockResolvedValue([tag(7, "drawdown", "rose")]);
    mCreateAssign.mockResolvedValue(assignment(123, 7, "drawdown", "rose"));
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    await waitFor(() => expect(mListTags).toHaveBeenCalled());

    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /add tag/i }));
    });
    const listbox = await screen.findByRole("listbox");
    const drawdownOption = Array.from(listbox.querySelectorAll("button")).find(
      b => b.textContent?.includes("drawdown"),
    )!;
    await act(async () => { fireEvent.click(drawdownOption); });

    await waitFor(() => expect(mCreateAssign).toHaveBeenCalledWith({
      tag_id: 7, entity_type: "weekly_retro", entity_id: 42,
    }));
    // Pill rendered above (was unmounted from listbox after dropdown close).
    expect(await screen.findByText("drawdown")).toBeInTheDocument();
  });

  test("POST failure removes the optimistic pill and shows error toast", async () => {
    mListTags.mockResolvedValue([tag(7, "drawdown", "rose")]);
    mCreateAssign.mockResolvedValue({ error: "boom" } as any);
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    await waitFor(() => expect(mListTags).toHaveBeenCalled());

    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /add tag/i }));
    });
    const listbox = await screen.findByRole("listbox");
    const opt = Array.from(listbox.querySelectorAll("button")).find(
      b => b.textContent?.includes("drawdown"),
    )!;
    await act(async () => { fireEvent.click(opt); });

    // Pill briefly appeared optimistically, then was removed on failure.
    await waitFor(() => expect(screen.queryByText("drawdown")).not.toBeInTheDocument());
    expect(await screen.findByRole("status")).toHaveTextContent(/boom/i);
  });

  test("clicking '+ Create' POSTs the tag then the assignment", async () => {
    mListTags.mockResolvedValue([]);
    mCreateTag.mockResolvedValue(tag(50, "newTag", "sky"));
    mCreateAssign.mockResolvedValue(assignment(500, 50, "newTag", "sky"));
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    await waitFor(() => expect(mListTags).toHaveBeenCalled());

    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /add tag/i }));
    });
    const input = await screen.findByPlaceholderText(/search or create tag/i);
    await act(async () => { fireEvent.change(input, { target: { value: "newTag" } }); });
    const createBtn = await screen.findByRole("button", { name: /create.*newTag/i });
    await act(async () => { fireEvent.click(createBtn); });

    await waitFor(() => {
      expect(mCreateTag).toHaveBeenCalledWith({
        portfolio: "CanSlim", name: "newTag", color: "sky",
      });
      expect(mCreateAssign).toHaveBeenCalledWith({
        tag_id: 50, entity_type: "weekly_retro", entity_id: 42,
      });
    });
  });

  test("clicking pill X optimistically removes and DELETEs", async () => {
    mListAssignments.mockResolvedValue([assignment(123, 7, "drawdown", "rose")]);
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    expect(await screen.findByText("drawdown")).toBeInTheDocument();

    const removeBtn = screen.getByRole("button", { name: /remove drawdown/i });
    await act(async () => { fireEvent.click(removeBtn); });
    await waitFor(() => expect(mDeleteAssign).toHaveBeenCalledWith(123));
    expect(screen.queryByText("drawdown")).not.toBeInTheDocument();
  });

  test("DELETE failure restores the pill and shows error toast", async () => {
    mListAssignments.mockResolvedValue([assignment(123, 7, "drawdown", "rose")]);
    mDeleteAssign.mockResolvedValue({ error: "network down" } as any);
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    expect(await screen.findByText("drawdown")).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /remove drawdown/i }));
    });
    await waitFor(() => expect(mDeleteAssign).toHaveBeenCalled());
    // Pill restored.
    expect(await screen.findByText("drawdown")).toBeInTheDocument();
    expect(await screen.findByRole("status")).toHaveTextContent(/network down/i);
  });

  test("Esc closes the dropdown", async () => {
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    await waitFor(() => expect(mListTags).toHaveBeenCalled());
    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /add tag/i }));
    });
    const input = await screen.findByPlaceholderText(/search or create tag/i);
    await act(async () => { fireEvent.keyDown(input, { key: "Escape" }); });
    expect(screen.queryByPlaceholderText(/search or create tag/i)).not.toBeInTheDocument();
  });

  test("click outside closes the dropdown", async () => {
    render(<div>
      <div data-testid="outside">outside</div>
      <TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />
    </div>);
    await waitFor(() => expect(mListTags).toHaveBeenCalled());
    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /add tag/i }));
    });
    expect(await screen.findByPlaceholderText(/search or create tag/i)).toBeInTheDocument();
    await act(async () => {
      fireEvent.mouseDown(screen.getByTestId("outside"));
    });
    expect(screen.queryByPlaceholderText(/search or create tag/i)).not.toBeInTheDocument();
  });

  test("Enter on exact match assigns the existing tag", async () => {
    mListTags.mockResolvedValue([tag(7, "drawdown", "rose")]);
    mCreateAssign.mockResolvedValue(assignment(500, 7, "drawdown", "rose"));
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    await waitFor(() => expect(mListTags).toHaveBeenCalled());

    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /add tag/i }));
    });
    const input = await screen.findByPlaceholderText(/search or create tag/i);
    await act(async () => { fireEvent.change(input, { target: { value: "drawdown" } }); });
    await act(async () => { fireEvent.keyDown(input, { key: "Enter" }); });

    await waitFor(() => expect(mCreateAssign).toHaveBeenCalledWith({
      tag_id: 7, entity_type: "weekly_retro", entity_id: 42,
    }));
    expect(mCreateTag).not.toHaveBeenCalled();
  });

  test("Enter with no match creates and assigns", async () => {
    mListTags.mockResolvedValue([]);
    mCreateTag.mockResolvedValue(tag(50, "brandnew", "sky"));
    mCreateAssign.mockResolvedValue(assignment(500, 50, "brandnew", "sky"));
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    await waitFor(() => expect(mListTags).toHaveBeenCalled());

    await act(async () => {
      fireEvent.click(await screen.findByRole("button", { name: /add tag/i }));
    });
    const input = await screen.findByPlaceholderText(/search or create tag/i);
    await act(async () => { fireEvent.change(input, { target: { value: "brandnew" } }); });
    await act(async () => { fireEvent.keyDown(input, { key: "Enter" }); });

    await waitFor(() => {
      expect(mCreateTag).toHaveBeenCalledWith({
        portfolio: "CanSlim", name: "brandnew", color: "sky",
      });
      expect(mCreateAssign).toHaveBeenCalled();
    });
  });

  test("at 10 tags, '+ Add tag' button is disabled with tooltip", async () => {
    mListAssignments.mockResolvedValue(
      Array.from({ length: 10 }, (_, i) => assignment(i + 1, i + 1, `t${i + 1}`, "sky")),
    );
    render(<TagPicker entityType="weekly_retro" entityId={42} portfolio="CanSlim" />);
    // Wait for the 10 pills to render.
    await waitFor(() => expect(screen.getAllByText(/^t\d+$/)).toHaveLength(10));
    const btn = screen.getByRole("button", { name: /add tag/i });
    expect(btn).toHaveAttribute("title", "Maximum 10 tags per entry");

    // Clicking it does NOT open the dropdown; it surfaces the error toast.
    await act(async () => { fireEvent.click(btn); });
    expect(screen.queryByPlaceholderText(/search or create tag/i)).not.toBeInTheDocument();
    expect(await screen.findByRole("status")).toHaveTextContent(/Maximum 10 tags/i);
  });
});
