import { render, screen, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";

// Mock the API surface with BOTH the weekly and daily methods.
vi.mock("@/lib/api", () => ({
  api: {
    listWeeklyRetroSnapshots: vi.fn(),
    uploadWeeklyRetroSnapshot: vi.fn(),
    deleteWeeklyRetroSnapshot: vi.fn(),
    listDailyJournalCaptures: vi.fn(),
    uploadDailyJournalCapture: vi.fn(),
    deleteDailyJournalCapture: vi.fn(),
  },
}));

import { api } from "@/lib/api";
import { SnapshotGallery } from "./snapshot-gallery";

const mockApi = vi.mocked(api);

describe("SnapshotGallery — Phase 7 parameterization", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockApi.listWeeklyRetroSnapshots.mockResolvedValue([]);
    mockApi.listDailyJournalCaptures.mockResolvedValue([]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("entityType=weekly_retro routes list to the weekly endpoint", async () => {
    render(
      <SnapshotGallery
        entityType="weekly_retro"
        entityId={7}
        portfolio="CanSlim"
      />,
    );
    await waitFor(() => {
      expect(mockApi.listWeeklyRetroSnapshots).toHaveBeenCalledWith(7, "CanSlim");
    });
    expect(mockApi.listDailyJournalCaptures).not.toHaveBeenCalled();
  });

  test("entityType=daily_journal routes list to the daily endpoint", async () => {
    render(
      <SnapshotGallery
        entityType="daily_journal"
        entityId={42}
        portfolio="CanSlim"
      />,
    );
    await waitFor(() => {
      expect(mockApi.listDailyJournalCaptures).toHaveBeenCalledWith(42, "CanSlim");
    });
    expect(mockApi.listWeeklyRetroSnapshots).not.toHaveBeenCalled();
  });

  test("disabled state shows the daily-specific copy when entityType=daily_journal", () => {
    render(
      <SnapshotGallery
        entityType="daily_journal"
        entityId={null}
        portfolio="CanSlim"
      />,
    );
    expect(screen.getByText(/Save the journal entry first/i)).toBeInTheDocument();
    expect(screen.queryByText(/Save the retro first/i)).not.toBeInTheDocument();
  });

  test("disabled state shows the weekly-specific copy when entityType=weekly_retro", () => {
    render(
      <SnapshotGallery
        entityType="weekly_retro"
        entityId={null}
        portfolio="CanSlim"
      />,
    );
    expect(screen.getByText(/Save the retro first/i)).toBeInTheDocument();
    expect(screen.queryByText(/Save the journal entry first/i)).not.toBeInTheDocument();
  });

  test("disabledMessage prop overrides the default copy", () => {
    render(
      <SnapshotGallery
        entityType="daily_journal"
        entityId={null}
        portfolio="CanSlim"
        disabledMessage="Custom disabled copy."
      />,
    );
    expect(screen.getByText(/Custom disabled copy/i)).toBeInTheDocument();
  });
});
