import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

// localStorage shim — admin doesn't itself read it, but child config
// loads do. Same pattern as other test files in this directory.
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
  });
}

// getActivePortfolio is mocked as a vi.fn so individual tests can override
// the return value (e.g. for the defensive 'active not in list' test). The
// module-level default is "CanSlim" — matches the user's primary portfolio
// and the value PortfolioProvider seeds at app load in production.
const mockGetActivePortfolio = vi.fn(() => "CanSlim");

vi.mock("@/lib/api", () => ({
  api: {
    config: vi.fn().mockResolvedValue({ value: null }),
    setConfig: vi.fn(),
    events: vi.fn().mockResolvedValue([]),
    addEvent: vi.fn(),
    updateEvent: vi.fn(),
    deleteEvent: vi.fn(),
    audit: vi.fn().mockResolvedValue([]),
    cleanupMarketsurge: vi.fn(),
    rebuildMctSignals: vi.fn(),
    journalBackfillMetrics: vi.fn(),
    listStrategies: vi.fn(),
    createStrategy: vi.fn(),
    updateStrategy: vi.fn(),
    runDriftScan: vi.fn(),
    listPortfolios: vi.fn(),
  },
  getActivePortfolio: () => mockGetActivePortfolio(),
}));

import { api } from "@/lib/api";
import { Admin } from "./admin";

const SEED_STRATEGIES = [
  { name: "CanSlim",   description: "primary",   color: "#6366f1", is_active: true,  created_at: "2026-01-01" },
  { name: "StockTalk", description: "small-cap", color: "#d97706", is_active: true,  created_at: "2026-01-02" },
  { name: "Retired",   description: "old",       color: "#888888", is_active: false, created_at: "2026-01-03" },
];

// Three production portfolios. Drift-scan dropdown defaults to the user's
// active portfolio if it appears in this list; otherwise falls back to ""
// (= All Portfolios). Per-test mocks can override the list to exercise
// the defensive default path.
const SEED_PORTFOLIOS = [
  { id: 1, name: "CanSlim",       starting_capital: null, reset_date: null, created_at: "2026-01-01", cash_balance: 0 },
  { id: 2, name: "TQQQ Strategy", starting_capital: null, reset_date: null, created_at: "2026-01-01", cash_balance: 0 },
  { id: 3, name: "457B Plan",     starting_capital: null, reset_date: null, created_at: "2026-01-01", cash_balance: 0 },
];

beforeEach(() => {
  vi.clearAllMocks();
  // Reset the active-portfolio mock to the default. Tests that need a
  // different value override via mockGetActivePortfolio.mockReturnValue.
  mockGetActivePortfolio.mockReturnValue("CanSlim");
  vi.mocked(api.listStrategies).mockResolvedValue(SEED_STRATEGIES as any);
  vi.mocked(api.listPortfolios).mockResolvedValue(SEED_PORTFOLIOS as any);
  vi.mocked(api.createStrategy).mockResolvedValue({
    name: "Momentum", description: "swing", color: "#22c55e",
    is_active: true, created_at: "2026-05-08",
  } as any);
  vi.mocked(api.updateStrategy).mockResolvedValue({
    name: "CanSlim", description: "Updated", color: "#6366f1",
    is_active: true, created_at: "2026-01-01",
  } as any);
});

async function openStrategiesSection() {
  render(<Admin navColor="#6366f1" />);
  // Section header is collapsible — click it to reveal the table.
  const sectionHeader = await screen.findByText("Strategies");
  await act(async () => { fireEvent.click(sectionHeader); });
  await waitFor(() => expect(api.listStrategies).toHaveBeenCalled());
}

describe("Admin — Strategies section", () => {
  test("lists all strategies (including inactive) with their chips", async () => {
    await openStrategiesSection();

    // Each seeded strategy renders.
    expect(await screen.findByText("CanSlim")).toBeInTheDocument();
    expect(screen.getByText("StockTalk")).toBeInTheDocument();
    expect(screen.getByText("Retired")).toBeInTheDocument();
  });

  test("inline is_active toggle calls api.updateStrategy with the flipped value", async () => {
    await openStrategiesSection();

    const toggle = await screen.findByTestId("admin-strategy-active-CanSlim");
    expect(toggle).toBeChecked();

    await act(async () => { fireEvent.click(toggle); });

    await waitFor(() => expect(api.updateStrategy).toHaveBeenCalled());
    const [name, body] = vi.mocked(api.updateStrategy).mock.calls[0];
    expect(name).toBe("CanSlim");
    expect(body.is_active).toBe(false);
  });

  test("Add Strategy modal: submit calls api.createStrategy with form values", async () => {
    await openStrategiesSection();

    const addBtn = await screen.findByTestId("admin-add-strategy");
    await act(async () => { fireEvent.click(addBtn); });

    expect(await screen.findByTestId("admin-strategy-modal")).toBeInTheDocument();

    // Fill the form. Field labels render the input as the next sibling.
    const nameInput = screen.getByPlaceholderText("CanSlim") as HTMLInputElement;
    const colorInput = screen.getByPlaceholderText("#6366f1") as HTMLInputElement;
    const descInput = screen.getByPlaceholderText("Short prose") as HTMLInputElement;

    fireEvent.change(nameInput, { target: { value: "Momentum" } });
    fireEvent.change(colorInput, { target: { value: "#22c55e" } });
    fireEvent.change(descInput, { target: { value: "swing" } });

    await act(async () => {
      fireEvent.click(screen.getByText("Create Strategy"));
    });

    await waitFor(() => expect(api.createStrategy).toHaveBeenCalled());
    const body = vi.mocked(api.createStrategy).mock.calls[0][0];
    expect(body.name).toBe("Momentum");
    expect(body.color).toBe("#22c55e");
    expect(body.description).toBe("swing");
    expect(body.is_active).toBe(true);
  });

  test("Add Strategy modal: client-side hex validation blocks submit on bad color", async () => {
    await openStrategiesSection();

    await act(async () => {
      fireEvent.click(await screen.findByTestId("admin-add-strategy"));
    });

    const nameInput = screen.getByPlaceholderText("CanSlim") as HTMLInputElement;
    const colorInput = screen.getByPlaceholderText("#6366f1") as HTMLInputElement;

    fireEvent.change(nameInput, { target: { value: "Bad" } });
    fireEvent.change(colorInput, { target: { value: "not-hex" } });

    await act(async () => {
      fireEvent.click(screen.getByText("Create Strategy"));
    });

    // Inline error appears; createStrategy is NOT called.
    expect(await screen.findByText(/six-digit hex/i)).toBeInTheDocument();
    expect(api.createStrategy).not.toHaveBeenCalled();
  });

  test("Edit Strategy modal: submit calls api.updateStrategy with patched fields", async () => {
    await openStrategiesSection();

    const editBtn = await screen.findByTestId("admin-strategy-edit-CanSlim");
    await act(async () => { fireEvent.click(editBtn); });

    expect(await screen.findByTestId("admin-strategy-modal")).toBeInTheDocument();

    // Description input shows the existing value; change it.
    const descInput = screen.getByPlaceholderText("Short prose") as HTMLInputElement;
    expect(descInput.value).toBe("primary");
    fireEvent.change(descInput, { target: { value: "Updated description" } });

    await act(async () => {
      fireEvent.click(screen.getByText("Save Changes"));
    });

    await waitFor(() => expect(api.updateStrategy).toHaveBeenCalled());
    const [name, body] = vi.mocked(api.updateStrategy).mock.calls[0];
    expect(name).toBe("CanSlim");
    expect(body.description).toBe("Updated description");
  });
});

// ---------------------------------------------------------------------------
// Drift Scan section (Phase 2 Commit 8)
// ---------------------------------------------------------------------------

const DRIFT_CLEAN_RESPONSE = {
  scanned_at: "2026-05-09T18:00:00Z",
  portfolio_filter: null,
  check_filter: null,
  sample_limit: 10,
  checks: [
    { check_id: "summary_detail_rule_mismatch", description: "rule mismatch",
      severity: "warning", violation_count: 0, samples: [], remediation: "ok",
      duration_ms: 5, error: null },
    { check_id: "closed_with_nonzero_shares", description: "closed shares",
      severity: "error", violation_count: 0, samples: [], remediation: "ok",
      duration_ms: 6, error: null },
  ],
  summary: { total_checks: 2, passed: 2, warnings: 0, errors: 0 },
};

const DRIFT_MIXED_RESPONSE = {
  scanned_at: "2026-05-09T18:00:00Z",
  portfolio_filter: null,
  check_filter: null,
  sample_limit: 10,
  checks: [
    {
      check_id: "summary_detail_rule_mismatch",
      description: "rule mismatch",
      severity: "warning",
      violation_count: 1,
      samples: [{ trade_id: "202604-001", ticker: "NVDA", portfolio: "CanSlim",
                  summary_rule: "br3.1", detail_rule: "br3.2" }],
      remediation: "Recompute via Trade Manager",
      duration_ms: 7,
      error: null,
    },
    {
      check_id: "closed_with_nonzero_shares",
      description: "closed shares",
      severity: "error",
      violation_count: 2,
      samples: [
        { trade_id: "202601-003", ticker: "AAPL", portfolio: "CanSlim", shares: 5 },
        { trade_id: "202601-004", ticker: "MSFT", portfolio: "CanSlim", shares: 3 },
      ],
      remediation: "Recompute LIFO",
      duration_ms: 9,
      error: null,
    },
  ],
  summary: { total_checks: 2, passed: 0, warnings: 1, errors: 1 },
};

async function openDriftSection() {
  render(<Admin navColor="#6366f1" />);
  const sectionHeader = await screen.findByText("Drift Scan");
  await act(async () => { fireEvent.click(sectionHeader); });
}

describe("Admin — Drift Scan section", () => {
  test("clean state shows green check after running scan", async () => {
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_CLEAN_RESPONSE as any);
    await openDriftSection();

    const runBtn = await screen.findByTestId("drift-scan-run-all");
    await act(async () => { fireEvent.click(runBtn); });

    expect(await screen.findByTestId("drift-scan-clean")).toBeInTheDocument();
    // Tile counts add up: 2 total / 2 passed / 0 warnings / 0 errors.
    expect(screen.getByTestId("drift-scan-tile-passed")).toHaveTextContent("2");
    expect(screen.getByTestId("drift-scan-tile-errors")).toHaveTextContent("0");
  });

  test("mixed warnings/errors render badges + counts", async () => {
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_MIXED_RESPONSE as any);
    await openDriftSection();

    await act(async () => {
      fireEvent.click(await screen.findByTestId("drift-scan-run-all"));
    });

    // Each check renders one row, with violation count visible.
    expect(await screen.findByTestId("drift-scan-row-summary_detail_rule_mismatch")).toBeInTheDocument();
    expect(screen.getByTestId("drift-scan-row-closed_with_nonzero_shares")).toBeInTheDocument();

    // Summary tile arithmetic.
    expect(screen.getByTestId("drift-scan-tile-warnings")).toHaveTextContent("1");
    expect(screen.getByTestId("drift-scan-tile-errors")).toHaveTextContent("1");
    // Clean banner is NOT shown when there are violations.
    expect(screen.queryByTestId("drift-scan-clean")).not.toBeInTheDocument();
  });

  test("Re-run scan button calls api.runDriftScan with the active portfolio", async () => {
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_CLEAN_RESPONSE as any);
    await openDriftSection();

    // Wait for portfolio dropdown to populate before clicking Re-run, so
    // the default-selection effect has a chance to set selectedPortfolio.
    await screen.findByRole("option", { name: "CanSlim" });

    await act(async () => {
      fireEvent.click(await screen.findByTestId("drift-scan-run-all"));
    });

    await waitFor(() => expect(api.runDriftScan).toHaveBeenCalled());
    // After this commit, the dropdown defaults to the active portfolio
    // and the Re-run scan button passes it through to the API. (Was {}
    // before the portfolio-filter feature; the new contract is
    // {portfolio: <active>}.)
    expect(vi.mocked(api.runDriftScan).mock.calls[0][0]).toEqual({
      portfolio: "CanSlim",
    });
  });

  test("per-row Re-run button calls api.runDriftScan with check_id + active portfolio", async () => {
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_MIXED_RESPONSE as any);
    await openDriftSection();
    await screen.findByRole("option", { name: "CanSlim" });

    // Initial scan to populate the table.
    await act(async () => {
      fireEvent.click(await screen.findByTestId("drift-scan-run-all"));
    });

    // Then mock a single-check response and click that row's Re-run.
    vi.mocked(api.runDriftScan).mockResolvedValueOnce({
      ...DRIFT_MIXED_RESPONSE,
      check_filter: "closed_with_nonzero_shares",
      checks: [{ ...DRIFT_MIXED_RESPONSE.checks[1], violation_count: 0, samples: [] }],
      summary: { total_checks: 1, passed: 1, warnings: 0, errors: 0 },
    } as any);

    const rerunBtn = await screen.findByTestId("drift-scan-rerun-closed_with_nonzero_shares");
    await act(async () => { fireEvent.click(rerunBtn); });

    await waitFor(() => expect(api.runDriftScan).toHaveBeenCalledTimes(2));
    // Per-row Re-run preserves the selected portfolio + adds checkId.
    expect(vi.mocked(api.runDriftScan).mock.calls[1][0]).toEqual({
      portfolio: "CanSlim",
      checkId: "closed_with_nonzero_shares",
    });
  });

  test("expand row reveals samples", async () => {
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_MIXED_RESPONSE as any);
    await openDriftSection();

    await act(async () => {
      fireEvent.click(await screen.findByTestId("drift-scan-run-all"));
    });

    // Samples are hidden until expanded.
    expect(screen.queryByTestId("drift-scan-samples-summary_detail_rule_mismatch")).not.toBeInTheDocument();

    const expandBtn = await screen.findByTestId("drift-scan-expand-summary_detail_rule_mismatch");
    await act(async () => { fireEvent.click(expandBtn); });

    expect(await screen.findByTestId("drift-scan-samples-summary_detail_rule_mismatch")).toBeInTheDocument();
    // The sample's trade_id appears inside the expanded row.
    expect(screen.getByText("202604-001")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Drift Scan portfolio selector (post-Commit-8 follow-up)
// Backend has supported ?portfolio= since Commit 8; this UI exposes it so
// the user can scope a scan to one portfolio without losing the option to
// run unfiltered. Default selection is the active portfolio (typically
// 'CanSlim'), with a defensive fallback to "All Portfolios" when the
// active portfolio isn't present in the listed set.
// ---------------------------------------------------------------------------

describe("Admin — Drift Scan portfolio selector", () => {
  test("renders dropdown with All Portfolios + every listed portfolio", async () => {
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_CLEAN_RESPONSE as any);
    await openDriftSection();

    const select = await screen.findByTestId("drift-scan-portfolio-select") as HTMLSelectElement;
    // Wait for listPortfolios resolution to populate options.
    await screen.findByRole("option", { name: "CanSlim" });

    // 4 options: "All Portfolios" + 3 seeded portfolios.
    const options = Array.from(select.querySelectorAll("option"));
    expect(options).toHaveLength(4);
    expect(options.map(o => o.textContent)).toEqual([
      "All Portfolios", "CanSlim", "TQQQ Strategy", "457B Plan",
    ]);
  });

  test("defaults selection to the active portfolio when present in list", async () => {
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_CLEAN_RESPONSE as any);
    await openDriftSection();
    await screen.findByRole("option", { name: "CanSlim" });

    const select = await screen.findByTestId("drift-scan-portfolio-select") as HTMLSelectElement;
    expect(select.value).toBe("CanSlim");
  });

  test("defensive: defaults to All Portfolios when active portfolio not in list", async () => {
    // Simulate a stale active-portfolio name (renamed/deleted between
    // PortfolioProvider's load and the drift-scan mount). The dropdown
    // should NOT preselect a name the backend would reject — it falls
    // back to "" so the user explicitly picks a valid scope before
    // running.
    mockGetActivePortfolio.mockReturnValue("OldGhostPortfolio");
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_CLEAN_RESPONSE as any);
    await openDriftSection();
    await screen.findByRole("option", { name: "CanSlim" });

    const select = await screen.findByTestId("drift-scan-portfolio-select") as HTMLSelectElement;
    expect(select.value).toBe("");
  });

  test("changing selection + Re-run scan calls api with the chosen portfolio", async () => {
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_CLEAN_RESPONSE as any);
    await openDriftSection();
    await screen.findByRole("option", { name: "TQQQ Strategy" });

    const select = await screen.findByTestId("drift-scan-portfolio-select") as HTMLSelectElement;
    await act(async () => {
      fireEvent.change(select, { target: { value: "TQQQ Strategy" } });
    });

    await act(async () => {
      fireEvent.click(await screen.findByTestId("drift-scan-run-all"));
    });

    await waitFor(() => expect(api.runDriftScan).toHaveBeenCalled());
    expect(vi.mocked(api.runDriftScan).mock.calls[0][0]).toEqual({
      portfolio: "TQQQ Strategy",
    });
  });

  test("selecting All Portfolios + Re-run scan calls api without a portfolio key", async () => {
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_CLEAN_RESPONSE as any);
    await openDriftSection();
    await screen.findByRole("option", { name: "CanSlim" });

    const select = await screen.findByTestId("drift-scan-portfolio-select") as HTMLSelectElement;
    await act(async () => {
      fireEvent.change(select, { target: { value: "" } });
    });

    await act(async () => {
      fireEvent.click(await screen.findByTestId("drift-scan-run-all"));
    });

    await waitFor(() => expect(api.runDriftScan).toHaveBeenCalled());
    // Empty string ⇒ no portfolio key in the request opts ⇒ backend
    // scans all portfolios. Pin this contract: the API wrapper should
    // NOT see a portfolio: "" entry (would surface as ?portfolio= and
    // backend would reject with Unknown portfolio: "").
    expect(vi.mocked(api.runDriftScan).mock.calls[0][0]).toEqual({});
  });

  test("per-row Re-run preserves the selected portfolio", async () => {
    vi.mocked(api.runDriftScan).mockResolvedValue(DRIFT_MIXED_RESPONSE as any);
    await openDriftSection();
    await screen.findByRole("option", { name: "457B Plan" });

    // Switch from default CanSlim to 457B Plan, populate the table,
    // then click a per-row Re-run.
    const select = await screen.findByTestId("drift-scan-portfolio-select") as HTMLSelectElement;
    await act(async () => {
      fireEvent.change(select, { target: { value: "457B Plan" } });
    });

    await act(async () => {
      fireEvent.click(await screen.findByTestId("drift-scan-run-all"));
    });

    vi.mocked(api.runDriftScan).mockResolvedValueOnce({
      ...DRIFT_MIXED_RESPONSE,
      portfolio_filter: "457B Plan",
      check_filter: "closed_with_nonzero_shares",
      checks: [{ ...DRIFT_MIXED_RESPONSE.checks[1], violation_count: 0, samples: [] }],
      summary: { total_checks: 1, passed: 1, warnings: 0, errors: 0 },
    } as any);

    const rerunBtn = await screen.findByTestId("drift-scan-rerun-closed_with_nonzero_shares");
    await act(async () => { fireEvent.click(rerunBtn); });

    await waitFor(() => expect(api.runDriftScan).toHaveBeenCalledTimes(2));
    expect(vi.mocked(api.runDriftScan).mock.calls[1][0]).toEqual({
      portfolio: "457B Plan",
      checkId: "closed_with_nonzero_shares",
    });
  });
});
