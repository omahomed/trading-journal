import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

class ResizeObserverStub {
  observe(): void { /* noop */ }
  unobserve(): void { /* noop */ }
  disconnect(): void { /* noop */ }
}
(globalThis as any).ResizeObserver = ResizeObserverStub;

// localStorage polyfill — Log Sell reads ps_prefill_sell on mount via a
// try/catch, so we need the methods to exist even though these tests
// don't write to it. JSDOM's default stub throws on most methods.
const _lsStore = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    getItem: (k: string) => _lsStore.get(k) ?? null,
    setItem: (k: string, v: string) => { _lsStore.set(k, v); },
    removeItem: (k: string) => { _lsStore.delete(k); },
    clear: () => { _lsStore.clear(); },
    key: (i: number) => Array.from(_lsStore.keys())[i] ?? null,
    get length() { return _lsStore.size; },
  },
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(), replace: vi.fn(), refresh: vi.fn(),
    back: vi.fn(), forward: vi.fn(), prefetch: vi.fn(),
  }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    tradesOpen: vi.fn(),
    tradesOpenDetails: vi.fn(),
    getTradeLessons: vi.fn(),
    logSell: vi.fn(),
    uploadImage: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

vi.mock("@/lib/upload-with-timeout", () => ({
  uploadWithTimeout: vi.fn(),
  DEFAULT_UPLOAD_TIMEOUT_MS: 60_000,
}));

import { api } from "@/lib/api";
import { uploadWithTimeout } from "@/lib/upload-with-timeout";
import { LogSell } from "./log-sell";

const mUpload = vi.mocked(uploadWithTimeout);

// Five open campaigns covering the search + option scenarios the tests
// need. Mixes stocks and one option so the OPTION ×100 banner test has
// a deterministic target.
const STOCK_AAPL = {
  trade_id: "202604-001", ticker: "AAPL", shares: 100, avg_entry: 180.0,
  instrument_type: "STOCK", multiplier: 1,
} as any;
const STOCK_MSFT = {
  trade_id: "202604-013", ticker: "MSFT", shares: 50, avg_entry: 410.0,
  instrument_type: "STOCK", multiplier: 1,
} as any;
const STOCK_NVDA = {
  trade_id: "202605-007", ticker: "NVDA", shares: 200, avg_entry: 130.5,
  instrument_type: "STOCK", multiplier: 1,
} as any;
const OPTION_FIVN = {
  trade_id: "202605-028", ticker: "FIVN 261016 $25C", shares: 10, avg_entry: 3.20,
  instrument_type: "OPTION", multiplier: 100,
} as any;
const STOCK_DELL = {
  trade_id: "202604-099", ticker: "DELL", shares: 303, avg_entry: 142.0,
  instrument_type: "STOCK", multiplier: 1,
} as any;

const OPEN_TRADES = [STOCK_AAPL, STOCK_MSFT, STOCK_NVDA, OPTION_FIVN, STOCK_DELL];


function setupDefaults() {
  vi.mocked(api.tradesOpen).mockResolvedValue(OPEN_TRADES);
  vi.mocked(api.tradesOpenDetails).mockResolvedValue({ details: [], lot_closures: [] } as any);
  vi.mocked(api.getTradeLessons).mockResolvedValue({ lessons: {} } as any);
  vi.mocked(api.logSell).mockResolvedValue({ trx_id: "S1" } as any);
  vi.mocked(api.uploadImage).mockResolvedValue({ url: "" } as any);
}

// Open the campaign picker. Mirrors log-buy.test.tsx's pattern: find the
// FormField label, grab the first <button> inside it (the SearchSelect
// trigger), and click.
async function openCampaignPicker(): Promise<void> {
  const field = screen.getByText("Select Campaign");
  const trigger = field.parentElement?.querySelector("button") as HTMLButtonElement;
  if (!trigger) throw new Error("campaign trigger not found");
  await act(async () => { fireEvent.click(trigger); });
}

function fillByLabel(labelText: string, value: string): void {
  const label = screen.getByText(labelText);
  const input = label.parentElement?.querySelector("input, textarea") as
    | HTMLInputElement
    | HTMLTextAreaElement
    | null;
  if (!input) throw new Error(`No input found in Field "${labelText}"`);
  fireEvent.change(input, { target: { value } });
}


describe("LogSell — campaign combobox (Audit-c extraction)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    _lsStore.clear();
    setupDefaults();
  });

  test("combobox renders all open campaigns after mount", async () => {
    render(<LogSell navColor="#08a86b" />);
    // Wait for the tradesOpen fetch to settle and the form to render.
    await screen.findByText("Select Campaign");
    await openCampaignPicker();

    // All five campaigns visible — each labeled "ticker (shares unit) | trade_id"
    // with "contracts" for OPTION and "shares" otherwise.
    expect(screen.getByText("AAPL (100 shares) | 202604-001")).toBeInTheDocument();
    expect(screen.getByText("MSFT (50 shares) | 202604-013")).toBeInTheDocument();
    expect(screen.getByText("NVDA (200 shares) | 202605-007")).toBeInTheDocument();
    expect(screen.getByText("FIVN 261016 $25C (10 contracts) | 202605-028")).toBeInTheDocument();
    expect(screen.getByText("DELL (303 shares) | 202604-099")).toBeInTheDocument();
  });

  test("typing filters by ticker substring", async () => {
    render(<LogSell navColor="#08a86b" />);
    await screen.findByText("Select Campaign");
    await openCampaignPicker();

    // Type "AAPL" into the search input. SearchSelect autofocuses the
    // search input on open, so the screen-found "Type to search..."
    // placeholder is the right target.
    const searchInput = screen.getByPlaceholderText("Type to search...") as HTMLInputElement;
    await act(async () => {
      fireEvent.change(searchInput, { target: { value: "AAPL" } });
    });

    // Only AAPL visible; other tickers filtered out.
    expect(screen.getByText("AAPL (100 shares) | 202604-001")).toBeInTheDocument();
    expect(screen.queryByText("MSFT (50 shares) | 202604-013")).not.toBeInTheDocument();
    expect(screen.queryByText("DELL (303 shares) | 202604-099")).not.toBeInTheDocument();
  });

  test("typing filters by trade_id substring", async () => {
    render(<LogSell navColor="#08a86b" />);
    await screen.findByText("Select Campaign");
    await openCampaignPicker();

    // "202604" should match the three trades whose ID starts with 202604
    // (AAPL/MSFT/DELL) and hide the two 202605 trades (NVDA/FIVN).
    const searchInput = screen.getByPlaceholderText("Type to search...") as HTMLInputElement;
    await act(async () => {
      fireEvent.change(searchInput, { target: { value: "202604" } });
    });

    // The combobox dropdown contains entries in "ticker (N unit) | trade_id"
    // form; the right-side Open Campaigns panel renders the same tickers
    // separately as plain labels. We assert only on the dropdown-shape
    // labels to scope the filter check to the combobox.
    expect(screen.getByText("AAPL (100 shares) | 202604-001")).toBeInTheDocument();
    expect(screen.getByText("MSFT (50 shares) | 202604-013")).toBeInTheDocument();
    expect(screen.getByText("DELL (303 shares) | 202604-099")).toBeInTheDocument();
    expect(screen.queryByText("NVDA (200 shares) | 202605-007")).not.toBeInTheDocument();
    expect(screen.queryByText("FIVN 261016 $25C (10 contracts) | 202605-028")).not.toBeInTheDocument();
  });

  test("selecting a campaign sets selectedTrade + renders the hint", async () => {
    render(<LogSell navColor="#08a86b" />);
    await screen.findByText("Select Campaign");
    await openCampaignPicker();

    // Pick MSFT
    await act(async () => {
      fireEvent.click(screen.getByText("MSFT (50 shares) | 202604-013"));
    });

    // FormField hint surfaces "50 shares @ $410.00 avg" after selection.
    await waitFor(() => {
      expect(screen.getByText(/50 shares @ \$410\.00 avg/)).toBeInTheDocument();
    });
    // The combobox trigger now displays the selected label.
    const trigger = screen.getByText("Select Campaign").parentElement?.querySelector("button") as HTMLButtonElement;
    expect(trigger.textContent).toContain("MSFT (50 shares) | 202604-013");
  });

  test("selecting an option-ticker campaign triggers the OPTION ×100 banner", async () => {
    render(<LogSell navColor="#08a86b" />);
    await screen.findByText("Select Campaign");
    await openCampaignPicker();

    // Pick the option campaign — banner is gated on instrument_type=OPTION
    // OR ticker shape match. FIVN's row satisfies both.
    await act(async () => {
      fireEvent.click(screen.getByText("FIVN 261016 $25C (10 contracts) | 202605-028"));
    });

    await waitFor(() => {
      expect(screen.getByText(/OPTION ×100/)).toBeInTheDocument();
      expect(screen.getByText(/Proceeds and realized P&L shown as notional/)).toBeInTheDocument();
    });
    // Hint reflects "contracts" unit, not "shares".
    expect(screen.getByText(/10 contracts @ \$3\.20 avg/)).toBeInTheDocument();
  });

  test("submit body has trade_id matching the selection", async () => {
    render(<LogSell navColor="#08a86b" />);
    await screen.findByText("Select Campaign");
    await openCampaignPicker();

    await act(async () => {
      fireEvent.click(screen.getByText("NVDA (200 shares) | 202605-007"));
    });
    // Need to wait for hint to render before filling the form so the
    // Field components below the picker exist in the DOM.
    await waitFor(() => {
      expect(screen.getByText(/200 shares @ \$130\.50 avg/)).toBeInTheDocument();
    });

    fillByLabel("Shares to Sell", "50");
    fillByLabel("Sell Price ($)", "145.00");

    await act(async () => {
      fireEvent.click(screen.getByText("LOG SELL ORDER"));
    });

    await waitFor(() => expect(api.logSell).toHaveBeenCalled());
    const body = vi.mocked(api.logSell).mock.calls[0][0] as Record<string, unknown>;
    expect(body.trade_id).toBe("202605-007");
    expect(body.shares).toBe(50);
    expect(body.price).toBe(145);
  });
});


// ─────────────────────────────────────────────────────────────────────
// Background upload tracker — Log Sell side.
// Same UX as Log Buy: position-change uploads run in the background via
// uploadWithTimeout, surface per-file status in <UploadTracker>, and
// the submit chain doesn't await them.
// ─────────────────────────────────────────────────────────────────────
describe("LogSell — background upload tracker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.tradesOpen).mockResolvedValue([STOCK_NVDA] as any);
    vi.mocked(api.logSell).mockResolvedValue({ trx_id: "S1", remaining_shares: 150, is_closed: false, realized_pl: 720 } as any);
  });

  async function setupAndSubmitWithChart() {
    render(<LogSell navColor="#08a86b" />);
    await screen.findByText("Select Campaign");
    await openCampaignPicker();
    await act(async () => {
      fireEvent.click(screen.getByText("NVDA (200 shares) | 202605-007"));
    });
    await waitFor(() => {
      expect(screen.getByText(/200 shares @ \$130\.50 avg/)).toBeInTheDocument();
    });
    fillByLabel("Shares to Sell", "50");
    fillByLabel("Sell Price ($)", "145.00");

    // Attach a position-change chart via the hidden file input.
    const fileInputs = document.querySelectorAll('input[type="file"]') as NodeListOf<HTMLInputElement>;
    expect(fileInputs.length).toBeGreaterThan(0);
    const file = new File(["trim-bytes"], "trim-chart.png", { type: "image/png" });
    await act(async () => {
      fireEvent.change(fileInputs[0], { target: { files: [file] } });
    });

    await act(async () => {
      fireEvent.click(screen.getByText("LOG SELL ORDER"));
    });

    return file;
  }

  test("submit with chart fires uploadWithTimeout (position_change) and shows tracker", async () => {
    let resolveUpload: (v: { ok: boolean }) => void = () => {};
    mUpload.mockReturnValue(new Promise(r => { resolveUpload = r; }));

    const file = await setupAndSubmitWithChart();
    await waitFor(() => expect(api.logSell).toHaveBeenCalled());

    const tracker = await screen.findByTestId("upload-tracker");
    expect(tracker.textContent).toContain("trim-chart.png");
    expect(tracker.textContent).toContain("Uploading");

    expect(mUpload).toHaveBeenCalledTimes(1);
    const [calledFile, , , , kind] = mUpload.mock.calls[0];
    expect(calledFile).toBe(file);
    expect(kind).toBe("position_change");

    await act(async () => { resolveUpload({ ok: true }); await Promise.resolve(); });
    await waitFor(() => {
      expect(screen.getByTestId("upload-tracker").textContent).toContain("Uploaded");
    });
  });

  test("failed upload shows Retry button; clicking it re-fires uploadWithTimeout", async () => {
    mUpload.mockResolvedValueOnce({ ok: false, error: "R2 unreachable" });

    await setupAndSubmitWithChart();
    const tracker = await screen.findByTestId("upload-tracker");
    await waitFor(() => expect(tracker.textContent).toContain("Failed"));
    expect(tracker.textContent).toContain("R2 unreachable");

    const retryButton = tracker.querySelector('[data-testid^="upload-retry-"]') as HTMLButtonElement;
    expect(retryButton).toBeTruthy();

    mUpload.mockResolvedValueOnce({ ok: true });
    await act(async () => { fireEvent.click(retryButton); });

    await waitFor(() => expect(mUpload).toHaveBeenCalledTimes(2));
    await waitFor(() => {
      expect(screen.getByTestId("upload-tracker").textContent).toContain("Uploaded");
    });
  });

  test("submit chain does NOT await uploads (button re-enables even if upload stalls)", async () => {
    mUpload.mockReturnValue(new Promise(() => {})); // never resolves

    await setupAndSubmitWithChart();
    await waitFor(() => expect(api.logSell).toHaveBeenCalled());

    // Proxy for "submitting flipped back to false": the post-submit
    // tradesOpen refetch ran (mount call + this one = 2). If handleSubmit
    // were awaiting the upload, this second call would never fire.
    await waitFor(() => expect(api.tradesOpen).toHaveBeenCalledTimes(2));

    const submitBtn = screen.getByText(/LOG SELL ORDER/) as HTMLButtonElement;
    expect(submitBtn.disabled).toBe(false);
  });
});
