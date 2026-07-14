import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

class ResizeObserverStub {
  observe(): void { /* noop */ }
  unobserve(): void { /* noop */ }
  disconnect(): void { /* noop */ }
}
(globalThis as any).ResizeObserver = ResizeObserverStub;

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(), replace: vi.fn(), refresh: vi.fn(),
    back: vi.fn(), forward: vi.fn(), prefetch: vi.fn(),
  }),
}));

vi.mock("@/lib/api", () => ({
  api: {
    journalLatest: vi.fn(),
    rallyPrefix: vi.fn(),
    trailingAvgLoss: vi.fn(),
    priceLookup: vi.fn(),
  },
  getActivePortfolio: () => "CanSlim",
}));

import { api } from "@/lib/api";
import { NewEntry, computeNewEntry } from "./new-entry";

const mJournal = vi.mocked(api.journalLatest);
const mRally = vi.mocked(api.rallyPrefix);
const mAvgLoss = vi.mocked(api.trailingAvgLoss);
const mPrice = vi.mocked(api.priceLookup);


function setupDefaults(opts: {
  endNlv?: number;
  state?: string | null;
  trendCount?: number | null;
  avgLossPct?: number | null;
  sampleSize?: number;
  price?: number;
  atrPct?: number;
} = {}) {
  // Use `'key' in opts` — NOT `??` — so explicit `null` / 0 overrides
  // land instead of getting replaced by the fallback (the "no sample
  // yet" test needs a real null in the trailingAvgLoss payload).
  mJournal.mockResolvedValue({ end_nlv: "endNlv" in opts ? opts.endNlv : 1_000_000 } as any);
  mRally.mockResolvedValue({
    prefix: "",
    state: "state" in opts ? opts.state : "POWERTREND",
    trend_count: "trendCount" in opts ? opts.trendCount : 3,
    active_exits: [],
  } as any);
  mAvgLoss.mockResolvedValue({
    portfolio: "CanSlim",
    window_months: 12,
    avg_loss_pct: "avgLossPct" in opts ? (opts.avgLossPct as number | null) : -4.58,
    median_loss_pct: -3.7,
    sample_size: "sampleSize" in opts ? opts.sampleSize! : 204,
    as_of: "2026-07-13",
  });
  mPrice.mockResolvedValue({
    ticker: "ALAB",
    price: opts.price ?? 400,
    atr: 8,
    atr_pct: opts.atrPct ?? 2.0,
  } as any);
}


// ═══════════════════════════════════════════════════════════════════════
// Pure math — computeNewEntry
// The whole recommendation flows through this one function, so pin the
// math in unit tests before the render-level tests exercise the wiring.
// ═══════════════════════════════════════════════════════════════════════

describe("computeNewEntry — pure math", () => {
  test("ALAB canonical case: risk 0.75, denom 4.58, entry 400, NLV $1M → formula 16.4%, cap binds at 12.5%, ~312 shares", () => {
    // The user's motivating spec example. Actual ALAB 4/14 buy was
    // 300 shares; the model output was "~307 shares at the 12.5% cap".
    // With NLV=$1M / entry=$400: 12.5% × 1M = 125k → 125k/400 = 312.5
    // → floor = 312. This is 1% wider than the actual entry because
    // ALAB's real NLV that day was slightly less than $1M — the ratio
    // (denominator vs risk unit and cap-binding) is what pins the
    // shape of the recommendation.
    const r = computeNewEntry({
      entry: 400,
      atrPct: 2.0,
      nlv: 1_000_000,
      riskUnitPct: 0.75,     // Offense (POWERTREND)
      avgLossPct: 4.58,
      targetCapPct: 12.5,
    });
    expect(r.denominatorPct).toBeCloseTo(4.58, 4);
    expect(r.denominatorFloored).toBe(false);
    expect(r.formulaPct).toBeCloseTo(16.3755, 3);  // 0.75 / 4.58 × 100
    expect(r.capBound).toBe(true);
    expect(r.posSizePct).toBe(12.5);
    expect(r.shares).toBe(312);                    // floor(125_000 / 400)
  });

  test("4% floor binds when trailing avg loss is tighter than 4%", () => {
    // Trader tightened up; avg loss dropped below the floor. The
    // client-side minimum kicks in so a single position doesn't get
    // sized against a 1-day median stop of ~1.5%. Cap 20% is above
    // the resulting 18.75% formula, so formula wins.
    const r = computeNewEntry({
      entry: 100, atrPct: 2, nlv: 100_000,
      riskUnitPct: 0.75, avgLossPct: 2.5, targetCapPct: 20,
    });
    expect(r.denominatorPct).toBe(4.0);          // floored
    expect(r.denominatorFloored).toBe(true);
    expect(r.formulaPct).toBeCloseTo(18.75, 3);  // 0.75 / 4.0 × 100
    expect(r.capBound).toBe(false);              // 20% cap > 18.75% formula → cap does not bind
    expect(r.posSizePct).toBeCloseTo(18.75, 3);
    expect(r.shares).toBe(187);                  // floor(18,750 / 100)
  });

  test("cap binds when the formula exceeds the target cap", () => {
    // The default Overweight (12.5%) case for a canonical trader.
    const r = computeNewEntry({
      entry: 100, atrPct: 2, nlv: 100_000,
      riskUnitPct: 0.75, avgLossPct: 4.58, targetCapPct: 12.5,
    });
    expect(r.formulaPct).toBeGreaterThan(12.5);
    expect(r.capBound).toBe(true);
    expect(r.posSizePct).toBe(12.5);
  });

  test("formula wins when it lands under the cap (cap does not bind)", () => {
    // Wide realized-loss denominator → smaller formula pct. E.g. a
    // choppy 6-month window where the realized-loss average is 10%.
    const r = computeNewEntry({
      entry: 100, atrPct: 2, nlv: 100_000,
      riskUnitPct: 0.75, avgLossPct: 10.0, targetCapPct: 12.5,
    });
    expect(r.formulaPct).toBeCloseTo(7.5, 3);
    expect(r.capBound).toBe(false);
    expect(r.posSizePct).toBeCloseTo(7.5, 3);
  });

  test("gap-tail arithmetic: pos_size% × stop% / 100 (both are percents)", () => {
    // ALAB canonical: pos_size = 12.5%, stop_pct = 1.5 × 2.0 = 3.0%,
    // gap tail = 12.5 × 3.0 / 100 = 0.375% NLV.
    const r = computeNewEntry({
      entry: 400, atrPct: 2.0, nlv: 1_000_000,
      riskUnitPct: 0.75, avgLossPct: 4.58, targetCapPct: 12.5,
    });
    expect(r.stopPct).toBeCloseTo(3.0, 3);
    expect(r.stopPrice).toBeCloseTo(388.0, 2);  // 400 × (1 - 0.03)
    expect(r.gapTailPctNlv).toBeCloseTo(0.375, 3);
    expect(r.gapTailDollars).toBeCloseTo(3_750, 0);   // 0.375% of $1M
  });

  test("speculative-tier flag fires when 1.5× ATR21 > 8%", () => {
    // A 6% ATR stock: 1.5 × 6 = 9% > 8% → SPECULATIVE TIER.
    const r = computeNewEntry({
      entry: 50, atrPct: 6.0, nlv: 100_000,
      riskUnitPct: 0.75, avgLossPct: 4.58, targetCapPct: 12.5,
    });
    expect(r.stopPct).toBeCloseTo(9.0, 3);
    expect(r.isSpeculative).toBe(true);
  });

  test("speculative-tier flag does NOT fire at the boundary (stop_pct = 8%)", () => {
    // Exact-boundary case: atrPct = 8/1.5 = 5.3333 → stopPct = 8.0
    // exactly. The flag uses strict > so 8.0 is fine, 8.01 is not.
    const r = computeNewEntry({
      entry: 50, atrPct: 8 / 1.5, nlv: 100_000,
      riskUnitPct: 0.75, avgLossPct: 4.58, targetCapPct: 12.5,
    });
    expect(r.stopPct).toBeCloseTo(8.0, 3);
    expect(r.isSpeculative).toBe(false);
  });

  test("shares = floor(posDollars / entry) — never fractional", () => {
    // $10,000 position at $37 = 270.27... → 270 shares.
    const r = computeNewEntry({
      entry: 37, atrPct: 2, nlv: 100_000,
      riskUnitPct: 0.5, avgLossPct: 5.0, targetCapPct: 10,
    });
    expect(r.posDollars).toBeCloseTo(10_000, 0);
    expect(r.shares).toBe(270);
  });
});


// ═══════════════════════════════════════════════════════════════════════
// Component — mount context + rendering
// ═══════════════════════════════════════════════════════════════════════

describe("NewEntry — mount context", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("POWERTREND → auto-picks Offense (0.75%) and renders M-Factor source", async () => {
    setupDefaults({ state: "POWERTREND" });
    render(<NewEntry navColor="#08a86b" />);

    const indicator = await screen.findByTestId("new-entry-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Auto/);
      expect(indicator.textContent).toMatch(/Offense \(0\.75%\)/);
      expect(indicator.textContent).toMatch(/from M Factor POWERTREND/);
    });
    expect(screen.queryByTestId("new-entry-reset-to-auto")).not.toBeInTheDocument();
  });

  test("CORRECTION → auto-picks Pilot (0.25%)", async () => {
    setupDefaults({ state: "CORRECTION" });
    render(<NewEntry navColor="#08a86b" />);

    const indicator = await screen.findByTestId("new-entry-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Pilot \(0\.25%\)/);
      expect(indicator.textContent).toMatch(/from M Factor CORRECTION/);
    });
  });

  test("UPTREND → auto-picks Normal (0.50%)", async () => {
    setupDefaults({ state: "UPTREND" });
    render(<NewEntry navColor="#08a86b" />);

    const indicator = await screen.findByTestId("new-entry-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Normal \(0\.50%\)/));
  });
});


describe("NewEntry — Trend Count banner (passive, does not gate)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("negative trend_count → amber banner renders with the Down-Cycle Protocol copy", async () => {
    setupDefaults({ trendCount: -4 });
    render(<NewEntry navColor="#08a86b" />);
    const banner = await screen.findByTestId("new-entry-trend-count-banner");
    expect(banner.textContent).toMatch(/Trend Count negative/);
    expect(banner.textContent).toMatch(/-4/);
    expect(banner.textContent).toMatch(/Down-Cycle Protocol: SR8 cascade only/);
    expect(banner.textContent).toMatch(/does not block New Entry sizing/i);
  });

  test("positive trend_count → no banner", async () => {
    setupDefaults({ trendCount: 5 });
    render(<NewEntry navColor="#08a86b" />);
    await screen.findByTestId("new-entry-mode-indicator");
    expect(screen.queryByTestId("new-entry-trend-count-banner")).not.toBeInTheDocument();
  });

  test("null trend_count → no banner (unknown ≠ negative)", async () => {
    setupDefaults({ trendCount: null });
    render(<NewEntry navColor="#08a86b" />);
    await screen.findByTestId("new-entry-mode-indicator");
    expect(screen.queryByTestId("new-entry-trend-count-banner")).not.toBeInTheDocument();
  });

  test("banner does NOT gate the verdict — sizing still renders on negative trend_count", async () => {
    setupDefaults({ trendCount: -6 });
    render(<NewEntry navColor="#08a86b" />);
    // Enter a ticker so the price lookup + verdict path fires
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => {
      fireEvent.change(tickerInput, { target: { value: "ALAB" } });
    });
    // Advance past the 600ms debounce
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });
    // Verdict box renders even though the banner is visible
    await waitFor(() => expect(screen.getByTestId("new-entry-verdict")).toBeInTheDocument());
    expect(screen.getByTestId("new-entry-trend-count-banner")).toBeInTheDocument();
  });
});


describe("NewEntry — downward-only manual override", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults({ state: "POWERTREND" });
  });

  test("larger tier is disabled and cannot be selected via click (CORRECTION → Pilot only)", async () => {
    setupDefaults({ state: "CORRECTION" });
    render(<NewEntry navColor="#08a86b" />);

    // Auto-derived to Pilot. Normal + Offense radios exist but are disabled.
    const indicator = await screen.findByTestId("new-entry-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Pilot/));

    const offenseRadio = screen.getByRole("button", { name: /Offense/i });
    const normalRadio  = screen.getByRole("button", { name: /Normal/i });
    expect(offenseRadio).toBeDisabled();
    expect(normalRadio).toBeDisabled();

    // Clicking a disabled Radio is a no-op — indicator stays on Pilot.
    fireEvent.click(offenseRadio);
    await waitFor(() => expect(indicator.textContent).toMatch(/Pilot/));
    expect(screen.queryByTestId("new-entry-reset-to-auto")).not.toBeInTheDocument();
  });

  test("smaller tier IS allowed and flips indicator to Manual + shows Reset (POWERTREND → Pilot)", async () => {
    render(<NewEntry navColor="#08a86b" />);
    const indicator = await screen.findByTestId("new-entry-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Offense/));

    const pilotRadio = screen.getByRole("button", { name: /Pilot/i });
    expect(pilotRadio).not.toBeDisabled();
    await act(async () => { fireEvent.click(pilotRadio); });

    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Manual/);
      expect(indicator.textContent).toMatch(/Pilot/);
      expect(indicator.textContent).toMatch(/downward override/);
    });
    expect(screen.getByTestId("new-entry-reset-to-auto")).toBeInTheDocument();
  });

  test("Reset to auto restores the regime-mapped tier", async () => {
    render(<NewEntry navColor="#08a86b" />);
    // Downshift to Pilot
    const pilotRadio = await screen.findByRole("button", { name: /Pilot/i });
    await act(async () => { fireEvent.click(pilotRadio); });
    // Then reset
    const reset = await screen.findByTestId("new-entry-reset-to-auto");
    await act(async () => { fireEvent.click(reset); });

    const indicator = screen.getByTestId("new-entry-mode-indicator");
    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Auto/);
      expect(indicator.textContent).toMatch(/Offense/);
    });
    expect(screen.queryByTestId("new-entry-reset-to-auto")).not.toBeInTheDocument();
  });
});


describe("NewEntry — trailing avg-loss handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("null avg_loss_pct + sample=0 → tile shows 'No sample yet' + floor copy", async () => {
    setupDefaults({ avgLossPct: null, sampleSize: 0 });
    render(<NewEntry navColor="#08a86b" />);
    const tile = await screen.findByTestId("new-entry-avgloss-tile");
    await waitFor(() => {
      expect(tile.textContent).toMatch(/No sample yet/);
      expect(tile.textContent).toMatch(/uses the 4% floor/);
    });
  });

  test("avg_loss_pct present → tile shows the aggregate + sample size + effective denominator", async () => {
    setupDefaults({ avgLossPct: -4.58, sampleSize: 204 });
    render(<NewEntry navColor="#08a86b" />);
    const tile = await screen.findByTestId("new-entry-avgloss-tile");
    await waitFor(() => {
      expect(tile.textContent).toMatch(/−4\.58%/);
      expect(tile.textContent).toMatch(/204 closed equity losses/);
      expect(tile.textContent).toMatch(/Denominator used: 4\.58%/);
    });
  });

  test("avg_loss_pct tighter than floor → tile shows the floor override", async () => {
    setupDefaults({ avgLossPct: -2.4, sampleSize: 30 });
    render(<NewEntry navColor="#08a86b" />);
    const tile = await screen.findByTestId("new-entry-avgloss-tile");
    await waitFor(() => {
      expect(tile.textContent).toMatch(/−2\.40%/);
      expect(tile.textContent).toMatch(/Denominator used: 4\.00%/);
      expect(tile.textContent).toMatch(/floored at 4%/);
    });
  });
});


describe("NewEntry — verdict rendering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults({
      state: "POWERTREND",
      endNlv: 1_000_000,
      avgLossPct: -4.58,
      sampleSize: 204,
      price: 400,
      atrPct: 2.0,
    });
  });

  test("ALAB canonical case flows end-to-end — verdict shows 312 shares · 12.5% NLV · gap tail 0.38%", async () => {
    render(<NewEntry navColor="#08a86b" />);
    await screen.findByTestId("new-entry-mode-indicator");

    // Enter ALAB → auto-fills entry $400 + atr_pct 2.0
    const tickerInput = screen.getByPlaceholderText("XYZ");
    await act(async () => {
      fireEvent.change(tickerInput, { target: { value: "ALAB" } });
    });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    const verdict = await screen.findByTestId("new-entry-verdict");
    await waitFor(() => {
      expect(verdict.textContent).toMatch(/Buy 312 shares/);
      expect(verdict.textContent).toMatch(/12\.5% NLV/);
      expect(verdict.textContent).toMatch(/risk 0\.75% NLV/);
      expect(verdict.textContent).toMatch(/gap tail 0\.38% NLV/);
      expect(verdict.textContent).toMatch(/Offense risk unit/);
    });
  });

  test("cap-bound wording surfaces when target cap tightens formula output", async () => {
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "ALAB" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    const verdict = await screen.findByTestId("new-entry-verdict");
    // Default cap = 12.5% Overweight; formula = 16.4% → cap binds
    await waitFor(() => {
      expect(verdict.textContent).toMatch(/BOUND/);
      expect(verdict.textContent).toMatch(/cap tighter than formula/i);
    });
  });

  test("formula-under-cap wording when a very wide realized-loss denominator shrinks the formula below the cap", async () => {
    setupDefaults({
      state: "POWERTREND",
      endNlv: 1_000_000,
      avgLossPct: -15.0,   // ugly stretch, wide denominator
      sampleSize: 20,
      price: 400,
      atrPct: 2.0,
    });
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "ALAB" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    const verdict = await screen.findByTestId("new-entry-verdict");
    // Formula = 0.75 / 15 × 100 = 5% < 12.5% cap → formula wins
    await waitFor(() => expect(verdict.textContent).toMatch(/formula 5\.00% is under the cap/i));
  });

  test("SPECULATIVE TIER warning fires when 1.5× ATR21 > 8%", async () => {
    setupDefaults({
      state: "POWERTREND",
      endNlv: 1_000_000,
      avgLossPct: -4.58,
      sampleSize: 204,
      price: 50,
      atrPct: 6.0,  // 1.5 × 6 = 9% stop → speculative
    });
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "MEME" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    const warning = await screen.findByTestId("new-entry-speculative-warning");
    expect(warning.textContent).toMatch(/SPECULATIVE TIER/);
    expect(warning.textContent).toMatch(/9\.00%/);
  });

  test("SPECULATIVE TIER warning does NOT fire on normal-volatility tickers", async () => {
    // ALAB canonical (atr 2.0 → stop_pct 3.0%) is nowhere near the 8% trip.
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "ALAB" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });
    await screen.findByTestId("new-entry-verdict");
    expect(screen.queryByTestId("new-entry-speculative-warning")).not.toBeInTheDocument();
  });

  test("empty inputs → placeholder message renders instead of a verdict", async () => {
    render(<NewEntry navColor="#08a86b" />);
    // Auto-context lands but no ticker typed → guard fails
    await screen.findByTestId("new-entry-mode-indicator");
    expect(screen.queryByTestId("new-entry-verdict")).not.toBeInTheDocument();
    expect(screen.getByText(/Enter a ticker \+ entry price/)).toBeInTheDocument();
  });
});
