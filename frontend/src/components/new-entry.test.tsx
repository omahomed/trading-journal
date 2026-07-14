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
  // `'key' in opts` — NOT `??` — so explicit null / 0 land instead of
  // falling back to the default (the "no sample yet" test needs a real
  // null in trailingAvgLoss).
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
// ═══════════════════════════════════════════════════════════════════════

describe("computeNewEntry — non-speculative (default denominator: trailing avg loss)", () => {
  test("ALAB canonical: risk 0.75, denom 4.58, entry 400, NLV $1M → formula 16.4%, cap binds at 12.5%, 312 shares", () => {
    // Foundational case. Non-spec (2.0% ATR × 1.5 = 3% < 8% tripwire).
    // Formula = 0.75 / 4.58 × 100 = 16.375. Cap 12.5 < formula → cap binds.
    // shares = floor(12.5% × $1M / $400) = 312.
    const r = computeNewEntry({
      entry: 400, atrPct: 2.0, nlv: 1_000_000,
      riskUnitPct: 0.75, avgLossPct: 4.58, targetCapPct: 12.5,
    });
    expect(r.isSpeculative).toBe(false);
    expect(r.denominatorSource).toBe("avg-loss");
    expect(r.denominatorPct).toBeCloseTo(4.58, 4);
    expect(r.denominatorFloored).toBe(false);
    expect(r.formulaPct).toBeCloseTo(16.3755, 3);
    expect(r.capBound).toBe(true);
    expect(r.posSizePct).toBe(12.5);
    expect(r.shares).toBe(312);
    expect(r.blocked).toBe(false);
    // No stop validation banner for non-spec cases.
    expect(r.stopValidation).toBe("none");
  });

  test("4% floor binds when trailing avg loss is tighter than 4%", () => {
    const r = computeNewEntry({
      entry: 100, atrPct: 2, nlv: 100_000,
      riskUnitPct: 0.75, avgLossPct: 2.5, targetCapPct: 20,
    });
    expect(r.denominatorPct).toBe(4.0);
    expect(r.denominatorFloored).toBe(true);
    expect(r.formulaPct).toBeCloseTo(18.75, 3);
    expect(r.capBound).toBe(false);         // 20% cap > 18.75% formula
    expect(r.posSizePct).toBeCloseTo(18.75, 3);
    expect(r.shares).toBe(187);
  });

  test("null avg loss (no sample) → falls back to floor and marks it floored", () => {
    const r = computeNewEntry({
      entry: 100, atrPct: 2, nlv: 100_000,
      riskUnitPct: 0.75, avgLossPct: null, targetCapPct: 20,
    });
    expect(r.denominatorPct).toBe(4.0);
    expect(r.denominatorFloored).toBe(true);
    expect(r.denominatorSource).toBe("avg-loss");
  });

  test("cap binds when the formula exceeds the target cap", () => {
    const r = computeNewEntry({
      entry: 100, atrPct: 2, nlv: 100_000,
      riskUnitPct: 0.75, avgLossPct: 4.58, targetCapPct: 12.5,
    });
    expect(r.formulaPct!).toBeGreaterThan(12.5);
    expect(r.capBound).toBe(true);
    expect(r.posSizePct).toBe(12.5);
  });

  test("formula wins when it lands under the cap", () => {
    // Wide realized-loss denominator → smaller formula pct.
    const r = computeNewEntry({
      entry: 100, atrPct: 2, nlv: 100_000,
      riskUnitPct: 0.75, avgLossPct: 10.0, targetCapPct: 12.5,
    });
    expect(r.formulaPct).toBeCloseTo(7.5, 3);
    expect(r.capBound).toBe(false);
    expect(r.posSizePct).toBeCloseTo(7.5, 3);
  });

  test("gap-tail = position_size_% × active_denominator_% / 100", () => {
    // Non-spec: gap-tail uses the avg-loss denominator (the expected
    // exit loss), NOT the ATR stop.
    const r = computeNewEntry({
      entry: 400, atrPct: 2.0, nlv: 1_000_000,
      riskUnitPct: 0.75, avgLossPct: 4.58, targetCapPct: 12.5,
    });
    // gap_tail = 12.5 × 4.58 / 100 = 0.5725% NLV
    expect(r.gapTailPctNlv).toBeCloseTo(0.5725, 3);
    expect(r.gapTailDollars).toBeCloseTo(5_725, 0);
  });

  test("ATR catastrophe backstop still populated on non-spec cases (verdict card shows it as reference)", () => {
    const r = computeNewEntry({
      entry: 400, atrPct: 2.0, nlv: 1_000_000,
      riskUnitPct: 0.75, avgLossPct: 4.58, targetCapPct: 12.5,
    });
    expect(r.atrStopPct).toBeCloseTo(3.0, 3);       // 1.5 × 2.0
    expect(r.atrStopPrice).toBeCloseTo(388, 2);     // 400 × (1 - 0.03)
  });

  test("tech-stop math is populated whenever MA is provided (even non-spec)", () => {
    // Informational: on non-spec cases the tech stop is displayed as
    // one of the two catastrophe-backstop reference rows.
    const r = computeNewEntry({
      entry: 400, atrPct: 2.0, nlv: 1_000_000,
      riskUnitPct: 0.75, avgLossPct: 4.58, targetCapPct: 12.5,
      maLevel: 395, bufferPct: 1.0,
    });
    expect(r.techStopPrice).toBeCloseTo(391.05, 2);       // 395 × 0.99
    expect(r.techStopDistPct).toBeCloseTo(2.2375, 3);     // (400-391.05)/400 × 100
    expect(r.denominatorSource).toBe("avg-loss");         // not used for sizing
    // Non-spec cases have no stop-validation banner regardless of MA.
    expect(r.stopValidation).toBe("none");
  });

  test("shares = floor(posDollars / entry) — never fractional", () => {
    const r = computeNewEntry({
      entry: 37, atrPct: 2, nlv: 100_000,
      riskUnitPct: 0.5, avgLossPct: 5.0, targetCapPct: 10,
    });
    expect(r.posDollars).toBeCloseTo(10_000, 0);
    expect(r.shares).toBe(270);
  });
});


describe("computeNewEntry — speculative tier: no MA → 1.5× ATR21 fallback", () => {
  test("high-ATR name, no MA entered → sizes off 1.5× ATR21 (fallback), no validation banner", () => {
    // 6% ATR: 1.5 × 6 = 9% > 8% → speculative.
    const r = computeNewEntry({
      entry: 50, atrPct: 6.0, nlv: 100_000,
      riskUnitPct: 0.75, avgLossPct: 4.58, targetCapPct: 12.5,
      // maLevel intentionally omitted
    });
    expect(r.isSpeculative).toBe(true);
    expect(r.denominatorSource).toBe("atr-fallback");
    expect(r.denominatorPct).toBeCloseTo(9.0, 3);      // 1.5 × 6.0
    expect(r.formulaPct).toBeCloseTo(8.333, 2);        // 0.75 / 9.0 × 100
    expect(r.stopValidation).toBe("none");             // no banner without MA
    expect(r.blocked).toBe(false);
  });
});


describe("computeNewEntry — speculative tier: validation states with MA", () => {
  test("FIG worked example: entry 23.65, ATR 8.28%, MA 20.71, buffer 1.0 → tech stop 20.50, 1.61× ATR → AMBER, 527 shares @ 1.88% NLV", () => {
    // The user's motivating example from the refinement spec.
    // tech_stop     = 20.71 × 0.99 = 20.5029
    // tech_stop_dist = (23.65 - 20.5029) / 23.65 = 13.30%   (spec says 13.32; matches with buffer 1.0)
    // stop_atr_mult = 13.30 / 8.28 = 1.606
    // size          = 0.25 / 13.30 × 100 = 1.88% NLV
    const r = computeNewEntry({
      entry: 23.65, atrPct: 8.28, nlv: 664_444,
      riskUnitPct: 0.25, avgLossPct: 4.34, targetCapPct: 12.5,
      maLevel: 20.71, bufferPct: 1.0,
    });
    expect(r.isSpeculative).toBe(true);
    expect(r.denominatorSource).toBe("tech-stop");
    expect(r.techStopPrice!).toBeCloseTo(20.5029, 3);
    expect(r.techStopDistPct!).toBeCloseTo(13.30, 1);
    expect(r.stopAtrMult!).toBeCloseTo(1.606, 2);
    expect(r.stopValidation).toBe("amber");
    // Formula 1.88% ≪ cap 12.5% → formula wins.
    expect(r.formulaPct!).toBeCloseTo(1.88, 1);
    expect(r.capBound).toBe(false);
    expect(r.posSizePct!).toBeCloseTo(1.88, 1);
    // pos $ = 1.88% × 664,444 ≈ $12,489; shares = floor($12,489 / $23.65) ≈ 527
    expect(r.shares).toBe(527);
    // max_valid_entry = 20.5029 / (1 - 1.5 × 8.28/100) = 20.5029 / 0.8758 ≈ 23.41
    expect(r.maxValidEntry!).toBeCloseTo(23.41, 1);
    // gap tail = 1.88% × 13.30% / 100 ≈ 0.25% NLV
    expect(r.gapTailPctNlv!).toBeCloseTo(0.25, 1);
  });

  test("GREEN — stop_atr_mult inside [MIN, TARGET]", () => {
    // Choose MA so tech_stop_dist ≈ 10%, ATR = 8.5% → mult ≈ 1.18 (green).
    const r = computeNewEntry({
      entry: 100, atrPct: 8.5, nlv: 500_000,
      riskUnitPct: 0.25, avgLossPct: 4.5, targetCapPct: 12.5,
      maLevel: 90.91, bufferPct: 0,  // stop = 90.91 → dist ≈ 9.09%; mult ≈ 1.07
    });
    expect(r.isSpeculative).toBe(true);
    expect(r.stopValidation).toBe("green");
    expect(r.stopAtrMult!).toBeGreaterThanOrEqual(1.0);
    expect(r.stopAtrMult!).toBeLessThanOrEqual(1.5);
    expect(r.blocked).toBe(false);
    expect(r.maxValidEntry).toBeNull();  // no suggestion when green
  });

  test("RED (hard block) — stop_atr_mult below MIN → sizing suppressed", () => {
    // MA very close to entry → tight stop on volatile name → RED.
    // ATR 8.5% × 1.5 = 12.75% > 8% (spec tier). Tech stop dist = 3%
    // → mult 3/8.5 = 0.35 (well below 1.0).
    const r = computeNewEntry({
      entry: 100, atrPct: 8.5, nlv: 500_000,
      riskUnitPct: 0.25, avgLossPct: 4.5, targetCapPct: 12.5,
      maLevel: 98, bufferPct: 1.0,   // stop = 97.02, dist ≈ 2.98%
    });
    expect(r.stopValidation).toBe("red");
    expect(r.blocked).toBe(true);
    expect(r.formulaPct).toBeNull();
    expect(r.posSizePct).toBeNull();
    expect(r.shares).toBeNull();
    expect(r.gapTailPctNlv).toBeNull();
    // Tech stop numbers still returned so the verdict card can show them.
    expect(r.techStopPrice).not.toBeNull();
    expect(r.stopAtrMult!).toBeLessThan(1.0);
  });

  test("AMBER — max_valid_entry brings stop_atr_mult back to TARGET when applied", () => {
    // AMBER case: verify that entering the suggested max_valid_entry
    // would produce stop_atr_mult exactly at TARGET.
    const inputs = {
      entry: 23.65, atrPct: 8.28, nlv: 664_444,
      riskUnitPct: 0.25, avgLossPct: 4.34, targetCapPct: 12.5,
      maLevel: 20.71, bufferPct: 1.0,
    };
    const r1 = computeNewEntry(inputs);
    expect(r1.stopValidation).toBe("amber");
    // Re-run at the suggested entry.
    const r2 = computeNewEntry({ ...inputs, entry: r1.maxValidEntry! });
    expect(r2.stopAtrMult!).toBeCloseTo(1.5, 2);
    expect(r2.stopValidation).toBe("green");  // exactly at TARGET → green
  });

  test("gap tail on speculative tier uses tech-stop-distance, not the ATR stop", () => {
    // FIG case again: gap tail = 1.88% × 13.30% / 100 ≈ 0.25% NLV,
    // NOT 1.88% × 12.42% / 100 ≈ 0.23% (which the 1.5× ATR path would
    // give). This test pins that distinction.
    const r = computeNewEntry({
      entry: 23.65, atrPct: 8.28, nlv: 664_444,
      riskUnitPct: 0.25, avgLossPct: 4.34, targetCapPct: 12.5,
      maLevel: 20.71, bufferPct: 1.0,
    });
    expect(r.denominatorSource).toBe("tech-stop");
    expect(r.gapTailPctNlv!).toBeCloseTo(0.25, 1);
    // ATR-based number for reference (not used):
    const atrBased = (r.posSizePct! * (1.5 * 8.28)) / 100;
    expect(atrBased).toBeCloseTo(0.233, 2);
    expect(r.gapTailPctNlv).not.toBeCloseTo(atrBased, 2);
  });
});


// ═══════════════════════════════════════════════════════════════════════
// Component — mount + banner behavior
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
    await waitFor(() => expect(indicator.textContent).toMatch(/Pilot \(0\.25%\)/));
  });
});


describe("NewEntry — trend count banner (passive)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
  });

  test("negative trend_count → amber banner renders", async () => {
    setupDefaults({ trendCount: -4 });
    render(<NewEntry navColor="#08a86b" />);
    const banner = await screen.findByTestId("new-entry-trend-count-banner");
    expect(banner.textContent).toMatch(/Trend Count negative/);
    expect(banner.textContent).toMatch(/Down-Cycle Protocol: SR8 cascade only/);
  });

  test("positive trend_count → no banner", async () => {
    setupDefaults({ trendCount: 5 });
    render(<NewEntry navColor="#08a86b" />);
    await screen.findByTestId("new-entry-mode-indicator");
    expect(screen.queryByTestId("new-entry-trend-count-banner")).not.toBeInTheDocument();
  });
});


describe("NewEntry — downward-only manual override", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults({ state: "POWERTREND" });
  });

  test("smaller tier IS allowed and flips indicator to Manual", async () => {
    render(<NewEntry navColor="#08a86b" />);
    const indicator = await screen.findByTestId("new-entry-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Offense/));

    const pilotRadio = screen.getByRole("button", { name: /Pilot/i });
    await act(async () => { fireEvent.click(pilotRadio); });

    await waitFor(() => {
      expect(indicator.textContent).toMatch(/Manual/);
      expect(indicator.textContent).toMatch(/Pilot/);
    });
    expect(screen.getByTestId("new-entry-reset-to-auto")).toBeInTheDocument();
  });

  test("larger tier is disabled in CORRECTION", async () => {
    setupDefaults({ state: "CORRECTION" });
    render(<NewEntry navColor="#08a86b" />);
    await screen.findByTestId("new-entry-mode-indicator");
    expect(screen.getByRole("button", { name: /Offense/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Normal/i })).toBeDisabled();
  });

  test("Reset to auto restores the regime-mapped tier", async () => {
    render(<NewEntry navColor="#08a86b" />);
    const pilotRadio = await screen.findByRole("button", { name: /Pilot/i });
    await act(async () => { fireEvent.click(pilotRadio); });
    const reset = await screen.findByTestId("new-entry-reset-to-auto");
    await act(async () => { fireEvent.click(reset); });

    const indicator = screen.getByTestId("new-entry-mode-indicator");
    await waitFor(() => expect(indicator.textContent).toMatch(/Offense/));
    expect(screen.queryByTestId("new-entry-reset-to-auto")).not.toBeInTheDocument();
  });
});


describe("NewEntry — trailing avg-loss tile", () => {
  beforeEach(() => { vi.clearAllMocks(); });

  test("null avg_loss_pct + sample=0 → 'No sample yet' + floor copy", async () => {
    setupDefaults({ avgLossPct: null, sampleSize: 0 });
    render(<NewEntry navColor="#08a86b" />);
    const tile = await screen.findByTestId("new-entry-avgloss-tile");
    await waitFor(() => {
      expect(tile.textContent).toMatch(/No sample yet/);
      expect(tile.textContent).toMatch(/uses the 4% floor/);
    });
  });

  test("populated avg_loss_pct → tile shows aggregate + sample size + effective denominator", async () => {
    setupDefaults({ avgLossPct: -4.58, sampleSize: 204 });
    render(<NewEntry navColor="#08a86b" />);
    const tile = await screen.findByTestId("new-entry-avgloss-tile");
    await waitFor(() => {
      expect(tile.textContent).toMatch(/−4\.58%/);
      expect(tile.textContent).toMatch(/204 closed equity losses/);
      expect(tile.textContent).toMatch(/Denominator \(default path\): 4\.58%/);
    });
  });
});


describe("NewEntry — verdict (non-speculative, ALAB canonical)", () => {
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

  test("verdict header renders shares, %NLV, risk, gap tail, mode", async () => {
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "ALAB" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    const verdict = await screen.findByTestId("new-entry-verdict");
    await waitFor(() => {
      expect(verdict.textContent).toMatch(/Buy 312 shares/);
      expect(verdict.textContent).toMatch(/12\.5% NLV/);
      expect(verdict.textContent).toMatch(/risk 0\.75% NLV/);
      // Gap tail off avg-loss denominator (non-spec):
      // 12.5% × 4.58% / 100 = 0.5725% ≈ 0.57%
      expect(verdict.textContent).toMatch(/gap tail 0\.57% NLV/);
      expect(verdict.textContent).toMatch(/Offense risk unit/);
    });
  });

  test("denominator source line reads 'trailing avg loss' on non-spec cases", async () => {
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "ALAB" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    const source = await screen.findByTestId("new-entry-denominator-source");
    expect(source.textContent).toMatch(/trailing avg loss 4\.58%/);
    expect(source.textContent).toMatch(/204 closed losses/);
  });

  test("stop reference rows show 1.5× ATR21 stop and — when MA entered — the tech stop", async () => {
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "ALAB" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    // Only ATR row until MA is entered
    await screen.findByTestId("new-entry-atr-stop-row");
    expect(screen.queryByTestId("new-entry-tech-stop-row")).not.toBeInTheDocument();

    // Enter an MA level → tech-stop row appears
    const maInput = screen.getByLabelText(/Key MA level/);
    await act(async () => { fireEvent.change(maInput, { target: { value: "395" } }); });

    const techStop = await screen.findByTestId("new-entry-tech-stop-row");
    // MA 395 × (1 - 1.0/100) = 391.05 → dist ≈ 2.24% below $400
    expect(techStop.textContent).toMatch(/\$391\.05/);
  });

  test("no banner on non-speculative cases", async () => {
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "ALAB" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });
    await screen.findByTestId("new-entry-verdict");
    expect(screen.queryByTestId("new-entry-stop-validation-banner")).not.toBeInTheDocument();
  });
});


describe("NewEntry — verdict (speculative tier)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("no MA on speculative tier → sizes off 1.5× ATR21 fallback, no banner, source line calls it out", async () => {
    setupDefaults({
      state: "POWERTREND",
      endNlv: 1_000_000,
      avgLossPct: -4.58,
      sampleSize: 204,
      price: 50,
      atrPct: 6.0,   // spec tier: 1.5 × 6 = 9% > 8%
    });
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "MEME" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    const source = await screen.findByTestId("new-entry-denominator-source");
    expect(source.textContent).toMatch(/1\.5× ATR21/);
    expect(source.textContent).toMatch(/no MA entered/);
    expect(source.textContent).toMatch(/speculative tier/);
    // No stop-validation banner without MA.
    expect(screen.queryByTestId("new-entry-stop-validation-banner")).not.toBeInTheDocument();
  });

  test("FIG worked example — MA entered, 1.61× ATR → AMBER banner with max_valid_entry, 527 shares", async () => {
    setupDefaults({
      state: "UPTREND UNDER PRESSURE",   // → Pilot 0.25% auto
      endNlv: 664_444,
      avgLossPct: -4.34,
      sampleSize: 278,
      price: 23.65,
      atrPct: 8.28,
    });
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "FIG" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    // Enter MA 20.71 (buffer stays at default 1.0)
    const maInput = screen.getByLabelText(/Key MA level/);
    await act(async () => { fireEvent.change(maInput, { target: { value: "20.71" } }); });

    const banner = await screen.findByTestId("new-entry-stop-validation-banner");
    expect(banner.getAttribute("data-variant")).toBe("amber");
    expect(banner.textContent).toMatch(/1\.6\d× ATR/);
    expect(banner.textContent).toMatch(/wider than/);
    expect(banner.textContent).toMatch(/\$23\.4[01]/);   // max_valid_entry ≈ 23.41

    const verdict = screen.getByTestId("new-entry-verdict");
    expect(verdict.textContent).toMatch(/Buy 527 shares/);
    expect(verdict.textContent).toMatch(/1\.9% NLV/);   // formula rounded 1 dp
    expect(verdict.textContent).toMatch(/risk 0\.25% NLV/);
    expect(verdict.textContent).toMatch(/gap tail 0\.25% NLV/);

    const source = screen.getByTestId("new-entry-denominator-source");
    expect(source.textContent).toMatch(/tech stop 13\.\d+%/);
    expect(source.textContent).toMatch(/speculative tier/);
  });

  test("GREEN banner when stop_atr_mult is inside [MIN, TARGET]", async () => {
    // Choose price/ATR/MA to land in the green zone deliberately.
    setupDefaults({
      state: "POWERTREND",
      endNlv: 500_000,
      avgLossPct: -4.5,
      sampleSize: 100,
      price: 100,
      atrPct: 8.5,
    });
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "GRN" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    const maInput = screen.getByLabelText(/Key MA level/);
    const bufferInput = screen.getByLabelText(/Buffer/);
    await act(async () => { fireEvent.change(maInput, { target: { value: "90.91" } }); });
    await act(async () => { fireEvent.change(bufferInput, { target: { value: "0" } }); });

    const banner = await screen.findByTestId("new-entry-stop-validation-banner");
    expect(banner.getAttribute("data-variant")).toBe("green");
    expect(banner.textContent).toMatch(/Stop validated/);
    // Verdict still renders.
    expect(screen.getByTestId("new-entry-verdict")).toBeInTheDocument();
  });

  test("RED banner + BLOCKED verdict when tech stop is inside daily noise", async () => {
    setupDefaults({
      state: "POWERTREND",
      endNlv: 500_000,
      avgLossPct: -4.5,
      sampleSize: 100,
      price: 100,
      atrPct: 8.5,
    });
    render(<NewEntry navColor="#08a86b" />);
    const tickerInput = await screen.findByPlaceholderText("XYZ");
    await act(async () => { fireEvent.change(tickerInput, { target: { value: "TIGHT" } }); });
    await act(async () => { await new Promise(r => setTimeout(r, 700)); });

    // MA very close to entry → tight stop → RED
    const maInput = screen.getByLabelText(/Key MA level/);
    await act(async () => { fireEvent.change(maInput, { target: { value: "98" } }); });

    const banner = await screen.findByTestId("new-entry-stop-validation-banner");
    expect(banner.getAttribute("data-variant")).toBe("red");
    expect(banner.textContent).toMatch(/inside daily noise/);
    expect(banner.textContent).toMatch(/Widen the stop or skip/);

    // Verdict card is present but marked blocked (no shares number).
    const verdict = screen.getByTestId("new-entry-verdict");
    expect(verdict.getAttribute("data-blocked")).toBe("true");
    expect(verdict.textContent).toMatch(/blocked/i);
    expect(verdict.textContent).not.toMatch(/Buy \d+ shares/);
  });

  test("empty inputs → placeholder message instead of a verdict", async () => {
    setupDefaults();
    render(<NewEntry navColor="#08a86b" />);
    await screen.findByTestId("new-entry-mode-indicator");
    expect(screen.queryByTestId("new-entry-verdict")).not.toBeInTheDocument();
    expect(screen.getByText(/Enter a ticker \+ entry price/)).toBeInTheDocument();
  });
});
