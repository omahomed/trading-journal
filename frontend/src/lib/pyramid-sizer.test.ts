import { describe, it, expect } from "vitest";
import {
  computePyramidSizing,
  PyramidSizerError,
  PYRAMID_ADD_CAP_PCT,
  PYRAMID_CAMPAIGN_CEILING_PCT,
  PYRAMID_FULL_SIZE_TRIGGER_PCT,
  PYRAMID_LOCATION_ATR_MULTIPLE,
  type PyramidSizerInputs,
} from "./pyramid-sizer";

// Canonical inputs: NAV $400K, Normal mode (0.50%), a DELL-shaped calm
// name mid-pyramid. B1 100sh @ $150 stopped at $135, A1 50sh @ $170
// stopped at $158, considering an add at $180. 21 EMA ~$175, ATR 4.5%.
const HAPPY: PyramidSizerInputs = {
  equity: 400_000,
  entry: 180,
  atrPct: 4.5,
  ema21: 175,
  keyLevel: 175, // structural / MA anchor
  tolPct: 0.5,
  currentPrice: 180,
  lastHeldBuyPrice: 170,
  heldLots: [
    { shares: 100, entry: 150, stopLoss: 135, label: "B1" },
    { shares: 50, entry: 170, stopLoss: 158, label: "A1" },
  ],
};

describe("computePyramidSizing — happy path", () => {
  const r = computePyramidSizing(HAPPY);

  it("evaluates all four gates", () => {
    expect(r.location).toBeDefined();
    expect(r.progress).toBeDefined();
    expect(r.budget).toBeDefined();
    expect(r.ceiling).toBeDefined();
  });

  it("location: passes when price ≤ 21 EMA + 1 ATR", () => {
    // ATR $/share = 180 × 4.5% = $8.10. Ceiling = 175 + 8.10 = $183.10.
    // Price 180 < 183.10 → passes.
    expect(r.location.passed).toBe(true);
    expect(r.location.ceilingPrice).toBeCloseTo(183.1, 1);
  });

  it("progress: up 5.88% from A1 $170 → full-size multiplier", () => {
    // (180 - 170) / 170 * 100 = 5.88% ≥ 5 → multiplier = 1.
    expect(r.progress.passed).toBe(true);
    expect(r.progress.profitPct).toBeCloseTo(5.88, 1);
    expect(r.progress.multiplier).toBe(1);
  });

  it("budget: campaign_risk = B1 risk + A1 risk; headroom = budget − Σ", () => {
    // B1: 100 × (150 − 135) = $1500
    // A1:  50 × (170 − 158) = $600
    // campaign_risk = $2100
    // budget = 400K × 0.005 = $2000  → headroom = $2000 − $2100 = -$100
    // → blocks with reason "no headroom"
    expect(r.budget.campaignRisk).toBeCloseTo(2100, 2);
    expect(r.budget.budgetDollars).toBe(2000);
    expect(r.budget.headroom).toBeCloseTo(-100, 2);
    expect(r.budget.passed).toBe(false);
  });

  it("blocks the add because budget didn't pass; final_shares = 0", () => {
    expect(r.blocked).toBe(true);
    expect(r.finalShares).toBe(0);
    expect(r.bind).toBe("blocked");
  });

  it("still exposes composite stop for informational display", () => {
    // computeCompositeStop is imported from vol-sizer; regression-free
    // reuse means we just check the sign / rough magnitude here.
    expect(r.composite.price).toBeGreaterThan(0);
    expect(r.composite.distance).toBeGreaterThan(0);
  });
});

describe("computePyramidSizing — location gate (rule 1)", () => {
  it("blocks when price > 21 EMA + 1 ATR (extended)", () => {
    // Push price out to $190 while 21 EMA + 1 ATR sits at ~$183.
    const r = computePyramidSizing({ ...HAPPY, entry: 190, currentPrice: 190 });
    expect(r.location.passed).toBe(false);
    expect(r.blocked).toBe(true);
    expect(r.location.reason).toMatch(/Extended.*ATR above 21 EMA/);
  });

  it("passes exactly at the ceiling (equality is not extension)", () => {
    // Ceiling = 175 + 8.10 = 183.10. Price 183.10 → passes.
    const r = computePyramidSizing({ ...HAPPY, entry: 183.1, currentPrice: 183.1 });
    expect(r.location.passed).toBe(true);
  });

  it("blocks with a clear message when 21 EMA is unavailable", () => {
    const r = computePyramidSizing({ ...HAPPY, ema21: 0 });
    expect(r.location.passed).toBe(false);
    expect(r.location.reason).toMatch(/21 EMA unavailable/);
  });
});

describe("computePyramidSizing — progress gate (rule 2)", () => {
  it("blocks when current price < last held buy", () => {
    const r = computePyramidSizing({ ...HAPPY, entry: 165, currentPrice: 165 });
    expect(r.progress.passed).toBe(false);
    expect(r.progress.multiplier).toBe(0);
    expect(r.progress.profitPct).toBeLessThan(0);
    expect(r.progress.reason).toMatch(/Below last held buy/);
  });

  it("prorates the multiplier linearly between 0% and 5%", () => {
    // 2.5% up → multiplier 0.5
    const r = computePyramidSizing({
      ...HAPPY,
      entry: 170 * 1.025,
      currentPrice: 170 * 1.025,
      // Loosen budget for this test so budget doesn't clobber the check.
      heldLots: [{ shares: 10, entry: 150, stopLoss: 149, label: "B1" }],
      lastHeldBuyPrice: 170,
    });
    expect(r.progress.passed).toBe(true);
    expect(r.progress.multiplier).toBeCloseTo(0.5, 2);
  });

  it("clamps multiplier at 1 when profit% ≥ 5% (no super-multiplier)", () => {
    const r = computePyramidSizing({
      ...HAPPY,
      entry: 200,
      currentPrice: 200,
      lastHeldBuyPrice: 170,
    });
    expect(r.progress.passed).toBe(true);
    expect(r.progress.multiplier).toBe(1);
    // Constant exposed so the trigger value is easy to change globally.
    expect(PYRAMID_FULL_SIZE_TRIGGER_PCT).toBe(5);
  });

  it("passes at full-size when there's no prior held buy (fresh campaign)", () => {
    const r = computePyramidSizing({
      ...HAPPY,
      heldLots: [],
      lastHeldBuyPrice: 0,
    });
    expect(r.progress.passed).toBe(true);
    expect(r.progress.multiplier).toBe(1);
    expect(Number.isNaN(r.progress.profitPct)).toBe(true);
  });
});

describe("computePyramidSizing — budget gate (rule 3)", () => {
  it("sums per-lot risk = shares × max(0, entry − stopLoss)", () => {
    const r = computePyramidSizing({
      ...HAPPY,
      heldLots: [
        { shares: 100, entry: 150, stopLoss: 135, label: "B1" }, // $1500
        { shares: 50, entry: 170, stopLoss: 158, label: "A1" }, //  $600
      ],
    });
    expect(r.budget.campaignRisk).toBeCloseTo(2100, 2);
    expect(r.budget.lotRisks[0].risk).toBeCloseTo(1500, 2);
    expect(r.budget.lotRisks[1].risk).toBeCloseTo(600, 2);
  });

  it("treats lots with stop ≥ entry as risk-free (contributes $0)", () => {
    // Trailing stop rose above cost basis → position is 'risk-free' on
    // that lot. Rule 3 headroom should benefit accordingly.
    const r = computePyramidSizing({
      ...HAPPY,
      equity: 1_000_000,
      heldLots: [
        { shares: 100, entry: 150, stopLoss: 160, label: "B1" }, // stop above cost → risk-free
      ],
    });
    expect(r.budget.lotRisks[0].risk).toBe(0);
    expect(r.budget.campaignRisk).toBe(0);
    expect(r.budget.headroom).toBe(r.budget.budgetDollars);
    expect(r.budget.passed).toBe(true);
  });

  it("blocks when headroom ≤ 0", () => {
    const r = computePyramidSizing(HAPPY);
    // HAPPY has negative headroom by construction.
    expect(r.budget.passed).toBe(false);
    expect(r.budget.reason).toMatch(/No headroom/);
  });
});

describe("computePyramidSizing — size math (rule 4)", () => {
  it("size = min(headroom/dist, 5%/entry × NAV) × progress_multiplier", () => {
    // Big NAV so budget dominates, plus a keyLevel that makes the
    // composite lock at ATR floor for a predictable stop distance.
    const r = computePyramidSizing({
      equity: 10_000_000,
      entry: 100,
      atrPct: 5,
      ema21: 95,
      keyLevel: 95, // Key Level candidate = 95 − max(2.5, 1) = 92.5
      tolPct: 0.5,
      currentPrice: 96, // ≤ ema21 + 1 ATR = 100  ✓
      lastHeldBuyPrice: 90, // up 6.67% → multiplier 1
      heldLots: [], // no existing risk
    });
    // budget = 50K. stop = min(95, 92.5) = 92.5 → dist = 7.5
    // risk_bound = floor(50_000 / 7.5) = 6666
    // notional_cap = floor(5% × 10M / 100) = 5000
    // → binding = notional_cap
    expect(r.riskBoundShares).toBe(6666);
    expect(r.notionalCapShares).toBe(5000);
    expect(r.finalShares).toBe(5000);
    expect(r.bind).toBe("notional_cap");
    expect(PYRAMID_ADD_CAP_PCT).toBe(5);
  });

  it("risk-bound wins when notional cap is generous", () => {
    // Force a small headroom → risk binds before notional cap.
    const r = computePyramidSizing({
      equity: 400_000,
      entry: 100,
      atrPct: 5,
      ema21: 95,
      keyLevel: 92,
      tolPct: 0.25, // budget = $1000
      currentPrice: 96,
      lastHeldBuyPrice: 90,
      heldLots: [],
    });
    // budget = 1000, dist = 100 − min(95, 92 − 2.5) = 100 − 89.5 = 10.5
    // risk_bound = floor(1000 / 10.5) = 95
    // notional_cap = floor(5% × 400K / 100) = 200
    // → risk-bound wins
    expect(r.riskBoundShares).toBe(95);
    expect(r.notionalCapShares).toBe(200);
    expect(r.finalShares).toBe(95);
    expect(r.bind).toBe("risk");
  });

  it("progress multiplier further clips a would-be-100 to 50 at 2.5% up", () => {
    const r = computePyramidSizing({
      equity: 400_000,
      entry: 100,
      atrPct: 5,
      ema21: 95,
      keyLevel: 92,
      tolPct: 0.25,
      currentPrice: 100,
      lastHeldBuyPrice: 100 / 1.025, // gives +2.5% → multiplier 0.5
      heldLots: [],
    });
    // preProgress = min(risk_bound, notional_cap). At entry 100, atr 5%:
    //   dist = 100 - min(95, 92-2.5) = 100-89.5 = 10.5
    //   risk_bound = floor(1000/10.5) = 95; notional_cap = 200; preProgress = 95
    //   × 0.5 = 47 (floored)
    expect(r.progress.multiplier).toBeCloseTo(0.5, 2);
    expect(r.finalShares).toBe(47);
    expect(r.bind).toBe("progress");
  });
});

describe("computePyramidSizing — ceiling gate (rule 6)", () => {
  it("clips down to the 25% NAV cap", () => {
    // 200 existing shares × $100 = $20K = 5% of $400K. Room to 25% is
    // 20% of NAV = $80K → 800 shares. Try to add a full-batch pyramid
    // that would breach.
    const r = computePyramidSizing({
      equity: 400_000,
      entry: 100,
      atrPct: 5,
      ema21: 95,
      keyLevel: 95,
      tolPct: 0.75, // biggest budget
      currentPrice: 100,
      lastHeldBuyPrice: 95,
      // Zero-risk existing lot so budget stays wide open.
      heldLots: [{ shares: 200, entry: 90, stopLoss: 90, label: "B1" }],
    });
    // notional_cap by 5% = 200 shares; ceiling room = 800 shares;
    // risk_bound easily > 200 → notional wins. NO ceiling clip needed.
    // Now force ceiling to bind: increase existing shares.
    const r2 = computePyramidSizing({
      ...HAPPY,
      equity: 400_000,
      currentPrice: 100,
      entry: 100,
      atrPct: 5,
      ema21: 95,
      keyLevel: 95,
      tolPct: 0.75,
      lastHeldBuyPrice: 95,
      heldLots: [{ shares: 990, entry: 90, stopLoss: 90, label: "B1" }], // $99K = 24.75%
    });
    // Ceiling room: (25% × 400K − 99K) / 100 = 1000 / 100 = 10 shares.
    expect(r2.finalShares).toBe(10);
    expect(r2.bind).toBe("ceiling");
    void r; // silence unused-var if the first branch collapses
  });

  it("blocks entirely when existing position is already at ceiling", () => {
    const r = computePyramidSizing({
      equity: 400_000,
      entry: 100,
      atrPct: 5,
      ema21: 95,
      keyLevel: 95,
      tolPct: 0.5,
      currentPrice: 100,
      lastHeldBuyPrice: 95,
      heldLots: [{ shares: 1000, entry: 100, stopLoss: 100, label: "B1" }], // exactly 25% NAV
    });
    // 1000 × 100 = 100_000 = 25% × 400K → no room left.
    expect(r.blocked).toBe(true);
    expect(r.finalShares).toBe(0);
    expect(PYRAMID_CAMPAIGN_CEILING_PCT).toBe(25);
  });
});

describe("computePyramidSizing — validation & edges", () => {
  it("throws on non-positive inputs", () => {
    expect(() => computePyramidSizing({ ...HAPPY, equity: 0 })).toThrow(PyramidSizerError);
    expect(() => computePyramidSizing({ ...HAPPY, entry: 0 })).toThrow(PyramidSizerError);
    expect(() => computePyramidSizing({ ...HAPPY, atrPct: 0 })).toThrow(PyramidSizerError);
    expect(() => computePyramidSizing({ ...HAPPY, keyLevel: 0 })).toThrow(PyramidSizerError);
    expect(() => computePyramidSizing({ ...HAPPY, tolPct: 0 })).toThrow(PyramidSizerError);
    expect(() => computePyramidSizing({ ...HAPPY, currentPrice: 0 })).toThrow(PyramidSizerError);
  });

  it("PYRAMID_LOCATION_ATR_MULTIPLE constant is 1", () => {
    // Regression guard on the "1 ATR extension" policy.
    expect(PYRAMID_LOCATION_ATR_MULTIPLE).toBe(1);
  });

  it("existing shares are reflected in projectedShares and projectedNotional", () => {
    const r = computePyramidSizing({
      equity: 10_000_000,
      entry: 100,
      atrPct: 5,
      ema21: 95,
      keyLevel: 95,
      tolPct: 0.5,
      currentPrice: 100,
      lastHeldBuyPrice: 95,
      heldLots: [
        { shares: 100, entry: 90, stopLoss: 90, label: "B1" },
        { shares: 50, entry: 92, stopLoss: 92, label: "A1" },
      ],
    });
    expect(r.existingShares).toBe(150);
    expect(r.existingNotional).toBeCloseTo(15_000, 2);
    expect(r.projectedShares).toBe(r.existingShares + r.finalShares);
    expect(r.projectedNotional).toBeCloseTo(r.projectedShares * 100, 2);
  });
});
