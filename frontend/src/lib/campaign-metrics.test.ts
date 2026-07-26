import { describe, it, expect } from "vitest";
import {
  computeCampaignMetrics,
  MFE_ELIGIBILITY_PCT,
  SR1A_FIRE_ATR_MULT,
} from "./campaign-metrics";

// Base fixture: a mid-conviction pyramid campaign, MFE eligible.
// B lot $10K cost + $2K gain; A lot $5K cost + $1.5K gain.
// mfe_pct = 40% (above the 30% floor), b_return_pct = 20%,
// mae_atr = 0.5 (below SR1a threshold).
const BASE = {
  b_initial_cost: 10_000,
  a_initial_cost: 5_000,
  a_pnl: 1_500,
  total_pnl: 3_500,       // 2000 B + 1500 A
  b_return_pct: 20,
  mfe_pct: 40,
  mae_atr: 0.5,
};


describe("computeCampaignMetrics — happy path", () => {
  const r = computeCampaignMetrics(BASE);

  it("perfect_starter_usd = B Cost × MFE% / 100", () => {
    // 10,000 × 40 / 100 = 4,000
    expect(r.perfect_starter_usd).toBe(4_000);
  });

  it("b_capture = B Return % / MFE % (eligible when MFE ≥ 30)", () => {
    // 20 / 40 = 0.5
    expect(r.b_capture).toBeCloseTo(0.5, 4);
  });

  it("campaign_efficiency = Total P&L / perfect_starter_usd (eligible when MFE ≥ 30)", () => {
    // 3,500 / 4,000 = 0.875
    expect(r.campaign_efficiency).toBeCloseTo(0.875, 4);
  });

  it("add_efficiency_pct = (A P&L / A Cost) × 100 (no MFE floor)", () => {
    // (1500 / 5000) × 100 = 30.0
    expect(r.add_efficiency_pct).toBeCloseTo(30.0, 4);
  });

  it("deploy_ratio is always null in v1 (no stored sizer recommendation)", () => {
    expect(r.deploy_ratio).toBeNull();
  });

  it("campaign_score is always null in v1 (depends on deploy_ratio)", () => {
    expect(r.campaign_score).toBeNull();
  });

  it("sr1a_fire is false when |mae_atr| ≤ 0.75", () => {
    expect(r.sr1a_fire).toBe(false);
  });
});


describe("computeCampaignMetrics — MFE null (pre-reconciler / options)", () => {
  const r = computeCampaignMetrics({ ...BASE, mfe_pct: null });

  it("perfect_starter_usd → null (not 0 — MFE missing, base is unknown)", () => {
    expect(r.perfect_starter_usd).toBeNull();
  });

  it("b_capture → null", () => {
    expect(r.b_capture).toBeNull();
  });

  it("campaign_efficiency → null", () => {
    expect(r.campaign_efficiency).toBeNull();
  });

  it("add_efficiency_pct still computes (no MFE dependency)", () => {
    expect(r.add_efficiency_pct).toBeCloseTo(30.0, 4);
  });
});


describe("computeCampaignMetrics — MFE below eligibility floor", () => {
  it("MFE = 29.99 → b_capture null, campaign_efficiency null (below floor)", () => {
    const r = computeCampaignMetrics({ ...BASE, mfe_pct: 29.99 });
    expect(r.b_capture).toBeNull();
    expect(r.campaign_efficiency).toBeNull();
  });

  it("MFE = 30 EXACTLY → both computed (equality is inside the floor)", () => {
    const r = computeCampaignMetrics({ ...BASE, mfe_pct: 30 });
    expect(r.b_capture).not.toBeNull();
    expect(r.campaign_efficiency).not.toBeNull();
    // 20 / 30 = 0.667
    expect(r.b_capture!).toBeCloseTo(0.667, 2);
  });

  it("perfect_starter_usd computes REGARDLESS of eligibility (only needs mfe_pct)", () => {
    // Spec: "perfect_starter_usd = B Cost × MFE% / 100. Null if MFE% is null."
    // No floor on this one.
    const r = computeCampaignMetrics({ ...BASE, mfe_pct: 15 });
    expect(r.perfect_starter_usd).toBe(1_500);  // 10K × 15 / 100
  });

  it("MFE_ELIGIBILITY_PCT constant is 30 (regression guard on the spec value)", () => {
    expect(MFE_ELIGIBILITY_PCT).toBe(30);
  });
});


describe("computeCampaignMetrics — A Cost null / zero edge", () => {
  it("A Cost = 0 → add_efficiency_pct null (no add-ons to divide against)", () => {
    const r = computeCampaignMetrics({ ...BASE, a_initial_cost: 0, a_pnl: 0 });
    expect(r.add_efficiency_pct).toBeNull();
  });

  it("A P&L is negative (losing pyramid) → still computes, still divides", () => {
    const r = computeCampaignMetrics({ ...BASE, a_pnl: -500 });
    // (-500 / 5000) × 100 = -10.0
    expect(r.add_efficiency_pct).toBeCloseTo(-10.0, 4);
  });
});


describe("computeCampaignMetrics — sr1a_fire semantics", () => {
  it("|mae_atr| > 0.75 → fires (spec: broker-stop threshold)", () => {
    const r = computeCampaignMetrics({ ...BASE, mae_atr: 0.76 });
    expect(r.sr1a_fire).toBe(true);
  });

  it("|mae_atr| = 0.75 exactly → does NOT fire (strict > per spec)", () => {
    const r = computeCampaignMetrics({ ...BASE, mae_atr: 0.75 });
    expect(r.sr1a_fire).toBe(false);
  });

  it("mae_atr null → sr1a_fire null (unknown, not false)", () => {
    const r = computeCampaignMetrics({ ...BASE, mae_atr: null });
    expect(r.sr1a_fire).toBeNull();
  });

  it("SR1A_FIRE_ATR_MULT constant is 0.75", () => {
    expect(SR1A_FIRE_ATR_MULT).toBe(0.75);
  });
});


describe("computeCampaignMetrics — zero is a REAL value, never null", () => {
  it("total_pnl = 0 with MFE eligible → campaign_efficiency = 0 (not null)", () => {
    const r = computeCampaignMetrics({ ...BASE, total_pnl: 0 });
    expect(r.campaign_efficiency).toBe(0);
  });

  it("b_return_pct = 0 with MFE eligible → b_capture = 0 (not null)", () => {
    const r = computeCampaignMetrics({ ...BASE, b_return_pct: 0 });
    expect(r.b_capture).toBe(0);
  });

  it("a_pnl = 0 with A Cost > 0 → add_efficiency_pct = 0 (not null)", () => {
    const r = computeCampaignMetrics({ ...BASE, a_pnl: 0 });
    expect(r.add_efficiency_pct).toBe(0);
  });
});


describe("computeCampaignMetrics — divide-by-zero guards", () => {
  it("perfect_starter_usd = 0 (B Cost = 0 while MFE eligible) → campaign_efficiency null", () => {
    // Options-only campaigns get filtered upstream, but the guard
    // protects against a mis-shaped input reaching the exporter.
    const r = computeCampaignMetrics({ ...BASE, b_initial_cost: 0 });
    expect(r.perfect_starter_usd).toBe(0);
    expect(r.campaign_efficiency).toBeNull();
  });

  it("b_return_pct null (no B lots) → b_capture null", () => {
    const r = computeCampaignMetrics({ ...BASE, b_return_pct: null });
    expect(r.b_capture).toBeNull();
  });
});
