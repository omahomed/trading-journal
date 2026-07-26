// Campaign-level metrics appended to the CR CSV export. Pure math so
// it can be unit-tested without rendering the CR page or standing up
// the full row-derivation pipeline. Consumed by
// campaign-review.tsx:onExportCsv (2026-07-25).
//
// null semantics — critical. Every metric returns `null` (not 0) when
// its inputs disqualify it. The CSV writer maps null → empty cell so
// downstream analysis can filter properly. A zero is a real value for
// every one of these fields (a campaign with 0 realized P&L, a
// perfectly-captured MFE, etc.), so zeros must NEVER stand in for
// "not-applicable".
//
// MFE eligibility floor: campaigns whose MFE < 30% aren't scoring
// candidates for capture ratios — the base is too small for the ratio
// to be meaningful. Explicit >= 30 (equality is eligible).
//
// deploy_ratio + campaign_score are always null in v1: the Position
// Sizer's recommended share count isn't persisted anywhere (audited
// 2026-07-25). Columns exist so future rows populate without a
// schema-side reshuffle. Do NOT substitute a proxy.

export const MFE_ELIGIBILITY_PCT = 30;
export const SR1A_FIRE_ATR_MULT = 0.75;

export interface CampaignMetricsInput {
  /** Cost basis for B-series (initial) lots — shares × price × multiplier. */
  b_initial_cost: number;
  /** Cost basis for A-series (add-on) lots. Zero when no add-ons. */
  a_initial_cost: number;
  /** Realized + unrealized P&L attributable to A-series lots only. */
  a_pnl: number;
  /** Realized + unrealized P&L for the whole campaign (B + A). Uses
   *  live prices for open lots on the CR page. */
  total_pnl: number;
  /** B-series return %. Null when b_initial_cost is 0 (i.e. no B lots
   *  — shouldn't happen for a real campaign but we honor the input). */
  b_return_pct: number | null;
  /** Maximum favorable excursion % from B1 fill (trades_summary.mfe_pct
   *  via migration 046). Null for options / pre-2026-01 opens the
   *  reconciler hasn't swept / same-day fills without EOD tape. */
  mfe_pct: number | null;
  /** Maximum adverse excursion expressed as ATR21 multiples —
   *  computed on the CR page as |mae_pct| / atr21_entry_pct. Sign is
   *  always ≥ 0 by construction, but we defensively abs() again inside. */
  mae_atr: number | null;
}

export interface CampaignMetrics {
  /** B Cost × MFE% / 100. The dollar figure a "perfect starter" (B
   *  only, held to the MFE) would have booked. Null iff mfe_pct null. */
  perfect_starter_usd: number | null;
  /** B Return % / MFE %. What fraction of the available MFE the B-only
   *  lot captured. Null when mfe_pct null OR mfe_pct < 30 (below the
   *  eligibility floor). */
  b_capture: number | null;
  /** Total P&L / perfect_starter_usd. What fraction of the "perfect
   *  starter" outcome the actual campaign (with adds) delivered.
   *  Null when mfe_pct null OR mfe_pct < 30 OR perfect_starter_usd is 0. */
  campaign_efficiency: number | null;
  /** (A P&L / A Cost) × 100. Return on the ADD dollars specifically —
   *  isolates whether the pyramid dollars were productive. Null when
   *  A Cost is 0 (no add-ons). No MFE floor. */
  add_efficiency_pct: number | null;
  /** B Cost / recommended B-lot cost from the Position Sizer at entry.
   *  Always null in v1 (no stored sizer recommendation — see module
   *  docstring). */
  deploy_ratio: number | null;
  /** campaign_efficiency × min(deploy_ratio, 1.0). Always null in v1
   *  (deploy_ratio always null). */
  campaign_score: number | null;
  /** True iff |MAE ATR| > 0.75 — matches the SR1a broker-stop threshold.
   *  Null when MAE ATR is unavailable (pre-reconciler-sweep, options). */
  sr1a_fire: boolean | null;
}

export function computeCampaignMetrics(input: CampaignMetricsInput): CampaignMetrics {
  const mfeEligible = input.mfe_pct != null && input.mfe_pct >= MFE_ELIGIBILITY_PCT;

  const perfect_starter_usd = input.mfe_pct != null
    ? input.b_initial_cost * input.mfe_pct / 100
    : null;

  const b_capture = mfeEligible && input.b_return_pct != null && input.mfe_pct != null && input.mfe_pct !== 0
    ? input.b_return_pct / input.mfe_pct
    : null;

  // Guard against division by zero on perfect_starter_usd (only
  // possible when b_initial_cost is 0 while mfe_pct is set — an
  // options-only campaign with a stock summary would look like this
  // but they're filtered upstream). Emit null rather than Infinity.
  const campaign_efficiency = mfeEligible && perfect_starter_usd != null && perfect_starter_usd !== 0
    ? input.total_pnl / perfect_starter_usd
    : null;

  const add_efficiency_pct = input.a_initial_cost !== 0
    ? (input.a_pnl / input.a_initial_cost) * 100
    : null;

  // v1: no stored sizer recommendation → deploy_ratio and (by
  // dependency) campaign_score are unconditionally null. Do NOT
  // substitute a proxy value here — the spec explicitly rejects
  // percentile / NLV heuristics as substitutes. Columns land as
  // empty cells; downstream analysis filters accordingly.
  const deploy_ratio: number | null = null;
  const campaign_score: number | null = null;

  const sr1a_fire = input.mae_atr != null
    ? Math.abs(input.mae_atr) > SR1A_FIRE_ATR_MULT
    : null;

  return {
    perfect_starter_usd,
    b_capture,
    campaign_efficiency,
    add_efficiency_pct,
    deploy_ratio,
    campaign_score,
    sr1a_fire,
  };
}
