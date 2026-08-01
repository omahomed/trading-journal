import "./cycle-tracker-methodology.css";

/**
 * M Factor — Complete Rules Reference. Every rule the engine enforces,
 * organized by category and color-coded so the entry / exit / meta
 * categories are visually separable at a glance.
 *
 * Source of truth for each rule lives in api/mct_engine.py; this page is
 * kept 1:1 with the engine constants (CORRECTION_DRAWDOWN, UNDERCUT,
 * FTD_* windows, PT_ON_* bar counts, etc.). If you change a constant
 * there, mirror it here.
 *
 * Self-contained: styles scoped under .ctm-root in
 * cycle-tracker-methodology.css. Fonts come from the M Factor route via
 * next/font CSS vars (see app/(app)/m-factor/page.tsx).
 */
export function CycleTrackerMethodology() {
  return (
    <div className="ctm-root">

      {/* ── hero ── */}
      <header className="hero">
        <div className="wrap">
          <span className="eyebrow">M Factor · Complete Rules Reference</span>
          <h1>Every rule, one page</h1>
          <p className="lead">
            The M&nbsp;Factor engine watches NASDAQ and turns a set of deterministic rules
            into a market <em>state</em>, an <em>entry ladder</em>, and an <em>exit ladder</em>.
            This page lists every rule it enforces, color-coded by what it does. The
            source of truth is <span className="mono">api/mct_engine.py</span>; if this
            page and the engine ever disagree, the engine wins.
          </p>

          <div className="legend">
            <span className="legend-item"><span className="chip go">Entry</span> promotes the ladder</span>
            <span className="legend-item"><span className="chip stop">Exit / Correction</span> defensive trigger</span>
            <span className="legend-item"><span className="chip sky">Structure</span> ratchet / cycle mechanics</span>
            <span className="legend-item"><span className="chip caution">Rally hunt</span> post-correction recovery</span>
            <span className="legend-item"><span className="chip meta">Meta</span> override / manual</span>
          </div>
        </div>
      </header>

      {/* ── STATES ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§1 — THE FIVE STATES</span>
          <h2>What the engine calls the market</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Every bar resolves to exactly one state. The state drives the M Factor
              banner, the entry ladder&rsquo;s permission ceiling, and downstream
              consumers (Position Sizer&rsquo;s auto-mode, ACS badges, exit-ladder
              severity).
            </p>
          </div>

          <div className="state-grid">
            <div className="state-card stop">
              <div className="state-tag">Reset · exp 0</div>
              <div className="state-name">CORRECTION</div>
              <div className="state-when">10% off high + 2 closes &lt; 50 SMA, or user override</div>
              <div className="state-note">Resets rally-hunt state; all steps retire; cap-at-100 may set.</div>
            </div>
            <div className="state-card caution">
              <div className="state-tag">Rally hunt</div>
              <div className="state-name">RALLY MODE</div>
              <div className="state-when">Cycle&rsquo;s STEP_0 fired, no STEP_4 yet</div>
              <div className="state-note">Tentative — 20&ndash;60 exposure while the FTD confirms.</div>
            </div>
            <div className="state-card sky">
              <div className="state-tag">Post-Step-4</div>
              <div className="state-name">UPTREND UNDER PRESSURE</div>
              <div className="state-when">Step 4 ever fired, but currently retired by a violation</div>
              <div className="state-note">Cycle continues; ladder builds back as rungs relight.</div>
            </div>
            <div className="state-card sky solid">
              <div className="state-tag">Structure intact</div>
              <div className="state-name">UPTREND</div>
              <div className="state-when">Step 4 currently done, not in correction, PT off</div>
              <div className="state-note">Live ladder from steps 2-7 as they qualify.</div>
            </div>
            <div className="state-card go">
              <div className="state-tag">Full offense · up to 200</div>
              <div className="state-name">POWERTREND</div>
              <div className="state-when">All 4 PT-ON conditions met (see §6)</div>
              <div className="state-note">STEP_8 fires; exposure jumps to 200 (subject to other rungs).</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── CORRECTION ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">§2 — CORRECTION DECLARATION &amp; NULLIFICATION</span>
          <h2>Entering and exiting the reset state</h2>

          <div className="rule-card stop">
            <div className="rule-head">
              <span className="rule-badge stop">DECLARE</span>
              <span className="rule-name">CORRECTION_DECLARED</span>
              <span className="rule-source">api/mct_engine.py::_phase_declaration</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">1. Structure</span> 2 closes below the 50 SMA (tracked via <span className="mono">consec_below_50</span>; naturally resets to 0 whenever a close pops back above the SMA).</div>
                <div className="cond"><span className="cond-label">2. Depth</span> current bar&rsquo;s <b>intraday low</b> ≤ reference high × 0.90 (10% off the running peak, peak-to-trough on the low).</div>
              </div>
              <div className="rule-effect">Both true → declare. Resets rally state, retires steps, sets <span className="mono">correction_active=True</span>. May also flip <span className="mono">cap_at_100</span> if a violation follows.</div>
            </div>
          </div>

          <div className="rule-card go">
            <div className="rule-head">
              <span className="rule-badge go">RESET</span>
              <span className="rule-name">CORRECTION_NULLIFIED</span>
              <span className="rule-source">api/mct_engine.py::_phase_nullification</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Trigger</span> current close &gt; reference high.</div>
              </div>
              <div className="rule-effect">Clears <span className="mono">correction_active</span> AND <span className="mono">in_correction</span>. Arms the ratchet (starts moving on new highs). Retires steps 5/6/7 so a future correction can re-fire them. Releases <span className="mono">cap_at_100</span> if set.</div>
            </div>
          </div>

          <div className="rule-card meta">
            <div className="rule-head">
              <span className="rule-badge meta">OVERRIDE</span>
              <span className="rule-name">Force Correction (manual)</span>
              <span className="rule-source">POST /api/mct/override · api/main.py</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Trigger</span> user clicks &ldquo;Force Correction&rdquo; on M Factor page; supplies a reason (min 40 chars).</div>
              </div>
              <div className="rule-effect">
                Engine re-runs with <span className="mono">force_correction_at_date</span> pinned to the override date. Fresh CORRECTION cycle seeded on that bar (no depth threshold required). Displayed with an <span className="mono">OVERRIDE</span> badge on the banner.
                <div className="rule-note">Auto-clears when systematic state recovers to POWERTREND / UPTREND, or when the systematic rule itself declares CORRECTION (rule caught up). Manual clear also available.</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── REFERENCE HIGH ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">§3 — REFERENCE HIGH RATCHET</span>
          <h2>The anchor everything is measured against</h2>

          <div className="rule-card sky">
            <div className="rule-head">
              <span className="rule-badge sky">RATCHET</span>
              <span className="rule-name">reference_high update</span>
              <span className="rule-source">api/mct_engine.py::_phase_ratchet</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Gate</span> not in correction (<span className="mono">correction_active=False AND in_correction=False</span>) AND ratchet armed.</div>
                <div className="cond"><span className="cond-label">Update</span> if current bar&rsquo;s intraday <b>high</b> &gt; stored <span className="mono">reference_high</span>, ratchet the stored value up.</div>
              </div>
              <div className="rule-effect">Ratchet arms after the first CORRECTION_NULLIFIED. Uses <b>intraday high</b> — the drawdown display (peak-to-trough) and the correction depth trigger both pair this with the intraday low, so display and trigger share the same signal.</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── RALLY CYCLE ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§4 — RALLY CYCLE RULES</span>
          <h2>How a fresh cycle starts after a correction</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Once correction_active is True, the engine hunts for a new rally cycle.
              These are the rules that fire STEP_0 → STEP_1 (FTD) → drive rally-day
              counting until STEP_4 lands (leaving the cycle in UPTREND).
            </p>
          </div>

          <div className="rule-card caution">
            <div className="rule-head">
              <span className="rule-badge caution">STEP_0</span>
              <span className="rule-name">Rally Day</span>
              <span className="rule-source">api/mct_engine.py::_phase_rally_hunt</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Gate</span> <span className="mono">correction_active=True</span>, no active rally.</div>
                <div className="cond"><span className="cond-label">Trigger</span> first bar in the correction that sets a new lower <span className="mono">running_min_low</span>, then closes up. Fires STEP_0_RALLY_DAY.</div>
              </div>
              <div className="rule-effect">Opens the cycle. Exposure → 20 (step 0 credit). Sets <span className="mono">rally_day_low</span> as invalidation floor.</div>
              <div className="rule-note">Pink rally day = same day where close is in the upper half of the intraday range (<span className="mono">position_in_range &gt; 0.5</span>). Distinct label; same STEP_0 event.</div>
            </div>
          </div>

          <div className="rule-card caution">
            <div className="rule-head">
              <span className="rule-badge caution">STEP_1</span>
              <span className="rule-name">Follow-Through Day (FTD)</span>
              <span className="rule-source">api/mct_engine.py::_phase_rally_hunt</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Rally day window</span> IXIC rally_count between 4 and 25 (<span className="mono">FTD_WINDOW_START</span>=4, <span className="mono">FTD_WINDOW_END</span>=25).</div>
                <div className="cond"><span className="cond-label">Pre-2026-07-31</span> IXIC close &gt; prior close by ≥ 1% (<span className="mono">FTD_PCT_THRESHOLD</span>=0.01). Volume not consulted.</div>
                <div className="cond"><span className="cond-label">On/after 2026-07-31</span> (<span className="mono">FTD_DUAL_INDEX_START</span>) fires when <b>either</b>:
                  <ul className="mt-1 ml-3 list-disc">
                    <li>IXIC close ≥ +1% <b>and</b> IXIC volume &gt; prior-day volume, or</li>
                    <li>SPY close ≥ +1% <b>and</b> SPY volume &gt; prior-day volume.</li>
                  </ul>
                </div>
                <div className="cond"><span className="cond-label">Missing SPY data</span> refuses to fire — waits for the nightly SPY ingest to land rather than falling back to IXIC-only.</div>
              </div>
              <div className="rule-effect">Fires STEP_1_FTD. Stores <span className="mono">ftd_close</span> and <span className="mono">ftd_low</span> as invalidation anchors (<b>always IXIC&apos;s</b> — IXIC stays state authority for the soft-fail check regardless of which index confirmed). Exposure → 40 (step 0 + 1). Meta carries <span className="mono">confirmed_by</span> (<span className="mono">ixic</span> / <span className="mono">spy</span> / <span className="mono">both</span> / <span className="mono">ixic_legacy</span>) rendered as a badge on the M Factor page.</div>
              <div className="rule-note">FTD is a cycle event (banked; not retired by pullbacks).</div>
            </div>
          </div>

          <div className="rule-card stop">
            <div className="rule-head">
              <span className="rule-badge stop">INVALIDATE</span>
              <span className="rule-name">RALLY_INVALIDATED</span>
              <span className="rule-source">api/mct_engine.py::_phase_rally_hunt</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Trigger</span> any intraday low &lt; <span className="mono">rally_day_low</span>.</div>
              </div>
              <div className="rule-effect">Kills the rally cycle. Resets STEP_0/1 flags. New STEP_0 must fire before rally-hunt resumes.</div>
              <div className="rule-note">Fires on intraday undercut, not close — a real breach of the low invalidates whether or not the close recovers.</div>
            </div>
          </div>

          <div className="rule-card stop">
            <div className="rule-head">
              <span className="rule-badge stop">SOFT FAIL</span>
              <span className="rule-name">POST_FTD_SOFT_FAIL</span>
              <span className="rule-source">api/mct_engine.py::_fire_post_ftd_soft_fail</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Gate</span> STEP_1 done (FTD confirmed), inside correction context.</div>
                <div className="cond"><span className="cond-label">Trigger</span> close &lt; <span className="mono">ftd_low</span> — a close (not intraday) below the FTD bar&rsquo;s low.</div>
              </div>
              <div className="rule-effect">Retires STEP_1, resets rally count. Engine returns to hunting a fresh FTD.</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── ENTRY LADDER ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§5 — THE ENTRY LADDER</span>
          <h2>Nine rungs, sum to exposure</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Each rung awards points when its condition is true. Exposure = sum of
              live points (capped at 200). Events (steps 0, 1, 8) latch for the cycle
              — banked once earned. Live steps (2-7) recheck every bar.
            </p>
          </div>

          <table className="ladder">
            <thead>
              <tr><th>#</th><th>Signal</th><th>Kind</th><th>Points</th><th>Type</th></tr>
            </thead>
            <tbody>
              <tr className="ev caution"><td className="st">0</td><td className="nm">Rally Day<small>First bar of a new rally cycle (see §4).</small></td><td>Latched cycle event</td><td className="pts">+20</td><td><span className="kind evt">Event</span></td></tr>
              <tr className="ev caution"><td className="st">1</td><td className="nm">Follow-Through Day<small>Confirms the rally is real (see §4).</small></td><td>Latched cycle event</td><td className="pts">+20</td><td><span className="kind evt">Event</span></td></tr>
              <tr className="ev go"><td className="st">2</td><td className="nm">Close &gt; 21 EMA<small>Current bar&rsquo;s close is above the 21 EMA.</small></td><td>Live condition</td><td className="pts">+20</td><td><span className="kind live">Live</span></td></tr>
              <tr className="ev go"><td className="st">3</td><td className="nm">Low &gt; 21 EMA<small>Stronger — didn&rsquo;t even dip below intraday.</small></td><td>Live condition</td><td className="pts">+20</td><td><span className="kind live">Live</span></td></tr>
              <tr className="ev go"><td className="st">4</td><td className="nm">Low &gt; 21 EMA for 3 days<small>The hold above 21 EMA is durable. Once fired, this cycle can re-enter UPTREND from UUP later.</small></td><td>Live condition (arms cycle)</td><td className="pts">+20</td><td><span className="kind live">Live</span></td></tr>
              <tr className="ev go"><td className="st">5</td><td className="nm">Low &gt; 50 SMA for 3 days<small>Medium-term average is holding.</small></td><td>Live condition</td><td className="pts">+20</td><td><span className="kind live">Live</span></td></tr>
              <tr className="ev go"><td className="st">6</td><td className="nm">21 EMA &gt; 50 SMA &gt; 200 SMA<small>Medium/long averages stacked in bullish order.</small></td><td>Live condition</td><td className="pts">+20</td><td><span className="kind live">Live</span></td></tr>
              <tr className="ev go"><td className="st">7</td><td className="nm">8 EMA &gt; 21 EMA &gt; 50 SMA &gt; 200 SMA<small>Full stack, all in order.</small></td><td>Live condition</td><td className="pts">+20</td><td><span className="kind live">Live</span></td></tr>
              <tr className="ev caution"><td className="st">8</td><td className="nm">Power-Trend ON<small>All 4 PT-ON conditions met (see §6).</small></td><td>Latched cycle event</td><td className="pts">+40</td><td><span className="kind evt">Event</span></td></tr>
            </tbody>
          </table>

          <div className="wrap" style={{ padding: 0 }}>
            <p className="mini-note">
              <b>Sticky base.</b> Steps 0, 1, and 8 (Rally Day, FTD, Power-Trend) are cycle events —
              once earned they stay lit until CORRECTION_NULLIFIED clears the cycle. That&rsquo;s the
              <b> 80-point sticky base</b> of any cycle that reaches Power-Trend.
              <br /><br />
              <b>Cap-at-100.</b> Once <span className="mono">cap_at_100</span> flips true (via a violation
              inside an active correction window), Step 8 cannot promote exposure to 200 — the ceiling
              stays at 100 until CORRECTION_NULLIFIED releases the cap.
            </p>
          </div>
        </div>
      </section>

      {/* ── POWER-TREND ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">§6 — POWER-TREND RULES</span>
          <h2>How PT turns on and off</h2>

          <div className="rule-card go">
            <div className="rule-head">
              <span className="rule-badge go">PT ON</span>
              <span className="rule-name">STEP_8_POWERTREND_ON</span>
              <span className="rule-source">api/mct_engine.py::_phase_post_step4</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Gate</span> step_4_done AND not power_trend AND not cap_at_100.</div>
                <div className="cond"><span className="cond-label">1</span> 21 EMA &gt; 50 SMA for last <b>5</b> bars (<span className="mono">consec_21_above_50 ≥ 5</span>).</div>
                <div className="cond"><span className="cond-label">2</span> low &gt; 21 EMA for last <b>10</b> bars (<span className="mono">consec_low_above_21 ≥ 10</span>).</div>
                <div className="cond"><span className="cond-label">3</span> today&rsquo;s close ≥ yesterday&rsquo;s close (close up).</div>
                <div className="cond"><span className="cond-label">4</span> today&rsquo;s 50 SMA &gt; yesterday&rsquo;s 50 SMA (SMA rising).</div>
              </div>
              <div className="rule-effect">All 4 met → power_trend = True. Exposure → 200 (via Step 8&rsquo;s +40 on top of the other rungs). Anchors <span className="mono">pt_on_idx</span> for the &ldquo;PT Day N&rdquo; display.</div>
            </div>
          </div>

          <div className="rule-card stop">
            <div className="rule-head">
              <span className="rule-badge stop">PT OFF</span>
              <span className="rule-name">POWERTREND_OFF</span>
              <span className="rule-source">api/mct_engine.py::_phase_pt_off</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">1</span> yesterday&rsquo;s 21 EMA ≥ yesterday&rsquo;s 50 SMA AND today&rsquo;s 21 EMA &lt; today&rsquo;s 50 SMA (21 EMA crossed BELOW 50 SMA on this bar).</div>
                <div className="cond"><span className="cond-label">2</span> today&rsquo;s close &lt; yesterday&rsquo;s close (down close).</div>
              </div>
              <div className="rule-effect">Both true → power_trend = False. Clears <span className="mono">pt_on_idx</span>. Step 8 goes dark; exposure drops by 40.</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── EXIT LADDER ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§7 — EXIT LADDER (VIOLATIONS)</span>
          <h2>Non-negotiable defense rules</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              These fire on <em>current-bar</em> conditions and drive the exit-ladder cards
              on the M Factor page + Position Sizer&rsquo;s auto-mode. Ordered by severity.
            </p>
          </div>

          <div className="rule-card caution">
            <div className="rule-head">
              <span className="rule-badge caution">WATCH</span>
              <span className="rule-name">21 EMA Watch</span>
              <span className="rule-source">mct_endpoint_adapter::_build_active_exits</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Trigger</span> exactly 1 close below 21 EMA (<span className="mono">consec_below_21 == 1</span>).</div>
              </div>
              <div className="rule-effect">Informational. Next-day intraday undercut &gt;1% → Violation. Next-day close below 21 EMA → Confirmed Break.</div>
            </div>
          </div>

          <div className="rule-card stop">
            <div className="rule-head">
              <span className="rule-badge stop">VIOLATE</span>
              <span className="rule-name">VIOLATION_21EMA</span>
              <span className="rule-source">api/mct_engine.py::_phase_violations</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Anchor</span> the day the 21 EMA close-break streak started (<span className="mono">anchor_21_low</span>).</div>
                <div className="cond"><span className="cond-label">Trigger</span> subsequent intraday low undercuts anchor by ≥ 1% (<span className="mono">UNDERCUT</span>=0.01).</div>
              </div>
              <div className="rule-effect">Target exposure 50%. Latches <span className="mono">violation_21_fired</span>. If fired inside correction_active → sets <span className="mono">cap_at_100</span>.</div>
            </div>
          </div>

          <div className="rule-card stop solid">
            <div className="rule-head">
              <span className="rule-badge stop">CONFIRM</span>
              <span className="rule-name">21 EMA Confirmed Break</span>
              <span className="rule-source">mct_endpoint_adapter::_build_active_exits</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Trigger</span> 2+ consecutive closes below 21 EMA (<span className="mono">consec_below_21 ≥ 2</span>).</div>
              </div>
              <div className="rule-effect">Target exposure 30%. Supersedes the Violation card — sustained weakness earns the deeper cut.</div>
            </div>
          </div>

          <div className="rule-card caution">
            <div className="rule-head">
              <span className="rule-badge caution">WATCH</span>
              <span className="rule-name">50 SMA Watch</span>
              <span className="rule-source">mct_endpoint_adapter::_build_active_exits</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Trigger</span> 1+ closes below 50 SMA (<span className="mono">consec_below_50 ≥ 1</span>).</div>
              </div>
              <div className="rule-effect">Informational. Next-day intraday undercut &gt;1% → Violation.</div>
            </div>
          </div>

          <div className="rule-card stop solid">
            <div className="rule-head">
              <span className="rule-badge stop">CRITICAL</span>
              <span className="rule-name">VIOLATION_50SMA</span>
              <span className="rule-source">api/mct_engine.py::_phase_violations</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Anchor</span> the day the 50 SMA close-break streak started (<span className="mono">anchor_50_low</span>).</div>
                <div className="cond"><span className="cond-label">Trigger</span> subsequent intraday low undercuts anchor by ≥ 1%.</div>
              </div>
              <div className="rule-effect">Target exposure 0% (all out). Latches <span className="mono">violation_50_fired</span>. If fired inside correction_active → sets <span className="mono">cap_at_100</span>.</div>
            </div>
          </div>

          <div className="rule-card meta">
            <div className="rule-head">
              <span className="rule-badge meta">SIDE-EFFECT</span>
              <span className="rule-name">CAP_AT_100_ACTIVATED</span>
              <span className="rule-source">api/mct_engine.py::_phase_violations</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Trigger</span> VIOLATION_21EMA or VIOLATION_50SMA fires while <span className="mono">correction_active=True</span>.</div>
              </div>
              <div className="rule-effect">Sets <span className="mono">cap_at_100</span>. Even a subsequent STEP_8 can&rsquo;t push exposure above 100 until CORRECTION_NULLIFIED clears the cap.</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── TREND COUNT ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">§8 — TREND COUNT</span>
          <h2>Signed session count anchored to STEP_4</h2>

          <div className="rule-card sky">
            <div className="rule-head">
              <span className="rule-badge sky">METRIC</span>
              <span className="rule-name">trend_count</span>
              <span className="rule-source">api/mct_engine.py + adapter</span>
            </div>
            <div className="rule-body">
              <div className="rule-conds">
                <div className="cond"><span className="cond-label">Anchor</span> the bar where STEP_4 armed last (<span className="mono">trend_anchor_idx</span>) + sign (<span className="mono">trend_sign</span>: +1 while positive leg holds, −1 after a violation).</div>
                <div className="cond"><span className="cond-label">Value</span> signed count of sessions from anchor to today. Blank when trend_sign = 0 (pre-first-Step-4).</div>
              </div>
              <div className="rule-effect">Surfaced in the M Factor banner as &ldquo;Trend Count: +N / −N.&rdquo; Journal Log persists per-day via the same value; heal path re-stamps when engine output changes.</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── GLOSSARY ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">§9 — GLOSSARY &amp; CONSTANTS</span>
          <h2>Every symbol, defined once</h2>

          <table className="constants">
            <thead>
              <tr><th>Constant</th><th>Value</th><th>Meaning</th></tr>
            </thead>
            <tbody>
              <tr><td className="mono">CORRECTION_DRAWDOWN</td><td className="mono">0.10</td><td>10% depth threshold for correction declaration (intraday low vs reference high).</td></tr>
              <tr><td className="mono">UNDERCUT</td><td className="mono">0.01</td><td>1% intraday undercut threshold for VIOLATION firing (both 21 EMA and 50 SMA).</td></tr>
              <tr><td className="mono">FTD_PCT_THRESHOLD</td><td className="mono">0.01</td><td>Minimum close % gain (vs prior close) to confirm a Follow-Through Day (applied to IXIC and SPY under the dual-index gate).</td></tr>
              <tr><td className="mono">FTD_WINDOW_START</td><td className="mono">4</td><td>Earliest rally_count where FTD is eligible to fire.</td></tr>
              <tr><td className="mono">FTD_WINDOW_END</td><td className="mono">25</td><td>Latest rally_count where FTD is eligible.</td></tr>
              <tr><td className="mono">FTD_DUAL_INDEX_START</td><td className="mono">2026-07-31</td><td>Cutover date: bars on/after require IXIC OR SPY to pass BOTH price ≥ +1% AND volume &gt; prior-day. Pre-cutover bars use the legacy IXIC-price-only rule.</td></tr>
              <tr><td className="mono">PT_ON_21_ABOVE_50_BARS</td><td className="mono">5</td><td>Bars of 21 EMA &gt; 50 SMA required to arm Power-Trend condition 1.</td></tr>
              <tr><td className="mono">PT_ON_LOW_ABOVE_21_BARS</td><td className="mono">10</td><td>Bars of low &gt; 21 EMA required to arm Power-Trend condition 2.</td></tr>
              <tr><td className="mono">PINK_RALLY_DAY_POS_IN_RANGE</td><td className="mono">0.5</td><td>Close ≥ midpoint of intraday range → &ldquo;pink&rdquo; label on a rally day.</td></tr>
              <tr><td className="mono">EXPOSURE_STEP_8</td><td className="mono">200</td><td>Exposure ceiling once STEP_8 fires (subject to cap_at_100).</td></tr>
            </tbody>
          </table>

          <dl className="gloss">
            <dt>reference_high</dt>
            <dd>The peak the drawdown is measured against — ratcheted up on new intraday highs when not in correction.</dd>

            <dt>correction_active vs in_correction</dt>
            <dd>Two flags. <span className="mono">correction_active</span> = the current declared correction cycle. <span className="mono">in_correction</span> = broader &ldquo;we&rsquo;re inside a correction context&rdquo; that persists through soft resets. Nullification clears both.</dd>

            <dt>rally_day_low / rally_day_idx</dt>
            <dd>The low and index of STEP_0. Any subsequent intraday low below <span className="mono">rally_day_low</span> invalidates the rally.</dd>

            <dt>ftd_close / ftd_low</dt>
            <dd>Close and intraday low of the STEP_1 bar. Close &lt; ftd_low fires POST_FTD_SOFT_FAIL.</dd>

            <dt>anchor_21_low / anchor_50_low</dt>
            <dd>The low of the day a close-break streak started against 21 EMA / 50 SMA. VIOLATION fires when a subsequent intraday low undercuts the anchor by ≥ 1%.</dd>

            <dt>consec_below_21 / consec_below_50</dt>
            <dd>Running streak counters — consecutive closes below the respective average. Reset to 0 on any close back above.</dd>

            <dt>consec_low_above_21 / consec_low_above_50 / consec_21_above_50</dt>
            <dd>Running &ldquo;persistence&rdquo; counters used by Step 4, Step 5, Step 8, and PT-ON conditions.</dd>

            <dt>cap_at_100</dt>
            <dd>Ceiling flag. Set by a Violation inside correction_active. Released on CORRECTION_NULLIFIED. Blocks exposure &gt; 100 even if STEP_8 conditions later re-arm.</dd>

            <dt>pt_on_idx / cycle_start_idx / trend_anchor_idx</dt>
            <dd>Three anchor bars the M Factor banner uses for &ldquo;Power-Trend Day N,&rdquo; &ldquo;Cycle Day N,&rdquo; and &ldquo;Trend Count.&rdquo;</dd>

            <dt>force_correction_at_date</dt>
            <dd>EngineConfig field the manual override sets. When present, engine forces a CORRECTION declaration on that exact bar regardless of the systematic depth threshold.</dd>

            <dt>Cycle event vs Live condition</dt>
            <dd>Steps 0, 1, 8 are events — latched for the cycle&rsquo;s life. Steps 2-7 are live — re-checked every bar. Nullification clears every latch.</dd>
          </dl>
        </div>
      </section>

      <footer className="ctm-footer">
        <div className="wrap">
          <p className="mono">M Factor · complete rules reference · api/mct_engine.py is source of truth · if this page and the engine disagree, the engine wins</p>
        </div>
      </footer>

    </div>
  );
}

export default CycleTrackerMethodology;
