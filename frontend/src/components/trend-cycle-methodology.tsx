import "./cycle-tracker-methodology.css";

/**
 * Trend Cycle Methodology — companion to the M Factor Complete Rules Reference.
 *
 * The Trend Cycle is one specific piece of the M Factor engine: the signed
 * count of the current 21-EMA leg. This page documents its arm / flip / re-arm
 * mechanics and links to the full evidence study (Study 01) that measures how
 * deep NASDAQ historically goes between the flip and the next Step-4 arm.
 *
 * Uses the same scoped styles as cycle-tracker-methodology.css. Kept as a
 * standalone page rather than folded into the M Factor page because the flip
 * has its own decision framework and reads better as a dedicated reference.
 *
 * Source of truth for the mechanics lives in api/mct_engine.py; if a constant
 * changes there, mirror it here.
 */
export function TrendCycleMethodology() {
  return (
    <div className="ctm-root">

      {/* ── hero ── */}
      <header className="hero">
        <div className="wrap">
          <span className="eyebrow">MO Trading Methodology · Study 01</span>
          <h1>The Trend Cycle</h1>
          <p className="lead">
            A signed count that tracks the state of the current 21-EMA leg. When it
            <em> flips</em> from <span className="mono">+1</span> to <span className="mono">−1</span>,
            NASDAQ has historically fallen a median of <strong>8.8%</strong> from ref-high
            to trough before Step 4 re-arms — with a <strong>44% chance</strong> the drawdown
            reaches ≥10%. This page documents the mechanics; the study behind the
            numbers is linked below.
          </p>

          <div className="legend">
            <span className="legend-item"><span className="chip go">+1 leg armed</span> uptrend intact</span>
            <span className="legend-item"><span className="chip stop">−1 leg broken</span> 21-EMA violated</span>
            <span className="legend-item"><span className="chip sky">Ref high</span> the peak that anchors dd</span>
            <span className="legend-item"><span className="chip caution">Step 4</span> re-arms next leg</span>
          </div>
        </div>
      </header>

      {/* ── §1 — Two states ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§1 — TWO STATES</span>
          <h2>Positive leg or broken leg</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              The Trend Cycle is always in exactly one of two states. It moves between
              them via two hard-coded edges, and its sign is what the M Factor page
              renders as <span className="mono">Trend Count: +N / −N</span> below the
              Suggested Exposure banner.
            </p>
          </div>

          <div className="state-grid">
            <div className="state-card go">
              <div className="state-tag">+1 · Leg armed</div>
              <div className="state-name">Trend intact</div>
              <div className="state-when">
                Three consecutive bars where the intraday low sits above the 21 EMA,
                closed on an up day.
              </div>
              <div className="state-note">
                The uptrend has demonstrated it can hold its own technical support.
                Ratchet the ref-high on every subsequent bar.
              </div>
            </div>
            <div className="state-card stop">
              <div className="state-tag">−1 · Leg broken</div>
              <div className="state-name">Trend violated</div>
              <div className="state-when">
                After a close below the 21 EMA anchors that day&rsquo;s low, a subsequent
                bar&rsquo;s low undercuts the anchor by ≥1%.
              </div>
              <div className="state-note">
                Sellers took technical support out. The leg is broken until Step 4
                arms a fresh +1 leg.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── §2 — The edges ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§2 — THE FOUR EDGES</span>
          <h2>Exact rules that fire the transitions</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Each edge below has a single, deterministic firing condition. Nothing is
              inferred; nothing is smoothed. The engine simply evaluates these against
              each bar as it arrives.
            </p>
          </div>

          <div className="state-grid">
            <div className="state-card go">
              <div className="state-tag">ARM · Step 4</div>
              <div className="state-name">+1 leg begins</div>
              <div className="state-when">
                <span className="mono">consec_low_above_21 ≥ 3</span> AND{" "}
                <span className="mono">close &gt; prev_close</span>
              </div>
              <div className="state-note">
                Fires on the first up-close that lands with three straight days of
                intraday support above the 21 EMA. Also re-arms after a broken −1 leg.
              </div>
            </div>

            <div className="state-card sky">
              <div className="state-tag">RATCHET · every bar in +1</div>
              <div className="state-name">Ref high updates</div>
              <div className="state-when">
                <span className="mono">ref_high = max(ref_high, high)</span>
              </div>
              <div className="state-note">
                Rolling maximum of the intraday high from the arm date through today.
                This is the peak all downstream drawdown numbers measure against.
              </div>
            </div>

            <div className="state-card stop">
              <div className="state-tag">FLIP · +1 → −1</div>
              <div className="state-name">21-EMA violation</div>
              <div className="state-when">
                First <span className="mono">close &lt; ema_21</span> sets{" "}
                <span className="mono">anchor_low = low</span>. Any later bar with{" "}
                <span className="mono">low &lt; anchor_low × 0.99</span> flips the sign.
              </div>
              <div className="state-note">
                The anchor stays fixed on subsequent below-21 bars — only a close back
                above the 21 EMA clears it. The 1% undercut prevents whipsaws on a bar
                that merely re-tests the anchor.
              </div>
            </div>

            <div className="state-card go">
              <div className="state-tag">RE-ARM · −1 → +1</div>
              <div className="state-name">Same as ARM</div>
              <div className="state-when">
                Same conditions as the initial arm. The −1 leg ends on the bar the new
                +1 leg begins.
              </div>
              <div className="state-note">
                Median 19 sessions from flip to re-arm (25–75th pctile: 10–35).
                The −1 leg is typically a month-long episode.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── §3 — Evidence summary ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§3 — WHAT THE FLIP SIGNALS</span>
          <h2>Evidence from 144 completed cycles</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Study 01 tested 36.5 years of NASDAQ history (1990-2026): 144 completed
              +1&nbsp;→&nbsp;−1&nbsp;→&nbsp;re-arm cycles and 68 distinct ≥9% correction events.
              Two claims were tested:
            </p>
            <p>
              <strong>Necessary-condition claim (holds).</strong> 94.1% of past corrections
              had a Trend Cycle flip in the preceding 5 months; 98.5% within 12 months. If
              I haven&rsquo;t seen a flip in ~5 months, a correction is historically
              improbable — the flip is a genuine precondition.
            </p>
            <p>
              <strong>Sufficient-condition claim (partially holds).</strong> When the flip
              fires, the median trough drawdown from ref-high is <strong>8.8%</strong>. But
              the distribution is wide: <strong>44%</strong> of cycles reach ≥10%,{" "}
              <strong>22%</strong> reach ≥15%, and <strong>8%</strong> become full bear-
              market-magnitude declines (≥25%). The flip is a reliable warning that
              meaningful downside is coming — but not a precise forecast of depth.
            </p>
          </div>

          <div className="state-grid">
            <div className="state-card sky">
              <div className="state-tag">Median outcome</div>
              <div className="state-name">−8.8% dd</div>
              <div className="state-when">
                ref-high to intraday trough before next arm
              </div>
              <div className="state-note">
                25–75th percentile: −5.9% to −13.3%
              </div>
            </div>
            <div className="state-card caution">
              <div className="state-tag">≥10% tail</div>
              <div className="state-name">44% of cycles</div>
              <div className="state-when">technical correction</div>
              <div className="state-note">
                Approximately 2× per year historically
              </div>
            </div>
            <div className="state-card stop">
              <div className="state-tag">≥15% tail</div>
              <div className="state-name">22% of cycles</div>
              <div className="state-when">deep pullback</div>
              <div className="state-note">
                Weak-leg flips concentrate here (25% chance vs 10% for strong-leg)
              </div>
            </div>
            <div className="state-card stop solid" style={{ background: "linear-gradient(180deg, #fff 0%, #F4E1DD 100%)" }}>
              <div className="state-tag">≥25% tail</div>
              <div className="state-name">8% of cycles</div>
              <div className="state-when">bear-market magnitude</div>
              <div className="state-note">
                COVID, dot-com, 2008 lived here
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── §4 — Leg quality ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§4 — LEG QUALITY AT FLIP</span>
          <h2>Median outcome is stable; tail risk splits sharply</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Segmenting the 144 flips by the health of the +1 leg at the moment it
              broke, the <em>median</em> drawdown is basically the same across all three
              categories. The <em>tail</em> is not.
            </p>
          </div>

          <div className="state-grid">
            <div className="state-card stop">
              <div className="state-tag">Weak leg</div>
              <div className="state-name">P(≥15%): 25%</div>
              <div className="state-when">&lt;40 days OR &lt;5% gain · n=99</div>
              <div className="state-note">
                Median dd 9.1% · false-start flips that never really earned their
                trend break hardest.
              </div>
            </div>
            <div className="state-card caution">
              <div className="state-tag">Healthy leg</div>
              <div className="state-name">P(≥15%): 13%</div>
              <div className="state-when">≥40 days AND ≥5% gain · n=45</div>
              <div className="state-note">
                Median dd 8.1% · established trends correct less severely on average.
              </div>
            </div>
            <div className="state-card go">
              <div className="state-tag">Strong leg</div>
              <div className="state-name">P(≥15%): 10%</div>
              <div className="state-when">≥60 days AND ≥10% gain · n=30</div>
              <div className="state-note">
                Median dd 8.6% · the safest tail. Deep bear moves from a strong leg
                are rare but not impossible (COVID happened from a strong leg).
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── §5 — The rule ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§5 — THE RULE I TRADE</span>
          <h2>How the flip changes my exposure</h2>

          <div className="wrap" style={{ padding: 0 }}>
            <p>
              The flip alone is not a permission to sell everything — the median outcome
              is a ~9% drawdown that fully recovers within a month. But it <em>is</em> a
              permission to size down, tighten stops, and stop adding to positions.
            </p>
            <p>
              <strong>Base rule.</strong> When the Trend Cycle flips from +1 to −1,
              reduce exposure to the tier defined by the M Factor state. The tail risk
              (22% chance of ≥15%, 8% chance of ≥25%) is the trade that matters — I
              take those cheaply if I&rsquo;ve already de-risked.
            </p>
            <p>
              <strong>Tail-risk overlay.</strong> When the flip fires from a{" "}
              <em>weak leg</em>, lean harder defensive. Tail risk is 2.5× vs strong-leg
              flips (25% vs 10%) — even if the median outcome is similar, the shape of
              the distribution is meaningfully different.
            </p>
            <p>
              <strong>The absence of a flip is also information.</strong> If no flip in
              5 months, corrections are improbable (98.5% of the last 68 corrections
              had a flip within a year). That&rsquo;s a green light for aggressive
              positioning under UPTREND / POWERTREND.
            </p>
          </div>
        </div>
      </section>

      {/* ── §6 — Limitations ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§6 — WHAT IT DOESN&rsquo;T TELL ME</span>
          <h2>Honest limits</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              <strong>Depth.</strong> The distribution is wide. Median ~9%, modal
              5-15%, tail to 48%. The flip tells me a drop is coming; it doesn&rsquo;t
              tell me how deep.
            </p>
            <p>
              <strong>Timing at the top.</strong> The median flip fires ~5% below the
              ref high. If I&rsquo;m waiting for the flip to sell my last share at the
              top, I miss the top by a meaningful margin. This is a leg-management
              tool, not a top-picker.
            </p>
            <p>
              <strong>Shallow cycles happen.</strong> 34% of flips resolve at &lt;7%
              dd — the &ldquo;false alarms&rdquo; from a correction-hunter&rsquo;s
              perspective. I accept giving up some upside on shallow cycles to be
              safely positioned for deep ones. Asymmetric-payoff instrument by design.
            </p>
            <p>
              <strong>Leg quality shifts probabilities; it doesn&rsquo;t cap outcomes.</strong>{" "}
              The 2020 COVID crash fired from a strong 90-day / 20.7% leg and delivered
              32.6% dd. Leg quality isn&rsquo;t a shield.
            </p>
          </div>
        </div>
      </section>

      {/* ── §7 — Full study link ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">§7 — THE EVIDENCE</span>
          <h2>Full study, distribution charts, every flip since 2018</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Study 01 walks through the methodology (including the two wrong
              measurements I tried before landing on the right one), the complete 144-
              cycle distribution with bar chart, the current live cycle placed against
              history, and every flip on NASDAQ since 2018 with its ref-high, trough,
              and drawdown.
            </p>
            <p>
              <a href="https://claude.ai/code/artifact/7e9d8774-9078-42a7-a027-fbf323594e6b"
                 target="_blank" rel="noreferrer">
                Open Study 01: The Trend Cycle Flip →
              </a>
            </p>
            <p style={{ fontSize: "0.9rem", color: "var(--slate)" }}>
              Reproducibility script: <span className="mono">scratchpad/trend_cycle_leg_trough.py</span>.
              Cycle mechanics: <span className="mono">api/mct_engine.py</span> phase 8, trend-sign block.
              Data: yfinance <span className="mono">^IXIC</span> daily OHLCV, 1990-01-02
              through 2026-07-29.
            </p>
          </div>
        </div>
      </section>

    </div>
  );
}

export default TrendCycleMethodology;
