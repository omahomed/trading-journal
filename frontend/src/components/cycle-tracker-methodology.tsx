import "./cycle-tracker-methodology.css";

/**
 * Cycle Tracker Methodology — the full editorial explainer of the M Factor
 * exposure model. Self-contained: all styling is scoped under .ctm-root in
 * cycle-tracker-methodology.css, and the palette is pinned locally so it
 * renders identically regardless of the app's light/dark theme.
 *
 * Fonts (Newsreader / IBM Plex Sans / IBM Plex Mono) are provided by the
 * M Factor route via next/font CSS vars (--font-newsreader, --font-plex-sans,
 * --font-plex-mono). See app/(app)/m-factor/page.tsx.
 */
export function CycleTrackerMethodology() {
  return (
    <div className="ctm-root">

      {/* ── hero ── */}
      <header className="hero">
        <div className="wrap">
          <span className="eyebrow">The Cycle Tracker · Methodology</span>
          <h1>How the M&nbsp;Factor reads the market</h1>
          <p className="lead">
            The page answers one question: <em>given what the market is doing right now,
            how much of your account should be exposed?</em> This explains how it decides,
            what every signal means, and how to use it without fighting it.
          </p>

          <div className="gauge-card">
            <div className="gauge-top">
              <span className="lbl">Suggested exposure</span>
              <span className="gauge-num">100<span>/ 200</span></span>
            </div>
            <div className="seglane" aria-hidden="true">
              <div className="seg on" /><div className="seg on" /><div className="seg on" />
              <div className="seg on" /><div className="seg on" /><div className="seg" />
              <div className="seg" /><div className="seg" /><div className="seg" /><div className="seg" />
            </div>
            <div className="scalerow"><span>0</span><span>40</span><span>80</span><span>120</span><span>160</span><span>200</span></div>
            <div className="gauge-foot">
              Each lit block is a market signal that&rsquo;s currently true. In the reading shown, <b>five of
              ten</b> are lit &mdash; a total of <b>100</b>. Exposure isn&rsquo;t a switch you flip; it&rsquo;s a total you add up.
            </div>
          </div>
        </div>
      </header>

      {/* ── 01 ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">01 — THE CORE IDEA</span>
          <h2>Exposure is a sum, not a switch</h2>
          <p>
            Most people picture market exposure as on or off &mdash; you&rsquo;re in, or you&rsquo;re out.
            The M&nbsp;Factor doesn&rsquo;t work that way. It watches a fixed checklist of <strong>nine
            market signals</strong>. Every signal that&rsquo;s currently true adds points. Your
            <strong> suggested exposure</strong> is simply the total of those points, on a scale from 0 to 200.
          </p>
          <ul className="principles">
            <li><b>Every signal green → 200.</b> Maximum offense; the environment justifies full leverage.</li>
            <li><b>Some green, some not → a number in between.</b> And it moves, day by day, as conditions change.</li>
            <li><b>An official correction → 0.</b> No cycle is running; the slate is wiped clean.</li>
          </ul>
          <p>
            Why a sum instead of a switch? Because markets rarely flip cleanly from bad to good. A
            recovery happens in stages &mdash; first the selling stops, then the market reclaims its
            moving averages, then the whole structure stacks up in order. A running total lets your
            exposure <em>climb and fall in step</em> with that, instead of lurching between all-in and
            all-out and forcing you to guess the exact turning point.
          </p>
          <div className="note">
            <span className="tag">Read the number this way</span>
            <b>100</b> = fully invested, no leverage. <b>Above 100</b> = the environment would tolerate
            margin. <b>Below 100</b> = hold some cash. <b>200</b> = the maximum the system will ever suggest.
          </div>
        </div>
      </section>

      {/* ── 02 ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">02 — THE MARKET CYCLE</span>
          <h2>Where you are in the cycle</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Growth markets move in a repeating rhythm. The M&nbsp;Factor names where you are in it
              with one of four <strong>states</strong>. A full cycle runs from one correction to the
              next, and everything on the page is measured relative to the cycle you&rsquo;re currently in.
            </p>
          </div>

          <figure>
            <svg viewBox="0 0 880 250" role="img" aria-label="The market cycle: correction resets to zero, a rally day and follow-through start the climb, the uptrend builds structure, power-trend reaches full offense, then a pullback de-levers back toward a new correction.">
              <defs>
                <marker id="ctm-ar" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#6B7079" /></marker>
              </defs>
              <line x1="40" y1="200" x2="840" y2="200" stroke="#E4DFD5" strokeWidth="1" />
              <circle cx="90" cy="200" r="9" fill="#B23A2B" />
              <text x="90" y="228" textAnchor="middle" fontSize="12" fontWeight="600" fill="#B23A2B">CORRECTION</text>
              <text x="90" y="244" textAnchor="middle" fontSize="11" fill="#6B7079">exp 0</text>
              <circle cx="260" cy="160" r="9" fill="#BB7A0E" />
              <text x="260" y="140" textAnchor="middle" fontSize="12" fontWeight="600" fill="#BB7A0E">RALLY MODE</text>
              <text x="260" y="124" textAnchor="middle" fontSize="11" fill="#6B7079">rally day → FTD</text>
              <circle cx="450" cy="110" r="9" fill="#3D6E91" />
              <text x="450" y="90" textAnchor="middle" fontSize="12" fontWeight="600" fill="#3D6E91">UPTREND</text>
              <text x="450" y="74" textAnchor="middle" fontSize="11" fill="#6B7079">structure builds</text>
              <circle cx="650" cy="64" r="11" fill="#0A8F5E" />
              <text x="650" y="44" textAnchor="middle" fontSize="12" fontWeight="600" fill="#0A8F5E">POWER-TREND</text>
              <text x="650" y="28" textAnchor="middle" fontSize="11" fill="#6B7079">full offense · up to 200</text>
              <circle cx="790" cy="150" r="8" fill="#6B7079" />
              <text x="800" y="150" fontSize="11" fill="#6B7079">pullback</text>
              <text x="800" y="166" fontSize="11" fill="#6B7079">de-levers</text>
              <path d="M99,196 L251,164" stroke="#6B7079" strokeWidth="1.4" markerEnd="url(#ctm-ar)" />
              <path d="M269,156 L441,114" stroke="#6B7079" strokeWidth="1.4" markerEnd="url(#ctm-ar)" />
              <path d="M459,106 L640,68" stroke="#6B7079" strokeWidth="1.4" markerEnd="url(#ctm-ar)" />
              <path d="M660,72 L783,143" stroke="#6B7079" strokeWidth="1.4" markerEnd="url(#ctm-ar)" />
              <path d="M790,158 C790,235 250,250 96,209" fill="none" stroke="#B23A2B" strokeWidth="1.4" strokeDasharray="5 4" markerEnd="url(#ctm-ar)" />
              <text x="430" y="244" textAnchor="middle" fontSize="11" fill="#B23A2B">if the pullback deepens to an official correction → reset to 0, new cycle</text>
            </svg>
            <figcaption><b>The cycle.</b> Exposure rises as the market proves itself and falls as that proof breaks down. Most pullbacks de-lever and recover without resetting &mdash; only an <em>official correction</em> wipes the slate and starts the count over.</figcaption>
          </figure>

          <div className="wrap" style={{ padding: 0 }}>
            <h3>The four states</h3>
            <ul className="principles">
              <li><b>Correction</b> &mdash; the market has fallen far enough, and broken enough structure, that the prior advance is over. Exposure resets to zero and the cycle starts fresh.</li>
              <li><b>Rally Mode</b> &mdash; the selling has stopped and the market has made a first attempt to turn up. Tentative, small exposure, hunting for confirmation.</li>
              <li><b>Uptrend</b> &mdash; the turn is confirmed and the market is rebuilding its moving-average structure. Exposure climbs as each piece falls into place.</li>
              <li><b>Power-Trend</b> &mdash; the strongest state. The trend is established and durable; the system permits full offense.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* ── 03 ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">03 — THE NINE SIGNALS</span>
          <h2>The Entry Ladder</h2>
          <p>The checklist has nine rungs, numbered 0 through 8. They come in two kinds, and the difference matters:</p>
          <p>
            <strong>Events</strong> are things that <em>happened</em> and stay true for the rest of the
            cycle. Once the market earns one, it&rsquo;s banked until the next correction &mdash; it won&rsquo;t
            flicker off on an ordinary pullback. <strong>Conditions</strong> are things that are either
            true or false <em>right now</em>, rechecked every single day.
          </p>

          <table className="ladder">
            <thead>
              <tr><th>#</th><th>Signal</th><th>Type</th><th className="pts">Adds</th></tr>
            </thead>
            <tbody>
              <tr className="ev"><td className="st">0</td><td className="nm">Rally Day<small>The market&rsquo;s first strong up-day off a low &mdash; the spark that starts a new cycle.</small></td><td><span className="kind evt">Event</span></td><td className="pts">+20</td></tr>
              <tr className="ev"><td className="st">1</td><td className="nm">Follow-Through Day<small>A powerful confirmation day (O&rsquo;Neil&rsquo;s FTD) that says the rally is the real thing, not a dead-cat bounce.</small></td><td><span className="kind evt">Event</span></td><td className="pts">+20</td></tr>
              <tr><td className="st">2</td><td className="nm">Close above the 21 EMA<small>The market finishes the day above its short-term trend line.</small></td><td><span className="kind live">Live</span></td><td className="pts">+20</td></tr>
              <tr><td className="st">3</td><td className="nm">Low above the 21 EMA<small>Stronger &mdash; it never even dipped below the line intraday.</small></td><td><span className="kind live">Live</span></td><td className="pts">+20</td></tr>
              <tr><td className="st">4</td><td className="nm">Low above the 21 EMA, 3 days running<small>The hold above the short-term line is durable, not a one-day fluke.</small></td><td><span className="kind live">Live</span></td><td className="pts">+20</td></tr>
              <tr><td className="st">5</td><td className="nm">Low above the 50 SMA, 3 days running<small>The market is holding cleanly above its medium-term average too.</small></td><td><span className="kind live">Live</span></td><td className="pts">+20</td></tr>
              <tr><td className="st">6</td><td className="nm">21 EMA &gt; 50 SMA &gt; 200 SMA<small>The medium and long averages are stacked in bullish order.</small></td><td><span className="kind live">Live</span></td><td className="pts">+20</td></tr>
              <tr><td className="st">7</td><td className="nm">8 &gt; 21 &gt; 50 &gt; 200<small>The full stack &mdash; every average from fastest to slowest, all in order.</small></td><td><span className="kind live">Live</span></td><td className="pts">+20</td></tr>
              <tr className="ev"><td className="st">8</td><td className="nm">Power-Trend ON<small>The regime flag for an established, durable trend. Worth double.</small></td><td><span className="kind evt">Event</span></td><td className="pts">+40</td></tr>
            </tbody>
          </table>

          <div className="note go">
            <span className="tag">The sticky base</span>
            Steps <b>0, 1, and 8</b> are the events. Once a cycle has logged its Rally Day and
            Follow-Through and turned its Power-Trend on, those <b>80 points are banked</b> for the life
            of the cycle. They don&rsquo;t drop out when the market has a rough week &mdash; only an official
            correction clears them.
          </div>

          <h3>Rungs fill independently</h3>
          <p>
            The live conditions don&rsquo;t have to fill in order. During a choppy stretch the market can
            satisfy step 6 (the medium-term stack is intact) while <em>failing</em> steps 4 and 5 (it
            hasn&rsquo;t held cleanly above the averages for three straight days). The ladder shows exactly
            that &mdash; gaps and all. A green rung high on the ladder with dark rungs below it isn&rsquo;t a
            bug; it&rsquo;s an honest picture of a market that&rsquo;s structurally sound but still choppy.
          </p>

          <div className="note">
            <span className="tag">Worked example</span>
            Say the lit rungs are <b>0</b> (Rally Day, banked), <b>1</b> (FTD, banked), <b>6</b>
            (medium stack intact), and <b>8</b> (Power-Trend on). That&rsquo;s 20 + 20 + 20 + 40 = <b className="mono">100</b>.
            Steps 2&ndash;5 and 7 are dark &mdash; the market is holding above its 50-day but hasn&rsquo;t yet
            put together a clean multi-day hold above the 21-day after a pullback.
          </div>
        </div>
      </section>

      {/* ── 04 ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">04 — HOW EXPOSURE MOVES</span>
          <h2>How exposure breathes</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>Because exposure is a running total of live signals, it rises and falls on its own as the market does its thing. Here&rsquo;s a full cycle, start to finish:</p>
          </div>

          <figure>
            <svg viewBox="0 0 880 340" role="img" aria-label="Exposure curve across one cycle, climbing from zero through the rally and uptrend to a 200 peak in power-trend, then stepping down through a pullback before a new correction resets it to zero.">
              <g fill="#9aa0a6">
                <line x1="70" y1="40" x2="850" y2="40" stroke="#EDE9E1" /><text x="40" y="44" fontSize="11">200</text>
                <line x1="70" y1="120" x2="850" y2="120" stroke="#EDE9E1" /><text x="40" y="124" fontSize="11">150</text>
                <line x1="70" y1="200" x2="850" y2="200" stroke="#E4DFD5" /><text x="40" y="204" fontSize="11">100</text>
                <line x1="70" y1="280" x2="850" y2="280" stroke="#EDE9E1" /><text x="44" y="284" fontSize="11">50</text>
                <line x1="70" y1="300" x2="850" y2="300" stroke="#E4DFD5" /><text x="50" y="316" fontSize="11">0</text>
              </g>
              <path d="M70,300 L150,300 L150,268 L240,268 L240,172 L330,172 L330,140 L420,140 L420,72 L520,72 L520,40 L600,40 L600,140 L660,140 L660,172 L720,172 L720,268 L780,268 L780,300 L850,300"
                    fill="none" stroke="#0A8F5E" strokeWidth="2.5" strokeLinejoin="round" />
              <path d="M70,300 L150,300 L150,268 L240,268 L240,172 L330,172 L330,140 L420,140 L420,72 L520,72 L520,40 L600,40 L600,140 L660,140 L660,172 L720,172 L720,268 L780,268 L780,300 Z"
                    fill="#0A8F5E" opacity="0.07" />
              <g fill="#454C54">
                <circle cx="150" cy="268" r="3.5" fill="#BB7A0E" /><text x="150" y="258" textAnchor="middle" fontSize="10.5" fill="#BB7A0E">rally day</text>
                <circle cx="240" cy="172" r="3.5" fill="#BB7A0E" /><text x="240" y="162" textAnchor="middle" fontSize="10.5" fill="#BB7A0E">FTD +steps</text>
                <circle cx="520" cy="40" r="4" fill="#0A8F5E" /><text x="520" y="30" textAnchor="middle" fontSize="10.5" fill="#0A8F5E">full stack · 200</text>
                <circle cx="600" cy="140" r="3.5" fill="#6B7079" /><text x="610" y="135" fontSize="10.5" fill="#6B7079">loses 21 EMA</text>
                <circle cx="720" cy="172" r="3.5" fill="#6B7079" /><text x="720" y="192" textAnchor="middle" fontSize="10.5" fill="#6B7079">21&lt;50 · PT off</text>
                <circle cx="780" cy="300" r="4" fill="#B23A2B" /><text x="790" y="296" fontSize="10.5" fill="#B23A2B">correction · reset</text>
              </g>
              <text x="460" y="334" textAnchor="middle" fontSize="11" fill="#9aa0a6">one cycle  →  time</text>
            </svg>
            <figcaption><b>The shape of a cycle.</b> Notice the staircase: exposure climbs one rung at a time as the market proves itself, and steps <em>down</em> the same way as that proof breaks. You never have to call the top &mdash; the ladder de-levers for you, automatically, in 20-point steps.</figcaption>
          </figure>

          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Reading it left to right: a correction sits the market at <b className="mono">0</b>. A Rally
              Day starts the cycle at 20. The Follow-Through usually lands alongside its first live steps,
              jumping the reading to 80. As the market reclaims and holds its averages, the live rungs fill
              in and exposure climbs through 120 and 160. Power-Trend turns on, and with the full stack
              aligned the reading tops out at <b className="mono">200</b> &mdash; maximum offense.
            </p>
            <p>
              Then a pullback. Price loses the 21&nbsp;EMA, so the rungs that depend on it (2, 3, 4) go
              dark and exposure steps down &mdash; but the 80-point event base and the still-intact averages
              hold the line near 100&ndash;120. If the slide deepens and the 21&nbsp;EMA crosses below the
              50, Power-Trend ends and the reading falls toward 40. Only if it becomes an official
              correction does it reset to 0 and the next cycle begin.
            </p>
            <div className="note caution">
              <span className="tag">Power-Trend is not automatically 200</span>
              Turning Power-Trend on adds 40 points &mdash; it does not jump you to the ceiling. Reaching
              <b> 200</b> still requires every other rung to be green. A <em>young</em> Power-Trend whose
              50-day average hasn&rsquo;t yet climbed above its 200-day will read <b>160</b>, not 200 &mdash;
              and that&rsquo;s correct. The structure isn&rsquo;t fully built yet, so the system doesn&rsquo;t pretend it is.
            </div>
          </div>
        </div>
      </section>

      {/* ── 05 ── */}
      <section className="sec">
        <div className="wide">
          <span className="secnum">05 — WHAT COUNTS AS A CORRECTION</span>
          <h2>Why the reset is hard to trigger</h2>
          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Not every dip is a correction. The M&nbsp;Factor is deliberately strict about the reset,
              because declaring a false correction throws away a perfectly good cycle and forces the
              market to re-prove everything from scratch. There are three depths, and only the middle
              one resets the cycle.
            </p>
          </div>

          <figure>
            <svg viewBox="0 0 880 230" role="img" aria-label="Three correction tiers by depth: a shallow pullback de-levers but does not reset; an official correction requires 10 percent down plus a confirmed 50-day break and does reset; a bear market is deeper than 20 percent.">
              <text x="40" y="34" fontSize="11" fill="#6B7079">high</text>
              <line x1="64" y1="40" x2="64" y2="200" stroke="#E4DFD5" strokeWidth="1.5" />
              <g fill="#9aa0a6">
                <line x1="58" y1="40" x2="64" y2="40" stroke="#9aa0a6" /><text x="22" y="44" fontSize="11">0%</text>
                <line x1="58" y1="110" x2="64" y2="110" stroke="#9aa0a6" /><text x="16" y="114" fontSize="11">-10%</text>
                <line x1="58" y1="180" x2="64" y2="180" stroke="#9aa0a6" /><text x="16" y="184" fontSize="11">-20%</text>
              </g>
              <rect x="90" y="44" width="760" height="62" rx="7" fill="#F6EDD6" />
              <text x="106" y="68" fontSize="12.5" fontWeight="600" fill="#BB7A0E">UNDER PRESSURE — a pullback</text>
              <text x="106" y="90" className="s" fontSize="12.5" fill="#454C54">Below the 50-day, but down less than 10%. Live rungs drop and you de-lever — but the cycle does NOT reset.</text>
              <rect x="90" y="112" width="760" height="62" rx="7" fill="#F4E1DD" stroke="#B23A2B" strokeWidth="1.5" />
              <text x="106" y="136" fontSize="12.5" fontWeight="600" fill="#B23A2B">OFFICIAL CORRECTION — resets the cycle</text>
              <text x="106" y="158" className="s" fontSize="12.5" fill="#454C54">Down ≥10% from the high AND a confirmed close below the 50-day (two closes, not one poke). Power-Trend off, reset to 0, new rally hunt.</text>
              <rect x="90" y="180" width="760" height="34" rx="7" fill="#EDE9E1" />
              <text x="106" y="202" fontSize="12.5" fontWeight="600" fill="#454C54">BEAR — down more than 20%. A deeper version of the same reset.</text>
            </svg>
            <figcaption><b>Two conditions, both required.</b> An official correction needs real <em>depth</em> (≥10%) AND broken <em>structure</em> (a confirmed 50-day break) at the same time.</figcaption>
          </figure>

          <div className="wrap" style={{ padding: 0 }}>
            <p>
              Why insist on both? Because either one alone is a false alarm. A fast 10% air-pocket that
              the 50-day average holds is a shakeout, not a trend change. A shallow drift just below the
              50-day while the market is only off 4&ndash;5% is noise. It takes <em>both</em> &mdash;
              genuine depth and broken structure &mdash; to conclude the prior advance is actually over.
            </p>
            <div className="note stop">
              <span className="tag">This replaced a known flaw</span>
              An earlier version reset the cycle on a simple &ldquo;<b>down 7%</b>&rdquo; rule. That fired
              on ordinary wobbles &mdash; wiping the day-count and faking a fresh cycle while the 50-day
              average was never even broken. The 10%-plus-confirmed-50 rule is the fix: the count only
              resets when the market has genuinely changed character.
            </div>
          </div>
        </div>
      </section>

      {/* ── 06 ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">06 — SUGGESTIONS VS RULES</span>
          <h2>The two ladders</h2>
          <p>
            The Entry Ladder you&rsquo;ve just learned tells you how much you <em>may</em> carry. It&rsquo;s a
            <strong> suggestion</strong> &mdash; the ceiling the environment justifies. What you actually
            hold is your call, driven by your individual positions and conviction.
          </p>
          <p>
            There is a second ladder, and it is <em>not</em> a suggestion. The <strong>Exit Ladder</strong>
            is a set of rules. When its triggers fire, you act &mdash; they&rsquo;re not advisory.
          </p>
          <ul className="principles">
            <li><b>Tier 1</b> &mdash; first close below the 21&nbsp;EMA, then a next-day intraday undercut of more than 1%: come off margin, stop opening new positions.</li>
            <li><b>Tier 2</b> &mdash; two consecutive closes below the 21&nbsp;EMA: audit the book and cull everything that isn&rsquo;t a core monster position.</li>
            <li><b>Tier 3</b> &mdash; two consecutive closes below the 50&nbsp;SMA: escalate &mdash; even the core positions get a hard look.</li>
          </ul>
          <div className="note">
            <span className="tag">The asymmetry is the point</span>
            <b>Offense is optional</b> &mdash; you can always choose to carry less than the suggested
            exposure. <b>Defense is not</b> &mdash; when the exit rules trigger, hesitating is exactly how
            a manageable drawdown compounds into a damaging one. The suggestions are where you exercise
            judgment; the rules are where you don&rsquo;t.
          </div>
        </div>
      </section>

      {/* ── 07 ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">07 — READING THE PAGE</span>
          <h2>What you&rsquo;re looking at</h2>
          <p>A quick tour of the M&nbsp;Factor page itself:</p>
          <ul className="reading">
            <li><span className="ri">›</span><span className="rt"><b>The state banner.</b> The large regime label (e.g. <span className="mono">POWER-TREND</span>) and a one-line subtitle. &ldquo;Power-Trend active — Day 52&rdquo; means it&rsquo;s been 52 trading sessions since this cycle&rsquo;s Rally Day.</span></li>
            <li><span className="ri">›</span><span className="rt"><b>Suggested Exposure.</b> The headline number, 0 to 200 &mdash; the sum of every green rung, read as a percentage of your account.</span></li>
            <li><span className="ri"><span className="ico ok">✓</span></span><span className="rt"><b>Green check.</b> This signal is true and contributing its points right now.</span></li>
            <li><span className="ri"><span className="ico open">○</span></span><span className="rt"><b>Open circle.</b> A live condition that&rsquo;s currently false. It can turn green on the next day&rsquo;s bar if price cooperates.</span></li>
            <li><span className="ri"><span className="ico lock">🔒</span></span><span className="rt"><b>Lock.</b> A latched event not yet earned this cycle. You can&rsquo;t get it from a single day&rsquo;s move &mdash; the cycle has to produce it. (A lock on Rally Day means no cycle is running at all.)</span></li>
            <li><span className="ri">›</span><span className="rt"><b>The contribution labels</b> (<span className="mono">+20</span> / <span className="mono">+40</span>). Each rung shows what it&rsquo;s worth. <b>The green rungs always sum to the headline number</b> &mdash; if they don&rsquo;t, the page is lying about something.</span></li>
          </ul>
        </div>
      </section>

      {/* ── 08 ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">08 — HOW TO USE IT</span>
          <h2>A coach, not an autopilot</h2>
          <ul className="principles">
            <li><b>It tells you what the environment justifies &mdash; not what to buy.</b> The M&nbsp;Factor doesn&rsquo;t place trades or size positions. That&rsquo;s the position sizer and your own rules. It sets the weather, not the itinerary.</li>
            <li><b>Treat the suggested exposure as a ceiling, not a target.</b> A clean Power-Trend gives you permission to be aggressive; it never obligates you to be.</li>
            <li><b>De-lever into strength, not weakness.</b> The ladder is designed to let you reduce on the way up &mdash; trimming while rungs are still green &mdash; rather than forcing a fire-sale after they&rsquo;ve all gone dark.</li>
            <li><b>When the Exit Ladder triggers, follow it.</b> Judgment lives in the suggestions. The rules are where judgment gets you in trouble.</li>
            <li><b>Let it absorb the noise.</b> A single intraday dip resets nothing &mdash; that&rsquo;s by design. Let the ladder ignore the ticks so you don&rsquo;t have to.</li>
          </ul>
        </div>
      </section>

      {/* ── 09 ── */}
      <section className="sec">
        <div className="wrap">
          <span className="secnum">09 — GLOSSARY</span>
          <h2>Every term, defined</h2>
          <dl className="gloss">
            <dt>Exposure / Suggested Exposure</dt>
            <dd>How much of the account the market environment justifies having at risk, as a percentage. 0 = all cash; 100 = fully invested; 200 = maximum, with leverage. It is a suggestion, not an instruction.</dd>

            <dt>EMA — Exponential Moving Average</dt>
            <dd>A moving average that weights recent prices more heavily, so it reacts faster. The page uses the <span className="mono">8 EMA</span> (very fast) and <span className="mono">21 EMA</span> (short-term trend).</dd>

            <dt>SMA — Simple Moving Average</dt>
            <dd>A plain average of the last N closes. The page uses the <span className="mono">50 SMA</span> (medium-term trend) and <span className="mono">200 SMA</span> (long-term trend).</dd>

            <dt>The stack</dt>
            <dd>The moving averages lined up in order, fast to slow: 8 above 21 above 50 above 200. A fully stacked market is the textbook picture of a healthy uptrend.</dd>

            <dt>Rally Day</dt>
            <dd>The first strong up-day off a low that begins a new cycle. It opens the count but is only a tentative signal until confirmed.</dd>

            <dt>Follow-Through Day (FTD)</dt>
            <dd>William O&rsquo;Neil&rsquo;s confirmation signal &mdash; a high-conviction up-day, on heavier volume, that arrives a few days after a Rally Day and signals the rally is real rather than a bounce.</dd>

            <dt>Power-Trend</dt>
            <dd>The regime flag for an established, durable uptrend. It&rsquo;s worth 40 points and stays on until the 21&nbsp;EMA crosses below the 50&nbsp;SMA or an official correction hits.</dd>

            <dt>Cycle / Cycle Day</dt>
            <dd>One full run from correction to correction. Cycle Day counts trading sessions since the current cycle&rsquo;s Rally Day.</dd>

            <dt>Event (latched)</dt>
            <dd>A signal that, once earned, stays true for the rest of the cycle &mdash; it won&rsquo;t flicker off on a pullback. Steps 0, 1, and 8.</dd>

            <dt>Condition (live)</dt>
            <dd>A signal that&rsquo;s re-checked every day and is simply true or false right now. Steps 2 through 7.</dd>

            <dt>Sticky base</dt>
            <dd>The 80 points from the three events (Rally Day, FTD, Power-Trend). Banked for the life of the cycle and cleared only by an official correction.</dd>

            <dt>Reference high</dt>
            <dd>The peak the current drawdown is measured against &mdash; the input to the &ldquo;down 10%&rdquo; half of the correction test.</dd>

            <dt>Drawdown</dt>
            <dd>How far the market (or the account) is below its recent high, in percent.</dd>

            <dt>Undercut</dt>
            <dd>A move below a prior low or a moving-average line. A &gt;1% intraday undercut of the 21&nbsp;EMA is the trigger for the first exit tier.</dd>

            <dt>Off margin</dt>
            <dd>Carrying no borrowed money &mdash; exposure at or below 100. The first thing the Exit Ladder asks you to do under pressure.</dd>

            <dt>21&lt;50 cross</dt>
            <dd>When the 21&nbsp;EMA falls below the 50&nbsp;SMA &mdash; the trigger that ends Power-Trend.</dd>

            <dt>Official correction</dt>
            <dd>Down ≥10% from the reference high <em>and</em> a confirmed (two-close) break below the 50&nbsp;SMA. The only event that resets the cycle to 0.</dd>

            <dt>Entry Ladder</dt>
            <dd>The nine-rung checklist that sums to suggested exposure. A set of suggestions.</dd>

            <dt>Exit Ladder</dt>
            <dd>The tiered defense rules (Tier 1&ndash;3) that tell you when to de-lever, cull, and escalate. Not optional.</dd>

            <dt>Core / monster position</dt>
            <dd>A large winner held under the strictest sell rules. The last thing the Exit Ladder asks you to touch.</dd>
          </dl>
        </div>
      </section>

      <footer className="ctm-footer">
        <div className="wrap">
          <p className="mono">The Cycle Tracker — M Factor methodology · NASDAQ-based · this page sits on top of M&nbsp;Factor, it is not a replacement for it.</p>
          <p>Suggestions tell you what you may do. Rules tell you what you must. Know which is which.</p>
        </div>
      </footer>

    </div>
  );
}

export default CycleTrackerMethodology;
