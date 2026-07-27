// Pure helpers for the Game Plan section (migration 052). The section
// wraps a plain textarea keyed off a journal row's `day`; editability is
// governed by "how far into the future does the lockdown push?"
//
// Rule: plan for day X is editable while today (America/Chicago) is
// STRICTLY BEFORE the next weekday after X.
//   Mon-Thu → editable that day only; locks at X+1 00:00 CT
//   Fri → editable Fri + Sat + Sun; locks at Mon 00:00 CT
//   Sat/Sun → editable Sat/Sun; locks at following Mon 00:00 CT
//
// v1 ignores market holidays. If a holiday falls after a Thursday, the
// Thursday plan will lock on Friday 00:00 even though the market is
// closed. Acceptable trade-off vs. wiring a holiday calendar client-side
// (the server matches this behavior so the two sides stay coherent).
//
// Server side lives at api/main.py:_game_plan_lock_date; both sides need
// to agree on the calendar. If we ever add a holiday map, do it in one
// place and mirror.

/** Return the ISO date string (YYYY-MM-DD) at which day X's plan becomes
 *  locked. Editable while today_ct < return value. Pure calendar math —
 *  no `Date.now()` reads inside so it's safe to call from tests. */
export function gamePlanLockDate(dayIso: string): string {
  const [y, m, d] = dayIso.slice(0, 10).split("-").map(Number);
  // Build a UTC anchor for the input date; we only ever compare pure
  // calendar dates (YYYY-MM-DD), so any TZ offset would cancel out.
  const anchor = new Date(Date.UTC(y, m - 1, d));
  // getUTCDay(): 0=Sun, 1=Mon, ..., 6=Sat
  const dow = anchor.getUTCDay();
  let add: number;
  if (dow >= 1 && dow <= 4) add = 1;       // Mon-Thu → +1
  else if (dow === 5) add = 3;             // Fri → Mon
  else if (dow === 6) add = 2;             // Sat → Mon
  else /* dow === 0 (Sun) */ add = 1;      // Sun → Mon
  const lock = new Date(Date.UTC(y, m - 1, d + add));
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${lock.getUTCFullYear()}-${pad(lock.getUTCMonth() + 1)}-${pad(lock.getUTCDate())}`;
}

/** true iff `todayCt` (YYYY-MM-DD in America/Chicago) is strictly before
 *  the lock date for `dayIso`. Both inputs must be pre-formatted so this
 *  function stays pure and testable. */
export function isGamePlanEditable(dayIso: string, todayCt: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(dayIso)) return false;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(todayCt)) return false;
  return todayCt < gamePlanLockDate(dayIso);
}
