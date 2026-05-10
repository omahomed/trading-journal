import { parseOptionTicker } from "./options";
import type { TradePosition } from "./api";

/**
 * Match a single search token against a trade. Token semantics, in order:
 *
 *  1. Equity ticker exact match — `token === t.ticker`
 *  2. Option underlying match — `token === parseOptionTicker(t.ticker)?.underlying`
 *     (so searching "DOCN" surfaces DOCN equity AND every "DOCN ..." option)
 *  3. Trade ID exact match — `token === t.trade_id` (e.g. "202605-013")
 *  4. Trade ID prefix match — `t.trade_id.startsWith(token)`, only when the
 *     token starts with a digit. Trade IDs always begin with `YYYYMM`, so a
 *     digit-prefixed token unambiguously addresses an ID, never a ticker.
 *     This is what powers partial-month queries like "202605".
 *
 * Equity ticker comparison is case-sensitive to preserve pre-existing
 * filter behavior (the input is always uppercased upstream, and stored
 * tickers are uppercase). Option underlying comparison upper-cases the
 * token defensively, since parseOptionTicker normalises its result.
 */
export function matchesTradeQuery(
  trade: Pick<TradePosition, "ticker" | "trade_id">,
  token: string,
): boolean {
  const tk = (trade.ticker || "").trim();
  const id = (trade.trade_id || "").trim();
  const q = token.trim();
  if (!q) return false;
  if (tk === q) return true;
  const parsed = parseOptionTicker(tk);
  if (parsed && parsed.underlying === q.toUpperCase()) return true;
  if (id === q) return true;
  if (/^\d/.test(q) && id.startsWith(q)) return true;
  return false;
}

/** True iff any token in the list matches the trade. Empty list → false. */
export function matchesAnyTradeQuery(
  trade: Pick<TradePosition, "ticker" | "trade_id">,
  tokens: readonly string[],
): boolean {
  return tokens.some(t => matchesTradeQuery(trade, t));
}
