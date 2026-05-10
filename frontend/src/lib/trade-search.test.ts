import { describe, it, expect } from "vitest";
import { matchesTradeQuery, matchesAnyTradeQuery } from "./trade-search";

const equity = { ticker: "DOCN", trade_id: "202605-013" };
const docnCall = { ticker: "DOCN 260515 $105C", trade_id: "202605-014" };
const aaplPut = { ticker: "AAPL 260620 $180P", trade_id: "202604-007" };
const noTicker = { ticker: "", trade_id: "202603-001" };

describe("matchesTradeQuery — ticker semantics", () => {
  it("matches an equity ticker on exact equality", () => {
    expect(matchesTradeQuery(equity, "DOCN")).toBe(true);
  });
  it("matches an option's underlying when only the underlying is typed", () => {
    expect(matchesTradeQuery(docnCall, "DOCN")).toBe(true);
  });
  it("still matches the full option ticker when typed verbatim", () => {
    expect(matchesTradeQuery(docnCall, "DOCN 260515 $105C")).toBe(true);
  });
  it("does not cross-match different underlyings", () => {
    expect(matchesTradeQuery(aaplPut, "DOCN")).toBe(false);
  });
  it("returns false for an unrelated equity ticker", () => {
    expect(matchesTradeQuery(equity, "AAPL")).toBe(false);
  });
});

describe("matchesTradeQuery — trade ID semantics", () => {
  it("matches a trade by its exact ID", () => {
    expect(matchesTradeQuery(equity, "202605-013")).toBe(true);
  });
  it("matches a trade ID by month prefix", () => {
    expect(matchesTradeQuery(equity, "202605")).toBe(true);
  });
  it("matches a trade ID by year prefix", () => {
    expect(matchesTradeQuery(equity, "2026")).toBe(true);
  });
  it("does not match a different month's ID", () => {
    expect(matchesTradeQuery(equity, "202604")).toBe(false);
  });
  it("does not treat a non-digit-prefixed token as a trade ID prefix", () => {
    // Hypothetical: a ticker happens to share characters with a trade ID.
    // Since the prefix branch only fires on digit-leading tokens, a plain
    // ticker query never accidentally matches an ID.
    expect(matchesTradeQuery({ ticker: "X", trade_id: "X02605-013" }, "X")).toBe(true); // ticker match
    expect(matchesTradeQuery({ ticker: "Y", trade_id: "X02605-013" }, "X")).toBe(false); // not a prefix branch
  });
});

describe("matchesTradeQuery — edge cases", () => {
  it("returns false for an empty token", () => {
    expect(matchesTradeQuery(equity, "")).toBe(false);
    expect(matchesTradeQuery(equity, "   ")).toBe(false);
  });
  it("trims whitespace from the token", () => {
    expect(matchesTradeQuery(equity, "  DOCN  ")).toBe(true);
    expect(matchesTradeQuery(equity, " 202605-013 ")).toBe(true);
  });
  it("handles trades with an empty ticker (trade ID still searchable)", () => {
    expect(matchesTradeQuery(noTicker, "202603-001")).toBe(true);
    expect(matchesTradeQuery(noTicker, "DOCN")).toBe(false);
  });
});

describe("matchesAnyTradeQuery", () => {
  it("returns true if any token in the list matches", () => {
    expect(matchesAnyTradeQuery(docnCall, ["AAPL", "DOCN"])).toBe(true);
  });
  it("returns false when no token matches", () => {
    expect(matchesAnyTradeQuery(docnCall, ["AAPL", "MSFT"])).toBe(false);
  });
  it("returns false on an empty token list", () => {
    expect(matchesAnyTradeQuery(docnCall, [])).toBe(false);
  });
  it("mixes ticker and trade-ID tokens cleanly", () => {
    expect(matchesAnyTradeQuery(equity, ["AAPL", "202605-013"])).toBe(true);
    expect(matchesAnyTradeQuery(docnCall, ["NVDA", "202605"])).toBe(true);
  });
});
