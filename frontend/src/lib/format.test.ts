import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { formatCurrency, setFocusModeActive } from "./format";

describe("formatCurrency — defaults (2 decimals, sign before $)", () => {
  it("renders a positive value with comma separators", () => {
    expect(formatCurrency(1234.56)).toBe("$1,234.56");
  });
  it("renders a negative value with leading ASCII minus before $", () => {
    expect(formatCurrency(-1234.56)).toBe("-$1,234.56");
  });
  it("renders zero with two trailing decimals", () => {
    expect(formatCurrency(0)).toBe("$0.00");
  });
  it("renders large values with grouping at every thousand", () => {
    expect(formatCurrency(1234567.89)).toBe("$1,234,567.89");
  });
  it("preserves a sub-cent value padded to 2 decimals", () => {
    expect(formatCurrency(0.01)).toBe("$0.01");
  });
});

describe("formatCurrency — decimals option", () => {
  it("rounds half-up at decimals=0 (1234.5 → 1235)", () => {
    expect(formatCurrency(1234.5, { decimals: 0 })).toBe("$1,235");
  });
  it("rounds down at decimals=0 (1234.4 → 1234)", () => {
    expect(formatCurrency(1234.4, { decimals: 0 })).toBe("$1,234");
  });
  it("renders 4 decimals when requested", () => {
    expect(formatCurrency(1.2345, { decimals: 4 })).toBe("$1.2345");
  });
});

describe("formatCurrency — showSign", () => {
  it("prepends + to strictly positive values", () => {
    expect(formatCurrency(1234, { showSign: true })).toBe("+$1,234.00");
  });
  it("retains the negative sign for negative values", () => {
    expect(formatCurrency(-1234, { showSign: true })).toBe("-$1,234.00");
  });
  it("does NOT prepend + for zero (zero is unsigned)", () => {
    expect(formatCurrency(0, { showSign: true })).toBe("$0.00");
  });
});

describe("formatCurrency — signGlyph", () => {
  it("uses Unicode minus (U+2212) when signGlyph: unicode", () => {
    expect(formatCurrency(-1234, { signGlyph: "unicode" })).toBe("−$1,234.00");
  });
  it("combines unicode glyph with showSign for positives unchanged", () => {
    expect(formatCurrency(1234, { signGlyph: "unicode", showSign: true })).toBe("+$1,234.00");
  });
});

describe("formatCurrency — compact mode", () => {
  it("renders sub-1000 values without a suffix and no decimals", () => {
    expect(formatCurrency(999, { compact: true })).toBe("$999");
  });
  it("scales 1500 to 1.5k", () => {
    expect(formatCurrency(1500, { compact: true })).toBe("$1.5k");
  });
  it("scales 1.5M to $1.5M", () => {
    expect(formatCurrency(1_500_000, { compact: true })).toBe("$1.5M");
  });
  it("scales 1.5B to $1.5B", () => {
    expect(formatCurrency(1_500_000_000, { compact: true })).toBe("$1.5B");
  });
  it("places the negative sign before $ in compact mode", () => {
    expect(formatCurrency(-1500, { compact: true })).toBe("-$1.5k");
  });
  it("renders zero as $0 (no suffix, no decimals) in compact mode", () => {
    expect(formatCurrency(0, { compact: true })).toBe("$0");
  });
});

describe("formatCurrency — zeroAsDash", () => {
  it("renders zero as the null display when zeroAsDash is true", () => {
    expect(formatCurrency(0, { zeroAsDash: true })).toBe("—");
  });
  it("leaves non-zero values untouched when zeroAsDash is true", () => {
    expect(formatCurrency(0.01, { zeroAsDash: true })).toBe("$0.01");
  });
});

describe("formatCurrency — null / undefined / NaN", () => {
  it("renders null as the default em dash", () => {
    expect(formatCurrency(null)).toBe("—");
  });
  it("renders undefined as the default em dash", () => {
    expect(formatCurrency(undefined)).toBe("—");
  });
  it("renders NaN as the default em dash (does not throw)", () => {
    expect(formatCurrency(Number.NaN)).toBe("—");
  });
  it("respects a custom nullDisplay", () => {
    expect(formatCurrency(null, { nullDisplay: "n/a" })).toBe("n/a");
  });
});

describe("formatCurrency — combinations", () => {
  it("combines showSign + compact for positives", () => {
    expect(formatCurrency(1500, { showSign: true, compact: true })).toBe("+$1.5k");
  });
  it("combines showSign + compact for negatives (sign still before $)", () => {
    expect(formatCurrency(-1500, { showSign: true, compact: true })).toBe("-$1.5k");
  });
  it("respects an explicit decimals: 0 in compact mode (1500 → $2k)", () => {
    expect(formatCurrency(1500, { compact: true, decimals: 0 })).toBe("$2k");
  });
  it("respects an explicit decimals: 2 in compact mode (1500 → $1.50k)", () => {
    expect(formatCurrency(1500, { compact: true, decimals: 2 })).toBe("$1.50k");
  });
});

describe("formatCurrency — focusMode", () => {
  beforeEach(() => setFocusModeActive(false));
  afterEach(() => setFocusModeActive(false));

  it("masks positive values to '$•••'", () => {
    setFocusModeActive(true);
    expect(formatCurrency(1234.56)).toBe("$•••");
  });
  it("preserves negative sign before masked digits", () => {
    setFocusModeActive(true);
    expect(formatCurrency(-1234.56)).toBe("-$•••");
  });
  it("preserves +sign on showSign:true", () => {
    setFocusModeActive(true);
    expect(formatCurrency(1234, { showSign: true })).toBe("+$•••");
  });
  it("preserves unicode minus glyph", () => {
    setFocusModeActive(true);
    expect(formatCurrency(-1234, { signGlyph: "unicode" })).toBe("−$•••");
  });
  it("drops compact suffix when masking", () => {
    setFocusModeActive(true);
    expect(formatCurrency(1500, { compact: true })).toBe("$•••");
  });
  it("leaves zero unmasked (info-free)", () => {
    setFocusModeActive(true);
    expect(formatCurrency(0)).toBe("$0.00");
  });
  it("leaves null unmasked (info-free)", () => {
    setFocusModeActive(true);
    expect(formatCurrency(null)).toBe("—");
  });
  it("respects zeroAsDash even with focus on", () => {
    setFocusModeActive(true);
    expect(formatCurrency(0, { zeroAsDash: true })).toBe("—");
  });
});
