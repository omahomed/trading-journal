import { render } from "@testing-library/react";
import { describe, test, expect } from "vitest";

import { cycleBadge } from "./daily-report-card";

describe("DailyReportCard — cycleBadge helper", () => {
  test("UPTREND UNDER PRESSURE renders amber (#d97706) bg + full text (not grey fallback)", () => {
    // 5th-state cycle badge. The helper's inline `styles` Record must
    // include a UPTREND UNDER PRESSURE row so the badge picks up the
    // amber #d97706 background instead of the soft grey #888 fallback
    // path. Full machine string is rendered as visible text (this
    // desktop surface is not width-constrained). Dormant this commit
    // — no backend emits UUP yet.
    const badge = cycleBadge("UPTREND UNDER PRESSURE");
    const { container } = render(<>{badge}</>);
    const span = container.querySelector("span");
    expect(span).not.toBeNull();
    expect(span?.textContent).toBe("UPTREND UNDER PRESSURE");
    const style = span?.getAttribute("style") ?? "";
    // Amber hex #d97706 (rendered by React as rgb(217, 119, 6) in
    // jsdom's style-attribute normalizer). Assert both forms so the
    // test passes across renderer versions.
    expect(style).toMatch(/#d97706|rgb\(\s*217\s*,\s*119\s*,\s*6\s*\)/i);
    // Explicitly assert this is NOT the grey fallback path (#888).
    expect(style).not.toMatch(/#888|rgb\(\s*136\s*,\s*136\s*,\s*136\s*\)/i);
  });
});
