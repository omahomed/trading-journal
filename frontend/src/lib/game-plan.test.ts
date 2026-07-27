import { describe, it, expect } from "vitest";
import { gamePlanLockDate, isGamePlanEditable } from "./game-plan";

describe("gamePlanLockDate", () => {
  // Reference weekdays used across the tests below (2026-07 calendar):
  //   Mon 2026-07-27, Tue 07-28, Wed 07-29, Thu 07-30, Fri 07-31,
  //   Sat 2026-08-01, Sun 08-02
  it("Monday plan locks Tuesday", () => {
    expect(gamePlanLockDate("2026-07-27")).toBe("2026-07-28");
  });

  it("Thursday plan locks Friday", () => {
    expect(gamePlanLockDate("2026-07-30")).toBe("2026-07-31");
  });

  it("Friday plan locks the following Monday (skips Sat+Sun)", () => {
    expect(gamePlanLockDate("2026-07-31")).toBe("2026-08-03");
  });

  it("Saturday plan locks the following Monday", () => {
    expect(gamePlanLockDate("2026-08-01")).toBe("2026-08-03");
  });

  it("Sunday plan locks the following Monday", () => {
    expect(gamePlanLockDate("2026-08-02")).toBe("2026-08-03");
  });

  it("Handles a year boundary correctly", () => {
    // Thu 2026-12-31 → +1 = Fri 2027-01-01 (crosses years)
    expect(gamePlanLockDate("2026-12-31")).toBe("2027-01-01");
    // Fri 2028-12-29 → +3 = Mon 2029-01-01 (crosses years + skips weekend)
    expect(gamePlanLockDate("2028-12-29")).toBe("2029-01-01");
  });
});

describe("isGamePlanEditable", () => {
  it("Friday plan editable Fri/Sat/Sun; locked Monday", () => {
    // Friday 2026-07-31 plan window.
    expect(isGamePlanEditable("2026-07-31", "2026-07-31")).toBe(true);   // same day
    expect(isGamePlanEditable("2026-07-31", "2026-08-01")).toBe(true);   // Sat
    expect(isGamePlanEditable("2026-07-31", "2026-08-02")).toBe(true);   // Sun
    expect(isGamePlanEditable("2026-07-31", "2026-08-03")).toBe(false);  // Mon → locked
    expect(isGamePlanEditable("2026-07-31", "2026-08-10")).toBe(false);  // way past
  });

  it("Monday plan editable only that Monday", () => {
    expect(isGamePlanEditable("2026-07-27", "2026-07-27")).toBe(true);
    expect(isGamePlanEditable("2026-07-27", "2026-07-28")).toBe(false);  // Tue → locked
    expect(isGamePlanEditable("2026-07-27", "2026-07-26")).toBe(true);   // day before is still editable
    // Note: writing a plan for a future date isn't blocked here (the
    // client gates it via the date picker); this helper only encodes
    // the lockdown boundary.
  });

  it("Invalid date strings return false (fail-closed)", () => {
    expect(isGamePlanEditable("", "2026-07-27")).toBe(false);
    expect(isGamePlanEditable("2026-07-27", "")).toBe(false);
    expect(isGamePlanEditable("not-a-date", "2026-07-27")).toBe(false);
    expect(isGamePlanEditable("2026-07-27", "not-a-date")).toBe(false);
  });
});
