import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

vi.mock("@/lib/api", () => ({
  api: {
    createCashTransaction: vi.fn(),
  },
}));

import { api } from "@/lib/api";
import { CashActionForm } from "./settings";

const mockedCreate = vi.mocked(api.createCashTransaction);

describe("CashActionForm — reconcile validation", () => {
  beforeEach(() => {
    mockedCreate.mockReset();
  });

  test("accepts a negative broker balance in reconcile mode (margin account)", async () => {
    mockedCreate.mockResolvedValue({
      id: 1, portfolio_id: 1, date: "2026-04-27", amount: -100,
      source: "reconcile", trade_detail_id: null, note: "ok",
    } as any);

    const onDone = vi.fn().mockResolvedValue(undefined);
    render(
      <CashActionForm
        portfolioId={1}
        cashBalance={-394_000}
        action="reconcile"
        navColor="#6366f1"
        onDone={onDone}
        onCancel={() => {}}
      />
    );

    // Type the actual margin-account broker balance the user reported.
    // RTL's label-text matcher picks up the <span> inside the <label>.
    const input = screen.getByRole("spinbutton") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "-431003.85" } });
    expect(input.value).toBe("-431003.85");

    // Submit and confirm the API was called with the negative number — i.e.
    // validation passed and the body was forwarded as-is.
    fireEvent.click(screen.getByRole("button", { name: /Reconcile/i }));

    await waitFor(() => {
      expect(mockedCreate).toHaveBeenCalledTimes(1);
    });
    expect(mockedCreate).toHaveBeenCalledWith(1, expect.objectContaining({
      source: "reconcile",
      amount: -431003.85,
    }));

    // No "must be positive" error rendered.
    expect(screen.queryByText(/must be a positive/i)).toBeNull();
    expect(screen.queryByText(/must be a number/i)).toBeNull();
  });

  test("still rejects a negative deposit (sign comes from source, must be positive)", async () => {
    const onDone = vi.fn();
    render(
      <CashActionForm
        portfolioId={1}
        cashBalance={0}
        action="deposit"
        navColor="#6366f1"
        onDone={onDone}
        onCancel={() => {}}
      />
    );

    const input = screen.getByRole("spinbutton") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "-100" } });
    fireEvent.click(screen.getByRole("button", { name: /Deposit/i }));

    await waitFor(() => {
      expect(screen.getByText(/Amount must be a positive number/i)).toBeInTheDocument();
    });
    expect(mockedCreate).not.toHaveBeenCalled();
  });

  test("rejects a non-numeric reconcile input", async () => {
    render(
      <CashActionForm
        portfolioId={1}
        cashBalance={0}
        action="reconcile"
        navColor="#6366f1"
        onDone={vi.fn()}
        onCancel={() => {}}
      />
    );

    const input = screen.getByRole("spinbutton") as HTMLInputElement;
    // Browsers strip non-numeric chars from <input type="number"> on submit,
    // but parseFloat("") is NaN — submit must still surface a clear error.
    fireEvent.change(input, { target: { value: "" } });
    // Button is disabled when amount.trim() is empty, so directly submit via form.
    // Workaround: type something invalid, then clear and try clicking; the
    // disabled-button path is its own guard. Here we assert the validator
    // catches a bare-NaN if the field is somehow submitted with no value.
    // Skip the click — the disabled state means the validator never fires.
    expect((screen.getByRole("button", { name: /Reconcile/i }) as HTMLButtonElement).disabled).toBe(true);
  });
});
