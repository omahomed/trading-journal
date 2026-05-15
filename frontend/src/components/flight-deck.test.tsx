import { render, screen } from "@testing-library/react";
import { describe, test, expect } from "vitest";

import { FlightDeck } from "./flight-deck";

describe("FlightDeck — Phase 5 relocated activity tiles", () => {
  test("renders 4 tiles with the supplied values", () => {
    render(<FlightDeck totalTickets={18} uniqueTickers={7} buys={12} sellsTrims={6} />);
    expect(screen.getByText("Total Tickets")).toBeInTheDocument();
    expect(screen.getByText("Unique Tickers")).toBeInTheDocument();
    expect(screen.getByText("Buys")).toBeInTheDocument();
    expect(screen.getByText("Sells / Trims")).toBeInTheDocument();
    expect(screen.getByText("18")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByText("6")).toBeInTheDocument();
  });

  test("isOveractive flags Total Tickets value in red", () => {
    const { container } = render(
      <FlightDeck totalTickets={20} uniqueTickers={6} buys={14} sellsTrims={6} isOveractive />
    );
    // Find the Total Tickets value cell — it's the sibling of the label.
    const label = screen.getByText("Total Tickets");
    const value = label.nextElementSibling as HTMLElement;
    expect(value.textContent).toBe("20");
    // Inline style includes the red overactivity color.
    expect((value as HTMLElement).style.color).toMatch(/#e5484d|rgb\(229,\s*72,\s*77\)/);
    // Other tiles must NOT have the alert color.
    const buysLabel = screen.getByText("Buys");
    const buysVal = buysLabel.nextElementSibling as HTMLElement;
    expect(buysVal.style.color).not.toMatch(/#e5484d|rgb\(229,\s*72,\s*77\)/);
    // Avoid unused-var warning on container.
    expect(container).toBeTruthy();
  });

  test("zero-state renders zeros without crashing", () => {
    render(<FlightDeck totalTickets={0} uniqueTickers={0} buys={0} sellsTrims={0} />);
    const zeros = screen.getAllByText("0");
    expect(zeros.length).toBe(4);
  });
});
