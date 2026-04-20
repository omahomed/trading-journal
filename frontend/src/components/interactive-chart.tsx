"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, type IChartApi, CandlestickSeries, HistogramSeries, ColorType, type Time, createSeriesMarkers, type SeriesMarker } from "lightweight-charts";
import { api, type TradeDetail } from "@/lib/api";

interface Props {
  ticker: string;
  tradeId: string;
  openDate: string;
  closedDate?: string;
  details: TradeDetail[];
  navColor: string;
}

export function InteractiveChart({ ticker, tradeId, openDate, closedDate, details, navColor }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!containerRef.current) return;

    let cancelled = false;

    const build = async () => {
      try {
        // Calculate date range: 20 days before open, 20 days after close (or today)
        const openD = new Date(openDate);
        const startD = new Date(openD);
        startD.setDate(startD.getDate() - 20);
        const start = startD.toISOString().slice(0, 10);

        let end: string | undefined;
        if (closedDate) {
          const closeD = new Date(closedDate);
          closeD.setDate(closeD.getDate() + 20);
          end = closeD.toISOString().slice(0, 10);
        }

        const result = await api.chartOhlcv(ticker, start, end, closedDate ? undefined : "6mo");
        if (cancelled) return;

        if ("error" in result || !result.candles?.length) {
          setError("No chart data available");
          setLoading(false);
          return;
        }

        // Clean up previous chart
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
        }

        // Detect theme
        const isDark = document.documentElement.classList.contains("dark") ||
          getComputedStyle(document.documentElement).getPropertyValue("--bg").trim().startsWith("#1");

        const chart = createChart(containerRef.current!, {
          width: containerRef.current!.clientWidth,
          height: 400,
          layout: {
            background: { type: ColorType.Solid, color: "transparent" },
            textColor: isDark ? "#9ca3af" : "#6b7280",
            fontFamily: "var(--font-jetbrains), monospace",
            fontSize: 11,
          },
          grid: {
            vertLines: { color: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)" },
            horzLines: { color: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)" },
          },
          crosshair: {
            vertLine: { labelBackgroundColor: navColor },
            horzLine: { labelBackgroundColor: navColor },
          },
          rightPriceScale: {
            borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)",
          },
          timeScale: {
            borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)",
            timeVisible: false,
          },
        });
        chartRef.current = chart;

        // Candlestick series (v5 API)
        const candleSeries = chart.addSeries(CandlestickSeries, {
          upColor: "#22c55e",
          downColor: "#ef4444",
          borderDownColor: "#ef4444",
          borderUpColor: "#22c55e",
          wickDownColor: "#ef4444",
          wickUpColor: "#22c55e",
        });

        candleSeries.setData(result.candles.map(c => ({
          time: c.time as Time,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        })));

        // Volume series (v5 API)
        const volumeSeries = chart.addSeries(HistogramSeries, {
          priceFormat: { type: "volume" },
          priceScaleId: "volume",
        });
        chart.priceScale("volume").applyOptions({
          scaleMargins: { top: 0.85, bottom: 0 },
        });
        volumeSeries.setData(result.candles.map(c => ({
          time: c.time as Time,
          value: c.volume,
          color: c.close >= c.open
            ? (isDark ? "rgba(34,197,94,0.15)" : "rgba(34,197,94,0.2)")
            : (isDark ? "rgba(239,68,68,0.15)" : "rgba(239,68,68,0.2)"),
        })));

        // Build a time lookup map for matching trade dates to candle times
        const dateToTime = new Map<string, number>();
        for (const c of result.candles) {
          const d = new Date(c.time * 1000);
          const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
          dateToTime.set(key, c.time);
        }

        // Add buy/sell markers from trade details (v5 API)
        const tradeDetails = details.filter(d => d.trade_id === tradeId);
        const markers: SeriesMarker<Time>[] = [];

        for (const tx of tradeDetails) {
          const txDate = String(tx.date || "").slice(0, 10);
          const candleTime = dateToTime.get(txDate);
          if (!candleTime) continue;

          const isBuy = String(tx.action).toUpperCase() === "BUY";
          const price = parseFloat(String(tx.amount || 0));
          const shares = Math.abs(parseFloat(String(tx.shares || 0)));
          const label = tx.trx_id || (isBuy ? "B" : "S");

          markers.push({
            time: candleTime as Time,
            position: isBuy ? "belowBar" : "aboveBar",
            color: isBuy ? "#22c55e" : "#ef4444",
            shape: isBuy ? "arrowUp" : "arrowDown",
            text: `${label} ${shares}@$${price.toFixed(2)}`,
          });
        }

        // Sort markers by time (required by lightweight-charts)
        markers.sort((a, b) => (a.time as number) - (b.time as number));
        if (markers.length > 0) {
          createSeriesMarkers(candleSeries, markers);
        }

        // Add stop loss line from latest buy transaction
        const buyTxns = tradeDetails.filter(d => String(d.action).toUpperCase() === "BUY" && parseFloat(String(d.stop_loss || 0)) > 0);
        if (buyTxns.length > 0) {
          const latestStop = parseFloat(String(buyTxns[buyTxns.length - 1].stop_loss || 0));
          if (latestStop > 0) {
            candleSeries.createPriceLine({
              price: latestStop,
              color: "#f59e0b",
              lineWidth: 1,
              lineStyle: 2, // dashed
              axisLabelVisible: true,
              title: "Stop",
            });
          }
        }

        // Fit content
        chart.timeScale().fitContent();

        // Resize observer
        const ro = new ResizeObserver(() => {
          if (containerRef.current && chartRef.current) {
            chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
          }
        });
        ro.observe(containerRef.current!);

        setLoading(false);

        return () => {
          ro.disconnect();
        };
      } catch (err: any) {
        if (!cancelled) {
          setError(err.message || "Failed to load chart");
          setLoading(false);
        }
      }
    };

    build();

    return () => {
      cancelled = true;
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [ticker, tradeId, openDate, closedDate, details, navColor]);

  if (error) {
    return (
      <div className="px-5 py-4 text-[12px]" style={{ color: "var(--ink-4)" }}>
        Chart unavailable: {error}
      </div>
    );
  }

  return (
    <div className="px-5 py-4">
      {loading && (
        <div className="flex items-center gap-2 text-[12px] mb-2" style={{ color: "var(--ink-4)" }}>
          <div className="w-3 h-3 border-2 rounded-full animate-spin" style={{ borderColor: "var(--border)", borderTopColor: navColor }} />
          Loading chart...
        </div>
      )}
      <div ref={containerRef} style={{ width: "100%", minHeight: loading ? 0 : 400 }} />
    </div>
  );
}
