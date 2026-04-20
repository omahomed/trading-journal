"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { createChart, type IChartApi, CandlestickSeries, BarSeries, HistogramSeries, LineSeries, AreaSeries, ColorType, type Time, createSeriesMarkers, type SeriesMarker } from "lightweight-charts";
import { api, type TradeDetail } from "@/lib/api";

type ChartType = "bar" | "candlestick" | "line" | "area";
type Interval = "1d" | "1wk" | "1mo";
type VisibleRange = "5d" | "1m" | "3m" | "1y" | "5y" | "all";

const MA_COLORS = {
  ema8: "#22c55e",   // green
  ema21: "#f59e0b",  // orange
  sma50: "#ef4444",  // red
  sma200: "#a855f7", // purple/magenta
};

function calcEMA(closes: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  const k = 2 / (period + 1);
  let ema: number | null = null;
  for (let i = 0; i < closes.length; i++) {
    if (i < period - 1) { result.push(null); continue; }
    if (ema === null) {
      ema = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
    } else {
      ema = closes[i] * k + ema * (1 - k);
    }
    result.push(ema);
  }
  return result;
}

function calcSMA(closes: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  for (let i = 0; i < closes.length; i++) {
    if (i < period - 1) { result.push(null); continue; }
    const sum = closes.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
    result.push(sum / period);
  }
  return result;
}

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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [chartType, setChartType] = useState<ChartType>("bar");
  const [interval, setInterval] = useState<Interval>("1d");
  const [maVisible, setMaVisible] = useState({ ema8: true, ema21: true, sma50: true, sma200: true });
  const [visibleRange, setVisibleRange] = useState<VisibleRange>("3m");
  const candlesRef = useRef<{ time: number }[]>([]);

  const applyVisibleRange = useCallback((chart: IChartApi, candles: { time: number }[], range: VisibleRange) => {
    if (!candles.length) { chart.timeScale().fitContent(); return; }
    if (range === "all") { chart.timeScale().fitContent(); return; }

    const lastTime = candles[candles.length - 1].time;
    const daysMap: Record<string, number> = { "5d": 5, "1m": 30, "3m": 90, "1y": 365, "5y": 1825 };
    const days = daysMap[range] || 90;
    const fromTime = lastTime - days * 86400;

    chart.timeScale().setVisibleRange({
      from: fromTime as Time,
      to: lastTime as Time,
    });
  }, []);

  const handleRangeChange = useCallback((range: VisibleRange) => {
    setVisibleRange(range);
    if (chartRef.current && candlesRef.current.length) {
      applyVisibleRange(chartRef.current, candlesRef.current, range);
    }
  }, [applyVisibleRange]);

  const buildChart = useCallback(async () => {
    if (!containerRef.current) return;
    setLoading(true);
    setError("");

    try {
      // Calculate date range
      const openD = new Date(openDate);
      const startD = new Date(openD);
      // More history for weekly/monthly, and for MA calculation
      const lookback = interval === "1mo" ? 365 : interval === "1wk" ? 300 : 200;
      startD.setDate(startD.getDate() - lookback);
      const start = startD.toISOString().slice(0, 10);

      let end: string | undefined;
      if (closedDate) {
        const closeD = new Date(closedDate);
        closeD.setDate(closeD.getDate() + 20);
        end = closeD.toISOString().slice(0, 10);
      }

      const result = await api.chartOhlcv(ticker, start, end, closedDate ? undefined : "2y", interval);

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
        height: 420,
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
          rightOffset: 12,
        },
      });
      chartRef.current = chart;

      const candles = result.candles;
      const closes = candles.map(c => c.close);

      // Main price series based on chart type
      let priceSeries: any;
      if (chartType === "bar") {
        priceSeries = chart.addSeries(BarSeries, {
          upColor: "#22c55e",
          downColor: "#ef4444",
        });
        priceSeries.setData(candles.map(c => ({
          time: c.time as Time, open: c.open, high: c.high, low: c.low, close: c.close,
        })));
      } else if (chartType === "candlestick") {
        priceSeries = chart.addSeries(CandlestickSeries, {
          upColor: "#22c55e",
          downColor: "#ef4444",
          borderDownColor: "#ef4444",
          borderUpColor: "#22c55e",
          wickDownColor: "#ef4444",
          wickUpColor: "#22c55e",
        });
        priceSeries.setData(candles.map(c => ({
          time: c.time as Time, open: c.open, high: c.high, low: c.low, close: c.close,
        })));
      } else if (chartType === "line") {
        priceSeries = chart.addSeries(LineSeries, {
          color: navColor,
          lineWidth: 2,
        });
        priceSeries.setData(candles.map(c => ({ time: c.time as Time, value: c.close })));
      } else {
        priceSeries = chart.addSeries(AreaSeries, {
          topColor: navColor + "40",
          bottomColor: navColor + "05",
          lineColor: navColor,
          lineWidth: 2,
        });
        priceSeries.setData(candles.map(c => ({ time: c.time as Time, value: c.close })));
      }

      // Volume
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });
      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      });
      volumeSeries.setData(candles.map(c => ({
        time: c.time as Time,
        value: c.volume,
        color: c.close >= c.open
          ? (isDark ? "rgba(34,197,94,0.12)" : "rgba(34,197,94,0.18)")
          : (isDark ? "rgba(239,68,68,0.12)" : "rgba(239,68,68,0.18)"),
      })));

      // Moving averages
      const addMA = (values: (number | null)[], color: string, title: string) => {
        const s = chart.addSeries(LineSeries, {
          color,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
          title,
        });
        const data = candles
          .map((c, i) => values[i] !== null ? { time: c.time as Time, value: values[i]! } : null)
          .filter(Boolean) as { time: Time; value: number }[];
        s.setData(data);
      };

      if (maVisible.ema8) addMA(calcEMA(closes, 8), MA_COLORS.ema8, "8E");
      if (maVisible.ema21) addMA(calcEMA(closes, 21), MA_COLORS.ema21, "21E");
      if (maVisible.sma50) addMA(calcSMA(closes, 50), MA_COLORS.sma50, "50S");
      if (maVisible.sma200) addMA(calcSMA(closes, 200), MA_COLORS.sma200, "200S");

      // Build time lookup for markers (use UTC to avoid timezone shift)
      const dateToTime = new Map<string, number>();
      for (const c of candles) {
        const d = new Date(c.time * 1000);
        const key = `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-${String(d.getUTCDate()).padStart(2, "0")}`;
        dateToTime.set(key, c.time);
      }

      // Buy/sell markers — just trx_id, all above bar
      const tradeDetails = details.filter(d => d.trade_id === tradeId);
      const markers: SeriesMarker<Time>[] = [];
      for (const tx of tradeDetails) {
        const txDate = String(tx.date || "").slice(0, 10);
        const candleTime = dateToTime.get(txDate);
        if (!candleTime) continue;
        const isBuy = String(tx.action).toUpperCase() === "BUY";
        const label = tx.trx_id || (isBuy ? "B" : "S");
        markers.push({
          time: candleTime as Time,
          position: "aboveBar",
          color: isBuy ? "#22c55e" : "#ef4444",
          shape: "arrowDown",
          text: label,
        });
      }
      markers.sort((a, b) => (a.time as number) - (b.time as number));
      if (markers.length > 0) {
        createSeriesMarkers(priceSeries, markers);
      }

      // Stop loss line
      const buyTxns = tradeDetails.filter(d => String(d.action).toUpperCase() === "BUY" && parseFloat(String(d.stop_loss || 0)) > 0);
      if (buyTxns.length > 0) {
        const latestStop = parseFloat(String(buyTxns[buyTxns.length - 1].stop_loss || 0));
        if (latestStop > 0) {
          priceSeries.createPriceLine({
            price: latestStop,
            color: "#f59e0b",
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: true,
            title: "Stop",
          });
        }
      }

      // Store candles ref for range changes
      candlesRef.current = candles;

      // Set visible range based on selected preset
      applyVisibleRange(chart, candles, visibleRange);

      // Resize observer
      const ro = new ResizeObserver(() => {
        if (containerRef.current && chartRef.current) {
          chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
        }
      });
      ro.observe(containerRef.current!);
      setLoading(false);
    } catch (err: any) {
      setError(err.message || "Failed to load chart");
      setLoading(false);
    }
  }, [ticker, tradeId, openDate, closedDate, details, navColor, chartType, interval, maVisible, visibleRange, applyVisibleRange]);

  useEffect(() => {
    buildChart();
    return () => {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [buildChart]);

  const toggleMA = (key: keyof typeof maVisible) => {
    setMaVisible(prev => ({ ...prev, [key]: !prev[key] }));
  };

  if (error) {
    return (
      <div className="px-5 py-4 text-[12px]" style={{ color: "var(--ink-4)" }}>
        Chart unavailable: {error}
      </div>
    );
  }

  const btnStyle = (active: boolean): React.CSSProperties => ({
    background: active ? "var(--surface)" : "transparent",
    color: active ? "var(--ink)" : "var(--ink-4)",
    boxShadow: active ? "0 1px 2px rgba(0,0,0,0.04)" : "none",
    border: "none",
    cursor: "pointer",
  });

  const maBtn = (key: keyof typeof maVisible, label: string, color: string): React.ReactNode => (
    <button key={key} onClick={() => toggleMA(key)}
            className="px-2 py-1 rounded-md text-[10px] font-semibold transition-all flex items-center gap-1"
            style={{
              background: maVisible[key] ? `${color}18` : "transparent",
              color: maVisible[key] ? color : "var(--ink-4)",
              border: `1px solid ${maVisible[key] ? `${color}40` : "var(--border)"}`,
              cursor: "pointer",
              opacity: maVisible[key] ? 1 : 0.5,
            }}>
      <span className="w-2.5 h-0.5 rounded-full inline-block" style={{ background: color, opacity: maVisible[key] ? 1 : 0.3 }} />
      {label}
    </button>
  );

  return (
    <div className="px-5 py-4">
      {/* Toolbar */}
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        {/* Timeframe */}
        <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          {([["1d", "D"], ["1wk", "W"], ["1mo", "M"]] as [Interval, string][]).map(([val, label]) => (
            <button key={val} onClick={() => setInterval(val)}
                    className="px-2.5 py-1 rounded-md text-[10px] font-semibold transition-all"
                    style={btnStyle(interval === val)}>
              {label}
            </button>
          ))}
        </div>

        {/* Visible range presets */}
        <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          {([["5d", "5d"], ["1m", "1m"], ["3m", "3m"], ["1y", "1y"], ["5y", "5y"], ["all", "All"]] as [VisibleRange, string][]).map(([val, label]) => (
            <button key={val} onClick={() => handleRangeChange(val)}
                    className="px-2.5 py-1 rounded-md text-[10px] font-semibold transition-all"
                    style={btnStyle(visibleRange === val)}>
              {label}
            </button>
          ))}
        </div>

        {/* Chart type */}
        <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          {([["bar", "Bar"], ["candlestick", "Candle"], ["line", "Line"], ["area", "Area"]] as [ChartType, string][]).map(([val, label]) => (
            <button key={val} onClick={() => setChartType(val)}
                    className="px-2.5 py-1 rounded-md text-[10px] font-semibold transition-all"
                    style={btnStyle(chartType === val)}>
              {label}
            </button>
          ))}
        </div>

        {/* MA toggles */}
        <div className="flex gap-1.5 ml-auto">
          {maBtn("ema8", "8 EMA", MA_COLORS.ema8)}
          {maBtn("ema21", "21 EMA", MA_COLORS.ema21)}
          {maBtn("sma50", "50 SMA", MA_COLORS.sma50)}
          {maBtn("sma200", "200 SMA", MA_COLORS.sma200)}
        </div>
      </div>

      {/* Chart */}
      {loading && (
        <div className="flex items-center gap-2 text-[12px] mb-2" style={{ color: "var(--ink-4)" }}>
          <div className="w-3 h-3 border-2 rounded-full animate-spin" style={{ borderColor: "var(--border)", borderTopColor: navColor }} />
          Loading chart...
        </div>
      )}
      <div ref={containerRef} style={{ width: "100%", minHeight: 420 }} />
    </div>
  );
}
