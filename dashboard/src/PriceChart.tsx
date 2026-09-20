import { useEffect, useRef } from "react";
import { ColorType, createChart, CrosshairMode, type IChartApi, type UTCTimestamp } from "lightweight-charts";
import type { MarketTrade, ShockEvent } from "./types";
import { formatPercent } from "./data";

interface PriceChartProps {
  instrument: string;
  trades: MarketTrade[];
  shocks: ShockEvent[];
  positive: boolean;
  compact?: boolean;
}

function asUnixSeconds(value: string): UTCTimestamp | null {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? (Math.floor(timestamp / 1000) as UTCTimestamp) : null;
}

function collapseTrades(trades: MarketTrade[]) {
  const latest = new Map<number, number>();
  for (const trade of trades) {
    const time = asUnixSeconds(trade.timestamp);
    if (time !== null && Number.isFinite(trade.price)) latest.set(time, trade.price);
  }
  return [...latest.entries()]
    .sort(([left], [right]) => left - right)
    .map(([time, value]) => ({ time: time as UTCTimestamp, value }));
}

export function PriceChart({ instrument, trades, shocks, positive, compact = false }: PriceChartProps) {
  const element = useRef<HTMLDivElement | null>(null);
  const chart = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!element.current) return;
    const instance = createChart(element.current, {
      autoSize: true,
      height: compact ? 176 : 232,
      layout: {
        background: { type: ColorType.Solid, color: "#0b1116" },
        textColor: "#7d8da1",
        fontFamily: "IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(151, 171, 189, 0.08)" },
        horzLines: { color: "rgba(151, 171, 189, 0.08)" },
      },
      rightPriceScale: { borderColor: "rgba(151, 171, 189, 0.13)" },
      timeScale: {
        borderColor: "rgba(151, 171, 189, 0.13)",
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: { mode: CrosshairMode.Normal },
      handleScroll: false,
      handleScale: false,
    });
    chart.current = instance;
    return () => {
      if (chart.current === instance) chart.current = null;
      instance.remove();
    };
  }, [compact]);

  useEffect(() => {
    if (!chart.current) return;
    const series = chart.current.addAreaSeries({
      lineColor: positive ? "#25d6a2" : "#ff6678",
      topColor: positive ? "rgba(37, 214, 162, 0.24)" : "rgba(255, 102, 120, 0.20)",
      bottomColor: "rgba(9, 16, 22, 0)",
      lineWidth: 2,
      crosshairMarkerRadius: 3,
      priceLineVisible: true,
      lastValueVisible: true,
    });
    const points = collapseTrades(trades);
    series.setData(points);
    const markers = shocks.flatMap((shock) => {
      const time = asUnixSeconds(shock.timestamp);
      if (time === null || !points.some((point) => point.time === time)) return [];
      return [{
        time,
        position: shock.phase === "active" ? "aboveBar" as const : "belowBar" as const,
        color: shock.classification === "systematic" ? "#f2b84b" : "#8b9cff",
        shape: shock.phase === "active" ? "arrowDown" as const : "circle" as const,
        text: shock.phase === "active" ? shock.type.replaceAll("_", " ") : "scheduled",
      }];
    });
    series.setMarkers(markers);
    chart.current.timeScale().fitContent();
    return () => chart.current?.removeSeries(series);
  }, [trades, shocks, positive]);

  const first = trades[0]?.price;
  const last = trades.at(-1)?.price;
  const move = first && last ? (last - first) / first : null;
  return (
    <div className="chart-frame" aria-label={`${instrument} price chart`}>
      <div className="chart-kicker">
        <span>{trades.length ? `${trades.length} prints` : "Awaiting first print"}</span>
        <strong className={positive ? "positive" : "negative"}>{formatPercent(move)}</strong>
      </div>
      <div className="chart-canvas" ref={element} />
    </div>
  );
}
