import type {
  ArtifactPayload,
  MarketSummary,
  OrderLevel,
  ParsedBook,
  ShockEvent,
} from "./types";

const addedOrder = /Adding order to OrderBook: Order\(id=([^,]+), agent=([^,]+), side=(BUY|SELL), type=([^,]+), qty=(\d+)\/(\d+), price=([^,\)]+)/;
const cancelledOrder = /Order canceled: Order\(id=([^,]+)/;
const executedTrade = /Trade executed: (.+?) bought (\d+) shares from (.+?) at \$([\d.]+)/;
const simulatedTime = /TIME_TICK message: ([\dT:+\-]+)/;
const emittedShock = /emitted (announcement|active) ([\w-]+).*?type=([^,]+), class=([^,]+), severity=([\d.]+)/;

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function getMarketSummaries(payload: ArtifactPayload): Record<string, MarketSummary> {
  const report = asRecord(payload.analysis_reports.ecology_market_summary);
  const markets = asRecord(report?.markets);
  if (markets) {
    return Object.fromEntries(
    Object.entries(markets).flatMap(([instrument, value]) => {
      const market = asRecord(value);
      if (!market || !Array.isArray(market.trades)) return [];
      return [[instrument, market as unknown as MarketSummary]];
    }),
    );
  }
  if (Object.keys(payload.live_markets).length) {
    return Object.fromEntries(
      Object.entries(payload.live_markets).map(([instrument, book]) => {
        const trades = book.trades.map(({ timestamp, price, quantity }) => ({ timestamp, price, quantity }));
        const startPrice = trades[0]?.price ?? 0;
        const endPrice = trades.at(-1)?.price ?? 0;
        return [instrument, {
          trade_count: trades.length,
          volume: trades.reduce((total, trade) => total + trade.quantity, 0),
          start_price: startPrice,
          end_price: endPrice,
          return: startPrice ? (endPrice - startPrice) / startPrice : 0,
          trades,
        } satisfies MarketSummary];
      }),
    );
  }
  return Object.fromEntries(
    Object.entries(payload.order_logs).map(([instrument, log]) => {
      const parsed = parseOrderBook(log);
      const trades = parsed.trades.map(({ timestamp, price, quantity }) => ({ timestamp, price, quantity }));
      const startPrice = trades[0]?.price ?? 0;
      const endPrice = trades.at(-1)?.price ?? 0;
      return [instrument, {
        trade_count: trades.length,
        volume: trades.reduce((total, trade) => total + trade.quantity, 0),
        start_price: startPrice,
        end_price: endPrice,
        return: startPrice ? (endPrice - startPrice) / startPrice : 0,
        trades,
      } satisfies MarketSummary];
    }),
  );
}

export function getInstruments(payload: ArtifactPayload): string[] {
  const summaries = Object.keys(getMarketSummaries(payload));
  if (summaries.length) return summaries;
  if (Object.keys(payload.order_logs).length) return Object.keys(payload.order_logs);
  if (Object.keys(payload.live_markets).length) return Object.keys(payload.live_markets);
  const simulationInfo = asRecord(payload.summary?.simulation_info);
  return Array.isArray(simulationInfo?.instruments)
    ? simulationInfo.instruments.filter((value): value is string => typeof value === "string")
    : [];
}

export function parseOrderBook(log: string): ParsedBook {
  const liveOrders = new Map<string, { side: "BUY" | "SELL"; quantity: number; price: number }>();
  const trades: ParsedBook["trades"] = [];
  let currentTime = "";
  for (const line of log.split("\n")) {
    const timeMatch = line.match(/time=(\d{4}-\d{2}-\d{2}T[\d:.+\-]+)/);
    if (timeMatch) currentTime = timeMatch[1];
    const addMatch = line.match(addedOrder);
    if (addMatch) {
      const [, id, , side, type, quantity, , rawPrice] = addMatch;
      const price = Number(rawPrice);
      if (type === "LIMIT" && Number.isFinite(price)) {
        liveOrders.set(id, { side: side as "BUY" | "SELL", quantity: Number(quantity), price });
      }
      continue;
    }
    const cancelMatch = line.match(cancelledOrder);
    if (cancelMatch) {
      liveOrders.delete(cancelMatch[1]);
      continue;
    }
    const tradeMatch = line.match(executedTrade);
    if (tradeMatch) {
      const [, buyer, rawQuantity, seller, rawPrice] = tradeMatch;
      trades.push({
        timestamp: currentTime,
        buyer,
        seller,
        quantity: Number(rawQuantity),
        price: Number(rawPrice),
      });
    }
  }
  const aggregate = (side: "BUY" | "SELL"): OrderLevel[] => {
    const values = new Map<number, OrderLevel>();
    for (const order of liveOrders.values()) {
      if (order.side !== side) continue;
      const current = values.get(order.price) ?? { price: order.price, quantity: 0, orderCount: 0 };
      current.quantity += order.quantity;
      current.orderCount += 1;
      values.set(order.price, current);
    }
    return [...values.values()].sort((a, b) => side === "BUY" ? b.price - a.price : a.price - b.price);
  };
  return { bids: aggregate("BUY"), asks: aggregate("SELL"), trades };
}

export function parseShockEvents(log: string): ShockEvent[] {
  const events: ShockEvent[] = [];
  let currentTime = "";
  for (const line of log.split("\n")) {
    const timeMatch = line.match(simulatedTime);
    if (timeMatch) currentTime = timeMatch[1];
    const shockMatch = line.match(emittedShock);
    if (!shockMatch) continue;
    const [, phase, id, type, classification, severity] = shockMatch;
    events.push({
      timestamp: currentTime,
      id,
      phase: phase as ShockEvent["phase"],
      type,
      classification,
      severity: Number(severity),
    });
  }
  return events;
}

export function marketReturn(market?: MarketSummary): number {
  return typeof market?.return === "number" ? market.return : 0;
}

export function formatMoney(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: decimals,
    minimumFractionDigits: decimals,
  }).format(value);
}

export function formatCompact(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
}

export function shortTime(value: string): string {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp)
    ? new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(timestamp)
    : "--";
}
