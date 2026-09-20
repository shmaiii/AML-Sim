import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Bot,
  CircleCheck,
  CirclePlay,
  CircleX,
  Clock3,
  Database,
  FlaskConical,
  Layers3,
  LineChart,
  LoaderCircle,
  Network,
  Pencil,
  Plus,
  Radio,
  RefreshCw,
  Settings2,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { getArtifacts, getCatalog, launchRun, subscribeToRun } from "./api";
import {
  formatCompact,
  formatMoney,
  formatPercent,
  getInstruments,
  getMarketSummaries,
  marketReturn,
  parseOrderBook,
  parseShockEvents,
  shortTime,
} from "./data";
import { PriceChart } from "./PriceChart";
import type {
  AgentAddition,
  AgentReport,
  ArtifactPayload,
  Catalog,
  ConfiguredAgent,
  MarketSummary,
  OrderLevel,
  ParsedBook,
  ShockEvent,
} from "./types";

const EMPTY_CATALOG: Catalog = { scenarios: [], scenario_details: [], runs: [], agent_types: [], rabbitmq_reachable: false };

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function metric(report: AgentReport | undefined, name: string): number | null {
  return readNumber(report?.metrics?.[name]);
}

function displayScenario(value: string): string {
  return value.replace(/^scenarios\//, "").replace(/\.yaml$/, "").replaceAll("_", " ");
}

function makeRunId(): string {
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "").slice(2);
  return `dashboard_${stamp}`;
}

function roleLabel(type: string): string {
  return type.replace(/^AML_/, "").replaceAll("_", " ");
}

function scenarioFromPayload(payload: ArtifactPayload): string | null {
  const metadata = payload.metadata?.scenario;
  if (!metadata || typeof metadata !== "object") return null;
  const sourcePath = (metadata as Record<string, unknown>).source_path;
  if (typeof sourcePath !== "string") return null;
  const marker = "/mai-AML-Sim/";
  const index = sourcePath.lastIndexOf(marker);
  return index >= 0 ? sourcePath.slice(index + marker.length) : null;
}

function configForAgent(agentId: string, configured: ConfiguredAgent[]): ConfiguredAgent | undefined {
  const direct = configured.find((agent) => agent.id === agentId);
  if (direct) return direct;
  return configured.find((agent) => agentId === agent.id || agentId.startsWith(`${agent.id}_`));
}

function statusLabel(status: string): string {
  return status === "completed" ? "Complete" : status === "failed" ? "Run error" : status === "running" ? "Live" : status === "starting" ? "Starting" : "Ready";
}

function statusIcon(status: string) {
  if (status === "completed") return <CircleCheck size={14} />;
  if (status === "failed") return <CircleX size={14} />;
  if (status === "running") return <Radio size={14} />;
  if (status === "starting") return <LoaderCircle className="spin" size={14} />;
  return <Activity size={14} />;
}

export function App() {
  const urlRunId = new URLSearchParams(window.location.search).get("run")?.trim();
  const [catalog, setCatalog] = useState<Catalog>(EMPTY_CATALOG);
  const [runId, setRunId] = useState(urlRunId || "local_multi_agent");
  const [scenario, setScenario] = useState("scenarios/research/financial_ecology_stock_future_bond_c2_connected.yaml");
  const [artifacts, setArtifacts] = useState<ArtifactPayload | null>(null);
  const [selectedInstrument, setSelectedInstrument] = useState("");
  const [composerOpen, setComposerOpen] = useState(false);
  const [agentAdditions, setAgentAdditions] = useState<AgentAddition[]>([]);
  const [isLaunching, setIsLaunching] = useState(false);
  const [launchingRunId, setLaunchingRunId] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [streamStatus, setStreamStatus] = useState<"connected" | "reconnecting">("reconnecting");

  const refreshCatalog = useCallback(async () => {
    try {
      const next = await getCatalog();
      setCatalog(next);
      if (!catalog.scenarios.length && next.scenarios.length && !next.scenarios.includes(scenario)) {
        setScenario(next.scenarios.at(-1) ?? next.scenarios[0]);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not load the dashboard catalog.");
    }
  }, [catalog.scenarios.length, scenario]);

  useEffect(() => {
    void refreshCatalog();
  }, [refreshCatalog]);

  useEffect(() => {
    if (!runId) return;
    let active = true;
    let closeStream: () => void = () => {};
    const update = (payload: ArtifactPayload) => {
      if (!active || payload.run_id !== runId) return;
      setArtifacts(payload);
      if (payload.job.status === "failed" || (payload.exists && payload.configured_agents.length)) {
        setLaunchingRunId(null);
      }
      const sourceScenario = scenarioFromPayload(payload);
      if (sourceScenario) setScenario(sourceScenario);
      setStreamStatus("connected");
      const instruments = getInstruments(payload);
      setSelectedInstrument((current) => current || instruments[0] || "");
      if (payload.job.status === "completed" || payload.job.status === "failed") {
        closeStream();
      }
    };
    closeStream = subscribeToRun(runId, update, () => active && setStreamStatus("reconnecting"));
    void getArtifacts(runId).then(update).catch(() => active && setStreamStatus("reconnecting"));
    return () => {
      active = false;
      closeStream();
    };
  }, [runId]);

  useEffect(() => {
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set("run", runId);
    window.history.replaceState(null, "", nextUrl);
  }, [runId]);

  const marketSummaries = useMemo(() => artifacts ? getMarketSummaries(artifacts) : {}, [artifacts]);
  const instruments = useMemo(() => artifacts ? getInstruments(artifacts) : [], [artifacts]);
  const selectedMarket = marketSummaries[selectedInstrument];
  const selectedBook = useMemo<ParsedBook>(
    () => artifacts?.live_markets[selectedInstrument] ?? parseOrderBook(artifacts?.order_logs[selectedInstrument] ?? ""),
    [artifacts?.live_markets, artifacts?.order_logs, selectedInstrument],
  );
  const shocks = useMemo(
    () => artifacts?.shock_events.length ? artifacts.shock_events : parseShockEvents(artifacts?.shock_log ?? ""),
    [artifacts?.shock_events, artifacts?.shock_log],
  );
  const currentStatus = launchingRunId === runId ? "starting" : artifacts?.job.status ?? "idle";
  const allTradeCount = Object.values(marketSummaries).reduce((total, market) => total + (market.trade_count || 0), 0);
  const allVolume = Object.values(marketSummaries).reduce((total, market) => total + (market.volume || 0), 0);
  const activeAgentCount = artifacts?.configured_agents.reduce((total, agent) => total + agent.count, 0) ?? 0;
  const slowLoops = artifacts?.slow_loop_status;
  const currentPrice = selectedMarket?.end_price ?? selectedBook.trades.at(-1)?.price;
  const bestBid = selectedBook.bids[0]?.price;
  const bestAsk = selectedBook.asks[0]?.price;
  const spread = bestBid !== undefined && bestAsk !== undefined ? bestAsk - bestBid : null;

  const reloadCurrent = useCallback(async () => {
    try {
      const next = await getArtifacts(runId);
      setArtifacts(next);
      setNotice("");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not refresh run data.");
    }
  }, [runId]);

  async function startRun() {
    setIsLaunching(true);
    setNotice("");
    try {
      const requestedRunId = runId && !artifacts?.exists ? runId : makeRunId();
      const job = await launchRun({ scenario, runId: requestedRunId, agentAdditions });
      const launchedRunId = job.run_id || requestedRunId;
      setLaunchingRunId(launchedRunId);
      setArtifacts(null);
      setSelectedInstrument("");
      setRunId(launchedRunId);
      setComposerOpen(false);
      setAgentAdditions([]);
      setNotice(`Simulation started: ${job.run_id || requestedRunId}`);
      void refreshCatalog();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The simulation could not be started.");
    } finally {
      setIsLaunching(false);
    }
  }

  function chooseRun(nextRunId: string) {
    setRunId(nextRunId);
    setSelectedInstrument("");
    setNotice("");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><Activity size={18} strokeWidth={2.5} /></div>
          <div>
            <div className="brand-name">AML Market Lab</div>
            <div className="brand-subtitle">Adaptive multi-agent financial simulation</div>
          </div>
        </div>

        <div className="topbar-controls">
          <div className={`connection ${currentStatus === "completed" ? "archived" : streamStatus}`} title={currentStatus === "completed" ? "Completed run archive" : streamStatus === "connected" ? "Live stream connected" : "Reconnecting to live stream"}>
            <span className="status-dot" />
            {currentStatus === "completed" ? "Archived" : streamStatus === "connected" ? "Streaming" : "Reconnecting"}
          </div>
          <label className="compact-field scenario-field">
            <span>Scenario</span>
            <select value={scenario} onChange={(event) => setScenario(event.target.value)}>
              {catalog.scenarios.map((entry) => <option value={entry} key={entry}>{displayScenario(entry)}</option>)}
            </select>
          </label>
          <label className="compact-field run-field">
            <span>Run</span>
            <input value={runId} onChange={(event) => {
              setRunId(event.target.value.replace(/[^a-zA-Z0-9_-]/g, ""));
              setArtifacts(null);
              setSelectedInstrument("");
            }} aria-label="Run identifier" />
          </label>
          <button className="icon-button" onClick={() => setComposerOpen(true)} title="Configure participants" aria-label="Configure participants">
            <SlidersHorizontal size={18} />
          </button>
          <button className="primary-button" disabled={isLaunching || currentStatus === "running" || currentStatus === "starting"} onClick={() => void startRun()}>
            {isLaunching ? <LoaderCircle className="spin" size={17} /> : <CirclePlay size={17} />}
            {currentStatus === "running" ? "Simulation running" : currentStatus === "starting" ? "Starting simulation" : "Run simulation"}
          </button>
        </div>
      </header>

      {!catalog.rabbitmq_reachable && (
        <div className="system-banner"><AlertTriangle size={15} /> RabbitMQ is not reachable on localhost:5672. A new simulation will not start until it is running.</div>
      )}
      {!!slowLoops?.failed && (
        <div className="error-banner"><AlertTriangle size={15} />
          <span>{slowLoops.failed} slow-loop {slowLoops.failed === 1 ? "failure" : "failures"}. The fast loops are continuing with their last validated strategies.</span>
          <strong>{slowLoops.recent_failures.at(-1)?.agent_id}: {slowLoops.recent_failures.at(-1)?.reason}</strong>
        </div>
      )}
      {notice && <div className="notice-banner"><span>{notice}</span><button onClick={() => setNotice("")} title="Dismiss" aria-label="Dismiss"><X size={15} /></button></div>}

      <section className="market-strip">
        <Stat label="Run" value={statusLabel(currentStatus)} state={currentStatus} icon={statusIcon(currentStatus)} />
        <Stat label="Session" value={sessionLabel(artifacts)} icon={<Clock3 size={15} />} />
        <Stat label="Trades" value={formatCompact(allTradeCount)} icon={<Activity size={15} />} />
        <Stat label="Volume" value={formatCompact(allVolume)} icon={<BarChart3 size={15} />} />
        <Stat label="Participants" value={String(activeAgentCount || "--")} icon={<Bot size={15} />} />
        <Stat label="Shock events" value={String(shocks.length || "--")} icon={<ShieldAlert size={15} />} />
        <Stat label="Links" value={linkCount(artifacts)} icon={<Network size={15} />} />
        <Stat label="Strategy loops" value={slowLoopLabel(slowLoops)} state={slowLoops?.failed ? "failed" : undefined} icon={<Sparkles size={15} />} />
      </section>

      <main className="workspace">
        <aside className="sidebar-stack">
          <section className="panel market-panel">
            <PanelTitle icon={<Layers3 size={16} />} title="Markets" detail={`${instruments.length || 0} venues`} />
            <div className="asset-list">
              {instruments.length ? instruments.map((instrument) => (
                <MarketRow
                  key={instrument}
                  instrument={instrument}
                  market={marketSummaries[instrument]}
                  selected={instrument === selectedInstrument}
                  onSelect={() => setSelectedInstrument(instrument)}
                />
              )) : <EmptyState icon={<LineChart size={20} />} text="No market path has been written yet." />}
            </div>
          </section>

          <section className="panel participant-panel">
            <PanelTitle icon={<Bot size={16} />} title="Participants" detail={`${activeAgentCount || 0} configured`} />
            <ParticipantList configured={artifacts?.configured_agents ?? []} reports={artifacts?.agent_reports ?? {}} />
          </section>
        </aside>

        <section className="center-stack">
          <section className="chart-grid">
            {instruments.length ? instruments.map((instrument) => {
              const market = marketSummaries[instrument];
              return (
                <article
                  key={instrument}
                  className={`panel chart-panel ${instrument === selectedInstrument ? "selected" : ""}`}
                  onClick={() => setSelectedInstrument(instrument)}
                  onKeyDown={(event) => event.key === "Enter" && setSelectedInstrument(instrument)}
                  role="button"
                  tabIndex={0}
                >
                  <div className="chart-header">
                    <div>
                      <div className="eyebrow">{assetClass(instrument)}</div>
                      <h2>{instrument}</h2>
                    </div>
                    <div className="chart-price">
                      <strong>{formatMoney(market?.end_price ?? null)}</strong>
                      <span className={marketReturn(market) >= 0 ? "positive" : "negative"}>{formatPercent(market?.return)}</span>
                    </div>
                  </div>
                  <PriceChart instrument={instrument} trades={market?.trades ?? []} shocks={shocks} positive={marketReturn(market) >= 0} compact />
                </article>
              );
            }) : <section className="panel empty-canvas"><EmptyState icon={currentStatus === "starting" || currentStatus === "running" ? <LoaderCircle className="spin" size={24} /> : <Database size={24} />} text={currentStatus === "starting" || currentStatus === "running" ? "Starting the exchanges and waiting for the first market snapshot." : "Choose a completed run or start a new simulation."} /></section>}
          </section>

          <section className="analysis-grid">
            <ShockTimeline shocks={shocks} />
            <EcologyPanel payload={artifacts} />
          </section>

          <section className="panel performance-panel">
            <PanelTitle icon={<BarChart3 size={16} />} title="Participant performance" detail="mark-to-market" />
            <PerformanceTable configured={artifacts?.configured_agents ?? []} reports={artifacts?.agent_reports ?? {}} />
          </section>
        </section>

        <aside className="right-stack">
          <section className="panel orderbook-panel">
            <PanelTitle icon={<LineChart size={16} />} title={`${selectedInstrument || "Market"} order book`} detail={spread !== null ? `spread ${spread.toFixed(2)}` : "awaiting quotes"} />
            <div className="book-summary">
              <div><span>Best bid</span><strong className="positive">{formatMoney(bestBid ?? null)}</strong></div>
              <div><span>Mark</span><strong>{formatMoney(currentPrice ?? null)}</strong></div>
              <div><span>Best ask</span><strong className="negative">{formatMoney(bestAsk ?? null)}</strong></div>
            </div>
            <OrderBookSide title="Asks" levels={selectedBook.asks.slice(0, 8)} side="ask" />
            <div className="book-mid"><span>Depth</span><strong>{selectedBook.bids.length + selectedBook.asks.length} levels</strong><span>Queue</span></div>
            <OrderBookSide title="Bids" levels={selectedBook.bids.slice(0, 8)} side="bid" />
          </section>

          <section className="panel tape-panel">
            <PanelTitle icon={<Activity size={16} />} title="Trade tape" detail={`${selectedBook.trades.length} prints`} />
            <div className="tape-head"><span>Time</span><span>Price</span><span>Size</span></div>
            <div className="tape-list">
              {selectedBook.trades.slice(-10).reverse().map((trade, index) => (
                <div className="tape-row" key={`${trade.timestamp}-${index}`}>
                  <span>{shortTime(trade.timestamp)}</span>
                  <strong className={trade.buyer.includes("maker") ? "positive" : "negative"}>{formatMoney(trade.price)}</strong>
                  <span>{trade.quantity}</span>
                </div>
              ))}
              {!selectedBook.trades.length && <EmptyState icon={<Activity size={18} />} text="No fills yet." compact />}
            </div>
          </section>

          <section className="panel run-panel">
            <PanelTitle icon={<FlaskConical size={16} />} title="Run archive" detail={artifacts?.exists ? "archived" : "pending"} />
            <label className="archive-select">
              <span>Saved sessions</span>
              <select value={runId} onChange={(event) => chooseRun(event.target.value)}>
                <option value={runId}>{runId}</option>
                {catalog.runs.filter((run) => run.run_id !== runId).map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id}</option>)}
              </select>
            </label>
            <div className="run-actions">
              <button className="secondary-button" onClick={() => void reloadCurrent()}><RefreshCw size={15} /> Refresh</button>
              <span>{artifacts?.available_reports.length ?? 0} reports</span>
            </div>
          </section>
        </aside>
      </main>

      <section className="console-panel">
        <div className="console-heading"><span><Activity size={14} /> Simulation log</span><span>{currentStatus === "running" ? "updating" : "latest output"}</span></div>
        <pre>{artifacts?.job.log_tail?.slice(-8).join("\n") || "No process output for this run."}</pre>
      </section>

      {composerOpen && (
        <ExperimentComposer
          instrumentOptions={catalog.scenario_details.find((entry) => entry.path === scenario)?.instruments ?? inferScenarioInstruments(scenario)}
          agentTypes={catalog.agent_types}
          additions={agentAdditions}
          onChange={setAgentAdditions}
          onClose={() => setComposerOpen(false)}
          onRun={() => void startRun()}
          isLaunching={isLaunching}
        />
      )}
    </div>
  );
}

function Stat({ label, value, icon, state }: { label: string; value: string; icon: ReactNode; state?: string }) {
  return <div className="strip-stat"><span>{icon}{label}</span><strong className={state ? `status-${state}` : ""}>{value}</strong></div>;
}

function PanelTitle({ icon, title, detail }: { icon: ReactNode; title: string; detail: string }) {
  return <div className="panel-title"><h2>{icon}{title}</h2><span>{detail}</span></div>;
}

function EmptyState({ icon, text, compact = false }: { icon: ReactNode; text: string; compact?: boolean }) {
  return <div className={`empty-state ${compact ? "compact" : ""}`}>{icon}<span>{text}</span></div>;
}

function MarketRow({ instrument, market, selected, onSelect }: { instrument: string; market?: MarketSummary; selected: boolean; onSelect: () => void }) {
  const positive = marketReturn(market) >= 0;
  return <button className={`asset-row ${selected ? "selected" : ""}`} onClick={onSelect}>
    <div><strong>{instrument}</strong><span>{assetClass(instrument)}</span></div>
    <div className="asset-price"><strong>{formatMoney(market?.end_price ?? null)}</strong><span className={positive ? "positive" : "negative"}>{formatPercent(market?.return)}</span></div>
  </button>;
}

function ParticipantList({ configured, reports }: { configured: ConfiguredAgent[]; reports: Record<string, AgentReport> }) {
  const ids = Object.keys(reports).length ? Object.keys(reports) : configured.flatMap((agent) => agent.count > 1 ? Array.from({ length: agent.count }, (_, index) => `${agent.id}_${index + 1}`) : [agent.id]);
  return <div className="participant-list">
    {ids.slice(0, 10).map((id) => {
      const config = configForAgent(id, configured);
      const report = reports[id];
      const pnl = metric(report, "Profit per Trade");
      return <div className="participant-row" key={id}>
        <div className="participant-name"><strong>{id}</strong><span>{roleLabel(config?.type ?? "Agent")}</span></div>
        <div className="participant-meta"><span>{config?.slow_strategist === "openai" ? "LLM" : "Frozen"}</span><strong className={(pnl ?? 0) >= 0 ? "positive" : "negative"}>{pnl === null ? "--" : formatMoney(pnl)}</strong></div>
      </div>;
    })}
    {!ids.length && <EmptyState icon={<Bot size={19} />} text="Participant reports will appear during the run." compact />}
  </div>;
}

function OrderBookSide({ title, levels, side }: { title: string; levels: OrderLevel[]; side: "bid" | "ask" }) {
  const maxDepth = Math.max(...levels.map((level) => level.quantity), 1);
  return <div className={`book-side ${side}`}>
    <div className="book-label"><span>{title} / price</span><span>Size</span><span>Orders</span></div>
    {levels.length ? levels.map((level) => <div className="book-row" key={`${side}-${level.price}`}>
      <div className="book-fill" style={{ width: `${Math.max(8, (level.quantity / maxDepth) * 100)}%` }} />
      <strong>{formatMoney(level.price)}</strong><span>{level.quantity}</span><em>{level.orderCount}</em>
    </div>) : <div className="book-empty">No resting {title.toLowerCase()}</div>}
  </div>;
}

function ShockTimeline({ shocks }: { shocks: ShockEvent[] }) {
  return <section className="panel shock-panel">
    <PanelTitle icon={<ShieldAlert size={16} />} title="Shock timeline" detail={`${shocks.length} observed`} />
    <div className="shock-list">
      {shocks.map((shock, index) => <div className="shock-row" key={`${shock.id}-${shock.phase}-${index}`}>
        <span className={`shock-kind ${shock.classification === "systematic" ? "macro" : "micro"}`}>{shock.classification === "systematic" ? "Macro" : "Micro"}</span>
        <div><strong>{shock.id.replaceAll("_", " ")}</strong><span>{shock.phase} at {shortTime(shock.timestamp)}</span></div>
        <span className="shock-severity">{shock.severity ? `${Math.round(shock.severity * 100)}%` : "--"}</span>
      </div>)}
      {!shocks.length && <EmptyState icon={<ShieldAlert size={19} />} text="Shock events will be indexed here as they are emitted." compact />}
    </div>
  </section>;
}

function EcologyPanel({ payload }: { payload: ArtifactPayload | null }) {
  const manifest = payload?.analysis_reports.ecology_manifest as Record<string, unknown> | undefined;
  const ledger = payload?.analysis_reports.ecology_channel_ledger as Record<string, unknown> | undefined;
  const counts = ledger?.channel_event_counts && typeof ledger.channel_event_counts === "object" ? ledger.channel_event_counts as Record<string, unknown> : {};
  const relationships = Array.isArray(manifest?.relationships) ? manifest.relationships as Array<Record<string, unknown>> : [];
  return <section className="panel ecology-panel">
    <PanelTitle icon={<Network size={16} />} title="Market ecology" detail={`${relationships.length} configured links`} />
    <div className="relationship-list">
      {relationships.map((relationship) => <div className="relationship-row" key={String(relationship.id)}>
        <div className="relationship-route"><span>{String(relationship.source)}</span><ArrowDownRight size={14} /><span>{String(relationship.target)}</span></div>
        <span>{Array.isArray(relationship.channels) ? relationship.channels.join(" / ") : "--"}</span>
      </div>)}
      {!relationships.length && <EmptyState icon={<Network size={19} />} text="Cross-market links are reported for ecology scenarios." compact />}
    </div>
    <div className="channel-counts">
      {Object.entries(counts).map(([channel, value]) => <div key={channel}><span>{channel.replaceAll("_", " ")}</span><strong>{String(value)}</strong></div>)}
      {!Object.keys(counts).length && <div><span>Channel events</span><strong>--</strong></div>}
    </div>
  </section>;
}

function PerformanceTable({ configured, reports }: { configured: ConfiguredAgent[]; reports: Record<string, AgentReport> }) {
  const rows = Object.entries(reports).map(([id, report]) => ({
    id,
    role: configForAgent(id, configured)?.type ?? "Agent",
    value: metric(report, "Last Portfolio Value") ?? report.last_portfolio_value ?? null,
    pnl: totalPnl(report),
    sharpe: metric(report, "Annualized Sharpe Ratio"),
    exposure: metric(report, "Gross Exposure"),
    fills: report.executed_order_count ?? 0,
  })).sort((left, right) => (right.pnl ?? -Infinity) - (left.pnl ?? -Infinity));
  return <div className="performance-table"><div className="performance-head"><span>Participant</span><span>Equity</span><span>Total PnL</span><span>Sharpe</span><span>Exposure</span><span>Fills</span></div>
    {rows.map((row) => <div className="performance-row" key={row.id}>
      <div><strong>{row.id}</strong><span>{roleLabel(row.role)}</span></div><span>{formatMoney(row.value)}</span><strong className={(row.pnl ?? 0) >= 0 ? "positive" : "negative"}>{formatMoney(row.pnl)}</strong><span>{row.sharpe?.toFixed(2) ?? "--"}</span><span>{formatMoney(row.exposure)}</span><span>{row.fills}</span>
    </div>)}
    {!rows.length && <EmptyState icon={<BarChart3 size={19} />} text="Performance metrics are written as participants trade." compact />}
  </div>;
}

function ExperimentComposer({ instrumentOptions, agentTypes, additions, onChange, onClose, onRun, isLaunching }: {
  instrumentOptions: string[];
  agentTypes: Catalog["agent_types"];
  additions: AgentAddition[];
  onChange: (next: AgentAddition[]) => void;
  onClose: () => void;
  onRun: () => void;
  isLaunching: boolean;
}) {
  const firstType = agentTypes[0];
  const makeDraft = (type = firstType?.type ?? "AML_Market_Maker"): AgentAddition => {
    const definition = agentTypes.find((item) => item.type === type) ?? firstType;
    return {
      id: `custom_${type.replace(/^AML_/, "").replaceAll("_", "").toLowerCase()}`,
      type,
      instrument: instrumentOptions[0] ?? "AAPL",
      count: 1,
      slow_strategist: "frozen",
      parameters: Object.fromEntries((definition?.parameters ?? []).map((parameter) => [parameter.key, parameter.default])),
    };
  };
  const [draft, setDraft] = useState<AgentAddition>(() => makeDraft());
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const role = agentTypes.find((item) => item.type === draft.type) ?? firstType;
  const cleanId = /^[a-z][a-z0-9_]{2,47}$/.test(draft.id);
  const duplicateId = additions.some((addition, index) => addition.id === draft.id && index !== editingIndex);
  const canSave = cleanId && !duplicateId && !!draft.instrument && !!role && (editingIndex !== null || additions.length < 12);

  function changeRole(type: string) {
    const previousDefaultId = makeDraft(draft.type).id;
    const next = makeDraft(type);
    setDraft({ ...next, id: draft.id === previousDefaultId ? next.id : draft.id, instrument: draft.instrument, count: draft.count, slow_strategist: draft.slow_strategist });
  }

  function saveParticipant() {
    if (!canSave) return;
    if (editingIndex === null) onChange([...additions, draft]);
    else onChange(additions.map((addition, index) => index === editingIndex ? draft : addition));
    setEditingIndex(null);
    setDraft(makeDraft(draft.type));
  }

  function editParticipant(index: number) {
    setEditingIndex(index);
    setDraft({ ...additions[index], parameters: { ...additions[index].parameters } });
  }

  function removeParticipant(index: number) {
    onChange(additions.filter((_, itemIndex) => itemIndex !== index));
    if (editingIndex === index) {
      setEditingIndex(null);
      setDraft(makeDraft());
    }
  }

  return <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="composer-modal agent-studio" role="dialog" aria-modal="true" aria-label="Agent Studio" onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span className="eyebrow">Scenario composition</span><h2>Agent Studio</h2></div><button className="icon-button" onClick={onClose} title="Close" aria-label="Close"><X size={18} /></button></header>
      <div className="studio-layout">
        <div className="studio-editor">
          <div className="role-picker" aria-label="Agent role">
            {agentTypes.map((type) => <button className={type.type === draft.type ? "selected" : ""} key={type.type} onClick={() => changeRole(type.type)}><Bot size={16} /><span>{type.label}</span></button>)}
          </div>
          <div className="composer-form identity-form">
            <label><span>Agent id</span><input value={draft.id} aria-invalid={!cleanId || duplicateId} onChange={(event) => setDraft({ ...draft, id: event.target.value.toLowerCase().replace(/[^a-z0-9_]/g, "") })} /></label>
            <label><span>Instrument</span><select value={draft.instrument} onChange={(event) => setDraft({ ...draft, instrument: event.target.value })}>{instrumentOptions.map((instrument) => <option key={instrument}>{instrument}</option>)}</select></label>
            <label><span>Instances</span><input type="number" min="1" max="25" value={draft.count} onChange={(event) => setDraft({ ...draft, count: Math.min(25, Math.max(1, Number(event.target.value) || 1)) })} /></label>
            <label><span>Strategist</span><select value={draft.slow_strategist} onChange={(event) => setDraft({ ...draft, slow_strategist: event.target.value as AgentAddition["slow_strategist"] })}><option value="frozen">Frozen</option><option value="openai">OpenAI</option></select></label>
          </div>
          <div className="parameter-heading"><div><span className="eyebrow">Role behavior</span><strong>{role?.label}</strong></div><span>{role?.description}</span></div>
          <div className="parameter-grid">
            {(role?.parameters ?? []).map((parameter) => <label key={parameter.key}>
              <span>{parameter.label}</span>
              {parameter.kind === "select" ? <select value={String(draft.parameters[parameter.key] ?? parameter.default)} onChange={(event) => setDraft({ ...draft, parameters: { ...draft.parameters, [parameter.key]: event.target.value } })}>{parameter.options?.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select> : <input type="number" min={parameter.min} max={parameter.max} step={parameter.step} value={Number(draft.parameters[parameter.key] ?? parameter.default)} onChange={(event) => setDraft({ ...draft, parameters: { ...draft.parameters, [parameter.key]: Number(event.target.value) } })} />}
            </label>)}
          </div>
          {(!cleanId || duplicateId) && <div className="field-error">{duplicateId ? "Agent id is already in the roster." : "Use 3-48 lowercase letters, numbers, or underscores; begin with a letter."}</div>}
          <div className="editor-actions"><button className="secondary-button" disabled={!canSave} onClick={saveParticipant}>{editingIndex === null ? <Plus size={16} /> : <Pencil size={15} />}{editingIndex === null ? "Add to roster" : "Update agent"}</button></div>
        </div>
        <aside className="studio-roster">
          <div className="roster-heading"><div><span className="eyebrow">Simulation roster</span><strong>{additions.reduce((total, addition) => total + addition.count, 0)} custom agents</strong></div><span>{additions.length}/12 groups</span></div>
          <div className="addition-list">
            {additions.map((addition, index) => <div className={`addition-row ${editingIndex === index ? "editing" : ""}`} key={`${addition.id}-${index}`}><Bot size={16} /><div><strong>{addition.id}</strong><span>{roleLabel(addition.type)} · {addition.instrument} · {addition.count}x · {addition.slow_strategist}</span></div><button className="icon-button mini" onClick={() => editParticipant(index)} title="Edit agent" aria-label={`Edit ${addition.id}`}><Pencil size={14} /></button><button className="icon-button mini danger" onClick={() => removeParticipant(index)} title="Remove agent" aria-label={`Remove ${addition.id}`}><Trash2 size={14} /></button></div>)}
            {!additions.length && <div className="composer-empty"><Settings2 size={19} /><span>No custom agents in this variant.</span></div>}
          </div>
          <div className="config-preview"><span>Generated overrides</span><pre>{additions.length ? JSON.stringify(additions, null, 2) : "[]"}</pre></div>
        </aside>
      </div>
      <footer><span>Base scenario + validated custom roster</span><div><button className="secondary-button" onClick={onClose}>Cancel</button><button className="primary-button" disabled={isLaunching} onClick={onRun}>{isLaunching ? <LoaderCircle className="spin" size={16} /> : <CirclePlay size={16} />} Run simulation</button></div></footer>
    </section>
  </div>;
}

function sessionLabel(artifacts: ArtifactPayload | null): string {
  const info = artifacts?.summary?.simulation_info;
  return info && typeof info === "object" && typeof (info as Record<string, unknown>).duration === "string" ? (info as Record<string, string>).duration : "--";
}

function slowLoopLabel(status: ArtifactPayload["slow_loop_status"] | undefined): string {
  if (!status?.recorded) return "--";
  return status.failed ? `${status.completed}/${status.recorded}` : formatCompact(status.recorded);
}

function linkCount(artifacts: ArtifactPayload | null): string {
  const report = artifacts?.analysis_reports.ecology_manifest;
  return report && Array.isArray(report.relationships) ? String(report.relationships.length) : "--";
}

function totalPnl(report: AgentReport): number | null {
  const value = report.metrics?.["Total P&L"];
  if (typeof value === "number") return value;
  if (value && typeof value === "object") return Object.values(value as Record<string, unknown>).reduce<number>((total, item) => total + (readNumber(item) ?? 0), 0);
  return null;
}

function assetClass(instrument: string): string {
  if (instrument.includes("FUT")) return "Future";
  if (/UST|BOND|YLD/.test(instrument)) return "Bond";
  if (instrument.includes("ETF")) return "ETF";
  return "Stock";
}

function inferScenarioInstruments(scenario: string): string[] {
  if (scenario.includes("bond")) return ["AAPL", "AAPL_FUT", "UST10Y"];
  if (scenario.includes("future")) return ["AAPL", "AAPL_FUT"];
  return ["AAPL"];
}
