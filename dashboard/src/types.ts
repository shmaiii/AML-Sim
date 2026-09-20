export type JsonRecord = Record<string, unknown>;

export type JobStatus = "idle" | "starting" | "running" | "completed" | "failed";

export interface RunJob {
  run_id?: string;
  scenario?: string;
  status: JobStatus;
  command?: string[];
  started_at?: number;
  finished_at?: number | null;
  return_code?: number | null;
  error?: string | null;
  log_tail?: string[];
  dashboard_url?: string;
}

export interface ConfiguredAgent {
  id: string;
  type: string;
  count: number;
  instruments: string[];
  slow_strategist: string;
}

export interface MarketTrade {
  timestamp: string;
  price: number;
  quantity: number;
}

export interface MarketSummary {
  trade_count: number;
  volume: number;
  start_price: number;
  end_price: number;
  return: number;
  trades: MarketTrade[];
}

export interface AgentMetric {
  [key: string]: unknown;
}

export interface AgentReport {
  metrics?: AgentMetric;
  portfolio_timeseries?: Array<{ timestamp: string; value: number }>;
  last_portfolio_value?: number;
  pending_order_count?: number;
  executed_order_count?: number;
}

export interface SlowLoopStatus {
  recorded: number;
  completed: number;
  failed: number;
  rejected: number;
  recent_failures: Array<{ agent_id: string; timestamp: string; reason: string }>;
}

export interface ArtifactPayload {
  run_id: string;
  exists: boolean;
  missing: string[];
  summary: JsonRecord | null;
  metadata: JsonRecord | null;
  actions_report: JsonRecord | null;
  configured_agents: ConfiguredAgent[];
  agent_reports: Record<string, AgentReport>;
  order_log: string;
  order_logs: Record<string, string>;
  live_markets: Record<string, ParsedBook>;
  shock_log: string;
  shock_events: ShockEvent[];
  analysis_reports: Record<string, JsonRecord>;
  available_reports: string[];
  slow_loop_status: SlowLoopStatus;
  job: RunJob;
  server_time?: number;
}

export interface RunListItem {
  run_id: string;
  created_at: string | null;
  scenario: string | null;
  completed: boolean;
  instruments: string[];
}

export interface AgentTypeOption {
  type: string;
  label: string;
  description: string;
  parameters: AgentParameterDefinition[];
}

export interface AgentParameterOption {
  value: string;
  label: string;
}

export interface AgentParameterDefinition {
  key: string;
  label: string;
  kind: "number" | "integer" | "select";
  default: number | string;
  min?: number;
  max?: number;
  step?: number;
  options?: AgentParameterOption[];
}

export interface ScenarioOption {
  path: string;
  instruments: string[];
}

export interface Catalog {
  scenarios: string[];
  scenario_details: ScenarioOption[];
  runs: RunListItem[];
  agent_types: AgentTypeOption[];
  rabbitmq_reachable: boolean;
}

export interface AgentAddition {
  id: string;
  type: string;
  instrument: string;
  count: number;
  slow_strategist: "frozen" | "openai";
  parameters: Record<string, number | string>;
}

export interface OrderLevel {
  price: number;
  quantity: number;
  orderCount: number;
}

export interface ParsedBook {
  bids: OrderLevel[];
  asks: OrderLevel[];
  trades: Array<{ timestamp: string; price: number; quantity: number; buyer: string; seller: string }>;
}

export interface ShockEvent {
  timestamp: string;
  id: string;
  phase: "announcement" | "active";
  type: string;
  classification: string;
  severity?: number;
}
