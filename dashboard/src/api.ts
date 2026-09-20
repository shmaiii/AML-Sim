import type { AgentAddition, ArtifactPayload, Catalog, RunJob } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  const payload = (await response.json()) as T & { error?: string };
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with ${response.status}`);
  }
  return payload;
}

export function getCatalog(): Promise<Catalog> {
  return request<Catalog>("/api/catalog");
}

export function getArtifacts(runId: string): Promise<ArtifactPayload> {
  return request<ArtifactPayload>(`/api/artifacts?run_id=${encodeURIComponent(runId)}`);
}

export function launchRun(input: {
  scenario: string;
  runId: string;
  agentAdditions: AgentAddition[];
}): Promise<RunJob> {
  return request<RunJob>("/api/run", {
    method: "POST",
    body: JSON.stringify({
      scenario: input.scenario,
      run_id: input.runId,
      reports: true,
      agent_additions: input.agentAdditions,
    }),
  });
}

export function subscribeToRun(
  runId: string,
  onSnapshot: (payload: ArtifactPayload) => void,
  onFailure: () => void,
): () => void {
  const stream = new EventSource(`/api/live?run_id=${encodeURIComponent(runId)}`);
  stream.addEventListener("snapshot", (event) => {
    try {
      onSnapshot(JSON.parse((event as MessageEvent<string>).data) as ArtifactPayload);
    } catch {
      onFailure();
    }
  });
  stream.onerror = onFailure;
  return () => stream.close();
}
