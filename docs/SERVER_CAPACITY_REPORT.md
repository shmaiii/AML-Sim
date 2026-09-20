# AML-Sim server capacity smoke test

- **Test date:** 10 September 2026
- **Host:** Apple M2 MacBook Air, 8 CPU cores, 8 GB memory
- **Services:** Local AML-Sim, local RabbitMQ, and the React dashboard served by
  `dashboard_server.py`

## Purpose

This smoke test checks whether the current local architecture can start, run,
and report simulations with 10, 50, and 100 trading-agent processes. It is a
capacity indicator, not a production benchmark or a substitute for long-run
stability testing.

## Workload

Each trial used the same compact synthetic market:

- one AAPL order-book exchange;
- one frozen market maker;
- the remaining population as frozen retail agents;
- 30-second action and market ticks;
- six simulated ticks over three simulated minutes;
- active order submission, matching, logging, portfolio output, and reports;
- no OpenAI calls, external market data, or shock agent.

Frozen strategists isolate infrastructure cost from API latency and cost. Peak
CPU and memory represent the AML runner and its descendant exchange, clock, and
agent processes. RabbitMQ, the operating system, and unrelated applications are
not included. Aggregate RSS is an approximate process footprint and may count
some shared memory more than once.

## Results

| Trading agents | Startup grace | Completed ticks | Wall time | Peak CPU | Peak aggregate RSS | Agent reports | Trades | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 3 s | 6/6 | 43.67 s | 380% | 623 MB | 10/10 | 31 | Passed |
| 50 | 3 s | 6/6 | 43.36 s | 595% | 1.68 GB | 50/50 | 154 | Passed |
| 100 | 3 s | 1/6 | 193.13 s | 586% | 2.53 GB | 0/100 | 0 | Timed out |
| 100 | 20 s | 6/6 | 71.05 s | 551% | 2.82 GB | 100/100 | 355 | Passed |

`100%` CPU represents approximately one fully used CPU core, so the observed
peaks used between 3.8 and 6 cores. The successful 100-agent run started 104
processes at peak: the runner, 100 traders, exchange, clock, and process support.

## Dashboard read test

The completed 100-agent run was loaded through the dashboard API:

- archived artifact response: HTTP 200, 218 KB, approximately 12 ms locally;
- live stream: three complete snapshots in three seconds;
- live snapshot size: approximately 183 KB per second per connected browser.

The dashboard remained responsive. Archived report loading is inexpensive at
this scale. Live streaming is acceptable for a small number of local viewers,
but bandwidth and repeated JSON serialization grow with both run size and
connected viewers.

## Findings

### 10 Agents

This is comfortably within the current machine's capacity. CPU, memory, startup,
matching, and report generation completed without errors.

### 50 Agents

Fifty agents also completed without additional wall-clock delay in this short
trial. Memory increased to 1.68 GB and CPU reached roughly six cores, but all
agents produced reports and the exchange processed all ticks.

### 100 Agents

The machine can run 100 active frozen agents, but startup is sensitive to
RabbitMQ readiness. A three-second grace period started the clock before the
exchange had completed its queue binding. The first tick was missed, the clock
waited for an exchange response, and the run reached its 180-second wall-time
limit.

Increasing startup grace to 20 seconds allowed the same workload to complete.
The successful run used about 2.82 GB aggregate RSS and took 71 seconds including
the longer startup wait. Resource exhaustion was not observed.

## Engineering issues found

1. **Process alive is not service ready.** The launcher currently checks that
   processes exist, then relies on a fixed startup delay. At high populations,
   the clock can publish before RabbitMQ queues are fully bound.
2. **Timeouts can look successful.** The failed 100-agent run terminated after
   the wall-time limit but returned exit code 0 and generated empty reports.
   Run status should distinguish completion from timeout or forced termination.
3. **AML agents are not part of the clock barrier.** The clock log reports zero
   expected trader responses. This permits ticks to advance without proving
   every AML trader processed the previous tick, which can matter under heavier
   loads.
4. **Live streaming repeats full snapshots.** At 100 agents, one browser receives
   roughly 183 KB each second. Delta updates or lower-frequency participant
   summaries would scale better for multiple viewers.