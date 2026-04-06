"""
AgentTrace Telemetry Server
Streams GPU hardware telemetry + straggler analysis + agent events
via Server-Sent Events (SSE).

Runs on AMD HPC cluster alongside Chopper/Lit Silicon workloads.

Usage:
    # On AMD cluster:
    uv pip install fastapi uvicorn
    python telemetry_server.py

    # From laptop (SSH tunnel):
    ssh -NL 8765:localhost:8765 th273073@hpcfund.amd.com

    # Open AgentTrace dashboard — it auto-connects to localhost:8765
"""

import asyncio
import json
import math
import os
import time
from collections import deque
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="AgentTrace Telemetry Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try AMD SMI (only on AMD cluster with ROCm)
HAS_AMDSMI = False
devices = []
gpu_map = {}
try:
    from amdsmi import (
        AmdSmiException,
        amdsmi_get_gpu_kfd_info,
        amdsmi_get_gpu_metrics_info,
        amdsmi_get_processor_handles,
        amdsmi_init,
    )
    amdsmi_init()
    devices = amdsmi_get_processor_handles()
    gpu_map = {
        amdsmi_get_gpu_kfd_info(dev)["node_id"]: dev for dev in devices
    }
    HAS_AMDSMI = True
    print(f"[AgentTrace] AMD SMI initialized - {len(devices)} GPUs detected")
except ImportError:
    print("[AgentTrace] AMD SMI not available - using mock data")
except Exception as e:
    print(f"[AgentTrace] AMD SMI error: {e} - using mock data")

START_TIME = time.time()

# ── Straggler Detection State ──
# Rolling window of per-GPU throughput for s-value computation
# s-value = how far behind each GPU is relative to the fastest (Lit Silicon metric)
THROUGHPUT_WINDOW = 10  # seconds of history for straggler detection
gpu_freq_history = {}  # gpu_id -> deque of (timestamp, gfxclk)

# ── Agent Event Log ──
# Stores agent-level events (LLM calls, tool executions) pushed via POST
agent_events = deque(maxlen=1000)
agent_lock = Lock()


def compute_straggler(gpus):
    """Compute straggler s-value per GPU based on recent frequency.

    s-value measures how far each GPU's average frequency is from the
    fastest GPU. Higher s-value = more of a straggler.
    Mirrors Chopper's get_straggler_df() but computed in real-time.
    """
    now = time.time()
    for g in gpus:
        gid = g["gpu"]
        if gid not in gpu_freq_history:
            gpu_freq_history[gid] = deque(maxlen=THROUGHPUT_WINDOW * 2)
        gpu_freq_history[gid].append((now, g["gfxclk"]))

    # Compute average frequency over window per GPU
    cutoff = now - THROUGHPUT_WINDOW
    avg_freq = {}
    for gid, hist in gpu_freq_history.items():
        recent = [f for t, f in hist if t >= cutoff and f > 0]
        avg_freq[gid] = sum(recent) / len(recent) if recent else 0

    if not avg_freq:
        return {}

    max_freq = max(avg_freq.values()) if avg_freq else 1
    if max_freq == 0:
        max_freq = 1

    # s-value: normalized lag from fastest GPU (0 = fastest, 1 = max lag)
    straggler = {}
    for gid, freq in avg_freq.items():
        straggler[gid] = round((max_freq - freq) / max_freq, 4)

    return straggler


def get_gpu_snapshot():
    """Read current GPU metrics from AMD SMI (same API as Chopper gpu.py)."""
    if HAS_AMDSMI:
        samples = []
        for gpu_id, dev in gpu_map.items():
            try:
                info = amdsmi_get_gpu_metrics_info(dev)
                samples.append({
                    "gpu": gpu_id,
                    "gfxclk": info.get("current_gfxclk", 0),
                    "memclk": info.get("current_uclk", 0),
                    "power": info.get("current_socket_power", 0),
                    "temp": info.get("temperature_hotspot", 0),
                })
            except AmdSmiException:
                samples.append({
                    "gpu": gpu_id,
                    "gfxclk": 0, "memclk": 0, "power": 0, "temp": 0,
                })
        return samples
    else:
        import random
        n_gpus = int(os.environ.get("MOCK_GPUS", "8"))
        t = time.time() - START_TIME
        return [
            {
                "gpu": i,
                # Simulate Lit Silicon: GPU 0 throttled (straggler),
                # GPU 7 hottest (thermal straggler)
                "gfxclk": int(2100
                           - (200 if i == 0 else 0)
                           - (100 if i == 7 and t > 30 else 0)
                           + 50 * math.sin(t * 0.3 + i)),
                "memclk": int(1300 + 30 * math.sin(t * 0.2 + i)),
                "power": int(550
                          - (80 if i == 0 else 0)
                          + (30 if i == 7 else 0)
                          + 20 * math.sin(t * 0.4 + i)
                          + random.randint(-5, 5)),
                "temp": int(65 + i * 3
                         + (15 if i == 7 and t > 30 else 0)
                         + 5 * math.sin(t * 0.1 + i)
                         + random.randint(-1, 1)),
            }
            for i in range(n_gpus)
        ]


async def hardware_stream():
    """Yield GPU telemetry + straggler analysis as SSE events at 1Hz."""
    while True:
        gpus = get_gpu_snapshot()
        straggler = compute_straggler(gpus)

        # Find lead GPU (lowest s-value) and straggler (highest)
        if straggler:
            lead_gpu = min(straggler, key=straggler.get)
            straggler_gpu = max(straggler, key=straggler.get)
        else:
            lead_gpu = straggler_gpu = None

        # Compute aggregate stats
        powers = [g["power"] for g in gpus if g["power"] > 0]
        temps = [g["temp"] for g in gpus if g["temp"] > 0]

        event = {
            "ts": time.time(),
            "elapsed": round(time.time() - START_TIME, 1),
            "gpus": gpus,
            "straggler": {
                "s_values": straggler,
                "lead_gpu": lead_gpu,
                "straggler_gpu": straggler_gpu,
                "max_s_value": round(max(straggler.values()), 4) if straggler else 0,
            },
            "aggregate": {
                "total_power": sum(powers),
                "avg_temp": round(sum(temps) / len(temps), 1) if temps else 0,
                "temp_spread": (max(temps) - min(temps)) if temps else 0,
            },
            "source": "amdsmi" if HAS_AMDSMI else "mock",
        }
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(1.0)


async def agent_stream():
    """Yield agent-level events as SSE. Events are pushed via POST /api/agent/event."""
    last_idx = 0
    while True:
        with agent_lock:
            events = list(agent_events)
        if len(events) > last_idx:
            for evt in events[last_idx:]:
                yield f"data: {json.dumps(evt)}\n\n"
            last_idx = len(events)
        await asyncio.sleep(0.5)


@app.get("/stream/hardware")
async def stream_hardware():
    """SSE endpoint for live GPU telemetry + straggler analysis."""
    return StreamingResponse(
        hardware_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/stream/agent")
async def stream_agent():
    """SSE endpoint for agent-level events (LLM calls, tool executions)."""
    return StreamingResponse(
        agent_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/agent/event")
async def post_agent_event(request: Request):
    """Push an agent event (LLM call start/end, tool execution start/end).

    Expected JSON body:
        {"type": "llm_start"|"llm_end"|"tool_start"|"tool_end",
         "name": "description", "tokens_in": N, "tokens_out": N}
    """
    body = await request.json()
    body["ts"] = time.time()
    body["elapsed"] = round(time.time() - START_TIME, 1)
    with agent_lock:
        agent_events.append(body)
    return {"status": "ok", "event_count": len(agent_events)}


@app.get("/api/status")
async def api_status():
    """Health check endpoint."""
    return {
        "status": "ok",
        "gpu_count": len(gpu_map) if HAS_AMDSMI
                     else int(os.environ.get("MOCK_GPUS", "8")),
        "source": "amdsmi" if HAS_AMDSMI else "mock",
        "uptime_s": round(time.time() - START_TIME, 1),
        "agent_events": len(agent_events),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8765"))
    print(f"[AgentTrace] Telemetry server starting on port {port}")
    print(f"[AgentTrace] Endpoints:")
    print(f"  GET  /stream/hardware   — live GPU telemetry + straggler")
    print(f"  GET  /stream/agent      — agent event stream")
    print(f"  POST /api/agent/event   — push agent events")
    print(f"  GET  /api/status        — health check")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
