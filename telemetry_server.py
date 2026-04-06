"""
AgentTrace Telemetry Server
Streams GPU hardware telemetry via Server-Sent Events (SSE).
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
import os
import time

from fastapi import FastAPI
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
                # Simulate Lit Silicon power reduction on GPU 0
                "gfxclk": int(2100 - (200 if i == 0 else 0)
                           + 50 * __import__("math").sin(t * 0.3 + i)),
                "memclk": int(1300 + 30 * __import__("math").sin(t * 0.2 + i)),
                "power": int(550 - (80 if i == 0 else 0)
                          + 20 * __import__("math").sin(t * 0.4 + i)
                          + random.randint(-5, 5)),
                "temp": int(65 + i * 3
                         + 5 * __import__("math").sin(t * 0.1 + i)
                         + random.randint(-1, 1)),
            }
            for i in range(n_gpus)
        ]


async def hardware_stream():
    """Yield GPU telemetry as SSE events at 1Hz."""
    while True:
        event = {
            "ts": time.time(),
            "elapsed": round(time.time() - START_TIME, 1),
            "gpus": get_gpu_snapshot(),
            "source": "amdsmi" if HAS_AMDSMI else "mock",
        }
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(1.0)


@app.get("/stream/hardware")
async def stream_hardware():
    """SSE endpoint for live GPU telemetry."""
    return StreamingResponse(
        hardware_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status")
async def api_status():
    """Health check endpoint."""
    return {
        "status": "ok",
        "gpu_count": len(gpu_map) if HAS_AMDSMI
                     else int(os.environ.get("MOCK_GPUS", "8")),
        "source": "amdsmi" if HAS_AMDSMI else "mock",
        "uptime_s": round(time.time() - START_TIME, 1),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8765"))
    print(f"[AgentTrace] Telemetry server starting on port {port}")
    print(f"[AgentTrace] Dashboard connects to http://localhost:{port}/stream/hardware")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
