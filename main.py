import asyncio
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.schemas.agent import AgentRequest
from app.tools.registry import registry
from app.core.engine import AsyncReActEngine

# =====================================================================
# REGISTER SAMPLE TOOLS
# =====================================================================

@registry.register(name="evaluate_telemetry")
async def evaluate_telemetry(asset_id: str, metric: str) -> dict:
    await asyncio.sleep(0.15)  # Simulate I/O query
    return {
        "asset_id": asset_id,
        "metric": metric,
        "value": 1.1,
        "unit": "bar",
        "status": "CRITICAL_PRESSURE_DROP",
        "threshold": 2.5
    }

@registry.register(name="dispatch_work_order")
async def dispatch_work_order(asset_id: str, priority: str, issue_code: str) -> dict:
    await asyncio.sleep(0.20)
    return {
        "work_order_id": f"WO-{int(time.time())}",
        "asset_id": asset_id,
        "priority": priority,
        "assigned_team": "Rapid Response Unit",
        "status": "DISPATCHED"
    }

# =====================================================================
# FASTAPI APP INITIALIZATION
# =====================================================================

app = FastAPI(
    title="Async Agent Core",
    description="Lightweight ReAct agent engine with cycle-detection guardrails.",
    version="1.0.0"
)

engine = AsyncReActEngine(tool_registry=registry)

@app.post("/api/v1/agent/stream")
async def stream_agent_execution(request: AgentRequest):
    return StreamingResponse(
        engine.run_loop_generator(request),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)