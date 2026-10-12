import asyncio
import inspect
import logging
import json
from typing import Any, Callable, Dict, Optional
from pydantic import BaseModel, create_model
from app.schemas.agent import ToolCall

logger = logging.getLogger("async-agent-core")

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, name: Optional[str] = None):
        def decorator(func: Callable):
            if not asyncio.iscoroutinefunction(func):
                raise ValueError(f"Tool '{func.__name__}' must be an asyn function (async def)")
            
            tool_name = name or func.__name__
            
            sig = inspect.signature(func)
            fields = {}
            
            for param_name, param in sig.parameters.items():
                param_type = param.annotation if param.annotation != inspect.Parameter.empty() else Any
                default_val = param.default if param.default != inspect.Parameter.empty() else ...
                fields[param_name] = (param_type, default_val)
                
            schema_model = create_model(f"{tool_name}_InputSchema", **fields)
            
            self._tools[tool_name] = func
            self._schemas[tool_name] = schema_model
            return func
        return decorator
    
    async def execute(self, tool_call: ToolCall) -> str:
        if tool_call.tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_call.tool_name}' is not registered.")
        
        tool_fn = self._tools[tool_call.tool_name]
        try:
            # Unpack tool_call.args into keyword arguments
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(**tool_call.args)
            else:
                result = tool_fn(**tool_call.args)
            
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            return str(result)
        
        except Exception as e:
            logger.error(f"Error executing tool '{tool_call.tool_name}': {str(e)}")
            return f"Tool Execution Error: {str(e)}"
        
    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        return self._schemas

registry = ToolRegistry()

@registry.register(name="check_water_utility_status")
async def check_water_utility_status(account_id: str) -> str:
    await asyncio.sleep(0.1)
    if account_id.startswith("UK-"):
        return f"Account {account_id}: Pressure nominal (4.2 bar), Telemetry status: OK, Active SLA: Tier-1"
    return f"Account {account_id}: Status unknown or invalid accound ID format"

@registry.register(name="fetch_sensor_telemetry")
async def fetch_sensor_telemetry(sensor_id: str, limit: int=5) -> str:
    await asyncio.sleep(0.15)
    return f"Sensor {sensor_id}: Last {limit} readings -> [22.4 C, 22.6 C, 23.1 C, 22.8 C, 22.5 C]; Status: Calibrated"