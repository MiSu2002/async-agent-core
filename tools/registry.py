import asyncio
import inspect
import logging
import json
from typing import Callable, List, Dict, Optional
from app.schemas.agent import ToolCall

logger = logging.getLogger("async-agent-core")

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, dict] = {}

    def register(self, name: Optional[str] = None, description: Optional[str] = None):
        def decorator(func: Callable):
            tool_name = name or func.__name__
            self._tools[tool_name] = func

            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or description or "No description provided."

            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                param_type = "string"
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == dict:
                    param_type = "object"

                properties[param_name] = {"type": param_type, "description": f"Parameter {param_name}"}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            # Anthropic tool schema format uses 'input_schema'
            self._schemas[tool_name] = {
                "name": tool_name,
                "description": doc,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
            return func
        return decorator
    
    def get_anthropic_tool_schemas(self) -> List[dict]:
        return list(self._schemas.values())
    
    async def execute(self, tool_call: ToolCall) -> str:
        if tool_call.tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_call.tool_name}' is not registered.")
        
        tool_fn = self._tools[tool_call.tool_name]
        try:
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(**tool_call.args)
            else:
                result = tool_fn(**tool_call.args)

            if isinstance(result, (dict, list)):
                return json.dumps(result)
            
            return str(result)
        
        except Exception as e:
            return f"Tool Execution Error: {str(e)}"

registry = ToolRegistry()
