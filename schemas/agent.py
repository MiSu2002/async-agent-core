from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class StepType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"
    SYSTEM = "SYSTEM"  
    ERROR = "ERROR"

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Name of the registered tool to execute")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass into the tool")

class AgentStep(BaseModel):
    step_num: int = Field(..., description="Chronological step index")
    step_type: StepType = Field(..., description="Type of ReAct step")
    content: str = Field(...,description="Reasoning text, tool observation or answer")
    tool_call: Optional[ToolCall] = Field(default=None, description="Present if step_type is ACTION")

class AgentRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Goal or instruction for the agent")
    max_iterations: int = Field(default=5, ge=1, le=20, description="Maximum allowed execution steps")
    timeout_secs: float = Field(default=10.0, gt=0.0, description="Latency budget timeout")

class AgentResponse(BaseModel):
    success: bool = Field(..., description="True if goal achieved without error or timeout")
    final_output: str = Field(..., description="Result answer or error details")
    trajectory: List[AgentStep] = Field(default_factory=List, description="Full execution history")
    total_steps: int = Field(..., description="Total steps taken")