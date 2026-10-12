import asyncio
import json
import hashlib
import time
from typing import AsyncGenerator, List, Optional
from app.schemas.agent import AgentRequest, AgentStep, StepType, ToolCall
from app.tools.registry import ToolRegistry

class CycleDetectedException(Exception):
    """Raised when an agent enters an infinite loop pattern."""
    pass

class AsyncReActEngine:
    def __init__(self, tool_registry: ToolRegistry, loop_history_window: int = 3):
        self.registry = tool_registry
        self.loop_history_window = loop_history_window

    def get_tool_signature_hash(self, tool_call: ToolCall) -> str:
        """Generates a deterministic signature hash for loop detection."""
        serialized = json.dumps({"tool": tool_call.tool_name, "args": tool_call.args}, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def mock_llm_reasoning_step(
        self, step_idx: int, prompt: str, observation_history: List[str]
    ) -> tuple[str, Optional[ToolCall]]:
        """Simulates dynamic LLM reasoning steps."""
        await asyncio.sleep(0.1)  # Simulate network latency
        
        if step_idx == 1:
            thought = "Inbound telemetry alert received. Querying SCADA state for Asset UK-9021."
            action = ToolCall(
                tool_name="evaluate_telemetry",
                args={"asset_id": "UK-9021", "metric": "PRESSURE"}
            )
            return thought, action

        elif step_idx == 2:
            thought = "Telemetry confirms critical pressure drop. Dispatching Field Service Work Order."
            action = ToolCall(
                tool_name="dispatch_work_order",
                args={"asset_id": "UK-9021", "priority": "HIGH", "issue_code": "PRESS_DROP_CRIT"}
            )
            return thought, action

        else:
            thought = "Work Order successfully dispatched. Triage trajectory completed."
            return thought, None

    async def run_loop_generator(self, request: AgentRequest) -> AsyncGenerator[str, None]:
        """Main ReAct execution loop emitting Server-Sent Events (SSE)."""
        step_counter = 0
        execution_history_hashes: List[str] = []
        observation_history: List[str] = []
        start_time = time.time()

        try:
            # Emit System Initialization Step
            init_step = AgentStep(
                step_num=step_counter,
                step_type=StepType.SYSTEM,
                content=f"Agent execution initiated. Budget: {request.timeout_secs}s. Max iterations: {request.max_iterations}."
            )
            yield f"data: {init_step.model_dump_json()}\n\n"

            while step_counter < request.max_iterations:
                # Check latency budget before each step
                elapsed_time = time.time() - start_time
                if elapsed_time > request.timeout_secs:
                    raise asyncio.TimeoutError()

                step_counter += 1

                # 1. Reasoning Phase
                thought, tool_call = await self.mock_llm_reasoning_step(
                    step_counter, request.prompt, observation_history
                )

                thought_step = AgentStep(
                    step_num=step_counter,
                    step_type=StepType.THOUGHT,
                    content=thought,
                    tool_call=tool_call
                )
                yield f"data: {thought_step.model_dump_json()}\n\n"

                # Exit condition: Reasoning complete (Final Answer)
                if tool_call is None:
                    final_step = AgentStep(
                        step_num=step_counter,
                        step_type=StepType.FINAL_ANSWER,
                        content=thought
                    )
                    yield f"data: {final_step.model_dump_json()}\n\n"
                    break

                # 2. Cycle-Detection Guardrail
                call_hash = self.get_tool_signature_hash(tool_call)
                recent_window = execution_history_hashes[-self.loop_history_window:]
                if call_hash in recent_window:
                    raise CycleDetectedException(
                        f"Infinite loop detected! Tool '{tool_call.tool_name}' invoked repeatedly with identical arguments."
                    )
                execution_history_hashes.append(call_hash)

                # Stream Action Step
                action_step = AgentStep(
                    step_num=step_counter,
                    step_type=StepType.ACTION,
                    content=f"Executing tool '{tool_call.tool_name}'",
                    tool_call=tool_call
                )
                yield f"data: {action_step.model_dump_json()}\n\n"

                # 3. Tool Execution / Observation Phase
                tool_result = await self.registry.execute(tool_call)
                observation_history.append(tool_result)

                obs_step = AgentStep(
                    step_num=step_counter,
                    step_type=StepType.OBSERVATION,
                    content=tool_result
                )
                yield f"data: {obs_step.model_dump_json()}\n\n"

        except asyncio.TimeoutError:
            err_step = AgentStep(
                step_num=step_counter,
                step_type=StepType.ERROR,
                content=f"Execution halted: Latency budget ({request.timeout_secs}s) exceeded."
            )
            yield f"data: {err_step.model_dump_json()}\n\n"

        except CycleDetectedException as cde:
            err_step = AgentStep(
                step_num=step_counter,
                step_type=StepType.ERROR,
                content=f"Guardrail Triggered: {str(cde)}"
            )
            yield f"data: {err_step.model_dump_json()}\n\n"

        except Exception as e:
            err_step = AgentStep(
                step_num=step_counter,
                step_type=StepType.ERROR,
                content=f"Unhandled Execution Exception: {str(e)}"
            )
            yield f"data: {err_step.model_dump_json()}\n\n"