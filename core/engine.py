import os
import json
import hashlib
import asyncio
import time
from typing import AsyncGenerator, List, Optional
import anthropic

from app.schemas.agent import AgentRequest, AgentStep, StepType, ToolCall
from app.tools.registry import ToolRegistry

class CycleDetectedException(Exception):
    """Raised when an agent enters an infinite loop pattern."""
    pass

class AsyncReActEngine:
    def __init__(self, tool_registry: ToolRegistry, loop_history_window: int = 3):
        self.registry = tool_registry
        self.loop_history_window = loop_history_window
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def get_tool_signature_hash(self, tool_call: ToolCall) -> str:
        serialized = json.dumps({"tool": tool_call.tool_name, "args": tool_call.args}, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def run_loop_generator(self, request: AgentRequest) -> AsyncGenerator[str, None]:
        step_counter = 0
        execution_history_hashes: List[str] = []
        start_time = time.time()

        messages = [{"role": "user", "content": request.prompt}]
        tools = self.registry.get_anthropic_tool_schemas()

        try:
            init_step = AgentStep(
                step_num=step_counter,
                step_type=StepType.SYSTEM,
                content=f"Agent execution initiated with Claude. Budget: {request.timeout_secs}s. Max iterations: {request.max_iterations}."
            )
            yield f"data: {init_step.model_dump_json()}\n\n"

            while step_counter < request.max_iterations:
                if (time.time() - start_time) > request.timeout_secs:
                    raise asyncio.TimeoutError()

                step_counter += 1

                # Call Claude API
                response = await self.client.messages.create(
                    model="claude-haiku-5-5",
                    max_tokens=1024,
                    system="You are an autonomous utility SCADA triage agent. Use available tools when necessary to investigate telemetry alerts and resolve incidents.",
                    tools=tools if tools else [],
                    messages=messages
                )

                thought = ""
                tool_uses = []

                for content_block in response.content:
                    if content_block.type == "text":
                        thought += content_block.text
                    elif content_block.type == "tool_use":
                        tool_uses.append(content_block)

                if not thought:
                    thought = "Evaluating next operational step..."

                # Determine if a tool call was requested
                primary_tool_call = None
                if tool_uses:
                    first_use = tool_uses[0]
                    primary_tool_call = ToolCall(
                        tool_name=first_use.name,
                        args=first_use.input
                    )

                thought_step = AgentStep(
                    step_num=step_counter,
                    step_type=StepType.THOUGHT,
                    content=thought,
                    tool_call=primary_tool_call
                )
                yield f"data: {thought_step.model_dump_json()}\n\n"

                # Append assistant message containing all content blocks
                messages.append({"role": "assistant", "content": response.content})

                if not tool_uses:
                    final_step = AgentStep(
                        step_num=step_counter,
                        step_type=StepType.FINAL_ANSWER,
                        content=thought
                    )
                    yield f"data: {final_step.model_dump_json()}\n\n"
                    break

                # Process all tool uses and gather corresponding tool results
                tool_result_content_blocks = []
                
                for t_use in tool_uses:
                    current_call = ToolCall(tool_name=t_use.name, args=t_use.input)

                    # Guardrail check
                    call_hash = self.get_tool_signature_hash(current_call)
                    recent_window = execution_history_hashes[-self.loop_history_window:]
                    if call_hash in recent_window:
                        raise CycleDetectedException(
                            f"Infinite loop detected! Tool '{current_call.tool_name}' invoked repeatedly with identical arguments."
                        )
                    execution_history_hashes.append(call_hash)

                    action_step = AgentStep(
                        step_num=step_counter,
                        step_type=StepType.ACTION,
                        content=f"Executing tool '{current_call.tool_name}'",
                        tool_call=current_call
                    )
                    yield f"data: {action_step.model_dump_json()}\n\n"

                    # Execute tool
                    tool_result_str = await self.registry.execute(current_call)

                    obs_step = AgentStep(
                        step_num=step_counter,
                        step_type=StepType.OBSERVATION,
                        content=tool_result_str
                    )
                    yield f"data: {obs_step.model_dump_json()}\n\n"

                    tool_result_content_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": t_use.id,
                        "content": tool_result_str
                    })

                # Append all tool results in a single user message back to Claude
                messages.append({
                    "role": "user",
                    "content": tool_result_content_blocks
                })

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
