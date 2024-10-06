import json
import sys
import asyncio
from typing import Union
from langchain_core.messages import ToolMessage

from .ai_agent import DMAIAgent, InputState, OutputState, Message

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

__all__ = ["DMAioAIAgent"]


class DMAioAIAgent(DMAIAgent):
    agent_name = "AsyncAIAgent"

    async def run(self, messages: list[Message]) -> Union[str, OutputState]:
        state = await self._graph.ainvoke({"messages": messages})
        if self._return_context:
            return state
        return state["answer"]

    async def _invoke_llm_node(self, state: InputState) -> InputState:
        self._logger.debug("Run node: Invoke LLM")
        ai_response = await self._agent.ainvoke({"messages": state.inner_state.messages})
        state.inner_state.messages.append(ai_response)
        return state

    async def _execute_tool_node(self, state: InputState) -> InputState:
        self._logger.debug("Run node: Execute tool")
        tasks = []
        for tool_call in state.inner_state.messages[-1].tool_calls:
            tool_id = tool_call["id"]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            async def tool_callback(tool_id=tool_id, tool_name=tool_name, tool_args=tool_args) -> None:
                self._logger.debug("Invoke tool", tool_id=tool_id, tool_name=tool_name, tool_args=tool_args)
                if tool_name in self._tool_map:
                    try:
                        tool_response = await self._tool_map[tool_name].arun(tool_args)
                    except Exception as e:
                        self._logger.error(e, tool_id=tool_id)
                        tool_response = "Tool executed with an error!"
                else:
                    tool_response = f"Tool '{tool_name}' not found!"
                self._logger.debug(f"Tool response:\n{tool_response}", tool_id=tool_id)

                state.inner_state.context.append({"tool_name": tool_name,
                                                  "tool_args": json.dumps(tool_args, ensure_ascii=False),
                                                  "tool_response": tool_response})
                tool_message = ToolMessage(content=str(tool_response), name=tool_name, tool_call_id=tool_id)
                state.inner_state.messages.append(tool_message)

            tasks.append(asyncio.create_task(tool_callback()))

        await asyncio.gather(*tasks)
        return state
