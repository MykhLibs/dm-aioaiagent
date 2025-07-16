import uuid
import asyncio
from typing import Any
from langchain_core.messages import AIMessage, ToolMessage

from .ai_agent import DMAIAgent
from .types import *


class DMAioAIAgent(DMAIAgent):
    def __init__(
        self,
        *args,
        before_tool_call_callback: AsyncBeforeToolCallCallback = None,
        after_tool_call_callback: AsyncAfterToolCallCallback = None,
        **kwargs
    ):
        super().__init__(
            *args,
            before_tool_call_callback=before_tool_call_callback,
            after_tool_call_callback=after_tool_call_callback,
            **kwargs
        )

    async def run(self, query: str, *args, **kwargs) -> Union[str, TypedDict, BaseModel]:
        new_messages = await self.run_messages(messages=[TextMessage(role="user", content=query)], *args, **kwargs)

        last_message = new_messages[-1]
        if isinstance(last_message, AIMessage):
            return last_message.content
        return last_message

    async def run_messages(
        self,
        messages: list[InputMessage],
        *,
        ls_metadata: dict[str, Any] = None,
        ls_tags: list[str] = None,
        ls_run_id: uuid.UUID = None,
        ls_thread_id: uuid.UUID = None
    ) -> list[BaseMessage]:
        if ls_metadata is None:
            ls_metadata = {}
        if isinstance(ls_run_id, uuid.UUID):
            ls_run_id = ls_run_id
        if isinstance(ls_thread_id, uuid.UUID):
            ls_metadata["thread_id"] = ls_thread_id

        config_data = {
            "metadata": ls_metadata,
            "tags": ls_tags,
            "run_id": ls_run_id
        }
        state = await self._graph.ainvoke(
            input={"messages": messages, "new_messages": []},
            config={k: v for k, v in config_data.items() if v}
        )
        return state["new_messages"]

    async def _invoke_llm_node(self, state: State, second_attempt: bool = False) -> State:
        self._logger.debug("Run node: Invoke LLM")
        try:
            ai_response = await self._agent.ainvoke({"messages": state["messages"]})
        except Exception as e:
            self._logger.error(e)
            if second_attempt:
                if "invalid_image_url" in str(e):
                    response = self._response_if_invalid_image
                else:
                    response = self._response_if_request_fail

                ai_response = AIMessage(content=response)
                state["messages"].append(ai_response)
                state["new_messages"].append(ai_response)
                return state

            return await self._invoke_llm_node(state, second_attempt=True)

        state["messages"].append(ai_response)
        state["new_messages"].append(ai_response)
        return state

    async def _execute_tool_node(self, state: State) -> State:
        self._logger.debug("Run node: Execute tool")
        tasks = []

        for tool_call in state["messages"][-1].tool_calls:
            tool_id = tool_call["id"]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            async def tool_callback(tool_id=tool_id, tool_name=tool_name, tool_args=tool_args) -> None:
                if tool_name in self._tool_map:
                    try:
                        if self._before_tool_call_callback:
                            await self._before_tool_call_callback(tool_name, tool_args)
                    except Exception as e:
                        self._logger.error(str(e), callback_type="before_tool_call")

                    try:
                        self._logger.debug("Invoke tool", tool_id=tool_id, tool_name=tool_name, tool_args=tool_args)
                        tool_response = await self._tool_map[tool_name].arun(tool_args)
                    except Exception as e:
                        self._logger.error(str(e), tool_id=tool_id)
                        tool_response = "Tool executed with an error!"

                    try:
                        if self._after_tool_call_callback:
                            await self._after_tool_call_callback(tool_name, tool_args, tool_response)
                    except Exception as e:
                        self._logger.error(str(e), callback_type="after_tool_call")
                else:
                    tool_response = f"Tool '{tool_name}' not found!"
                self._logger.debug(f"Tool response:\n{tool_response}", tool_id=tool_id)

                tool_message = ToolMessage(content=str(tool_response), name=tool_name, tool_call_id=tool_id)
                state["messages"].append(tool_message)
                state["new_messages"].append(tool_message)

            tasks.append(asyncio.create_task(tool_callback()))

        await asyncio.gather(*tasks)
        return state
