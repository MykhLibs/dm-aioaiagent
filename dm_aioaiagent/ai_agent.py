import os
import uuid
from typing import Any
from pydantic import SecretStr
from itertools import dropwhile
from threading import Thread
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph
from dm_logger import DMLogger

from .types import *

__all__ = ["DMAIAgent"]


class DMAIAgent:
    MAX_MEMORY_MESSAGES = 20  # Only INT greater than 0
    _ALLOWED_ROLES = ("user", "ai")

    def __init__(
        self,
        system_message: str = "You are a helpful assistant.",
        tools: list[BaseTool] = None,
        *,
        model: str = "gpt-4o-mini",
        temperature: int = 1,
        parallel_tool_calls: bool = True,
        agent_name: str = "AIAgent",
        input_output_logging: bool = True,
        is_memory_enabled: bool = True,
        max_memory_messages: int = MAX_MEMORY_MESSAGES,
        save_tools_responses_in_memory: bool = True,
        llm_provider_api_key: str = "",
        response_if_request_fail: str = "I can't provide a response right now. Please try again later.",
        response_if_invalid_image: str = "The image is unavailable or the link is incorrect."
    ):
        self._logger = DMLogger(agent_name)
        self._input_output_logging = bool(input_output_logging)

        self._system_message = str(system_message)
        self._tools = tools or []
        self._is_tools_exists = bool(tools)
        self._model = str(model)
        self._temperature = int(temperature)
        self._parallel_tool_calls = bool(parallel_tool_calls)
        self._llm_provider_api_key = str(llm_provider_api_key)

        self._memory_messages = []
        self._is_memory_enabled = bool(is_memory_enabled)
        self._save_tools_responses_in_memory = bool(save_tools_responses_in_memory)
        self._max_memory_messages = self._validate_max_memory_messages(max_memory_messages)
        self._response_if_request_fail = str(response_if_request_fail)
        self._response_if_invalid_image = str(response_if_invalid_image)

        self._check_langsmith_envs()
        self._init_agent()
        self._init_graph()

    def run(self, query: str, **kwargs) -> str:
        new_messages = self.run_messages(messages=[{"role": "user", "content": query}], **kwargs)
        return new_messages[-1].content

    def run_messages(
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
        state = self._graph.invoke(input={"messages": messages, "new_messages": []},
                                   config={k: v for k, v in config_data.items() if v})
        return state["new_messages"]

    @property
    def memory_messages(self) -> list[BaseMessage]:
        return self._memory_messages

    def clear_memory_messages(self) -> None:
        self._memory_messages.clear()

    def _prepare_messages_node(self, state: State) -> State:
        messages = state["messages"] or [{"role": "user", "content": ""}]
        state["messages"] = []
        for item in messages:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if not role or role not in self._ALLOWED_ROLES or not content:
                    continue
                if role == "ai":
                    MessageClass = AIMessage
                else:
                    MessageClass = HumanMessage
                state["messages"].append(MessageClass(content))
            elif isinstance(item, BaseMessage):
                state["messages"].append(item)

        if self._input_output_logging:
            self._logger.debug(f'Query:\n{state["messages"][-1].content}')
        if self._is_memory_enabled:
            state["messages"] = self._memory_messages + state["messages"]
        return state

    def _invoke_llm_node(self, state: State, second_attempt: bool = False) -> State:
        self._logger.debug("Run node: Invoke LLM")
        try:
            ai_response = self._agent.invoke({"messages": state["messages"]})
        except Exception as e:
            self._logger.error(e)
            if second_attempt:
                response = self._response_if_invalid_image if "invalid_image_url" in str(e) else self._response_if_request_fail
                state["messages"].append(AIMessage(content=response))
                return state
            return self._invoke_llm_node(state, second_attempt=True)
        state["messages"].append(ai_response)
        state["new_messages"].append(ai_response)
        return state

    def _execute_tool_node(self, state: State) -> State:
        self._logger.debug("Run node: Execute tool")
        threads = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_id = tool_call["id"]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            def tool_callback(tool_id=tool_id, tool_name=tool_name, tool_args=tool_args) -> None:
                self._logger.debug("Invoke tool", tool_id=tool_id, tool_name=tool_name, tool_args=tool_args)
                if tool_name in self._tool_map:
                    try:
                        tool_response = self._tool_map[tool_name].run(tool_args)
                    except Exception as e:
                        self._logger.error(e, tool_id=tool_id)
                        tool_response = "Tool executed with an error!"
                else:
                    tool_response = f"Tool not found!"
                self._logger.debug(f"Tool response:\n{tool_response}", tool_id=tool_id)

                tool_message = ToolMessage(content=str(tool_response), name=tool_name, tool_call_id=tool_id)
                state["messages"].append(tool_message)
                state["new_messages"].append(tool_message)

            threads.append(Thread(target=tool_callback, daemon=True))

        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return state

    def _exit_node(self, state: State) -> State:
        if self._input_output_logging:
            self._logger.debug(f'Answer:\n{state["messages"][-1].content}')

        if self._is_memory_enabled:
            messages_to_memory = state["messages"][-self._max_memory_messages:]
            if self._save_tools_responses_in_memory:
                # drop ToolsMessages from start of list
                self._memory_messages = list(dropwhile(lambda x: isinstance(x, ToolMessage), messages_to_memory))
            else:
                self._memory_messages.clear()
                for mes in messages_to_memory:
                    if isinstance(mes, ToolMessage) or (isinstance(mes, AIMessage) and mes.tool_calls):
                        continue
                    self._memory_messages.append(mes)
        return state

    def _messages_router(self, state: State) -> str:
        if self._is_tools_exists and state["messages"][-1].tool_calls:
            route = "execute_tool"
        else:
            route = "exit"
        return route

    def _init_agent(self) -> None:
        if self._llm_provider_api_key:
            self._llm_provider_api_key = SecretStr(self._llm_provider_api_key)

        if self._model.startswith("gpt"):
            from langchain_openai import ChatOpenAI

            api_key = SecretStr(self._llm_provider_api_key or os.getenv("OPENAI_API_KEY"))
            llm = ChatOpenAI(model_name=self._model, temperature=self._temperature, openai_api_key=api_key)
        elif self._model.startswith("claude"):
            from langchain_anthropic import ChatAnthropic

            api_key = SecretStr(self._llm_provider_api_key or os.getenv("ANTHROPIC_API_KEY"))
            llm = ChatAnthropic(model=self._model, temperature=self._temperature, anthropic_api_key=api_key)
        else:
            raise ValueError(f"{self.__class__.__name__} not support this model: '{self._model}'")

        if self._is_tools_exists:
            self._tool_map = {t.name: t for t in self._tools}
            llm = llm.bind_tools(self._tools, parallel_tool_calls=self._parallel_tool_calls)

        prompt = ChatPromptTemplate.from_messages([SystemMessage(content=self._system_message),
                                                   MessagesPlaceholder(variable_name="messages")])

        self._agent = prompt | llm

    def _init_graph(self) -> None:
        workflow = StateGraph(State)
        workflow.add_node("Prepare messages", self._prepare_messages_node)
        workflow.add_node("Invoke LLM", self._invoke_llm_node)
        workflow.add_node("Execute tool", self._execute_tool_node)
        workflow.add_node("Exit", self._exit_node)

        workflow.add_edge("Prepare messages", "Invoke LLM")
        workflow.add_conditional_edges(source="Invoke LLM",
                                       path=self._messages_router,
                                       path_map={"execute_tool": "Execute tool", "exit": "Exit"})
        workflow.add_edge("Execute tool", "Invoke LLM")
        workflow.set_entry_point("Prepare messages")
        workflow.set_finish_point("Exit")
        self._graph = workflow.compile()

    @staticmethod
    def _check_langsmith_envs() -> None:
        if os.getenv("LANGCHAIN_API_KEY"):
            if not os.getenv("LANGCHAIN_TRACING_V2"):
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if not os.getenv("LANGCHAIN_ENDPOINT"):
                os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    @classmethod
    def _validate_max_memory_messages(cls, max_messages_in_memory: int) -> int:
        if isinstance(max_messages_in_memory, int) and max_messages_in_memory > 0:
            return max_messages_in_memory
        return cls.MAX_MEMORY_MESSAGES

    def print_graph(self) -> None:
        self._graph.get_graph().print_ascii()

    def save_graph_image(self, path: str) -> None:
        try:
            image = self._graph.get_graph().draw_mermaid_png()
            with open(str(path), "wb") as f:
                f.write(image)
        except Exception as e:
            self._logger.error(e)

    def set_logger(self, logger) -> None:
        if (
            hasattr(logger, "debug") and callable(logger.debug) and
            hasattr(logger, "info") and callable(logger.info) and
            hasattr(logger, "warning") and callable(logger.warning) and
            hasattr(logger, "error") and callable(logger.error)
        ):
            self._logger = logger
        else:
            print("Invalid logger")
