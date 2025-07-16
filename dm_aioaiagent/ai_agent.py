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


class DMAIAgent:
    MAX_MEMORY_MESSAGES = 20  # Only INT greater than 0
    _ALLOWED_ROLES = ("user", "ai")

    def __init__(
        self,
        # general
        system_message: str = "You are a helpful assistant.",
        model: str = "gpt-4o-mini",
        temperature: float = None,
        output_schema: OutputSchemaType = None,  # IF you set output_schema, tools and memory will be disabled
        agent_name: str = "AIAgent",
        # tools
        tools: list[BaseTool] = None,
        parallel_tool_calls: bool = None,
        before_tool_call_callback: BeforeToolCallCallback = None,
        after_tool_call_callback: AfterToolCallCallback = None,
        # memory
        is_memory_enabled: bool = True,
        save_tools_responses_in_memory: bool = True,
        max_memory_messages: int = MAX_MEMORY_MESSAGES,
        # other
        input_output_logging: bool = True,
        response_if_request_fail: str = "I can't provide a response right now. Please try again later.",
        response_if_invalid_image: str = "The image is unavailable or the link is incorrect.",
        # llm provider
        llm_provider_api_key: str = "",
        llm_provider_base_url: str = ""
    ):
        self._logger = DMLogger(agent_name)

        # general
        self._system_message = str(system_message)
        self._model = str(model)
        self._temperature = temperature
        self._output_schema = self._validate_output_schema(output_schema)
        if self._output_schema:
            tools = []
            parallel_tool_calls = None
            is_memory_enabled = False
        # tools
        self._tools = tools or []
        self._is_tools_exists = bool(tools)
        self._parallel_tool_calls = parallel_tool_calls
        self._before_tool_call_callback = before_tool_call_callback
        self._after_tool_call_callback = after_tool_call_callback
        # memory
        self._memory_messages = []
        self._is_memory_enabled = bool(is_memory_enabled)
        self._save_tools_responses_in_memory = bool(save_tools_responses_in_memory)
        self._max_memory_messages = self._validate_max_memory_messages(max_memory_messages)
        # other
        self._input_output_logging = bool(input_output_logging)
        self._response_if_request_fail = str(response_if_request_fail)
        self._response_if_invalid_image = str(response_if_invalid_image)
        # llm provider
        self._llm_provider_api_key = str(llm_provider_api_key)
        self._llm_provider_base_url = str(llm_provider_base_url)

        self._check_langsmith_envs()
        self._init_agent()
        self._init_graph()

    def run(self, query: str, **kwargs) -> Union[str, TypedDict, BaseModel]:
        new_messages = self.run_messages(messages=[TextMessage(role="user", content=query)], **kwargs)

        last_message = new_messages[-1]
        if isinstance(last_message, AIMessage):
            return last_message.content
        return last_message

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
        state = self._graph.invoke(
            input={"messages": messages, "new_messages": []},
            config={k: v for k, v in config_data.items() if v}
        )
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
                if "invalid_image_url" in str(e):
                    response = self._response_if_invalid_image
                else:
                    response = self._response_if_request_fail

                ai_response = AIMessage(content=response)
                state["messages"].append(ai_response)
                state["new_messages"].append(ai_response)
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
                if tool_name in self._tool_map:
                    try:
                        if self._before_tool_call_callback:
                            self._before_tool_call_callback(tool_name, tool_args)
                    except Exception as e:
                        self._logger.error(str(e), callback_type="before_tool_call")

                    try:
                        self._logger.debug("Invoke tool", tool_id=tool_id, tool_name=tool_name, tool_args=tool_args)
                        tool_response = self._tool_map[tool_name].run(tool_args)
                    except Exception as e:
                        self._logger.error(str(e), tool_id=tool_id)
                        tool_response = "Tool executed with an error!"

                    try:
                        if self._after_tool_call_callback:
                            self._after_tool_call_callback(tool_name, tool_args, tool_response)
                    except Exception as e:
                        self._logger.error(str(e), callback_type="after_tool_call")
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
            answer = state["messages"][-1]
            if isinstance(answer, AIMessage):
                answer = answer.content
            self._logger.debug(f'Answer:\n{answer}')

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
        if self._output_schema:
            return "exit"
        elif self._is_tools_exists and state["messages"][-1].tool_calls:
            return "execute_tool"
        return "exit"

    def _init_agent(self) -> None:
        base_kwargs = {"model": self._model}
        if isinstance(self._temperature, float):
            base_kwargs["temperature"] = self._temperature
        else:
            ValueError("Temperature must be a float value.")
        if self._llm_provider_api_key:
            base_kwargs["api_key"] = SecretStr(self._llm_provider_api_key)
        if self._llm_provider_base_url:
            base_kwargs["base_url"] = self._llm_provider_base_url

        if self._model.startswith("claude"):
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(**base_kwargs)
            bind_tool_kwargs = {"tool_choice": {"type": "auto"}}
            if isinstance(self._parallel_tool_calls, bool):
                bind_tool_kwargs["tool_choice"]["disable_parallel_tool_use"] = not self._parallel_tool_calls
        else:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(**base_kwargs)
            bind_tool_kwargs = {}
            if isinstance(self._parallel_tool_calls, bool):
                bind_tool_kwargs["parallel_tool_calls"] = self._parallel_tool_calls

        if self._is_tools_exists:
            self._tool_map = {t.name: t for t in self._tools}
            llm = llm.bind_tools(self._tools, **bind_tool_kwargs)

        if self._output_schema:
            llm = llm.with_structured_output(self._output_schema)

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

    @staticmethod
    def _validate_output_schema(schema: OutputSchemaType) -> OutputSchemaType:
        if schema is None:
            return None
        if isinstance(schema, type) and \
            (type(schema).__name__ == "_TypedDictMeta" or issubclass(schema, BaseModel)):
            return schema
        raise ValueError("Output schema must be a TypedDict or BaseModel type, or None.")

    def print_graph(self) -> None:
        self._graph.get_graph().print_ascii()

    def save_graph_image(self, path: str) -> None:
        try:
            image = self._graph.get_graph().draw_mermaid_png()
            with open(str(path), "wb") as f:
                f.write(image)
        except Exception as e:
            self._logger.error(e)
