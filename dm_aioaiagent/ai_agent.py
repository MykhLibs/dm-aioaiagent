import os
from itertools import dropwhile
from threading import Thread
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph
from dm_logger import DMLogger

from .types import *

__all__ = ["DMAIAgent"]


class DMAIAgent:
    agent_name = "AIAgent"
    _allowed_roles = ("user", "ai")
    MAX_MEMORY_MESSAGES = 20  # Only INT greater than 0

    def __init__(
        self,
        system_message: str = "You are a helpful assistant.",
        tools: list[BaseTool] = None,
        *,
        model: str = "gpt-4o-mini",
        temperature: int = 1,
        agent_name: str = None,
        input_output_logging: bool = True,
        is_memory_enabled: bool = True,
        max_memory_messages: int = None,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("'OPENAI_API_KEY' environment variable is not set!")

        self._logger = DMLogger(agent_name or self.agent_name)
        self._is_tools_exists = bool(tools)
        self._input_output_logging = bool(input_output_logging)
        self._is_memory_enabled = bool(is_memory_enabled)
        self._max_memory_messages = self._validate_max_memory_messages(max_memory_messages)

        prompt = ChatPromptTemplate.from_messages([SystemMessage(content=system_message),
                                                   MessagesPlaceholder(variable_name="messages")])
        llm = ChatOpenAI(model=str(model), temperature=int(temperature))
        if self._is_tools_exists:
            self._tool_map = {t.name: t for t in tools}
            llm = llm.bind_tools(tools)
        self._agent = prompt | llm
        self._memory = {}

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

    def run(self, input_messages: InputMessagesType, memory_id: str = None) -> ResponseType:
        state = self._graph.invoke({"input_messages": input_messages, "memory_id": memory_id})
        return state["response"]

    def get_memory_messages(
        self,
        memory_id: str = None,
        *,
        without_tool_m: bool = False,
        return_str_m: bool = False
    ) -> Union[list[BaseMessage], list[str]]:
        messages = self._memory.get(self._validate_memory_id(memory_id), [])
        if without_tool_m:
            messages = [m for m in messages if not (m.type == "tool" or (m.type == "ai" and m.tool_calls))]
        if return_str_m:
            messages = [m.content for m in messages]
        return messages

    def clear_memory(self, memory_id: str = None) -> None:
        self._memory[self._validate_memory_id(memory_id)] = []

    def _prepare_messages_node(self, state: State) -> State:
        state.memory_id = self._validate_memory_id(state.memory_id)
        state.input_messages = state.input_messages or [{"role": "user", "content": ""}]
        for item in state.input_messages:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if not role or role not in self._allowed_roles or not content:
                    continue
                if role == "ai":
                    MessageClass = AIMessage
                else:
                    MessageClass = HumanMessage
                state.messages.append(MessageClass(content))
            elif isinstance(item, BaseMessage):
                state.messages.append(item)

        if self._input_output_logging:
            self._logger.debug(f"Query:\n{state.messages[-1].content}", memory_id=state.memory_id)
        if self._is_memory_enabled:
            state.messages = self.get_memory_messages(state.memory_id) + state.messages
        return state

    def _invoke_llm_node(self, state: State) -> State:
        self._logger.debug("Run node: Invoke LLM")
        ai_response = self._agent.invoke({"messages": state.messages})
        state.messages.append(ai_response)
        return state

    def _execute_tool_node(self, state: State) -> State:
        self._logger.debug("Run node: Execute tool")
        threads = []
        for tool_call in state.messages[-1].tool_calls:
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
                state.messages.append(tool_message)

            threads.append(Thread(target=tool_callback, daemon=True))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return state

    def _exit_node(self, state: State) -> State:
        answer = state.messages[-1].content
        if self._input_output_logging:
            self._logger.debug(f"Answer:\n{answer}", memory_id=state.memory_id)

        if self._is_memory_enabled:
            messages_to_memory = state.messages[-self._max_memory_messages:]
            # drop ToolsMessages from start of list
            self._memory[state.memory_id] = list(dropwhile(lambda x: isinstance(x, ToolMessage), messages_to_memory))
            state.response = answer
        else:
            state.response = state.messages[len(state.input_messages):]
        return state

    def _messages_router(self, state: State) -> str:
        if self._is_tools_exists and state.messages[-1].tool_calls:
            route = "execute_tool"
        else:
            route = "exit"
        return route

    @staticmethod
    def _validate_memory_id(memory_id: Union[str, None]) -> Union[str, int]:
        return str(memory_id) if memory_id else 0

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
