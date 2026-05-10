import os
import uuid
from typing import Any, Literal
from pydantic import SecretStr
from itertools import dropwhile
from threading import Thread
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph
from dm_logger import DMLogger

from .output_image import OutputImage
from .types import *


ImageMemoryMode = Literal["drop", "keep_last", "keep_all"]
_VALID_IMAGE_MEMORY_MODES = ("drop", "keep_last", "keep_all")


INVALID_IMAGE_ERROR_MARKERS = (
    "invalid_image_url",
    "Could not process image",
    "Unable to process input image",
    "INVALID_ARGUMENT",
)


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
        # multimodal
        enable_image_generation: bool = False,
        image_memory_mode: ImageMemoryMode = "keep_last",
        # other
        input_output_logging: bool = True,
        node_execution_logging: bool = True,
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
        # multimodal
        self._enable_image_generation = bool(enable_image_generation)
        self._image_memory_mode = self._validate_image_memory_mode(image_memory_mode)
        self._images: list[OutputImage] = []
        # other
        self._input_output_logging = bool(input_output_logging)
        self._node_execution_logging = bool(node_execution_logging)
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
            return self._extract_text(last_message)
        return last_message

    @staticmethod
    def _extract_text(message: AIMessage) -> str:
        # AIMessage.content may be a plain string (legacy) or a list of standard
        # v1 content blocks (multimodal). For non-string content collect every
        # text block; if there are none, return empty string.
        if isinstance(message.content, str):
            return message.content
        return "".join(b["text"] for b in message.content_blocks if b.get("type") == "text")

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

    @property
    def images(self) -> tuple[OutputImage, ...]:
        return tuple(self._images)

    def clear_memory_messages(self) -> None:
        self._memory_messages.clear()
        self._images.clear()

    def _prepare_messages_node(self, state: State) -> State:
        messages = state["messages"] or [{"role": "user", "content": ""}]
        state["messages"] = []
        for item in messages:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if not role or role not in self._ALLOWED_ROLES or not content:
                    self._logger.debug("Skipped malformed input dict", role=role, content_type=type(content).__name__)
                    continue
                # content may be a plain str (legacy) or a list of v1 standard
                # content blocks (multimodal: text + image/audio/video/file).
                # Both shapes are accepted by HumanMessage/AIMessage as-is.
                MessageClass = AIMessage if role == "ai" else HumanMessage
                state["messages"].append(MessageClass(content=content))
            elif isinstance(item, BaseMessage):
                state["messages"].append(item)

        if self._input_output_logging and state["messages"]:
            self._logger.debug(f'Query:\n{state["messages"][-1].content}')
        if self._is_memory_enabled:
            state["messages"] = self._memory_messages + state["messages"]
        return state

    def _invoke_llm_node(self, state: State, second_attempt: bool = False) -> State:
        if self._node_execution_logging:
            self._logger.debug("Run node: Invoke LLM")
        try:
            ai_response = self._agent.invoke({"messages": state["messages"]})
        except Exception as e:
            self._logger.error(e)
            if second_attempt:
                err_str = str(e)
                if any(m in err_str for m in INVALID_IMAGE_ERROR_MARKERS):
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
        if self._node_execution_logging:
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
                        self._logger.error(e, callback_type="before_tool_call")

                    try:
                        self._logger.debug("Invoke tool", tool_id=tool_id, tool_name=tool_name, tool_args=tool_args)
                        tool_response = self._tool_map[tool_name].run(tool_args)
                    except Exception as e:
                        self._logger.error(e, tool_id=tool_id)
                        tool_response = "Tool executed with an error!"

                    try:
                        if self._after_tool_call_callback:
                            self._after_tool_call_callback(tool_name, tool_args, tool_response)
                    except Exception as e:
                        self._logger.error(e, callback_type="after_tool_call")
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
        base_kwargs = {"model": self._model, "output_version": "v1"}
        if self._temperature is not None:
            if not isinstance(self._temperature, (int, float)):
                raise ValueError("Temperature must be a float value.")
            base_kwargs["temperature"] = float(self._temperature)
        if self._llm_provider_api_key:
            base_kwargs["api_key"] = SecretStr(self._llm_provider_api_key)
        if self._llm_provider_base_url:
            base_kwargs["base_url"] = self._llm_provider_base_url

        # Pre-init: OpenAI image generation needs the Responses API.
        # Detect the requested provider from the model string (init_chat_model
        # itself does the same) so we can flip use_responses_api before init.
        if self._enable_image_generation and self._wants_openai_provider(self._model):
            base_kwargs["use_responses_api"] = True

        # Pre-init: Gemini image-output models must opt into the IMAGE modality.
        if self._is_gemini_image_model(self._model):
            base_kwargs["response_modalities"] = ["IMAGE", "TEXT"]

        llm = init_chat_model(**base_kwargs)

        provider = self._get_provider(llm)

        if self._enable_image_generation and self._output_schema:
            self._logger.warning(
                "output_schema disables tools — enable_image_generation will be ignored."
            )
        if self._enable_image_generation and provider == "anthropic":
            self._logger.warning("Claude does not support image generation; the flag is ignored.")

        if provider == "anthropic":
            bind_tool_kwargs = {"tool_choice": {"type": "auto"}}
            if isinstance(self._parallel_tool_calls, bool):
                bind_tool_kwargs["tool_choice"]["disable_parallel_tool_use"] = not self._parallel_tool_calls
        elif provider == "openai":
            bind_tool_kwargs = {}
            if isinstance(self._parallel_tool_calls, bool):
                bind_tool_kwargs["parallel_tool_calls"] = self._parallel_tool_calls
        else:
            bind_tool_kwargs = {}

        if self._is_tools_exists:
            self._tool_map = {t.name: t for t in self._tools}

        tools_to_bind: list = list(self._tools)
        if self._enable_image_generation and provider == "openai" and not self._output_schema:
            tools_to_bind.append({"type": "image_generation"})

        if tools_to_bind:
            llm = llm.bind_tools(tools_to_bind, **bind_tool_kwargs)

        if self._output_schema:
            llm = llm.with_structured_output(self._output_schema)

        prompt = ChatPromptTemplate.from_messages([SystemMessage(content=self._system_message),
                                                   MessagesPlaceholder(variable_name="messages")])
        self._agent = prompt | llm

    @staticmethod
    def _get_provider(llm) -> str:
        # Derive provider tag from the chat-model class name: strip "Chat" prefix
        # and lowercase. Vertex AI is aliased to Google Generative AI — they are
        # the same provider for our branching purposes.
        name = type(llm).__name__
        if name.startswith("Chat"):
            name = name[4:]
        provider = name.lower()
        if provider == "vertexai":
            provider = "googlegenerativeai"
        return provider

    @staticmethod
    def _wants_openai_provider(model: str) -> bool:
        # Used pre-init for kwargs that must be set on the constructor
        # (use_responses_api). OpenAI prefixes are stable; other providers go
        # through post-init _get_provider().
        if ":" in model:
            return model.split(":", 1)[0].lower() == "openai"
        m = model.lower()
        return m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))

    @staticmethod
    def _is_gemini_image_model(model: str) -> bool:
        # Model names like "gemini-2.5-flash-image" or
        # "gemini-2.0-flash-preview-image-generation" — must opt into the
        # IMAGE modality at init time, otherwise the model still answers in
        # text only.
        m = model.lower()
        if ":" in m:
            m = m.split(":", 1)[1]
        return m.startswith("gemini") and "image" in m

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
    def _validate_image_memory_mode(mode: str) -> str:
        if mode in _VALID_IMAGE_MEMORY_MODES:
            return mode
        raise ValueError(
            f"image_memory_mode must be one of {_VALID_IMAGE_MEMORY_MODES}, got {mode!r}."
        )

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
