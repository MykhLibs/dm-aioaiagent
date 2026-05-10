# DM-aioaiagent

## Urls

* [PyPI](https://pypi.org/project/dm-aioaiagent)
* [GitHub](https://github.com/MykhLibs/dm-aioaiagent)

### * Package contains both `asynchronous` and `synchronous` clients

## Installation

By default, the package ships with **OpenAI** support. Other providers are optional extras:

```bash
pip install dm-aioaiagent                       # OpenAI only
pip install dm-aioaiagent[anthropic]            # + Anthropic
pip install dm-aioaiagent[anthropic,gemini]     # several at once
pip install dm-aioaiagent[all]                  # every supported provider
```

Available extras: `anthropic`, `gemini`, `groq`, `mistral`, `deepseek`, `ollama`, `all`.

If you call a model from a provider whose package is not installed, `init_chat_model` will raise an `ImportError` with the exact `pip install` command you need.

## Providers

Provider resolution is delegated to LangChain's [`init_chat_model`](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) — the agent picks the provider automatically by model name prefix when possible. For everything else, use the `"provider:model"` mask.

```python
# Auto-detected from model prefix (rules come from LangChain's init_chat_model)
agent = DMAioAIAgent(model="gpt-4o-mini")              # → openai
agent = DMAioAIAgent(model="claude-3-5-sonnet-latest") # → anthropic
agent = DMAioAIAgent(model="gemini-2.0-flash")         # → google_vertexai (see note below)

# Explicit provider via "provider:model" mask
agent = DMAioAIAgent(model="google_genai:gemini-2.0-flash")
agent = DMAioAIAgent(model="groq:llama-3.1-70b-versatile")
agent = DMAioAIAgent(model="mistralai:mistral-large-latest")
agent = DMAioAIAgent(model="deepseek:deepseek-chat")
agent = DMAioAIAgent(model="ollama:llama3.1")

# OpenAI-compatible gateway (OpenRouter, Together, vLLM, LiteLLM proxy, ...)
# Works without installing any extra — just point to the OpenAI-compatible URL.
agent = DMAioAIAgent(
    model="meta-llama/llama-3.1-70b-instruct",
    llm_provider_base_url="https://openrouter.ai/api/v1",
    llm_provider_api_key="sk-or-...",
)
```

> **Note about Gemini.** LangChain's auto-detect maps the `gemini*` prefix to **`google_vertexai`** (Google Cloud Vertex AI, requires a GCP service account). If you have a regular **Google AI Studio** API key (`GOOGLE_API_KEY`), use the `google_genai:` mask explicitly:
>
> ```python
> agent = DMAioAIAgent(model="google_genai:gemini-2.0-flash")
> ```

Supported provider keys for the `"provider:model"` mask (list inherited from LangChain): `openai`, `anthropic`, `azure_openai`, `azure_ai`, `google_vertexai`, `google_genai`, `bedrock`, `bedrock_converse`, `cohere`, `fireworks`, `together`, `mistralai`, `huggingface`, `groq`, `ollama`, `google_anthropic_vertex`, `deepseek`, `ibm`, `nvidia`, `xai`, `perplexity`.

### Note about parallel tool calls

`parallel_tool_calls` is currently mapped only for **OpenAI** and **Anthropic** (their APIs use different formats). For other providers the parameter is silently ignored — extend per-provider mapping if you need it.

## Usage

Analogue to `DMAioAIAgent` is the synchronous client `DMAIAgent`.

### Windows Setup

```python
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

### Api Key Setup

Each provider reads its API key from a dedicated environment variable, e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, etc. Alternatively, pass the key explicitly via the `llm_provider_api_key` argument — useful for multi-tenant setups, custom gateways, or runtime key rotation.

**Use load_dotenv to load the `.env` file.**

```python
from dotenv import load_dotenv
load_dotenv()
```

### Use agent *with* inner memory and run *single* message

By default, agent use inner memory to store the conversation history.

(You can set *max count messages in memory* by `max_memory_messages` init argument)

```python
import asyncio
from dm_aioaiagent import DMAioAIAgent


async def main():
    # define a system message
    system_message = "Your custom system message with role, backstory and goal"

    # (optional) define a list of tools, if you want to use them
    tools = [...]

    # define a openai model, default is "gpt-4o-mini"
    model_name = "gpt-4o"

    # create an agent
    ai_agent = DMAioAIAgent(system_message, tools, model=model_name)
    # if you don't want to see the input and output messages from agent
    # you can set `input_output_logging=False` init argument

    # call an agent
    answer = await ai_agent.run("Hello!")

    # call an agent
    answer = await ai_agent.run("I want to know the weather in Kyiv")

    # get full conversation history
    conversation_history = ai_agent.memory_messages

    # clear conversation history
    ai_agent.clear_memory_messages()


if __name__ == "__main__":
    asyncio.run(main())
```

### Use agent *without* inner memory and run *multiple* messages

If you want to control the memory of the agent, you can disable it by setting `is_memory_enabled=False`

```python
import asyncio
from dm_aioaiagent import DMAioAIAgent


async def main():
    # define a system message
    system_message = "Your custom system message with role, backstory and goal"

    # (optional) define a list of tools, if you want to use them
    tools = [...]

    # define a openai model, default is "gpt-4o-mini"
    model_name = "gpt-4o"

    # create an agent
    ai_agent = DMAioAIAgent(system_message, tools, model=model_name,
                            is_memory_enabled=False)
    # if you don't want to see the input and output messages from agent
    # you can set input_output_logging=False

    # define the conversation message(s)
    messages = [
        {"role": "user", "content": "Hello!"}
    ]

    # call an agent
    new_messages = await ai_agent.run_messages(messages)

    # add new_messages to messages
    messages.extend(new_messages)

    # define the next conversation message
    messages.append(
        {"role": "user", "content": "I want to know the weather in Kyiv"}
    )

    # call an agent
    new_messages = await ai_agent.run_messages(messages)


if __name__ == "__main__":
    asyncio.run(main())
```

### Working with images — input

Use the `InputImage` helper to attach an image to a user message in a way that works **across providers** (OpenAI, Anthropic, Gemini). Each factory returns a ready-to-send `HumanMessage` whose `.content` is a list of LangChain v1 standard content blocks.

```python
from dm_aioaiagent import DMAIAgent, InputImage

agent = DMAIAgent(agent_name="image_vision", model="gpt-4o-mini")

# from a local file (mime type inferred from extension)
msg_file = InputImage.from_file("photo.png", text="What is in the picture?")

# from a remote URL
msg_url = InputImage.from_url("https://your.domain/image.png", text="Describe it.")

# from raw bytes / base64 (mime_type required)
with open("photo.png", "rb") as f:
    msg_bytes = InputImage.from_bytes(f.read(), mime_type="image/png", text="Describe.")
msg_b64 = InputImage.from_base64("aGVsbG8=", mime_type="image/png")

answer = agent.run_messages([msg_file])
print(answer[-1].content_blocks)  # list of standard blocks
```

**Multiple images per turn.** Each factory builds **one** image message. To attach several images to a single user turn, pass several messages:

```python
messages = [
    InputImage.from_file("front.png", text="Compare these two views:"),
    InputImage.from_file("back.png"),
]
agent.run_messages(messages)
```

> **`from_url` caveats.** Some providers (notably Anthropic and Gemini) may have stricter rules about remote URLs (allowed hosts, public reachability, redirects). When in doubt — read the file yourself and use `from_file` / `from_bytes`.
