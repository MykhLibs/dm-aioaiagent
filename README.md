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

### Image generation and edit

The agent can also produce images. The mechanism differs by provider, so two flavours of model are supported:

#### OpenAI — `enable_image_generation=True`

Pass the flag to a normal chat-capable OpenAI model (`gpt-4.1`, `gpt-5`, etc.). Under the hood the agent enables the **Responses API** and binds OpenAI's built-in `image_generation` tool — the model decides on its own when to call it. Plain text turns stay text.

```python
from dm_aioaiagent import DMAIAgent, OutputImage

agent = DMAIAgent(model="gpt-4.1", enable_image_generation=True)

agent.run("Draw a small red square on a white background.")

# Generated images surface on agent.images
for i, img in enumerate(agent.images):
    img.save(f"out_{i}.png")
```

The same flag can be combined with regular tools — they coexist. `enable_image_generation=True` is **safe** even when the user only asks for text: the model uses `tool_choice="auto"`.

#### Gemini — image-output models

For Gemini you pick a model whose name contains `image` — e.g. `gemini-2.5-flash-image` (Nano Banana). The agent auto-injects `response_modalities=["IMAGE","TEXT"]` so the model is allowed to draw.

```python
agent = DMAIAgent(model="google_genai:gemini-2.5-flash-image")

agent.run("Generate a small red square.")
agent.images[0].save("out.png")
```

> **Heads up.** A Gemini image-output model is **not** a general chat model — it tends to draw on every turn, including plain greetings. For mixed workloads use a **two-agent pattern**: a chat agent with the image agent attached as a tool. See [`agent.as_tool()`](#agentas_tool) below.

#### Anthropic — vision only

Claude **cannot generate** images. If you pass `enable_image_generation=True` to a Claude model, the flag is silently ignored and a warning is logged. Image input (vision) works as usual.

### Working with generated images — `OutputImage`

Generated images live in `agent.images` as `OutputImage` instances:

```python
img = agent.images[0]
img.bytes        # raw image bytes
img.mime_type    # e.g. "image/png"
img.save("out.png")
img.to_base64()
```

You can also extract images directly from any `AIMessage`:

```python
from dm_aioaiagent import OutputImage
images = OutputImage.extract_from(response_message)  # list[OutputImage]
```

### Image memory modes

Images in `agent.memory_messages` (the conversation history sent to the LLM on each turn) and in `agent.images` (the property exposing AI-generated images) follow the `image_memory_mode` constructor argument:

| Mode | Memory (history) | `agent.images` |
|---|---|---|
| `keep_last` *(default)* | last user-image kept; last AI-image kept; older → `[image]` / `[generated image]` placeholder | last AI-image kept; replaced when a new one arrives |
| `drop` | every image (user + AI) becomes a placeholder right after the turn | only the AI-image of the **current** turn (then wiped on the next call) |
| `keep_all` | nothing is stripped — full multimodal history | every AI-image accumulates |

```python
agent = DMAIAgent(model="gpt-4o-mini", image_memory_mode="keep_last")
agent.run_messages([InputImage.from_file("photo.png", text="Describe.")])
agent.run("What colour was dominant?")  # answers based on the image
```

`agent.clear_memory_messages()` clears both `memory_messages` and `images`.

> Only **AI-generated** images populate `agent.images`. Images you upload via `InputImage` go into history per the rules above but are not exposed on the `images` property.

### `agent.as_tool()`

Wrap any agent as a `StructuredTool` so a *parent* agent can call it like any other tool — the basis for multi-agent composition. Default name is derived from `agent_name` (lowercased, non-alphanumerics replaced with `_`); `description` is required.

```python
from dm_aioaiagent import DMAIAgent

# specialised image agent
image_agent = DMAIAgent(
    agent_name="image_drawer",
    model="google_genai:gemini-2.5-flash-image",
)

# chat agent that delegates drawing to the image agent
chat_agent = DMAIAgent(
    model="google_genai:gemini-2.5-flash",
    tools=[image_agent.as_tool(description="Generates an image from a text prompt.")],
)

chat_agent.run("Hi! Please draw a small red square.")
# the chat agent picks the tool, the image agent draws, image lands in image_agent.images
image_agent.images[0].save("out.png")
```

The async client (`DMAioAIAgent.as_tool`) returns a tool with both `func` and `coroutine` set, so it can be invoked from sync or async parent agents.
