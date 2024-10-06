# DM-aioaiagent

## Urls

* [PyPI](https://pypi.org/project/dm-aioaiagent)
* [GitHub](https://github.com/MykhLibs/dm-aioaiagent)

### * Package contains both `asynchronous` and `synchronous` clients

## Usage

Analogue to `DMAioAIAgent` is the synchronous client `DMAIAgent`.

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
    # you can set input_output_logging=False

    # define the conversation messages
    messages = [
       {"role": "user", "content": "Hello!"},
       {"role": "ai", "content": "How can I help you?"},
       {"role": "user", "content": "I want to know the weather in Kyiv"},
    ]

    # call an agent
    answer = await ai_agent.run(messages)


if __name__ == "__main__":
    asyncio.run(main())
```

### Return context of the tools with answer
```python
import asyncio
from dm_aioaiagent import DMAioAIAgent


async def main():
    # define a system message
    system_message = "Your custom system message with role, backstory and goal"

    # define a list of tools, if you want to use them
    tools = [...]

    # create an agent
    ai_agent = DMAioAIAgent(system_message, tools, return_context=True)

    # define the conversation messages
    messages = [
       {"role": "user", "content": "Hello!"},
       {"role": "ai", "content": "How can I help you?"},
       {"role": "user", "content": "I want to know the weather in Kyiv"},
    ]

    # start the agent
    state = await ai_agent.run(messages)

    # if you define tools, you can see the context of the tools
    answer = state["answer"]
    context = state["context"]


if __name__ == "__main__":
    asyncio.run(main())
```

### Set custom logger

_If you want set up custom logger_

```python
from dm_aioaiagent import DMAioAIAgent


# create custom logger
class MyLogger:
    def debug(self, message):
        pass

    def info(self, message):
        pass

    def warning(self, message):
        print(message)

    def error(self, message):
        print(message)


# create an agent
ai_agent = DMAioAIAgent()

# set up custom logger for this agent
ai_agent.set_logger(MyLogger())
```
