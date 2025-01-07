# DM-aioaiagent

## Urls

* [PyPI](https://pypi.org/project/dm-aioaiagent)
* [GitHub](https://github.com/MykhLibs/dm-aioaiagent)

### * Package contains both `asynchronous` and `synchronous` clients

## Usage

Analogue to `DMAioAIAgent` is the synchronous client `DMAIAgent`.

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

### Image vision

```python
from dm_aioaiagent import DMAIAgent, OpenAIImageMessageContent


def main():
    # create an agent
    ai_agent = DMAIAgent(agent_name="image_vision", model="gpt-4o")

    # create an image message content
    # NOTE: text argument is optional
    img_content = OpenAIImageMessageContent(image_url="https://your.domain/image",
                                            text="Hello, what is shown in the photo?")

    # define the conversation messages
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "user", "content": img_content},
    ]

    # call an agent
    new_messages = ai_agent.run_messages(messages)
    answer = new_messages[-1].content


if __name__ == "__main__":
    main()
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
