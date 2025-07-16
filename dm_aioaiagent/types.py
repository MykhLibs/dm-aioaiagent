from typing import Literal, Union, Type, TypedDict, Callable, Coroutine, Any
from pydantic import BaseModel
from langchain_core.messages import BaseMessage

OutputSchemaType = Union[Type[TypedDict], Type[BaseModel], None]

BeforeToolCallCallback = Callable[[str, dict], None]
AfterToolCallCallback = Callable[[str, dict, str], None]
AsyncBeforeToolCallCallback = Callable[[str, dict], Coroutine[Any, Any, None]]
AsyncAfterToolCallCallback = Callable[[str, dict, str], Coroutine[Any, Any, None]]

class ImageMessageTextItem(TypedDict):
    type: Literal['text']
    text: str


class ImageMessageImageItem(TypedDict):
    type: Literal['image_url']
    image_url: dict


class ImageMessage(TypedDict):
    role: Literal["user"]
    content: list[Union[ImageMessageTextItem, ImageMessageImageItem]]


class TextMessage(TypedDict):
    role: Literal["user", "ai"]
    content: str


InputMessage = Union[TextMessage, ImageMessage, BaseMessage]


class State(TypedDict):
    messages: list[InputMessage]
    new_messages: list[BaseMessage]
