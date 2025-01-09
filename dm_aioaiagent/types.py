from typing import Literal, Union
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


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
