from typing import Literal, Union
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class ImageMessageTextMessage(TypedDict):
    type: Literal['text']
    text: str


class ImageMessageImageItem(TypedDict):
    type: Literal['image_url']
    image_url: dict


class ImageMessage(TypedDict):
    role: Literal["user"]
    content: list[Union[ImageMessageTextMessage, ImageMessageImageItem]]


class TextMessage(TypedDict):
    role: Literal["user", "ai"]
    content: str


InputMessagesType = list[Union[TextMessage, ImageMessage, BaseMessage]]

class State(TypedDict):
    messages: InputMessagesType
    new_messages: list[BaseMessage]
