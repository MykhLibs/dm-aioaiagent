from typing import Literal, Union, Type, TypedDict
from pydantic import BaseModel
from langchain_core.messages import BaseMessage

OutputSchemaType = Union[Type[TypedDict], Type[BaseModel], None]


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
