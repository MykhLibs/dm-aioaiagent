from typing import Optional, Literal, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class ImageMessageTextMessage(TypedDict):
    type: Literal['text']
    text: str


class ImageMessageImageItem(TypedDict):
    type: Literal['image_url']
    image_url: dict


ImageMessageContent = list[Union[ImageMessageTextMessage, ImageMessageImageItem]]


class ImageMessage(TypedDict):
    role: Literal["user"]
    content: ImageMessageContent


class TextMessage(TypedDict):
    role: Literal["user", "ai"]
    content: str


Message = Union[TextMessage, ImageMessage]

InputMessagesType = list[Union[Message, BaseMessage]]

ResponseType = Union[str, list[BaseMessage]]


class State(BaseModel):
    input_messages: InputMessagesType
    memory_id: Union[str, int, None] = Field(default=0)
    messages: Optional[list[BaseMessage]] = Field(default_factory=list)
    response: ResponseType = Field(default="")
