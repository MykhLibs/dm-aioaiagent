from typing import Optional, Literal, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class Message(TypedDict):
    role: Literal["user", "ai"]
    content: str


InputMessagesType = list[Union[Message, BaseMessage]]
ResponseType = Union[str, list[BaseMessage]]


class State(BaseModel):
    input_messages: InputMessagesType
    memory_id: Union[str, int, None] = Field(default=0)
    messages: Optional[list[BaseMessage]] = Field(default_factory=list)
    response: ResponseType = Field(default="")
