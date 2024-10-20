from typing import Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class Message(TypedDict):
    role: Literal["user", "ai"]
    content: str


class InnerState(BaseModel):
    messages: list[BaseMessage] = Field(default=[])
    context: list[Message] = Field(default=[])


class InputState(BaseModel):
    messages: list[Message]
    inner_state: Optional[InnerState] = Field(default=InnerState())


class OutputState(TypedDict):
    answer: str
    context: list[Message]
