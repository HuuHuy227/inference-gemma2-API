from dataclasses import dataclass

import time
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    # SYSTEM = "system"
    # FUNCTION = "function"
    # TOOL = "tool"


@unique
class Finish(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL = "tool_calls"

@dataclass
class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]

class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: Literal["owner"] = "owner"

class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard] = []

class Function(BaseModel):
    name: str
    arguments: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class FunctionAvailable(BaseModel):
    type: Literal["function", "code_interpreter"] = "function"
    function: Optional[FunctionDefinition] = None


class FunctionCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Function

class ChatMessage(BaseModel):
    role: Role
    content: Optional[str] = None
    tool_calls: Optional[List[FunctionCall]] = None


class ChatCompletionMessage(BaseModel):
    role: Optional[Role] = None
    content: Optional[str] = None
    tool_calls: Optional[List[FunctionCall]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[FunctionAvailable]] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Finish


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: ChatCompletionMessage
    finish_reason: Optional[Finish] = None


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

