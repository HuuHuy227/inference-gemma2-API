from typing import  Dict,  List, Optional, Union

from typing_extensions import Literal, NotRequired, TypedDict

class PytorchGenerateConfig(TypedDict, total=False):
    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int
    stream: bool
    max_tokens: int
    echo: bool
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[Union[int, List[int]]]
    stream_interval: int
    model: Optional[str]
    tools: Optional[List[Dict]]
    lora_name: Optional[str]
    stream_options: Optional[Union[dict, None]]
    request_id: Optional[str]

class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]