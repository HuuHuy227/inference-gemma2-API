# from typing import Optional, List, Iterator, Union
# from type import PytorchGenerateConfig, Response
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import logging
# logger = logging.getLogger(__name__)

# class ChatModel(): #PytorchModel, ChatModelMixin):
#     def __init__(self, model_id: str):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             torch_dtype=torch.float16,
#             device_map="auto",
#         )

#     def generate(
#         self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
#     ):
#         input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
#         if generate_config["do_sample"] == True:
#             temperature = generate_config["temperature"]
#             top_p = generate_config["top_p"]
#             top_k = generate_config["top_k"]
#         elif generate_config["do_sample"] == False:
#             temperature = None
#             top_p = None
#             top_k = None
        
#         generate_output = self.model.generate(
#             inputs = input_ids,
#             max_new_tokens=generate_config["max_tokens"],
#             eos_token_id = 1, 
#             do_sample = generate_config["do_sample"],
#             temperature = temperature,
#             top_p = top_p,
#             top_k = top_k 
#         )
#         prompt_length = input_ids.shape[-1]

#         response_ids = generate_output[:, prompt_length:]
#         response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         results = []
#         for i in range(len(response)):
#             eos_index = (response_ids[i] == self.tokenizer.eos_token_id).nonzero()
#             response_length = (eos_index[0].item() + 1) if len(eos_index) else len(response_ids[i])
#             results.append(
#                 Response(
#                     response_text=response[i],
#                     response_length=response_length,
#                     prompt_length=prompt_length,
#                     finish_reason="stop" if len(eos_index) else "length",
#                 )
#             )

#         return results

#     @torch.inference_mode()
#     def chat(
#         self,
#         prompt: str,
#         chat_history: Optional[List] = None,
#         generate_config: Optional[PytorchGenerateConfig] = None,
#     ):
#         full_prompt = self._get_full_prompt(prompt, chat_history)

#         res = self.generate(full_prompt, generate_config)
#         return res

#         # stream = generate_config.get("stream", False)
#         # if stream:
#         #     it = self.generate(full_prompt, generate_config)
#         #     assert isinstance(it, Iterator)
#         #     return self._to_chat_completion_chunks(it)
#         # else:
#         #     c = self.generate(full_prompt, generate_config)
#         #     assert not isinstance(c, Iterator)
#         #     return self._to_chat_completion(c)

#     # def load(self):
#     #     super().load()

#     # def _get_full_prompt(self, prompt, system_prompt, chat_history):
#     #     assert self.model_family.prompt_style is not None
#     #     prompt_style = self.model_family.prompt_style.copy()
#     #     if system_prompt:
#     #         prompt_style.system_prompt = system_prompt
#     #     chat_history = chat_history or []
#     #     full_prompt = ChatModelMixin.get_prompt(
#     #         prompt, chat_history, prompt_style
#     #     )
#     #     return full_prompt
    
#     def _get_full_prompt(self, prompt, chat_history) -> str:
#         chat_history = chat_history or []
#         chat_history.extend([
#             {"role": "user", "content": prompt},
#             {"role": "model", "content": ""}
#         ])
#         ret = ""
#         for message in chat_history:
#             content = message["content"]
#             role = message["role"] #get_role(message["role"])
#             ret += "<start_of_turn>" + role + "\n"
#             if content:
#                 ret += content + "<end_of_turn>\n"
#         return ret

#     # def prepare_batch_inference(self, req_list: List[InferenceRequest]):
#     #     super().prepare_batch_inference(req_list)
#     #     for r in req_list:
#     #         r.full_prompt = self._get_full_prompt(
#     #             r.prompt, r.system_prompt, r.chat_history, None
#     #         )

#     # def handle_batch_inference_results(self, req_list: List[InferenceRequest]):
#     #     for req in req_list:
#     #         if req.stream and req.error_msg is None:
#     #             if req.completion:
#     #                 results = []
#     #                 for i, c in enumerate(req.completion):
#     #                     if c == "<bos_stream>":
#     #                         results.append(
#     #                             self._get_first_chat_completion_chunk(
#     #                                 req.completion[i + 1]
#     #                             )
#     #                         )
#     #                     elif c == "<eos_stream>":
#     #                         break
#     #                     else:
#     #                         results.append(self._to_chat_completion_chunk(c))

#     #                 if req.stopped and req.include_usage:
#     #                     results.append(
#     #                         self._get_final_chat_completion_chunk(req.completion[-1])
#     #                     )
#     #                 req.completion = results

# Copyright 2024 THUDM and the LlamaFactory team.
#
# This code is inspired by the THUDM's ChatGLM implementation.
# https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence

from .utils import torch_gc
from .hf_engine import HuggingfaceEngine


if TYPE_CHECKING:
    from .base_engine import Response


def _start_background_loop(loop: "asyncio.AbstractEventLoop") -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ChatModel:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        # model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        self.engine = HuggingfaceEngine(
            model_path="Huy227/gemma2_vn",
            # enable_tensorizer=True,
            generate_config = {
                "max_tokens": 2048,
                "top_p":0.95,
                "top_k":40,
                "temperature":0.1,  
                "do_sample": True
            }
        )

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        **input_kwargs,
    ) -> List["Response"]:
        task = asyncio.run_coroutine_threadsafe(self.achat(messages, **input_kwargs), self._loop)
        return task.result()

    async def achat(
        self,
        messages: Sequence[Dict[str, str]],
        **input_kwargs,
    ) -> List["Response"]:
        return await self.engine.chat(messages, **input_kwargs)

    def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        **input_kwargs,
    ) -> Generator[str, None, None]:
        generator = self.astream_chat(messages, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        async for new_token in self.engine.stream_chat(messages, **input_kwargs):
            yield new_token

def run_chat() -> None:
    try:
        import platform

        if platform.system() != "Windows":
            import readline  # noqa: F401
    except ImportError:
        print("Install `readline` for a better experience.")

    chat_model = ChatModel()
    messages = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            torch_gc()
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "model", "content": response})