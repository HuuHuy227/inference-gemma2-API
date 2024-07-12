import asyncio
import concurrent.futures
import os
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple, Union
import json
import logging

import torch
from transformers import GenerationConfig, TextIteratorStreamer
from utils import (
    get_device_preferred_dtype,
    gpu_count,
    is_hf_accelerate_supported,
    select_device
)

from type import Response #BaseEngine, Response
# from type import PytorchGenerateConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

class HuggingfaceEngine(): #BaseEngine):
    def __init__(
        self,
        model_path: str,
        enable_tensorizer: bool = False,
        generate_config: Dict[str, Any] = None #Optional[PytorchGenerateConfig] = None,
    ) -> None: 
        
        self.generating_args = generate_config
        self.enable_tensorizer = enable_tensorizer
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            logger.warning("There is no current event loop, creating a new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.model_path = model_path
        self.semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT", "1")))
        
        logger.info(f"Loading model {self.model_path}")
        self._model, self._tokenizer =  self.load() #Loading tokenizer and model

    def load(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
            )
        # quantization = self.quantization
        num_gpus = gpu_count()
        device = "auto"
        self._device = select_device(device)

        kwargs = {}

        dtype = get_device_preferred_dtype(self._device)

        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        else:
            raise ValueError(f"Device {self._device} is not supported in temporary")

        is_device_map_auto = False

        # This is required for Intel GPU to actually work with accelerate device_map until
        # https://github.com/intel/intel-extension-for-pytorch/issues/522
        # is resolved
        max_memory_env = os.getenv("ACCELERATE_MAX_MEMORY", None)

        if max_memory_env is not None:
            max_memory_raw = json.loads(max_memory_env)
            max_memory = {
                int(k) if k.isdigit() else k: max_memory_raw[k] for k in max_memory_raw
            }
            kwargs["max_memory"] = max_memory

        if num_gpus > 0 and is_hf_accelerate_supported(self._device):
            kwargs.update({"device_map": "auto"})
            is_device_map_auto = True

        if self._check_tensorizer_integrity():
            model, tokenizer = self._load_tensorizer(**kwargs)
        else:
            model, tokenizer = self._load_model(**kwargs)

        if not is_device_map_auto:
            model.to(self._device)

        self._save_tensorizer(**kwargs)

        logger.debug(f"Model Memory: {model.get_memory_footprint()}")

        return model, tokenizer

    def _check_tensorizer_integrity(self):
        if not self.enable_tensorizer:
            return False

        from .tensorizer_utils import check_tensorizer_integrity

        integrity = check_tensorizer_integrity(
            self.model_path,
            [component[0] for component in self._get_components()],
        )
        logger.info(f"Tensorizer files integrity: {integrity} gemma2")
        return integrity

    def _load_tensorizer(self, **kwargs):
        enable_tensorizer = self.enable_tensorizer
        if enable_tensorizer:
            from .tensorizer_utils import load_from_tensorizer

            component_metadata = [
                (name, type, kwargs)
                for name, _, type, kwargs in self._get_components(**kwargs)
            ]
            model, tokenizer = load_from_tensorizer(
                self.model_path, component_metadata, self._get_model_class(), **kwargs
            )
            return model, tokenizer

    def _save_tensorizer(self, **kwargs):
        enable_tensorizer = self.enable_tensorizer
        if enable_tensorizer:
            from .tensorizer_utils import save_to_tensorizer

            components = [(name, obj) for name, obj, _, _ in self._get_components()]
            save_to_tensorizer(self.model_path, self._model, components, **kwargs)

    def _get_model_class(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM
    
    def _get_components(self, **kwargs):
        from transformers import AutoTokenizer

        return [
            (
                "tokenizer",
                getattr(self, "_tokenizer", None),
                AutoTokenizer,
                {
                    "use_fast": True,
                    "trust_remote_code": kwargs.get("trust_remote_code", True),
                    "revision": kwargs.get("revision"),
                    "code_revision": kwargs.get("code_revision", None),
                },
            )
        ]

    def _load_model(self, **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                    "Please make sure 'transformers' is installed. ",
                    "You can install it by `pip install transformers`\n",
                ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast = True
            )
        model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                **kwargs,
            )

        return model, tokenizer

    @staticmethod
    def _get_full_prompt(chat_history) -> str:
        # chat_history = chat_history or []
        # chat_history.extend([
        #     {"role": "user", "content": prompt},
        #     {"role": "model", "content": ""}
        # ])
        ret = ""
        for message in chat_history:
            content = message["content"]
            role = message["role"] #get_role(message["role"])
            ret += "<start_of_turn>" + role + "\n"
            if content:
                ret += content + "<end_of_turn>\n"
        return ret

    @staticmethod
    def _process_args(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        generating_args: Dict[str, Any],
        messages: Sequence[Dict[str, str]],
        input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Tuple[Dict[str, Any], int]:
        
        prompt_ids = HuggingfaceEngine._get_full_prompt(messages)

        # inputs = torch.tensor([prompt_ids], device=model.device)
        inputs = tokenizer.encode(prompt_ids , return_tensors="pt").to(model.device)
        prompt_length = inputs.shape[-1] #len(prompt_ids)
        # attention_mask = torch.ones_like(inputs, dtype=torch.bool)

        do_sample: Optional[bool] = input_kwargs.pop("do_sample", None)
        temperature: Optional[float] = input_kwargs.pop("temperature", None)
        top_p: Optional[float] = input_kwargs.pop("top_p", None)
        top_k: Optional[float] = input_kwargs.pop("top_k", None)
        num_return_sequences: int = input_kwargs.pop("num_return_sequences", 1)
        # repetition_penalty: Optional[float] = input_kwargs.pop("repetition_penalty", None)
        # length_penalty: Optional[float] = input_kwargs.pop("length_penalty", None)
        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        max_new_tokens: Optional[int] = input_kwargs.pop("max_new_tokens", None)

        generating_args = generating_args.copy()
        generating_args.update(
            dict(
                max_new_tokens = max_new_tokens if max_new_tokens is not None else generating_args["max_new_tokens"],
                do_sample=do_sample if do_sample is not None else generating_args["do_sample"],
                temperature=temperature if temperature is not None else generating_args["temperature"],
                top_p=top_p if top_p is not None else generating_args["top_p"],
                top_k=top_k if top_k is not None else generating_args["top_k"],
                num_return_sequences=num_return_sequences,
                # repetition_penalty=repetition_penalty
                # if repetition_penalty is not None
                # else generating_args["repetition_penalty"],
                # length_penalty=length_penalty if length_penalty is not None else generating_args["length_penalty"],
                eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        )
        print(max_new_tokens,top_k, top_p, temperature, do_sample)
        if isinstance(num_return_sequences, int) and num_return_sequences > 1:  # do_sample needs temperature > 0 else turn off do_sample=False
            generating_args["do_sample"] = True
            generating_args["temperature"] = generating_args["temperature"] or 1.0

        if not generating_args["temperature"]:
            generating_args["do_sample"] = False

        if not generating_args["do_sample"]:
            generating_args.pop("temperature", None)
            generating_args.pop("top_p", None)

        if max_length:
            generating_args.pop("max_new_tokens", None)
            generating_args["max_length"] = max_length

        if max_new_tokens:
            generating_args.pop("max_length", None)
            generating_args["max_new_tokens"] = max_new_tokens

        gen_kwargs = dict(
            inputs=inputs,
            # attention_mask=attention_mask,
            generation_config=GenerationConfig(**generating_args),
        )

        return gen_kwargs, prompt_length

    @staticmethod
    @torch.inference_mode()
    def _chat(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        generating_args: Dict[str, Any],
        messages: Sequence[Dict[str, str]],
        input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> List["Response"]:
        
        gen_kwargs, prompt_length = HuggingfaceEngine._process_args(
            model, tokenizer, generating_args, messages, input_kwargs
        )

        generate_output = model.generate(**gen_kwargs)
        response_ids = generate_output[:, prompt_length:]
        response = tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results = []
        for i in range(len(response)):
            eos_index = (response_ids[i] == tokenizer.eos_token_id).nonzero()
            response_length = (eos_index[0].item() + 1) if len(eos_index) else len(response_ids[i])
            results.append(
                Response(
                    response_text=response[i],
                    response_length=response_length,
                    prompt_length=prompt_length,
                    finish_reason="stop" if len(eos_index) else "length",
                )
            )

        return results

    async def chat(
        self,
        prompt: str,
        messages: Sequence[Dict[str, str]] = None,
        **input_kwargs,
    ) -> List["Response"]:

        messages = messages or []
        messages.extend([
            {"role": "user", "content": prompt},
            {"role": "model", "content": ""}
        ])

        # model, tokenizer = self._load_model

        loop = asyncio.get_running_loop()
        input_args = (
            self._model,
            self._tokenizer,
            self.generating_args,
            messages,
            input_kwargs,
        )
        async with self.semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, self._chat, *input_args)

    @staticmethod
    @torch.inference_mode()
    def _stream_chat(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        generating_args: Dict[str, Any],
        messages: Sequence[Dict[str, str]],
        input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Callable[[], str]:
        gen_kwargs, _ = HuggingfaceEngine._process_args(
            model, tokenizer, generating_args, messages, input_kwargs
        )

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        def stream():
            try:
                return streamer.__next__()
            except StopIteration:
                raise StopAsyncIteration()

        return stream
    
    async def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:

        loop = asyncio.get_running_loop()
        input_args = (
            self._model,
            self._tokenizer,
            self.generating_args,
            messages,
            input_kwargs,
        )
        async with self.semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                stream = self._stream_chat(*input_args)
                while True:
                    try:
                        yield await loop.run_in_executor(pool, stream)
                    except StopAsyncIteration:
                        break