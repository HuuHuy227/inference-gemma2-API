from typing import Optional, List, Iterator, Union
from type import PytorchGenerateConfig, Response
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
logger = logging.getLogger(__name__)

class ChatModel(): #PytorchModel, ChatModelMixin):
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(
        self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
    ):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        if generate_config["do_sample"] == True:
            temperature = generate_config["temperature"]
            top_p = generate_config["top_p"]
            top_k = generate_config["top_k"]
        elif generate_config["do_sample"] == False:
            temperature = None
            top_p = None
            top_k = None
        
        generate_output = self.model.generate(
            inputs = input_ids,
            max_new_tokens=generate_config["max_tokens"],
            eos_token_id = 1, 
            do_sample = generate_config["do_sample"],
            temperature = temperature,
            top_p = top_p,
            top_k = top_k 
        )
        prompt_length = input_ids.shape[-1]

        response_ids = generate_output[:, prompt_length:]
        response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results = []
        for i in range(len(response)):
            eos_index = (response_ids[i] == self.tokenizer.eos_token_id).nonzero()
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

    @torch.inference_mode()
    def chat(
        self,
        prompt: str,
        chat_history: Optional[List] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ):
        full_prompt = self._get_full_prompt(prompt, chat_history)

        res = self.generate(full_prompt, generate_config)
        return res

        # stream = generate_config.get("stream", False)
        # if stream:
        #     it = self.generate(full_prompt, generate_config)
        #     assert isinstance(it, Iterator)
        #     return self._to_chat_completion_chunks(it)
        # else:
        #     c = self.generate(full_prompt, generate_config)
        #     assert not isinstance(c, Iterator)
        #     return self._to_chat_completion(c)

    # def load(self):
    #     super().load()

    # def _get_full_prompt(self, prompt, system_prompt, chat_history):
    #     assert self.model_family.prompt_style is not None
    #     prompt_style = self.model_family.prompt_style.copy()
    #     if system_prompt:
    #         prompt_style.system_prompt = system_prompt
    #     chat_history = chat_history or []
    #     full_prompt = ChatModelMixin.get_prompt(
    #         prompt, chat_history, prompt_style
    #     )
    #     return full_prompt
    
    def _get_full_prompt(self, prompt, chat_history) -> str:
        chat_history = chat_history or []
        chat_history.extend([
            {"role": "user", "content": prompt},
            {"role": "model", "content": ""}
        ])
        ret = ""
        for message in chat_history:
            content = message["content"]
            role = message["role"] #get_role(message["role"])
            ret += "<start_of_turn>" + role + "\n"
            if content:
                ret += content + "<end_of_turn>\n"
        return ret

    # def prepare_batch_inference(self, req_list: List[InferenceRequest]):
    #     super().prepare_batch_inference(req_list)
    #     for r in req_list:
    #         r.full_prompt = self._get_full_prompt(
    #             r.prompt, r.system_prompt, r.chat_history, None
    #         )

    # def handle_batch_inference_results(self, req_list: List[InferenceRequest]):
    #     for req in req_list:
    #         if req.stream and req.error_msg is None:
    #             if req.completion:
    #                 results = []
    #                 for i, c in enumerate(req.completion):
    #                     if c == "<bos_stream>":
    #                         results.append(
    #                             self._get_first_chat_completion_chunk(
    #                                 req.completion[i + 1]
    #                             )
    #                         )
    #                     elif c == "<eos_stream>":
    #                         break
    #                     else:
    #                         results.append(self._to_chat_completion_chunk(c))

    #                 if req.stopped and req.include_usage:
    #                     results.append(
    #                         self._get_final_chat_completion_chunk(req.completion[-1])
    #                     )
    #                 req.completion = results