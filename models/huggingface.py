import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Union, Optional, List
from logger import logger
import gc

# TODO: support batch mode
# TODO: support give a model as input, and do not free the model for futher use
# TODO: different decode
class AutoCausalLM:
    def __init__(self, model_name) -> None:
         # Load pre-trained model and tokenizer
        self.config = transformers.AutoConfig.from_pretrained(model_name)
        def _get_dtype(
            dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
        ) -> torch.dtype:
            """Converts `dtype` from `str` to torch.dtype when possible."""
            if dtype is None and config is not None:
                _torch_dtype = config.torch_dtype
            elif isinstance(dtype, str) and dtype != "auto":
                # Convert `str` args torch dtype: `float16` -> `torch.float16`
                _torch_dtype = getattr(torch, dtype)
            else:
                _torch_dtype = dtype
            return _torch_dtype
        torch_dtype=_get_dtype(None, self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if torch.cuda.device_count() > 1:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch_dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name,  torch_dtype=torch_dtype)
        
        logger.debug(f'device: {self.model.device.type}')
        if self.model.device.type=='cuda':
            logger.debug(f'device_map: {self.model.hf_device_map}')
            
    def _decorate_prompt_for_qa(self, prompt):
        """some model like lmsys/vicuna-7b-v1.5, chavinlo/alpaca-13b, chavinlo/native will generate empty answer for plain question with ? at end so that need this decorating to gen ans."""
        return f"Question: {prompt}\nAnswer: "
        
    def generate_answer(self, question: str, free_model_when_exit: bool = True):
        # support give a model as input, and do not free the model for futher use
        
        # Check if CUDA (GPU support) is available and move the model to GPU if it is
        if self.model.device.type == 'cpu' and torch.cuda.is_available():
            self.model = self.model.to('cuda')
                        
        # Prepare the prompt
        prompt = self._decorate_prompt_for_qa(question)
        # prompt = "The Hubble Space Telescope, launched in 1990, has made significant contributions to astronomy by"

        # Encode the prompt and generate text
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')

        # Move the input tensors to the GPU if CUDA is available
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=False)

        # Decode and remove the prompt from the generate text
        prompt_length = len(self.tokenizer.encode(prompt))
        generated_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        # print(generated_text)

        if free_model_when_exit:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
        
        return generated_text