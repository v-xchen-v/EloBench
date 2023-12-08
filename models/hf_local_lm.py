import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Union, Optional, List
from logger import logger, info_logger
import gc
from models.lm import LM
from accelerate import find_executable_batch_size
import torch.nn.functional as F
from tqdm import tqdm

# TODO: support batch mode
# TODO: support give a model as input, and do not free the model for futher use
# TODO: different decode
class AutoCausalLM(LM):
    def __init__(self, model_name) -> None:
        """
        Initializes an instance of AutoCausalLM.

        Args:
            model_name (str): The name or path of the pre-trained model to load.

        Returns:
            None
        """
        self.model_name = model_name
        try:    
            self._create_model(model_name)
            self.model_parallel = False
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logger.debug('CUDA out of memory')
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
                logger.info('retry with model parallelism')
                self._create_model(model_name, use_model_parallel=True)
                self.model_parallel = True
            else:
                raise e
        
        # Set the model to evaluation mode
        self.model.eval()
        torch.set_grad_enabled(False)

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        # ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
        # ! What's the proper way to handle pad across multiple models
        # For now, use the preset pad_token and pad_token_id from the tokenizer, if not exist, use eos_token and eos_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # # Access the padding side from the tokenizer's configuration
        # padding_side = self.tokenizer.padding_side
        # print(f"Padding side: {padding_side}")
            
        self.max_batch_size = 1024
        self.max_new_tokens = 512
        self.batch_size=None # None means auto detect batch_size, otherwise, use the given batch_size
        
    def _create_model(self, model_name: str, use_model_parallel: bool = False):
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
        if torch.cuda.device_count() > 1 and use_model_parallel:
            # use multiple gpus by model parallelism
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch_dtype)
        elif torch.cuda.is_available():
        # if torch.cuda.is_available():
            # use gpu
            self.model = AutoModelForCausalLM.from_pretrained(model_name,  torch_dtype=torch_dtype).to('cuda')
        else:
            # use cpu if no gpu
            self.model = AutoModelForCausalLM.from_pretrained(model_name,  torch_dtype=torch_dtype)
            
        self._device = self.model.device
        logger.debug(f'device: {self.model.device.type}')
        if self.model.device.type=='cuda' and hasattr(self.model, 'hf_device_map'):
            logger.debug(f'device_map: {self.model.hf_device_map}')
            if "lm_head" in self.model.hf_device_map:
                self._device = self.model.hf_device_map['lm_head']            
        
    def _decorate_prompt_for_qa(self, prompt):
        """
        Decorates the prompt for question answering.

        Args:
            prompt (str): The prompt to decorate.

        Returns:
            str: The decorated prompt.
        """
        """some model like lmsys/vicuna-7b-v1.5, chavinlo/alpaca-13b, chavinlo/native will generate empty answer for plain question with ? at end so that need this decorating to gen ans."""
        return f"Question: {prompt}\nAnswer: "
        
    def generate_answer(self, question: str, free_model_when_exit: bool = True):
        """
        Generates an answer for the given question.

        Args:
            question (str): The question to generate an answer for.
            free_model_when_exit (bool): Whether to free the model for further use when exiting.

        Returns:
            str: The generated answer.
        """
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
    
    def _detect_batch_size(self, questions: list):
        """
        Detects the optimal batch size for the given questions.

        Args:
            questions (list): The questions to generate answers for.

        Returns:
            int: The optimal batch size.
        """
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available, unable to detect batch size')
        
        # Check if CUDA (GPU support) is available and move the model to GPU if it is
        if self.model.device.type == 'cpu':
            self.model = self.model.to('cuda')
            
        if self.model_parallel:
            raise RuntimeError('model parallelism is enabled, unable to detect batch size')
            
        longest_question = max([self._decorate_prompt_for_qa(question) for question in questions], key=len)
        longest_question_length = len(self.tokenizer.encode(longest_question))
        # print(longest_question_length)
        max_length = longest_question_length+self.max_new_tokens
        
        # deprecated, because it can not handle the OOM caused error when using model parallelism
        #  # if OOM, then halves batch_size and tries again
        # @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        # def forward_batch(batch_size):
        #     test_batch = torch.ones((batch_size, max_length), device=self._device).long()
        #     for _ in range(5):
        #         _ = F.log_softmax(self.model(test_batch)['logits'], dim=-1).cpu()
        #     return batch_size

        # batch_size = forward_batch()
        batch_size=self.max_batch_size
        def forward_batch(batch_size):
            try:
                test_batch = torch.ones((batch_size, max_length), device=self._device).long()
                for _ in range(5):
                    _ = F.log_softmax(self.model(test_batch)['logits'], dim=-1).cpu()
            except RuntimeError as e: # OOM
                logger.debug(e)
                del test_batch
                gc.collect()
                torch.cuda.empty_cache()
                raise e        
            
            return batch_size
        
        # try to detect batch_size, start from the max_batch_size, if OOM, then halves batch_size and tries again
        while True:
            try:
                batch_size = forward_batch(batch_size)
                info_logger.info(f"detected batch_size: {batch_size} on {self._device}")
                break
            except RuntimeError as e: # OOM
                batch_size //= 2
                info_logger.info(f'decrease batch size to {batch_size} due to OOM')
                logger.debug(e)
                if batch_size < 1:
                    raise RuntimeError('Unable to detect batch size, no enough memory for even single batch')
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
        self.batch_size = batch_size
        return batch_size
        
    def _generate_answers(self, questions: list, free_model_when_exit: bool=True):
        if self.model.device.type == 'cpu' and torch.cuda.is_available():
            self.model = self.model.to('cuda')
                                
        # Prepare the prompt
        prompts = [self._decorate_prompt_for_qa(question) for question in questions]
        
        # Encode the prompt and generate text
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        def remove_padding_from_batch_left_padding(output):
            # Find the index of the first non-padding token
            first_non_pad_token_index = next((i for i, token in enumerate(output) if token != self.tokenizer.pad_token_id), None)
            if first_non_pad_token_index is not None:
                non_pad_output = output[first_non_pad_token_index:]
                output = non_pad_output
            return output
        
        # Decode and process generated text
        answers = []
        for i in range(outputs.size(0)):
            prompt_length = len(self.tokenizer.encode(prompts[i]))
        
            # Remove padding tokens from the output
            output = remove_padding_from_batch_left_padding(outputs[i])
                
            continuation = output[prompt_length:]

            generated_text = self.tokenizer.decode(continuation, skip_special_tokens=True)
            answers.append(generated_text)
            
        if free_model_when_exit:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            
        return answers
    
    def batch_generate_answer(self, questions: list, free_model_when_exit: bool = False):
            """
            Generates answers for a batch of questions.

            Args:
                questions (list): A list of questions.
                free_model_when_exit (bool, optional): Whether to free the model when exiting. Defaults to True.

            Yields:
                str: The generated answer for each question in the batch.
            """
            if self.batch_size is None:
                info_logger.info('detecting batch_size...')
                self.batch_size = self._detect_batch_size(questions)
                self.max_batch_size = self.batch_size
                info_logger.info(f'detected batch_size: {self.batch_size}')
            
            # answers = []
            for batch_start in range(0, len(questions), self.batch_size):
                batch_end = min(batch_start+self.batch_size-1, len(questions)-1)
                batch_questions = questions[batch_start:batch_end+1]
                
                batch_answers = self._generate_answers(batch_questions, free_model_when_exit=False)
                for answer in batch_answers:
                    yield answer
                    
            if free_model_when_exit:
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
            
if __name__ == '__main__':
# TODO: move this test to test/test_hf_local_lm.py
    lm = AutoCausalLM('lmsys/vicuna-7b-v1.5')
    # try:
    questions = ['A new software developer spent 4 days to make a simple HTML button and CSS, should I fire him?', 'Are all illegal actions unethical? If not, what are the boundaries between illegal actions that are ethical and those that are not? What are the criteria that can be used to decide this? Who gets to decide this?']*10
    # answers = list(lm.batch_generate_answer(questions))
    # print(len(answers))
    # print(answers[0])
    # print(answers[1])
    batch_size = lm._detect_batch_size(questions)
    print(batch_size)
    lm.batch_size = batch_size
    for answer in tqdm(lm.batch_generate_answer(questions), total=len(questions), desc=f'loop {len(questions)} questions on {lm.model_name}'):
        print(answer)
    # answers = lm.generate_answers()
    # except RuntimeError as e:
    #     if 'CUDA out of memory' in str(e):
    #         print('CUDA out of memory')
    #         del lm.model
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #     else:
    #         raise e
    # print(len(answers))
    
# given a reference answers, generated answer once a time, the batch mode should have same answer with the non-batch mode
# ,question,lmsys/vicuna-7b-v1.5
# 0,"A new software developer spent 4 days to make a simple HTML button and CSS, should I fire him?","4 days to make a simple HTML button and CSS is an unacceptable amount of time. The developer should be able to complete this task in a matter of hours, not days. It's possible that the developer is inexperienced or not familiar with the necessary tools and technologies. It would be best to have a conversation with the developer to understand the issue and determine if they can improve their efficiency. If the issue persists, it may be necessary to consider letting the developer go."
# 1,"Are all illegal actions unethical? If not, what are the boundaries between illegal actions that are ethical and those that are not? What are the criteria that can be used to decide this? Who gets to decide this?","1. No, not all illegal actions are unethical. Illegal actions are those that violate the law, while unethical actions are those that violate moral principles.
# 2. The boundaries between illegal actions that are ethical and those that are not depend on the specific circumstances and the moral principles involved. For example, stealing may be illegal, but it may be considered ethical in certain circumstances, such as when it is done to prevent harm or save a life.
# 3. The criteria that can be used to decide whether an illegal action is ethical or not include:
# * The intentions behind the action: If the intention is to cause harm or to gain an unfair advantage, the action is likely to be considered unethical.
# * The consequences of the action: If the action causes harm to others or violates their rights, it is likely to be considered unethical.
# * The moral principles involved: Different moral principles may lead to different judgments about the ethicality of an action. For example, some people may believe that it is always wrong to lie, while others may believe that it is acceptable in certain circumstances.
# 4. Ultimately, it is up to individuals and society as a whole to decide what is ethical and what is not. This can be done through discussions, debates, and the development of ethical codes and laws. However, it is important to recognize that there may be disagreements and variations in opinions about what is ethical, and that different cultures and societies may have different moral frameworks."
