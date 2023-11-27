# TODO: support batch mode
# TODO: support give a model as input, and do not free the model for futher use
# TODO: different decode

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from tqdm import tqdm
from logger import logger

class AnswerGenerator:
    # @classmethod
    # def generate_answers(cls, hf_model_id: str, questions: list[str], progress_bar_class):
    #     # Load pre-trained model and tokenizer
    #     # model_name = "gpt2"
    #     model_name= hf_model_id
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     if torch.cuda.device_count() > 1:
    #         model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    #     else:
    #         model = AutoModelForCausalLM.from_pretrained(model_name)
        
    #     answers = []
    #     for q in progress_bar_class(questions, desc=f'loop {len(questions)} questions on {hf_model_id}', leave=False, position=1):
    #         logger.debug(f'Question:\n{q}')
    #         answer = cls.generate_answer_with_model(tokenizer, model, q, free_model_when_exit=False)
    #         logger.debug(f'Answer:\n{answer}')
    #         answers.append(answer)
        
    #     del model
    #     gc.collect()
    #     torch.cuda.empty_cache()
        
    #     return answers
        
    # @classmethod
    # def generate_answer(cls, hf_model_id: str, question: str, free_model_when_exit: bool = True):
    #     # Load pre-trained model and tokenizer
    #     # model_name = "gpt2"
    #     model_name= hf_model_id
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForCausalLM.from_pretrained(model_name)
        
    #     return cls.generate_answer_with_model(tokenizer, model, question, free_model_when_exit)

    @classmethod
    def decorate_prompt_for_qa(cls, prompt):
        """some model like lmsys/vicuna-7b-v1.5, chavinlo/alpaca-13b, chavinlo/native will generate empty answer for plain question with ? at end so that need this decorating to gen ans."""
        return f"Question: {prompt}\nAnswer: "
        
    @classmethod
    def generate_answer_with_model(cls, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, question: str, free_model_when_exit: bool = True):
        # support give a model as input, and do not free the model for futher use
        
        # Check if CUDA (GPU support) is available and move the model to GPU if it is
        if model.device.type == 'cpu' and torch.cuda.is_available():
            model = model.to('cuda')
                        
        # Prepare the prompt
        prompt = cls.decorate_prompt_for_qa(question)
        # prompt = "The Hubble Space Telescope, launched in 1990, has made significant contributions to astronomy by"

        # Encode the prompt and generate text
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Move the input tensors to the GPU if CUDA is available
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        outputs = model.generate(inputs, max_new_tokens=512, do_sample=False)

        # Decode and remove the prompt from the generate text
        prompt_length = len(tokenizer.encode(prompt))
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        # print(generated_text)

        if free_model_when_exit:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
        return generated_text
    