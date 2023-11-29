from .gpt import chat_completion
from .prompts import GPT4_GEN_ANS_PROMPT

class GPT4LM:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        
    def generate_answer(self, question, **kwargs) -> str:
        gpt_4_response = chat_completion(GPT4_GEN_ANS_PROMPT, gpt_name=self.model_name, question=question, max_tokens=512)
        gpt4_response_text = gpt_4_response['response']
        answer = gpt4_response_text
        return answer