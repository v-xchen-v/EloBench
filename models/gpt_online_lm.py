from openai_chat import chat_completion
from gpt_chat_prompts.prompts import GPT4_GEN_ANS_PROMPT
from models.lm import LM

class GPTOnlineLM(LM):
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        
    def generate_answer(self, question, **kwargs) -> str:
        """
        Generate an answer using the OPENAI GPT online language model.

        Args:
            question (str): The input question.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated answer.
        """
        gpt_4_response = chat_completion(GPT4_GEN_ANS_PROMPT, gpt_name=self.model_name, question=question, max_tokens=512)
        gpt4_response_text = gpt_4_response['response']
        answer = gpt4_response_text
        return answer