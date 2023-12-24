from openai_chat import chat_completion
from gpt_chat_prompts.prompts import GPT4_GEN_ANS_PROMPT
from models.lm import LM
from concurrent.futures import ThreadPoolExecutor

class GPTOnlineLM(LM):
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.max_new_tokens = 512
        self.batch_size = 2
        
    def generate_answer(self, question, **kwargs) -> str:
        """
        Generate an answer using the OPENAI GPT online language model.

        Args:
            question (str): The input question.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated answer.
        """
        gpt_4_response = chat_completion(GPT4_GEN_ANS_PROMPT, gpt_name=self.model_name, question=question, max_tokens=self.max_new_tokens)
        gpt4_response_text = gpt_4_response['response']
        answer = gpt4_response_text
        return answer
    
    def batch_generate_answer(self, questions, **kwargs) -> list:
        for  q in questions:
            yield self.generate_answer(q, **kwargs)
            
        # TODO: test the parallel logic and use it if it is faster
        # # Create a thread pool with {batch_size} worker threads
        # with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
        #     for batch_start in range(0, len(questions), self.batch_size):
        #         batch_end = min(batch_start+self.batch_size-1, len(questions)-1)
        #         batch_questions = questions[batch_start:batch_end+1]
            
        #         # Use executor.map to apply generate_answer to each question
        #         results = executor.map(lambda q: self.generate_answer(q, **kwargs), batch_questions)
            
        #         # Return results as a list
        #         for result in results:
        #             yield result
