"generate answers for questions on language model, cache the answers and questions in the tempcache_dir, could resume and rerun from the last cached state."


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gc
from pathlib import Path
from tqdm import tqdm
import torch
from typing import List

from logger import info_logger
from models import get_model, AutoCausalLM
from datamodel import QuestionAndAnswersCollection, LLMAnswer
from collections import defaultdict

# TODO: replace csv file to sqlite if necessary


# # a 2-dim matrix to record whether the answer is missing(not generated), the first dim is the question index, the second dim is the model index
# answer_isna_matrix = defaultdict(defaultdict(lambda: False))
# TODO: decouple generate answer logic from battle pipeline to here
def generate_answer(model_name:str, questions: List[str], q_and_as_collection: QuestionAndAnswersCollection, batch_mode: bool=True):
    lm = get_model(model_name)
    if not batch_mode or len(questions) == 1:
        info_logger.info(f'generate answers for {len(questions)} questions on {model_name} in non-batch mode')
        answers = []
        for q in tqdm(questions, desc=f'loop {len(questions)} questions on {model_name}', leave=False, position=1):
            answer = lm.generate_answer(q, free_model_when_exit=False)
            answers.append(answer)
            q_and_as_collection.add_answer(q, LLMAnswer(model_name, answer))
            
            # caching generated answers per ans
            q_and_as_collection.to_csv()
            
            info_logger.debug(f'Question:\n{q}')
            info_logger.debug(f'Answer:\n{answer}')
    else:
        answers = []
        info_logger.info(f'generate answers for {len(questions)} questions on {model_name} in batch mode')
        while len(answers) < len(questions):
            try:
                # avoid repeat generating answers when retry
                num_gen_ans = len(answers)
                questions = questions[num_gen_ans:]
                if num_gen_ans > 0:
                    info_logger.info(f'retry with remaining {len(questions)} questions')
                
                for answer in tqdm(lm.batch_generate_answer(questions, free_model_when_exit=False), total=len(questions), desc=f'loop {len(questions)} questions on {model_name}', leave=False, position=1):
                    answers.append(answer)
                    q = questions[len(answers)-1-num_gen_ans]

                    q_and_as_collection.add_answer(q, LLMAnswer(model_name, answer))
                    
                    if (len(answers) % lm.batch_size == 0 or len(answers) == len(questions)):
                        # caching generated answers per ans
                        q_and_as_collection.to_csv()  
                        
                    info_logger.debug(f'Question:\n{q}')
                    info_logger.debug(f'Answer:\n{answer}')                 
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    info_logger.warning(f'OOM error on {model_name}, free gpu resource and try again.')
                    
                    lm.batch_size = int(lm.batch_size / 2)
                    if lm.batch_size < 1:
                        raise e
                    info_logger.info(f'retry with half batch size: {int(lm.batch_size / 2)}')
                else:
                    raise e

    if isinstance(lm, AutoCausalLM):
        # free gpu resource
        del lm.model
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == '__main__':
    # model_name = 'lmsys/vicuna-33b-v1.3'
    model_name = 'gpt-35-turbo'
    dataset_name = 'google_quora_alpaca_10629'
    batch_mode = False
    questions = [ \
    "Does light travel forever or does it eventually fade?",
    "What is France's record in wars?",
    "Do you think Rihanna's image has influenced the fashion industry, and if so, how?",
    "Why does the United States have higher living standards than Europe?",
    'Can you name the main actors in the film "Green Book"?',
    "What are some crazy coincidences in history ?",
    "Is anything in England actually original? Tea, for example, comes from China. Fish & chips from Portugal. Football (soccer) originates in China. Gothic style architecture is French.",
    "Have you ever tried adding exotic fruits to your overnight oats?",
    "Can you name a few controversies that have involved Jeremy Clarkson?",
    "Can you tell me about the impact of the iPad 3 on the consumer electronics industry?",
    "Who are some actors that have worked with Kevin Hart in his comedy films?",
    "How does an Andrea Bocelli concert compare to other concerts you've been to?",
    "How has Katy Perry's image evolved over the years?",
    "Hi, I'm interested in learning to play badminton. Can you explain the game to me?",
    ]
    # q_and_as_collection = QuestionAndAnswersCollection.read_csv(Path('tempcache')/{dataset_name}/'q_and_as.csv')
    q_and_as_collection = QuestionAndAnswersCollection()
    dataset_name = 'test'
    q_and_as_collection.cache_filepath = Path('tempcache')/dataset_name/'q_and_as.csv'
    generate_answer(model_name, questions, q_and_as_collection, batch_mode)