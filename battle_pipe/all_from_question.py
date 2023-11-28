import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from pathlib import Path
from collections import defaultdict

from battle_pipe.battle_pipe import BattlePipeline
from datamodel import ArrangementStrategy
from datamodel import LLMAnswer
from datamodel import PairToBattle
from datamodel import QuestionAndAnswersCollection
from tqdm import tqdm
from logger import logger
import gc
import torch
from models.huggingface import AutoCausalLM
from models import get_model

class AllFromQuestionsBattlePipeline(BattlePipeline):
    def gen_model_answers(self) -> None:
        
        def group_questions_by_model(lst: list[PairToBattle]):
            """Group question by each model to avoid reload and free model frequently for efficiency"""
            grouped_data = defaultdict(list)
            for item in lst:
                if item.question not in grouped_data[item.model_a]:
                    grouped_data[item.model_a].append(item.question)
                if item.question not in grouped_data[item.model_b]:
                    grouped_data[item.model_b].append(item.question)
            return grouped_data
        
        questions_by_model = group_questions_by_model(self.battle_arrangements.battles_in_order)

        if not self.no_cache and (Path(self.tempcache_dir)/'q_and_as.csv').exists():
            self.question_and_answers_collection = QuestionAndAnswersCollection.read_csv(Path(self.tempcache_dir)/'q_and_as.csv')
            caching_q_and_as_collection = self.question_and_answers_collection
            
            # iterate the model: questions dictionary to remove cached question
            for model_name, questions in questions_by_model.items():
                questions_by_model[model_name] = [q for q in questions if not caching_q_and_as_collection.answer_exists(q, model_name)]
            
            # remove entries with empty question list
            questions_by_model = {k: v for k, v in questions_by_model.items() if len(v) > 0}
        
        for model_name, questions in tqdm(questions_by_model.items(), desc="Loop models and gen ans"):
        
            lm = get_model(model_name)

            answers = []
            for q in tqdm(questions, desc=f'loop {len(questions)} questions on {model_name}', leave=False, position=1):
                answer = lm.generate_answer(q, free_model_when_exit=False)
                answers.append(answer)
                self.question_and_answers_collection.add_answer(q, LLMAnswer(model_name, answer))
                
                if not self.no_cache:
                    # caching generated answers per ans
                    self.question_and_answers_collection.to_csv(Path(self.tempcache_dir)/'q_and_as.csv')
                
                logger.debug(f'Question:\n{q}')
                logger.debug(f'Answer:\n{answer}')
            
            if isinstance(lm, AutoCausalLM):
                # free gpu resource
                del lm.model
                gc.collect()
                torch.cuda.empty_cache()
            
            # deprecated batch mode for get ability to caching per ans
            # generate ans
            # answers = AnswerGenerator.generate_answers(hf_model_id=model_name, questions=questions, progress_bar_class=tqdm)
            
            # for q, ans in zip(questions, answers):
            #     self.question_and_answers_collection.add_answer(q, LLMAnswer(model_name, ans))

                
if __name__ == "__main__":
    # TODOs: design and dump config of battle in save folder with results
    # Register questions to battle pipeline
    questions = pd.read_csv(Path('data')/'quora_100'/'questions.csv')['question'].tolist()

    battle_pipe = AllFromQuestionsBattlePipeline(tempcache_dir='data/quora_100', save_dir=Path('results')/'quora_100_test4')
    battle_pipe.register_questions(questions)
    
    print(battle_pipe.question_collection)
    
    # Register models to battle
    models = [ \
        # 'gpt2',
        'huggyllama/llama-7b', 
        'huggyllama/llama-13b',
        'huggyllama/llama-30b',
        'huggyllama/llama-65b',
        'meta-llama/Llama-2-7b-hf',
        'meta-llama/Llama-2-13b-hf',
        'meta-llama/Llama-2-70b-hf',
        'lmsys/vicuna-7b-v1.5',
        'lmsys/vicuna-13b-v1.5',
        'lmsys/vicuna-33b-v1.3',
        'meta-llama/Llama-2-7b-chat-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'meta-llama/Llama-2-70b-chat-hf',
        'chavinlo/alpaca-native',
        'chavinlo/alpaca-13b',
        'gpt-4-turbo',
        'gpt-35-turbo',
    ]
    battle_pipe.register_models(models)
    print(battle_pipe)
    
    # Arrange battle rounds
    battle_pipe.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_Model, num_of_pair=300*3)
    
    # print(battle_pipe)
    
    # battle_pipe.arrange_battles(ArrangementStrategy.Reload_Existing_Arrangement)
    # print(battle_pipe)
    
    # Generate answer
    battle_pipe.gen_model_answers()
    print(battle_pipe)    
    
    battle_pipe.battle()

    battle_pipe.gen_elo()