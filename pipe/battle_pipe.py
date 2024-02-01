import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from pathlib import Path
from collections import defaultdict

from pipe.dummy_pipe import DummyPipeline
from datamodel import ArrangementStrategy
from datamodel import LLMAnswer
from datamodel import PairToBattle
from datamodel import QuestionAndAnswersCollection
from tqdm import tqdm
from logger import logger, info_logger
import gc
import torch
from models.hf_local_lm import AutoCausalLM
from models.gpt_online_lm import GPTOnlineLM
from models import get_model
from elo_rating.rating_evaluator import compute_predict_winrate, compute_acutal_winrate, evaluate_winrate, evaluate_rank_consistency
from typing import List

class BattlePipeline(DummyPipeline):
    def _gen_model_answers(self, model_name: str, questions: List[str], batch_mode):
        """Generate answers for each question with the given model"""
        lm = get_model(model_name)

        if not batch_mode or len(questions) == 1:
            info_logger.info(f'generate answers for {len(questions)} questions on {model_name} in non-batch mode')
            answers = []
            for q in tqdm(questions, desc=f'loop {len(questions)} questions on {model_name}', leave=False, position=1):
                answer = lm.generate_answer(q, free_model_when_exit=False)
                answers.append(answer)
                self.question_and_answers_collection.add_answer(q, LLMAnswer(model_name, answer))
                
                if not self.no_cache:
                    # caching generated answers per ans
                    self.question_and_answers_collection.to_csv(Path(self.tempcache_dir)/'q_and_as.csv')
                
                # logger.debug(f'Question:\n{q}')
                # logger.debug(f'Answer:\n{answer}')
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

                        self.question_and_answers_collection.add_answer(q, LLMAnswer(model_name, answer))
                        
                        if not self.no_cache and (len(answers) % lm.batch_size == 0 or len(answers) == len(questions)):
                            # caching generated answers per ans
                            self.question_and_answers_collection.to_csv(Path(self.tempcache_dir)/'q_and_as.csv')
                        
                        # logger.debug(f'Question:\n{q}')
                        # logger.debug(f'Answer:\n{answer}') 
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        logger.warning(f'OOM error on {model_name}, free gpu resource and try again.')
                        
                        lm.batch_size = int(lm.batch_size / 2)
                        if lm.batch_size < 1:
                            raise e
                        logger.info(f'retry with half batch size: {int(lm.batch_size / 2)}')
                    else:
                        raise e
        
        if isinstance(lm, AutoCausalLM):
            # free gpu resource
            del lm.model
            gc.collect()
            torch.cuda.empty_cache()
                
    def gen_model_answers(self, batch_mode=True) -> None:
        """
        Generate answers for each model based on the given questions.
        """
        
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
            self._gen_model_answers(model_name, questions, batch_mode)
                
if __name__ == "__main__":
    # TODOs: design and dump config of battle in save folder with results
    # # Register questions to battle pipeline
    # questions = pd.read_csv(Path('data')/'quora_100'/'questions.csv')['question'].tolist()

    # battle_pipe = BattlePipeline(tempcache_dir='data/quora_100', save_dir=Path('results')/'quora_100_test4')
    # battle_pipe.register_questions(questions)
    
    # print(battle_pipe.question_collection)
    
    # # Register models to battle
    # models = [ \
    #     # 'gpt2',
    #     'huggyllama/llama-7b', 
    #     'huggyllama/llama-13b',
    #     'huggyllama/llama-30b',
    #     'huggyllama/llama-65b',
    #     'meta-llama/Llama-2-7b-hf',
    #     'meta-llama/Llama-2-13b-hf',
    #     'meta-llama/Llama-2-70b-hf',
    #     'lmsys/vicuna-7b-v1.5',
    #     'lmsys/vicuna-13b-v1.5',
    #     'lmsys/vicuna-33b-v1.3',
    #     'meta-llama/Llama-2-7b-chat-hf',
    #     'meta-llama/Llama-2-13b-chat-hf',
    #     'meta-llama/Llama-2-70b-chat-hf',
    #     'chavinlo/alpaca-native',
    #     'chavinlo/alpaca-13b',
    #     'gpt-4-turbo',
    #     'gpt-35-turbo',
    # ]
    # battle_pipe.register_models(models)
    # print(battle_pipe)
    
    # # Arrange battle rounds
    # battle_pipe.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_Model, num_of_pair=300*3)
    
    tempcache_dir = "tempcache/quora_100"
    save_dir = "results/quora_100_test5_iterative_80_no_tie"
    battle_pipe = BattlePipeline(tempcache_dir,save_dir)
    battle_pipe.no_cache = False
    
    battle_pipe.reload_pipe()
    # questions = ["Give me a question about animal.", "Is Landon rainy?", 'Is Austrilia always sunny?']
    # models = ['huggyllama/llama-7b', 'meta-llama/Llama-2-7b-hf', 'mosaicml/mpt-7b']
    # battle_pipe.register_questions(questions)
    # battle_pipe.register_models(models)
    
    # # print(battle_pipe)
    
    # # battle_pipe.arrange_battles(ArrangementStrategy.Reload_Existing_Arrangement)
    # # print(battle_pipe)
    
    # battle_pipe.arrange_battles(ArrangementStrategy.Reload_Existing_Arrangement)
    # battle_pipe.gen_model_answers()
    
    # Generate answer
    battle_pipe.gen_model_answers()
    
    battle_pipe.battle()

    battle_pipe.gen_elo(with_history=True)
    
    for idx, battle_num in enumerate(battle_pipe.elo_rating_history.recorded_battle_num):
        # if battle_num > 0 and ((battle_num) % 100 == 0 or battle_num == battle_pipe.elo_rating_history.recorded_battle_num[-1]):
            
        history_point = battle_pipe.elo_rating_history.get_point(battle_num)
        predicted_winrate = compute_predict_winrate(history_point)
        actual_winrate = compute_acutal_winrate(battle_pipe.battled_pairs._to_df(battle_pipe.battled_pairs[0:battle_num]))
        # print(predicted_winrate)
        # print(actual_winrate)
        print(evaluate_winrate(actual_winrate, predicted_winrate))
        if idx > 0:
            prev_history_point_battle_num = battle_pipe.elo_rating_history.recorded_battle_num[idx-1]
            print(evaluate_rank_consistency(battle_pipe.elo_rating_history, prev_history_point_battle_num, battle_num))