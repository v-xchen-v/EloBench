import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datamodel.question_collection import QuestionCollection
from datamodel.pairwise_battle_arrangement import PairwiseBattleArrangement, ArrangementStrategy
from datamodel.battled_pairs import BattledPair, BattledPairs
from datamodel import PairToBattle
from datamodel.question_and_answers_collection import QuestionAndAnswersCollection, LLMAnswer
from typing import List
from judger import gpt_4_eval_and_score
from datamodel.battle_record import BattleRecord, BattleRecords
from datetime import datetime
from dataclasses import asdict
import pandas as pd
from tqdm import tqdm
from enum import Enum
import os
from pathlib import Path
from elo_rating import rating_helper
from logger import logger


class DumpyPipeline:
    def __init__(self, tempcache_dir: str, save_dir: str, start: int = 0,end: int = None, no_cache: bool = False) -> None:
        # allow adding questions multple times
        self.question_collection = None
        self.models = None
        self.battle_arrangements = None
        self.question_and_answers_collection = QuestionAndAnswersCollection()
        self.battled_pairs = BattledPairs()
        self.save_dir = save_dir
        self.tempcache_dir = tempcache_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.isdir(self.tempcache_dir):
            os.makedirs(self.tempcache_dir)
        self.end = end
        self.start = start
        self.no_cache = no_cache
        self.elo_ratings = None
    
    def register_questions(self, questions: List[str]):
        """collect questions"""
        self.question_collection = QuestionCollection(questions)
        
    def register_models(self, models: List[str]):
        self.models = models
    
    def arrange_battles(self, arrange_strategy: ArrangementStrategy, **kwargs):
        if self.battle_arrangements is None:
            self.battle_arrangements = PairwiseBattleArrangement(self.question_collection.questions, self.models)
        if arrange_strategy == ArrangementStrategy.Reload_Existing_Arrangement:
            self.battle_arrangements.arrange(arrange_strategy, battle_arrangement_file=Path(self.save_dir)/'battle_arrangement.csv')
        else:
            self.battle_arrangements.arrange(arrange_strategy, **kwargs)
        if not self.no_cache and arrange_strategy != ArrangementStrategy.Reload_Existing_Arrangement:
            self.battle_arrangements.to_csv(Path(self.save_dir)/'battle_arrangement.csv')
        # if 'preset' in kwargs and kwargs['preset'] == True:
        #     if 'preset_save_path' in kwargs:
        #         self.battle_arrangements = PairwiseBattleArrangement.read_csv(kwargs['preset_save_path'])
        #     elif 'preset_arrangment'  in kwargs:
        #         self.battle_arrangements = kwargs['preset_arrangment']
        # else:
        #     self.battle_arrangements = PairwiseBattleArrangement(questions=self.question_collection.questions, models=self.models)
        #     self.battle_arrangements.arrange_randomly_by_pairnumperquesiton(**kwargs)
        # pass
    
    def gen_model_answers(self) -> None:
        for rnd in self.battle_arrangements.battles_in_order:
            q = rnd.question
            m_a = rnd.model_a
            m_b = rnd.model_b
            
            # generate ans
            ans_a = 'dummy'
            ans_b = 'dummy'

            self.question_and_answers_collection.add_answer(q, LLMAnswer(m_a, ans_a))
            self.question_and_answers_collection.add_answer(q, LLMAnswer(m_b, ans_b))
            
            
    def _reload_cached_battle_records(self):
        tempcache_records_filepath = Path(self.tempcache_dir) / 'battle_records.csv'
        tempcache_records = None
        if not self.no_cache:
            if not (tempcache_records_filepath).exists():
                tempcache_records = BattleRecords([])
            else:
                tempcache_records = BattleRecords.from_csv(tempcache_records_filepath)

        return tempcache_records
        
    def battle(self, saving_per=50):
        # clear battle result
        self.battled_pairs.clear()
        
        def slice_list(input_list, start, end):
            # Set start and end to defaults if they are None
            start = 0 if start is None else start
            end = len(input_list) if end is None else end + 1  # end + 1 because list slicing is exclusive at the end

            # Return the sliced list
            return input_list[start:end]
            
        
        tempcache_records = self._reload_cached_battle_records()
        
        records = BattleRecords([])
        # records = []
        
        battles_in_order = slice_list(self.battle_arrangements.battles_in_order, self.start, self.end)
        
        new_battles_counter = 0
        
        # clear battle pairs before battle
        for idx, rnd in tqdm(enumerate(battles_in_order), total=len(battles_in_order)):
            record = None
            
            q = rnd.question
            m_a = rnd.model_a
            m_b = rnd.model_b
            
            # get result directly and skip this round if we can find result from caching
            if not self.no_cache:
                cached_rec = tempcache_records.get_record(PairToBattle(q, m_a, m_b))
                if cached_rec is not None:
                    record = cached_rec
                    records.add_record(record)
                    self.battled_pairs.add_pair(model_a=record.model_a, model_b=record.model_b, winner=record.winner)
                    logger.debug('Get record from cache, Skip!')
                    continue
                
            ans_a = self.question_and_answers_collection.get_answer(q, m_a)
            ans_b = self.question_and_answers_collection.get_answer(q, m_b)
            
            # skip non answer battle
            if str(ans_a) == 'nan' or str(ans_b) == 'nan':
                records.add_record(BattleRecord(model_a=m_a, model_b=m_b, winner=None, tstamp=datetime.now(), question=q, answer_a=ans_a, answer_b=ans_b, gpt_4_response=None, gpt_4_score=None, is_valid=False, judger=None))
                continue
            
            new_battles_counter += 1
            try:
                gpt4_response, gpt_4_score, gpt_4_winner = gpt_4_eval_and_score(question=q, model_a_ans=ans_a, model_b_ans=ans_b)
            except Exception as e:
                records.add_record(BattleRecord(model_a=m_a, model_b=m_b, winner=None, tstamp=datetime.now(), question=q, answer_a=ans_a, answer_b=ans_b, gpt_4_response=None, gpt_4_score=None, is_valid=False, judger=None))
                print(e)
                continue
            
            self.battled_pairs.add_pair(m_a, m_b, gpt_4_winner)
            
            # record
            records.add_record(BattleRecord(model_a=m_a, model_b=m_b, winner=gpt_4_winner, tstamp=datetime.now(), question=q, answer_a=ans_a, answer_b=ans_b, gpt_4_response=str(gpt4_response), gpt_4_score=str(gpt_4_score), is_valid=True, judger="GPT-4"))
            
            if new_battles_counter % saving_per == 0:
                logger.debug('caching gpt-4 eval and score result...')
                # self.battled_pairs.to_csv(Path(self.save_dir)/'battled_pairs.csv')
                # save_battle_records(records)
                records.to_csv(Path(self.save_dir)/'battle_records.csv')
                
                if not self.no_cache:
                    tempcache_records.add_records(records.records)
                    tempcache_records.to_csv(Path(self.tempcache_dir)/'battle_records.csv')
                    
                logger.debug('Done.')
        
        # save records
        # save_battle_records(records) 
        records.to_csv(Path(self.save_dir)/'battle_records.csv')
        records.to_csv(Path(self.save_dir)/'battled_pairs.csv', winner_only=True)
        if tempcache_records is not None:
            tempcache_records.add_records(records.records)
            tempcache_records.to_csv(Path(self.tempcache_dir)/'battle_records.csv')
        
    def gen_elo(self):
        if not self.no_cache and self.battled_pairs is None:
            self.battled_pairs = BattledPairs.read_csv(self.save_dir/'battled_pairs.csv')
            
        battled_pairs_dict = [asdict(obj) for obj in self.battled_pairs.battled_pairs_in_order]
        df = pd.DataFrame.from_dict(battled_pairs_dict)
        self.elo_ratings = rating_helper.get_elo_results_from_battles_data(df, K=4)
        self.elo_ratings.to_csv(Path(self.save_dir)/'elo_rating.csv')
    
    # def to_csv(self, save_dir: str):
    #     if not os.path.isdir(save_dir):
    #         os.makedirs(save_dir)
            
    #     self.battle_arrangements.to_csv(Path(save_dir)/'battle_arrangement.csv')
    #     self.question_and_answers_collection.to_csv(Path(save_dir)/'q_and_as.csv')
        
    def __repr__(self) -> str:
        repr_info = {
            "questions": str(self.question_collection),
            "models": str({'num_of_models:': len(self.models), 'models:': self.models}) if self.models is not None else "models not registered",
            "battle_arrangement": str(self.battle_arrangements) if self.battle_arrangements is not None else "battle not arranged",
            "model_answers": str(self.question_and_answers_collection) if self.question_and_answers_collection is not None else "answers not generated",
        }
        return str(repr_info)
        
if __name__ == '__main__':           
    # bp = BattlePipeline()
    # batch1_qs = ["Give me a question about animal.", "Is Landon rainy?", 'Is Austrilia always sunny?']
    # bp.register_questions(batch1_qs)
    # batch1_ms = ['huggyllama/llama-7b', 'meta-llama/Llama-2-7b-hf', 'mosaicml/mpt-7b']
    # bp.register_models(batch1_ms)
    # bp.arrange_battles(ArrangementStrategy.Random_N_Pairs_Per_Question, num_of_pair=2)
    # bp.gen_model_answers()
    # bp.battle()

    records = pd.read_csv(r'results/quora_100_test2_shuffle_ab/battle_records.csv')
    inclusive_cols = ['model_a', 'model_b', 'winner']
    records = records[inclusive_cols]
    ratings = rating_helper.get_elo_results_from_battles_data(records, K=4)
    print(ratings)
    
    # import numpy as np
    # arrangement_df= pd.read_csv(r'results/quora_100_test2/battle_arrangement.csv')
    # # Create a random boolean mask
    # mask = np.random.rand(len(arrangement_df)) > 0.5

    # # Shuffle using the mask
    # temp = arrangement_df['model_a'][mask].copy()
    # arrangement_df['model_a'][mask] = arrangement_df['model_b'][mask]
    # arrangement_df['model_b'][mask] = temp
    
    # arrangement_df.to_csv('results/quora_100_test2_shuffle_ab/battle_arrangement.csv', index=False)