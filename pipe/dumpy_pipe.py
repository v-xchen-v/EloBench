import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datamodel import QuestionCollection, ModelCollection
from datamodel import PairwiseBattleArrangement, ArrangementStrategy
from datamodel import BattleOutcome, BattleOutcomes
from datamodel import PairToBattle
from datamodel import QuestionAndAnswersCollection, LLMAnswer
from datamodel import BattleRecord, BattleRecords
from judger import gpt_4_eval_and_score, GPT_JUDGER_NAME
from elo_rating import rating_helper
from datamodel.elo_rating_history import EloRatingHistory

from typing import List
from datetime import datetime
from dataclasses import asdict
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
from logger import logger, battle_pipeline_logger, elo_rating_history_logger, info_logger
from openai import OpenAIError
from typing import Optional
from concurrent.futures import ThreadPoolExecutor


class DumpyPipeline:
    """
    A class representing a pipeline for conducting battles between models.

    Parameters:
    - tempcache_dir (str): The directory path for temporary caching.
    - save_dir (str): The directory path for saving battle records and results.
    - no_cache (bool, optional): Flag indicating whether to use caching. Defaults to False.
    """

    def __init__(self, tempcache_dir: str, save_dir: str, no_cache: bool = False) -> None:
        """
        Initialize the DumpyPipeline object.

        Args:
        - tempcache_dir (str): The directory path for temporary caching.
        - save_dir (str): The directory path for saving battle records and results.
        - no_cache (bool, optional): Flag indicating whether to use caching. Defaults to False.
        """
        self.save_dir = save_dir
        self.tempcache_dir = tempcache_dir
        self.question_filename = 'questions.csv'
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.isdir(self.tempcache_dir):
            os.makedirs(self.tempcache_dir)
        self.no_cache = no_cache
            
        # TODO: allow adding questions multple times
        self.clear()

    def register_questions(self, questions: Optional[List[str]]=None, reload=False):
        """
        Register questions to the system.
        if reload is True, then reload the questions from the csv file in save_dir.
        If reload is False, then register the given questions and save them to a CSV file.

        Args:
        - questions (List[str]): The list of questions to be registered.
        
        Returns:
            None
        """
        if reload == False:
            self.question_collection = QuestionCollection(questions)
            self.question_collection.to_csv(Path(self.save_dir) / self.question_filename)
            info_logger.info(f'{len(self.question_collection)} questions registered.')
        else:
            # Reload questions from csv file
            file_path = Path(self.save_dir) / self.question_filename 
            if not os.path.isfile(file_path):
                raise Exception(f'No {self.question_filename } found in {self.save_dir}')
            
            self.question_collection = QuestionCollection.read_csv(file_path)
            info_logger.info(f'{len(self.question_collection)} questions regiesterd by reloading.')

    def register_models(self, models: Optional[List[str]]=None, reload=False):
        """
        Register the given models to system.
        If reload is True, then reload the models from the csv file in save_dir.
        If reload is False, then register the given models and save them to a CSV file.

        Args:
            models (List[str]): A list of model names to register.

        Returns:
            None
        """
        if reload == False:
            self.model_collection = ModelCollection(models)
            self.model_collection.to_csv(Path(self.save_dir) / 'models.csv')
            info_logger.info(f'{len(self.model_collection)} models registered.')
        else:
            # Reload models from csv file
            file_path = Path(self.save_dir) / 'models.csv'
            if not os.path.isfile(file_path):
                raise Exception(f'No models.csv found in {self.save_dir}')
            
            self.model_collection = ModelCollection.read_csv(file_path)
            info_logger.info(f'{len(self.model_collection)} models regiestered by reloading.')

    def arrange_battles(self, arrange_strategy: ArrangementStrategy, **kwargs):
        """
        Arrange battles between the registered questions and models.

        Args:
        - arrange_strategy (ArrangementStrategy): The strategy for arranging battles.
        - kwargs: Additional keyword arguments for the arrangement strategy.
        """
        if self.battle_arrangements is None:
            self.battle_arrangements = PairwiseBattleArrangement(self.question_collection.questions, self.model_collection)
            
        if arrange_strategy == ArrangementStrategy.Reload_Existing_Arrangement:
            self.battle_arrangements.arrange(arrange_strategy, file=Path(self.save_dir) / 'battle_arrangement.csv')
        else:
            if arrange_strategy == ArrangementStrategy.Random_N_BattleCount_Each_CombPair:
                info_logger.info('Loading cached answers, to avoid arranging no answer question battles...')
                kwargs['q_and_as'] = QuestionAndAnswersCollection.read_csv(Path(self.tempcache_dir)/'q_and_as.csv')
            self.battle_arrangements.arrange(arrange_strategy, **kwargs)
        
        # save arrangement to CSV file.
        if arrange_strategy != ArrangementStrategy.Reload_Existing_Arrangement:
            self.battle_arrangements.to_csv(Path(self.save_dir) / 'battle_arrangement.csv')
        battle_pipeline_logger.info('Battles arranged.')

    def gen_model_answers(self) -> None:
        """
        Generate dummy answers for the registered questions and models.
        """
        for rnd in self.battle_arrangements.battles_in_order:
            q = rnd.question
            m_a = rnd.model_a
            m_b = rnd.model_b

            # generate ans
            ans_a = 'dummy'
            ans_b = 'dummy'

            self.question_and_answers_collection.add_answer(q, LLMAnswer(m_a, ans_a))
            self.question_and_answers_collection.add_answer(q, LLMAnswer(m_b, ans_b))
        battle_pipeline_logger.info('Model answers generated.')

    def _reload_cached_battle_records(self):
        """
        Reload the cached battle records.

        Returns:
        - tempcache_records: The reloaded cached battle records.
        """
        tempcache_records_filepath = Path(self.tempcache_dir) / 'battle_records.csv'
        tempcache_records = None
        if not self.no_cache:
            if not (tempcache_records_filepath).exists():
                tempcache_records = BattleRecords([])
            else:
                tempcache_records = BattleRecords.from_csv(tempcache_records_filepath)

        return tempcache_records
    
    def clear(self):
        self.question_collection = None
        self.model_collection = None
        self.battle_arrangements = None
        self.question_and_answers_collection = QuestionAndAnswersCollection()
        self.battled_pairs = BattleOutcomes()
        self.elo_ratings = None
        self.elo_rating_history = None
    
    def reload_pipe(self):
        self.clear()
        
        # Check if the files in save_dir is ready
        if os.path.isfile(Path(self.save_dir)/'questions.csv') and os.path.isfile(Path(self.save_dir)/'models.csv') and os.path.isfile(Path(self.save_dir)/'battle_arrangement.csv'):
            battle_pipeline_logger.info('Reloading pipeline...')
        else:
            battle_pipeline_logger.info('No existing pipeline found.')
            return
        
        # Register questions to battle pipeline
        questions = pd.read_csv(Path(self.save_dir)/'questions.csv')['question'].tolist()
        self.register_questions(questions)

        # Register models to battle pipeline
        models = pd.read_csv(Path(self.save_dir)/'models.csv')['model'].tolist()        
        self.register_models(models)
        
        # Arrange battles
        self.arrange_battles(ArrangementStrategy.Reload_Existing_Arrangement)
        battle_pipeline_logger.info('Pipeline reloaded.')
        
        
    def run(self, save_every=50, with_history=True):
        """
        Run the pipeline.

        Args:
        - save_every (int, optional): The number of battles after which to save the results. Defaults to 50.
        """
        self.gen_model_answers()
        self.battle(saving_per=save_every)
        self.gen_elo(with_history=with_history)
        battle_pipeline_logger.info('Pipeline finished.')
                                    
    def battle(self, saving_per=50):
        """
        Conduct battles between the registered questions and models.
        
        Args:
        - saving_per (int, optional): The number of battles after which to save the results. Defaults to 50.
        """
        # set up environment for launching a new battle
        self.battled_pairs.clear()
        tempcache_records = self._reload_cached_battle_records()
        records = BattleRecords([])
        new_battles_counter = 0

        # clear battle pairs before battle
        # TODO: change it to parallel processing to get winner with gpt-4 as judger
        m_as = []
        m_bs = []
        qs = []
        ans_as = []
        ans_bs = []
        to_battles = []
        for idx, rnd in tqdm(enumerate(self.battle_arrangements.battles_in_order), total=len(self.battle_arrangements.battles_in_order)):
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
            
            # all the questions with answers should have winner
            m_as.append(m_a)
            m_bs.append(m_b)
            qs.append(q)
            ans_as.append(ans_a)
            ans_bs.append(ans_b)
            to_battles = list(zip(m_as, m_bs, qs, ans_as, ans_bs))

        if len(to_battles) > 0:
            num_thread = 10 # quota 80K, 1k 10s per battle. 10 thread, 1000*10*6 = 60K, 60K/80K = 0.75
            with ThreadPoolExecutor(max_workers=num_thread) as executor:
                for batch_start in tqdm(range(0, len(to_battles), num_thread)):
                    batch_end = min(batch_start+num_thread, len(to_battles))
                    batch_to_battles = to_battles[batch_start:batch_end]
                    
                    new_records = executor.map(lambda to_battle: self._evaluate_and_record_battle(to_battle[0], to_battle[1], to_battle[2], to_battle[3], to_battle[4]), batch_to_battles)
                    for new_record in new_records:
                        records.add_record(new_record)
                    
                    new_battles_counter += 1*num_thread
                    
                    if new_battles_counter % saving_per < num_thread:
                        self._save_records(records, tempcache_records)
                    
        self._finalize_records(records, tempcache_records)
            
        # print information
        battle_pipeline_logger.debug(f'{new_battles_counter} new battles, {len(self.battle_arrangements.battles_in_order)-new_battles_counter} restored from cache.')
        battle_pipeline_logger.info('Battles conducted.')
            
    def _evaluate_and_record_battle(self, m_a, m_b, q, ans_a, ans_b):
        
        # evaluate
        is_valid = True
        gpt4_response = None
        gpt_4_score = None
        gpt_4_winner = None
        try:
            gpt4_response, gpt_4_score, gpt_4_winner = gpt_4_eval_and_score(question=q, model_a_ans=ans_a, model_b_ans=ans_b, judger_name=GPT_JUDGER_NAME)
            
            if gpt4_response is not None:
                # gpt reponse is none, is a in design but not expect return, so that record it, but only in full logs not the valid batted pairs
                self.battled_pairs.add_pair(m_a, m_b, gpt_4_winner)
        except OpenAIError as e:
            is_valid = False
            
        # record:
        # - openai error triggered record: is_valid=False, and None winner, score, reponse
        # - None reponse: is_valid=True, and None winner, score, reponse
        record = BattleRecord(model_a=m_a, model_b=m_b, winner=gpt_4_winner, tstamp=datetime.now(), question=q, answer_a=ans_a, answer_b=ans_b, gpt_4_response=str(gpt4_response), gpt_4_score=str(gpt_4_score), is_valid=is_valid, judger=GPT_JUDGER_NAME)

        return record
        
    def _save_records(self, records, tempcache_records):
        logger.debug('caching gpt-4 eval and score result...')
        records.to_csv(Path(self.save_dir) / 'battle_records.csv')
        records.to_csv(Path(self.save_dir) / 'battled_pairs.csv', winner_only=True)
        
        if not self.no_cache:
            tempcache_records.add_records(records.records)
            tempcache_records.to_csv(Path(self.tempcache_dir) / 'battle_records.csv')

        logger.debug('Done.')
        
    def _finalize_records(self, records, tempcache_records):
        # Finalize records logic
        records.to_csv(Path(self.save_dir) / 'battle_records.csv')
        records.to_csv(Path(self.save_dir) / 'battled_pairs.csv', winner_only=True)
        if tempcache_records is not None:
            tempcache_records.add_records(records.records)
            tempcache_records.to_csv(Path(self.tempcache_dir) / 'battle_records.csv')

    def gen_elo(self, with_history=False, use_bootstrap=False):
        """
        Generate Elo ratings based on the battled pairs.
        """
        if not self.no_cache and self.battled_pairs is None:
            self.battled_pairs = BattleOutcomes.read_csv(self.save_dir / 'battled_pairs.csv')

        battled_pairs_list = [asdict(obj) for obj in self.battled_pairs.battled_pairs_in_order]
        df = pd.DataFrame.from_dict(battled_pairs_list)
        # if use_bootstrap:
        #     self.elo_ratings = rating_helper.get_bootstrap_medium_elo(df, K=4, BOOTSTRAP_ROUNDS=100)
        # else:
        self.elo_ratings = rating_helper.get_elo_results_from_battles_data(df, K=4)
        self.elo_ratings.to_csv(Path(self.save_dir) / 'elo_rating.csv')
        battle_pipeline_logger.info('Elo ratings generated.')
        
        self.elo_rating_history = EloRatingHistory()
        if with_history:
            for idx_battle in tqdm(range(len(battled_pairs_list))):
                num_battle = idx_battle + 1
                if num_battle > 0 and (num_battle % 100 == 0 or idx_battle == len(battled_pairs_list)-1):
                    historypoint_battles_df = pd.DataFrame.from_dict(battled_pairs_list[:idx_battle+1])
                    if use_bootstrap:                     
                        historypoint_rating_df = rating_helper.get_bootstrap_medium_elo(historypoint_battles_df, K=4, BOOTSTRAP_ROUNDS=100)
                    else:
                        historypoint_rating_df = rating_helper.get_elo_results_from_battles_data(historypoint_battles_df, K=4)
                    # elo_rating_history_logger.debug(historypoint_rating_df)
                    self.elo_rating_history.add_point(historypoint_rating_df, idx_battle+1)
            self.elo_rating_history.to_csv(Path(self.save_dir) / 'elo_rating_history.csv')
            battle_pipeline_logger.info('Elo rating history generated.')
        
    def __repr__(self) -> str:
        """
        Return a string representation of the DumpyPipeline object.

        Returns:
        - str: The string representation of the DumpyPipeline object.
        """
        repr_info = {
            "questions": str(self.question_collection),
            "models": str({'num_of_models:': len(self.model_collection), 'models:': self.model_collection}) if self.model_collection is not None else "models not registered",
            "battle_arrangement": str(self.battle_arrangements) if self.battle_arrangements is not None else "battle not arranged",
            "model_answers": str(self.question_and_answers_collection) if self.question_and_answers_collection is not None else "answers not generated",
        }
        return str(repr_info)
        
if __name__ == '__main__':           
    dummy_bp = DumpyPipeline(tempcache_dir='tempcache/dummy', save_dir='results/dummy', no_cache=False)
    questions = ["Give me a question about animal.", "Is Landon rainy?", 'Is Austrilia always sunny?']
    dummy_bp.register_questions(questions)
    models = ['huggyllama/llama-7b', 'meta-llama/Llama-2-7b-hf', 'mosaicml/mpt-7b']
    dummy_bp.register_models(models)
    dummy_bp.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_CombPair, num_of_battle=2)
    dummy_bp.gen_model_answers()
    dummy_bp.battle()
    dummy_bp.gen_elo(with_history=True)

    # records = pd.read_csv(r'results/quora_100_test2_shuffle_ab/battle_records.csv')
    # inclusive_cols = ['model_a', 'model_b', 'winner']
    # records = records[inclusive_cols]
    # ratings = rating_helper.get_elo_results_from_battles_data(records, K=4)
    # print(ratings)
    
    # import numpy as np
    # arrangement_df= pd.read_csv(r'results/quora_100_test2/battle_arrangement.csv')
    # # Create a random boolean mask
    # mask = np.random.rand(len(arrangement_df)) > 0.5

    # # Shuffle using the mask
    # temp = arrangement_df['model_a'][mask].copy()
    # arrangement_df['model_a'][mask] = arrangement_df['model_b'][mask]
    # arrangement_df['model_b'][mask] = temp
    
    # arrangement_df.to_csv('results/quora_100_test2_shuffle_ab/battle_arrangement.csv', index=False)