from datamodel.question_collection import QuestionCollection
from datamodel.pairwise_battle_arrangement import PairwiseBattleArrangement
from datamodel.battled_pairs import BattledPair, BattledPairs
from datamodel.question_and_answers_collection import QuestionAndAnswersCollection, LLMAnswer
from typing import List
from judger import gpt_4_eval_and_score
from datamodel.log import LogBattlesWithGPT4AsJudger
from datetime import datetime
from dataclasses import asdict
import pandas as pd

class BattlePipeline:
    def __init__(self, save_path: str ='log.csv', num_of_max_battle: int = None) -> None:
        # allow adding questions multple times
        self.question_collection = None
        self.models = None
        self.battle_arrangements = None
        self.question_and_answers_collection = QuestionAndAnswersCollection()
        self.battled_pairs = BattledPairs()
        self.save_path = save_path
        self.num_of_max_battle = num_of_max_battle
    
    def register_questions(self, questions: List[str]):
        """collect questions"""
        self.question_collection = QuestionCollection(questions)
        
    def register_models(self, models: List[str]):
        self.models = models
    
    def arrange_battles(self, **kwargs):
        if kwargs['preset'] == True:
            if 'preset_save_path' in kwargs:
                self.battle_arrangements = PairwiseBattleArrangement.read_csv(kwargs['preset_save_path'])
            elif 'preset_arrangment'  in kwargs:
                self.battle_arrangements = kwargs['preset_arrangment']
        else:
            self.battle_arrangements = PairwiseBattleArrangement(questions=self.question_collection.questions, models=self.models)
            self.battle_arrangements.arrange_randomly_by_pairnumperquesiton(**kwargs)
        pass
    
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
        
    def battle(self):
        logs = []
        for idx, rnd in enumerate(self.battle_arrangements.battles_in_order):
            if self.num_of_max_battle and idx+1 > self.num_of_max_battle:
                break
            
            print(idx)
            q = rnd.question
            m_a = rnd.model_a
            m_b = rnd.model_b
            
            ans_a = self.question_and_answers_collection.get_answer(q, m_a)
            ans_b = self.question_and_answers_collection.get_answer(q, m_b)
            
            gpt4_response, gpt_4_score, gpt_4_winner = gpt_4_eval_and_score(question=q, model_a_ans=ans_a, model_b_ans=ans_b)
            
            self.battled_pairs.add_pair(m_a, m_b, gpt_4_winner)
            
            # record
            logs.append(LogBattlesWithGPT4AsJudger(model_a=m_a, model_b=m_b, winner=gpt_4_winner, tstamp=datetime.now(), question=q, answer_a=ans_a, answer_b=ans_b, gpt_4_reponse=str(gpt4_response), gpt_4_score=str(gpt_4_score)))
            
        
        # save records
        log_dicts = [asdict(obj) for obj in logs]
        pd.DataFrame.from_dict(log_dicts).to_csv(self.save_path)     
        
if __name__ == '__main__':           
    bp = BattlePipeline()
    batch1_qs = ["Give me a question about animal.", "Is Landon rainy?", 'Is Austrilia always sunny?']
    bp.register_questions(batch1_qs)
    batch1_ms = ['huggyllama/llama-7b', 'meta-llama/Llama-2-7b-hf', 'mosaicml/mpt-7b']
    bp.register_models(batch1_ms)
    bp.arrange_battles(num_of_pair=2)
    bp.gen_model_answers()
    bp.battle()