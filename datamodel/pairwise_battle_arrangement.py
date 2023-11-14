from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List
import pandas as pd
from .columns import QUESTION_COLUMN_NAME, MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME

@dataclass
class PairToBattle:
    question: str
    model_a: str
    model_b: str
        
class PairwiseBattleArrangement:
    def __init__(self, questions: List[str], models: List[str]) -> None:
        self.questions = questions
        self.models = models
        self.battles_in_order = []
    
    def arrange_randomly_by_pairnumperquesiton(self, num_of_pair: int):
        battles = []
        for question in self.questions:
            for _ in range(num_of_pair):
                model_pair = random.sample(self.models, 2)
                battles.append(PairToBattle(question=question, model_a=model_pair[0], model_b=model_pair[1]))
        self.battles_in_order = battles
        
    def to_csv(self, save_path):
        arrangments = []
        for battle in self.battles_in_order:
            arrangments.append([battle.question, battle.model_a, battle.model_b])
            
        arrangments_df = pd.DataFrame(arrangments, columns=[QUESTION_COLUMN_NAME, MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME])
        # print(arrangement)
        arrangments_df.to_csv(save_path)
    
    @classmethod
    def read_csv(cls, save_path) -> PairwiseBattleArrangement:
        df = pd.read_csv(save_path)
        models = set()
        questions = set()
        rounds = []
        for index, row in df.iterrows():
            q = row[QUESTION_COLUMN_NAME]
            m_a = row[MODEL_A_COLUMN_NAME]
            m_b = row[MODEL_B_COLUMN_NAME]
            rounds.append(PairToBattle(q, m_a, m_b))
            models.add(m_a)
            models.add(m_b)
            questions.add(q)
        battle_arrangemet = PairwiseBattleArrangement(questions=list(questions), models=list(models))
        battle_arrangemet.battles_in_order = rounds
        return battle_arrangemet
            
            
    
        
                