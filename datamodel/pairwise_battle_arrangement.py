from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List
import pandas as pd
from .columns import QUESTION_COLUMN_NAME, MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME
import itertools
from enum import Enum
from logger import logger
import numpy as np
from dataclasses import asdict

class ArrangementStrategy(Enum):
    Random_N_Pairs_Per_Question = 1,
    Random_N_BattleCount_Each_Model = 2,
    Random_N_Questions_Per_Pair = 3,
    All_Combination = 4,
    Reload_Existing_Arrangement = 5,
    Preset=6
    
@dataclass(frozen=True)
class PairToBattle:
    question: str
    model_a: str
    model_b: str
    
class PairwiseBattleArrangement:
    def __init__(self, questions: List[str], models: List[str]) -> None:
        self.questions = questions
        self.models = models
        self.battles_in_order = []
        self._battles_in_order_ele_as_dict = []
        
    def arrange(self, arrange_strategy: ArrangementStrategy, **kwargs):
        if arrange_strategy == ArrangementStrategy.Random_N_Pairs_Per_Question:
            self.arrange_randomly_by_pairnumperquesiton(**kwargs)
        elif arrange_strategy == ArrangementStrategy.Random_N_BattleCount_Each_Model:
            self.arrange_randomly_by_pairnumpermodel(**kwargs)
        elif arrange_strategy == ArrangementStrategy.All_Combination:
            self.arrange_all_combinations()
        elif arrange_strategy == ArrangementStrategy.Reload_Existing_Arrangement:
            self.arrange_by_existing_arrangement(**kwargs)
        else:
            raise NotImplementedError
    
    def _shuffle_battles(self, battles: List[PairToBattle]):
        battled_pairs_dict = [asdict(obj) for obj in battles]
        df = pd.DataFrame.from_dict(battled_pairs_dict)
        
        # Create a random boolean mask
        mask = np.random.rand(len(battles)) > 0.5

        # Shuffle using the mask
        temp = df['model_a'][mask].copy()
        df['model_a'][mask] = df['model_b'][mask]
        df['model_b'][mask] = temp
        
        shuffled_battles = []
        for _, item in df.iterrows():
            shuffled_battles.append(PairToBattle(item['question'], item['model_a'], item['model_b']))
        return shuffled_battles
        
    def arrange_all_combinations(self, shuffle=True):
        # generate all unique pairs from the list
        all_pairs = list(itertools.combinations(self.models, 2))
        battles = []
        for question in self.questions:
            for pair in all_pairs:
                battles.append(PairToBattle(question=question, model_a=pair[0], model_b=pair[1]))

        if shuffle:
            battles = self._shuffle_battles(battles)
        self.battles_in_order = battles
        self._battles_in_order_ele_as_dict = [asdict(obj) for obj in battles]
    
    def arrange_randomly_by_pairnumperquesiton(self, num_of_pair: int, shuffle=True):
        def sample_unique_pairs(lst, n):
            # Generate all unique pairs from the list
            all_pairs = list(itertools.combinations(lst, 2))

            # Shuffle the list of all pairs
            random.shuffle(all_pairs)

            # Select the first n pairs
            selected_pairs = all_pairs[:n] if n <= len(all_pairs) else all_pairs

            return selected_pairs
        
        battles = []
        for question in self.questions:
            for _ in range(num_of_pair):
                model_pair = sample_unique_pairs(self.models, 2)
                battles.append(PairToBattle(question=question, model_a=model_pair[0], model_b=model_pair[1]))
                
        if shuffle:
            battles = self._shuffle_battles(battles)
        self.battles_in_order = battles
        self._battles_in_order_ele_as_dict = [asdict(obj) for obj in battles]
    
    def arrange_randomly_by_pairnumpermodel(self, num_of_pair:int, shuffle=True):
        def valid_combination(model_paircounts, question_counts, num_of_pair):
            # check if each model meets the least num_of_pair and each question have at least involved in 1 combination
            return all(model_paircounts[m]>=num_of_pair for m in model_paircounts.keys()) and all(question_counts[q]>=1 for q in question_counts.keys())
            
        all_modelpairs = list(itertools.combinations(self.models, 2))
        
        # Create all possible (question, model pair) combinations
        all_combinations = list(itertools.product(self.questions, all_modelpairs))
        
        # Shuffle the combinations
        random.shuffle(all_combinations)
        
        model_counts = {model: 0 for model in self.models}
        question_counts = {question: 0 for question in self.questions}
        
        selected_combinations = []
        for comb in all_combinations:
            question, model_pair = comb
            # Increment the model counts
            for model in model_pair:
                model_counts[model] += 1
            question_counts[question] +=1
 
            selected_combinations.append(comb)
            
            # Check if all models meet the K requirement
            if valid_combination(model_counts, question_counts, num_of_pair):
                break
        
        battles = [PairToBattle(question=q, model_a=model_pair[0], model_b=model_pair[1]) for q, model_pair in selected_combinations]
        
        if shuffle:
            battles = self._shuffle_battles(battles)
        self.battles_in_order = battles
        self._battles_in_order_ele_as_dict = [asdict(obj) for obj in battles]
        
        
    def arrange_by_existing_arrangement(self, battle_arrangement_file: str):
        existing_arrangement = self.read_csv(battle_arrangement_file)
        if set(existing_arrangement.questions).issubset(set(self.questions)) and set(existing_arrangement.models).issubset(set(self.models)):
            self.battles_in_order = existing_arrangement.battles_in_order
            self._battles_in_order_ele_as_dict = [asdict(obj) for obj in existing_arrangement.battles_in_order]
        else:
            raise Exception('battle arrangement is not match question or models')
        
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
        battle_arrangemet._battles_in_order_ele_as_dict = [asdict(obj) for obj in rounds]
        
        return battle_arrangemet
    
    def __repr__(self) -> str:
        return str({'num_of_battle': len(self.battles_in_order)})
            
            
    def more_battles(self, pairs: List[PairToBattle], is_battlepair_in_list_despite_aborder=True):
        """The question and two models in each adding pairs should be contains in the battle arrangment. And adding the new and skip the existing pairs."""
        def is_equivalent_battlepair(pair1: PairToBattle, pair2: PairToBattle, battlepair_despite_aborder):
            if pair1.question != pair2.question:
                return False
            if battlepair_despite_aborder:
                return (pair1.model_a == pair2.model_a and pair1.model_b == pair2.model_b) or ((pair1.model_a == pair2.model_b and pair1.model_b == pair2.model_a))
            else:
                return pair1.model_a == pair2.model_a and pair1.model_b == pair2.model_b
        
        def is_battlepair_in_list_despite_aborder(pairs_list, pair_to_check):
            """Check if a list contains an equivalent PairToBattle instance, regardless of model order."""
            return any(is_equivalent_battlepair(pair, pair_to_check, battlepair_despite_aborder=True) for pair in pairs_list)


        new_battle_counter = 0         
        for pair in pairs:
            if pair.question not in self.questions or pair.model_a not in self.models or pair.model_b not in self.models:
                raise Exception("invalid adding pairs")
            if is_battlepair_in_list_despite_aborder(self.battles_in_order, pair):
                continue
            else:
                self.battles_in_order.append(pair)
                self._battles_in_order_ele_as_dict.append(asdict(pair))
                new_battle_counter+=1
        logger.debug(f'new {new_battle_counter} battle added!')
        
        return new_battle_counter
                
        
    def get_questions_to_arrange(self, model_a, model_b):
        # TODO: optimize the speed
        # battled_pairs_dict = [asdict(obj) for obj in self.battles_in_order]
        df = pd.DataFrame.from_dict(self._battles_in_order_ele_as_dict)
        
        arranged_questions = df[((df['model_a']==model_a) & (df['model_b']==model_b)) | (df['model_a']==model_b) & (df['model_b']==model_a)]['question'].unique().tolist()
        
        return list(set(self.questions) - set(arranged_questions))
        
        
    
        
                