from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Dict, Union
import pandas as pd
from datamodel.column_names import QUESTION_COLUMN_NAME, MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME
import itertools
from enum import Enum
from logger import logger, info_logger
import numpy as np
from dataclasses import asdict
from collections import defaultdict
from tqdm import tqdm

class ArrangementStrategy(Enum):
    Random_N_BattleCount_Each_CombPair = 0,
    Reload_Existing_Arrangement = 1,
    All_Combination = 2,
    Random_N_Pairs_Per_Question = 3,
    Random_N_BattleCount_Each_Model = 4,
    Random_N_Questions_Per_Pair = 5,
    
@dataclass(frozen=True)
class PairToBattle:
    question: str
    model_a: str
    model_b: str
    
class PairwiseBattleArrangement:
    """
    Represents a pairwise battle arrangement for comparing models on a set of questions.
    
    Attributes:
        questions (List[str]): The list of questions to be asked in the battles.
        models (List[str]): The list of models to be compared in the battles.
        battles_in_order (List[PairToBattle]): The list of battles in the arranged order.
        _battles_in_order_ele_as_dict (List[Dict[str, Union[str, int]]]): The list of battles as dictionaries.
    """
    
    def __init__(self, questions: List[str], models: List[str]) -> None:
        """
        Initializes a new instance of the PairwiseBattleArrangement class.
        
        Args:
            questions (List[str]): The list of questions to be asked in the battles.
            models (List[str]): The list of models to be compared in the battles.
        """
        self.questions: List[str] = questions
        self.models: List[str] = models
        # records how many times a question has been involved in a battle
        self.question_frequency = defaultdict(int)
        for q in self.questions:
            self.question_frequency[q] = 0
        self.battles_in_order: List[PairToBattle] = []
        self._battles_in_order_ele_as_dict: List[Dict[str, Union[str, int]]] = []
    def arrange(self, arrange_strategy: ArrangementStrategy, **kwargs):
        info_logger.info(f'arrange battles by {arrange_strategy}')
        """
        Arranges the battles based on the specified arrangement strategy.
        
        Args:
            arrange_strategy (ArrangementStrategy): The arrangement strategy to be used.
            **kwargs: Additional arguments specific to the arrangement strategy.
        """
        if arrange_strategy == ArrangementStrategy.Random_N_Pairs_Per_Question:
            self.arrange_randomly_by_pairnumperquesiton(**kwargs)
        elif arrange_strategy == ArrangementStrategy.Random_N_BattleCount_Each_Model:
            self.arrange_randomly_by_pairnumpermodel(**kwargs)
        elif arrange_strategy == ArrangementStrategy.All_Combination:
            self.arrange_all_combinations()
        elif arrange_strategy == ArrangementStrategy.Reload_Existing_Arrangement:
            self.arrange_by_existing_arrangement(**kwargs)
        elif arrange_strategy == ArrangementStrategy.Random_N_BattleCount_Each_CombPair:
            self.arrange_randomly_by_battlecountnumpercombpair(**kwargs)
        else:
            raise NotImplementedError
        info_logger.info(f'arrange battles by {arrange_strategy} done')
    
    def _shuffle_battles_ab(self, battles: List[PairToBattle]):
        """
        Shuffles the battles randomly.
        
        Args:
            battles (List[PairToBattle]): The list of battles to be shuffled.
        
        Returns:
            List[PairToBattle]: The shuffled list of battles.
        """
        battled_pairs_dict = [asdict(obj) for obj in battles]
        df = pd.DataFrame(battled_pairs_dict)
        
        # Create a random boolean mask
        mask = np.random.rand(len(battles)) > 0.5

        # Shuffle using the mask
        temp = df[MODEL_A_COLUMN_NAME][mask].copy()
        df[MODEL_A_COLUMN_NAME][mask] = df[MODEL_B_COLUMN_NAME][mask]
        df[MODEL_B_COLUMN_NAME][mask] = temp
        
        shuffled_battles = []
        for _, item in df.iterrows():
            shuffled_battles.append(PairToBattle(item[QUESTION_COLUMN_NAME], item[MODEL_A_COLUMN_NAME], item[MODEL_B_COLUMN_NAME]))
        return shuffled_battles
    
    def arrange_randomly_by_battlecountnumpercombpair(self, num_of_battle: int, shuffle=True):
        # generate all unique pairs from the list
        all_pairs = list(itertools.combinations(self.models, 2))
        
        if num_of_battle > len(self.questions):
            raise Exception('num_of_battle is larger than question number')
        # selected_questiosn = random.sample(self.questions, num_of_battle)
        
        battles = []
        # for question in selected_questiosn:
        for pair in tqdm(all_pairs, desc='comb pairs'):
            model_a = pair[0]
            model_b = pair[1]
            questions = self.random_select_question_to_arrange_by_frequency(model_a, model_b, num_of_battle)
            if questions is None:
                raise Exception('no more questions to arrange')
            
            for question in questions:
                battles.append(PairToBattle(question=question, model_a=model_a, model_b=model_b))
                self.question_frequency[question] += 1

        if shuffle:
            battles = self._shuffle_battles_ab(battles)
        self.battles_in_order = battles
        self._battles_in_order_ele_as_dict = [asdict(obj) for obj in battles]
        
        
    def arrange_all_combinations(self, shuffle=True):
        """
        Arranges the battles by considering all possible combinations of models for each question.
        
        Args:
            shuffle (bool): Indicates whether to shuffle the battles after arranging them. Default is True.
        """
        # generate all unique pairs from the list
        all_pairs = list(itertools.combinations(self.models, 2))
        battles = []
        for question in self.questions:
            for pair in all_pairs:
                battles.append(PairToBattle(question=question, model_a=pair[0], model_b=pair[1]))
                self.question_frequency[question] += 1

        if shuffle:
            battles = self._shuffle_battles_ab(battles)
        self.battles_in_order = battles
        self._battles_in_order_ele_as_dict = [asdict(obj) for obj in battles]
    
    def arrange_randomly_by_pairnumperquesiton(self, num_of_pair: int, shuffle=True):
        """
        Arranges the battles randomly by selecting a specific number of pairs per question.
        
        Args:
            num_of_pair (int): The number of pairs to be selected per question.
            shuffle (bool): Indicates whether to shuffle the battles after arranging them. Default is True.
        """
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
                self.question_frequency[question] += 1
                
        if shuffle:
            battles = self._shuffle_battles_ab(battles)
        self.battles_in_order = battles
        self._battles_in_order_ele_as_dict = [asdict(obj) for obj in battles]
    
    def arrange_randomly_by_pairnumpermodel(self, num_of_pair:int, shuffle=True):
        """
        Arranges the battles randomly by selecting a specific number of pairs for each model.
        
        Args:
            num_of_pair (int): The number of pairs to be selected for each model.
            shuffle (bool): Indicates whether to shuffle the battles after arranging them. Default is True.
        """
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
        
        battles = []
        for q, model_pair in selected_combinations:
            self.question_frequency[q] += 1
            battles.append(PairToBattle(question=q, model_a=model_pair[0], model_b=model_pair[1]))
        
        if shuffle:
            battles = self._shuffle_battles_ab(battles)
        self.battles_in_order = battles
        self._battles_in_order_ele_as_dict = [asdict(obj) for obj in battles]
        
        
    def arrange_by_existing_arrangement(self, file: str):
        """
        Arranges the battles based on an existing battle arrangement file.
        
        Args:
            battle_arrangement_file (str): The path to the battle arrangement file.
        
        Raises:
            Exception: If the battle arrangement does not match the questions or models.
        """
        existing_arrangement = self.read_csv(file)
        if set(existing_arrangement.questions).issubset(set(self.questions)) and set(existing_arrangement.models).issubset(set(self.models)):
            self.battles_in_order = existing_arrangement.battles_in_order
            self._battles_in_order_ele_as_dict = [asdict(obj) for obj in existing_arrangement.battles_in_order]
            self.question_frequency = existing_arrangement.question_frequency
        else:
            raise Exception('battle arrangement is not match question or models')
        
    def to_csv(self, save_path):
        """
        Saves the battle arrangement to a CSV file.
        
        Args:
            save_path (str): The path to save the CSV file.
        """
        arrangments = []
        for battle in self.battles_in_order:
            arrangments.append([battle.question, battle.model_a, battle.model_b])
            
        arrangments_df = pd.DataFrame(arrangments, columns=[QUESTION_COLUMN_NAME, MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME])
        arrangments_df.to_csv(save_path)
        
    @classmethod
    def read_csv(cls, save_path) -> PairwiseBattleArrangement:
        """
        Reads a battle arrangement from a CSV file.
        
        Args:
            save_path (str): The path to the CSV file.
        
        Returns:
            PairwiseBattleArrangement: The battle arrangement read from the CSV file.
        """
        df = pd.read_csv(save_path)
        models = set()
        questions = set()
        rounds = []
        question_frequency = defaultdict(int)
        for index, row in df.iterrows():
            q = row[QUESTION_COLUMN_NAME]
            m_a = row[MODEL_A_COLUMN_NAME]
            m_b = row[MODEL_B_COLUMN_NAME]
            rounds.append(PairToBattle(q, m_a, m_b))
            models.add(m_a)
            models.add(m_b)
            questions.add(q)
            question_frequency[q]+=1
            
        battle_arrangemet = PairwiseBattleArrangement(questions=list(questions), models=list(models))
        battle_arrangemet.battles_in_order = rounds
        battle_arrangemet._battles_in_order_ele_as_dict = [asdict(obj) for obj in rounds]
        battle_arrangemet.question_frequency = question_frequency
        
        return battle_arrangemet
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the PairwiseBattleArrangement object.
        
        Returns:
            str: The string representation of the object.
        """
        return str({'num_of_battle': len(self.battles_in_order)})
            
            
    def more_battles(self, pairs: List[PairToBattle], is_battlepair_in_list_despite_aborder=True, only_on_exists_q_and_models=True):
        """
        Adds more battles to the existing battle arrangement.
        
        Args:
            pairs (List[PairToBattle]): The list of PairToBattle instances to be added.
            is_battlepair_in_list_despite_aborder (bool): Indicates whether to consider battle pairs in the list regardless of model order. Default is True.
        
        Returns:
            int: The number of new battles added.
        """
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
            if only_on_exists_q_and_models and (pair.question not in self.questions or pair.model_a not in self.models or pair.model_b not in self.models):
                raise Exception("invalid adding pairs")
            if is_battlepair_in_list_despite_aborder(self.battles_in_order, pair):
                continue
            else:
                self.battles_in_order.append(pair)
                self._battles_in_order_ele_as_dict.append(asdict(pair))
                new_battle_counter+=1
                self.question_frequency[pair.question] += 1
        logger.debug(f'new {new_battle_counter} battle added!')
        
        return new_battle_counter
                
        
    def get_questions_to_arrange(self, model_a, model_b):
        """
        Gets the list of questions that need to be arranged for the specified model pair.
        
        Args:
            model_a (str): The first model in the pair.
            model_b (str): The second model in the pair.
        
        Returns:
            List[str]: The list of questions to be arranged for the model pair.
        """
        df = pd.DataFrame.from_dict(self._battles_in_order_ele_as_dict)
        
        if df.shape[0] > 0:
            arranged_questions = df[((df[MODEL_A_COLUMN_NAME]==model_a) & (df[MODEL_B_COLUMN_NAME]==model_b)) | (df[MODEL_A_COLUMN_NAME]==model_b) & (df[MODEL_B_COLUMN_NAME]==model_a)][QUESTION_COLUMN_NAME].unique().tolist()
        
            return list(set(self.questions) - set(arranged_questions))
        else:
            return self.questions
    
    def random_select_question_to_arrange_by_frequency(self, model_a, model_b, size=1):
        """
        Randomly selects a question to be arranged for the specified model pair based on the frequency of questions.
        
        Args:
            model_a (str): The first model in the pair.
            model_b (str): The second model in the pair.
        
        Returns:
            str: The selected question.
        """
        questions_to_arrange = self.get_questions_to_arrange(model_a, model_b)
        
        if len(questions_to_arrange) == 0 or size > len(questions_to_arrange):
            return None
        
        probabilities = list(self.question_frequency.values())
        
        # Add 1 to each probability to avoid zero probability
        # Invert to form a probability distribution that favors questions with lower frequency
        probabilities = [1/(i+1) for i in probabilities]

        
        # Normalize probabilities so they sum up to 1
        probabilities = [float(i)/sum(probabilities) for i in probabilities]

        selected_indexs = np.random.choice(len(probabilities), p=probabilities, replace=True, size=size)
        
        return list(np.array(list(self.question_frequency.keys()))[selected_indexs])
        
        
    
        
                