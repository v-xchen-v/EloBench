from dataclasses import dataclass, asdict
from typing import List, Tuple
import pandas as pd
from .column_names import MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME, WINNER_COLUMN_NAME
from collections import defaultdict
from collections import defaultdict

@dataclass
class BattleOutcome:
    """
    Represents the outcome of a battle between two models.

    Attributes:
        model_a (str): The name of model A.
        model_b (str): The name of model B.
        winner (str): The name of the winning model.
    """
    model_a: str
    model_b: str
    winner: str

class BattleOutcomes:
    """
    Class representing a collection of battle outcomes between models.
    """

    def __init__(self, battled_pairs: List[BattleOutcome] =[]) -> None:
        """
        Initialize the BattleOutcomes object.

        Parameters:
        - battled_pairs (List[BattleOutcome]): List of BattleOutcome objects representing the battled pairs.
        """
        self.battled_pairs_in_order = battled_pairs
        
    def add_pair(self, model_a:str, model_b: str, winner: str):
        """
        Add a battle pair to the collection.

        Parameters:
        - model_a (str): Name of model A.
        - model_b (str): Name of model B.
        - winner (str): Name of the winner model.
        """
        self.battled_pairs_in_order.append(BattleOutcome(model_a, model_b, winner))
        
    def add_pairs(self, pairs: List[BattleOutcome]):
        """
        Add multiple battle pairs to the collection.

        Parameters:
        - pairs (List[BattleOutcome]): List of BattleOutcome objects representing the battled pairs.
        """
        # ! seems useless
        for pair in pairs:
            self.add_pair(pair.model_a, pair.model_b, pair.winner)
        
    def to_csv(self, save_path):
        """
        Save the battle outcomes to a CSV file.

        Parameters:
        - save_path (str): Path to save the CSV file.
        """
        pd.DataFrame([list(asdict(pair).values()) for pair in self.battled_pairs_in_order], columns=[MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME, WINNER_COLUMN_NAME]).to_csv(save_path)
    
    def clear(self):
        """
        Clear the collection of battle outcomes.
        """
        self.battled_pairs_in_order = []
    
    @classmethod
    def read_csv(cls, save_path, nrows=None):        
        """
        Read battle outcomes from a CSV file.

        Parameters:
        - save_path (str): Path to the CSV file.

        Returns:
        - BattleOutcomes: BattleOutcomes object containing the battle outcomes read from the CSV file.
        """
        df = pd.read_csv(save_path, nrows=nrows)

        pairs = []
        for idx, row in df.iterrows():
            # TODO: check if the row is valid
            pairs.append(BattleOutcome(model_a=row[MODEL_A_COLUMN_NAME], model_b=row[MODEL_B_COLUMN_NAME], winner=row[WINNER_COLUMN_NAME]))
            
        return BattleOutcomes(pairs)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the BattleOutcomes object.

        Returns:
        - str: String representation of the BattleOutcomes object.
        """
        return str(self.battled_pairs_in_order)
    
    def frequency(self,despite_ab_order=True):
        """
        Calculate the frequency of battle outcomes.

        Parameters:
        - despite_ab_order (bool): Flag indicating whether to consider the order of model A and model B in the battle pairs.

        Returns:
        - dict: Dictionary containing the frequency of battle outcomes.
        """
        valid_winner = set(['model_a', 'model_b', 'tie', 'tie(all bad)'])
        valid_battled_pairs = [x for x in self.battled_pairs_in_order if x.winner in valid_winner]
        return self.get_frequency(valid_battled_pairs, despite_ab_order)
        
    def preety_print_frequency(self, frequery_dict):
        """
        Print the frequency of battle outcomes in a pretty format.

        Parameters:
        - frequery_dict (dict): Dictionary containing the frequency of battle outcomes.
        """
        print(pd.DataFrame.from_dict(frequery_dict))
        
    @classmethod
    def get_frequency(cls, battled_pairs: List[BattleOutcome], despite_ab_order=True) -> Tuple[dict, dict]:
        """
        Calculate the frequency of battle outcomes.

        Parameters:
        - battled_pairs (List[BattleOutcome]): List of BattleOutcome objects representing the battled pairs.
        - despite_ab_order (bool): Flag indicating whether to consider the order of model A and model B in the battle pairs.

        Returns:
        - dict: Dictionary containing the frequency of battle outcomes.
        """
        battled_pairs_dict = [asdict(obj) for obj in battled_pairs]
        battled_pairs_df = pd.DataFrame(battled_pairs_dict)

        all_counter_table: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
        no_tie_counter_table: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
        
        def get_a_battle_b_count(battles, a, b, default_as_na=False):
            valid_battles = battles[battles['winner']!='invalid']
            if a == b:
                return (pd.NA, pd.NA) if default_as_na else (0, 0)
            else:
                if despite_ab_order:
                    ab_battles = valid_battles[((valid_battles[MODEL_A_COLUMN_NAME]==a) & (valid_battles[MODEL_B_COLUMN_NAME]==b)) | (valid_battles[MODEL_A_COLUMN_NAME]==b) & (valid_battles[MODEL_B_COLUMN_NAME]==a)]
                else:
                    ab_battles = valid_battles[((valid_battles[MODEL_A_COLUMN_NAME]==a) & (valid_battles[MODEL_B_COLUMN_NAME]==b))]
                
                if ab_battles.shape[0] == 0 and default_as_na:
                    return pd.NA, pd.NA
                
                ab_battles.loc[:, WINNER_COLUMN_NAME] = ab_battles[WINNER_COLUMN_NAME].astype('str')
                ab_notie_battles = ab_battles[~ab_battles[WINNER_COLUMN_NAME].str.startswith('tie')]
                return ab_battles.shape[0], ab_notie_battles.shape[0]
        
        union_model_names = pd.concat([
        pd.Series(battled_pairs_df[MODEL_A_COLUMN_NAME].unique()), 
        pd.Series(battled_pairs_df[MODEL_B_COLUMN_NAME].unique())]).unique()
            
        for a in union_model_names:
            for b in union_model_names:
                if a == b:
                    continue
                
                battle_count, notie_battle_count = get_a_battle_b_count(battled_pairs_df, a, b, False)
                
                all_counter_table[a][b] = battle_count
                no_tie_counter_table[a][b] = notie_battle_count
        
        return all_counter_table, no_tie_counter_table
    
    def to_df(self):
        return self._to_df(self.battled_pairs_in_order)
    
    def _to_df(self, battled_pairs: List[BattleOutcome]):
        battled_pairs_dict = [asdict(obj) for obj in battled_pairs]
        battled_pairs_df = pd.DataFrame(battled_pairs_dict)
        return battled_pairs_df
        
    def __getitem__ (self, idx):
        return self.battled_pairs_in_order[idx]