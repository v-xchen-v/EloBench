from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple
import pandas as pd
from .column_names import MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME, WINNER_COLUMN_NAME
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import glob
from elo_rating.rating_helper import get_elo_results_from_battles_data

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
        return cls.read_df(df)
    
    @classmethod
    def read_df(cls, df: pd.DataFrame):
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
    
    def get_leaderboard(self, K: int):
        battled_outcomes_df = self.to_df()
        
        # filter valid battle outcomes        
        caring_winner = ['model_a', 'model_b', 'tie', 'tie(all bad)']
        battled_outcomes_df = battled_outcomes_df[battled_outcomes_df['winner'].isin(caring_winner)]

        return get_elo_results_from_battles_data(battled_outcomes_df, K)
    
class BootstrapedBattleOutcomes:
    """
    Represents a collection of bootstraped battle outcomes.

    Key Features:
    1. Bootstrap resampling on the nature battle outcomes.
    2. Save the bootstraped battle outcomes as CSV files.
    3. Reload the bootstraped battle outcomes from CSV files.
    4. Access the bootstraped battle outcomes as DataFrame objects.
    
    Attributes:
        nature_battle_outcomes (BattleOutcomes): The original nature battle outcomes.
        num_of_bootstrap (int): The number of bootstrap iterations.
        _bootstraped_battlecomes_dfs (list): A list of DataFrame objects representing the bootstraped battle outcomes.
        save_path_pattern (str): The pattern used to generate the save path for each bootstraped battle outcome.
    """

    def __init__(self, nature_battle_outcomes: BattleOutcomes=None, num_of_bootstrap: int=None) -> None:
        """
        Initializes a new instance of the BootstrapedBattleOutcomes class.

        Args:
            nature_battle_outcomes (BattleOutcomes, optional): The original nature battle outcomes. Defaults to None.
            num_of_bootstrap (int, optional): The number of bootstrap iterations. Defaults to None.
        """
        self.nature_battle_outcomes = nature_battle_outcomes
        self.num_of_bootstrap = num_of_bootstrap
        self._bootstraped_battlecomes_dfs = [] # list of dataframe
        
        if self.nature_battle_outcomes is not None and self.num_of_bootstrap is not None:
            self._do_bootstrap(num_of_bootstrap)
        
        self.save_path_pattern = '.bootstrap/battled_pairs_num_of_bootstrap.csv'
        
    def _do_bootstrap(self, num_of_bootstrap):
        """
        Performs bootstrap resampling on the nature battle outcomes.

        Args:
            num_of_bootstrap (int): The number of bootstrap iterations.
        
        Raises:
            Exception: If nature_battle_outcomes is None.
        """
        if self.nature_battle_outcomes is None:
            raise Exception('nature_battle_outcomes is None')
        
        nature_battle_comes_df = self.nature_battle_outcomes.to_df()
        np.random.seed(42)
        for _ in tqdm(range(num_of_bootstrap), desc="bootstrap"):
            # performing a random shuffle of the entire DataFrame
            bootstraped_battle_outcomes_df = nature_battle_comes_df.sample(frac=1.0, replace=False)
            self._bootstraped_battlecomes_dfs.append(bootstraped_battle_outcomes_df)
    
    def to_csv(self, save_dir):
        """
        Saves the bootstraped battle outcomes as CSV files.

        Args:
            save_dir (str): The directory where the CSV files will be saved.
        """
        if not os.path.exists(Path(save_dir)/'.bootstrap'):
            os.makedirs(Path(save_dir)/'.bootstrap')
            
        for i, battle_outcomes_df in tqdm(enumerate(self._bootstraped_battlecomes_dfs), desc='saving bootstrap battled pairs...'):
            save_path = Path(save_dir)/self.save_path_pattern.replace('num_of_bootstrap', str(i+1).zfill(5))
            battle_outcomes_df.to_csv(save_path)
            
    @classmethod
    def is_cached(cls, save_dir: str, num_of_bootstrap: int):
         # load cached bootstraped battle outcomes
        bootstrap_outcomes_files = glob.glob(str(Path(save_dir)/'.bootstrap/battled_pairs_*.csv'))  
        return len(bootstrap_outcomes_files) == num_of_bootstrap
          
    @classmethod
    def read_csv(cls, save_dir): 
        """
        Reads the bootstraped battle outcomes from CSV files.

        Args:
            save_dir (str): The directory where the CSV files are located.
        
        Returns:
            BootstrapedBattleOutcomes: An instance of the BootstrapedBattleOutcomes class with the loaded data.
        """
        bootstrap_battle_outcomes = BootstrapedBattleOutcomes()
        
        # load cached bootstraped battle outcomes
        bootstrap_outcomes_files = glob.glob(str(Path(save_dir)/'.bootstrap/battled_pairs_*.csv'))  
        bootstrap_outcomes_files.sort()  
        
        bootstrap_battle_outcomes.num_of_bootstrap = len(bootstrap_outcomes_files)
        for bootstrap_outcomes_file in tqdm(bootstrap_outcomes_files, desc='loading cached bootstraped battle outcomes'):
            bootstrap_battle_outcomes._bootstraped_battlecomes_dfs.append(pd.read_csv(bootstrap_outcomes_file))
            
        return bootstrap_battle_outcomes
        
    def __getitem__ (self, idx):
        return self._bootstraped_battlecomes_dfs[idx]
    
    def get_leaderboards(self, K: int) -> pd.DataFrame:
        elo_dfs = []
        # for i, battle_outcomes_df in tqdm(enumerate(self._bootstraped_battlecomes_dfs), desc='calculating bootstrap elo ratings', total=len(self._bootstraped_battlecomes_dfs)):
        for i, battle_outcomes_df in enumerate(self._bootstraped_battlecomes_dfs):
            elo_df = get_elo_results_from_battles_data(battle_outcomes_df, K)
            # Adding a new column with the same value for all rows
            elo_df['round_bootstrap'] = i+1
            elo_dfs.append(elo_df)

        elo_df = pd.concat(elo_dfs)
        return elo_df
    
    def get_leaderboard(self, K: int):
        elo_df = self.get_leaderboards(K)
        inclusive_columns = ['model', 'elo_rating']
        elo_df = elo_df[inclusive_columns]
        
        # calculate the median of elo ratings
        elo_df = elo_df.groupby('model').median().reset_index()
        elo_df["elo_rating"] = (elo_df["elo_rating"] + 0.5).astype(int)
        elo_df.sort_values(by=['elo_rating'], ascending=False, inplace=True)
        elo_df.reset_index(drop=True)
        return elo_df
    
    def get_first_n_rows(self, n_rows):
        first_n_rows_bootstrap_battle_outcomes =  BootstrapedBattleOutcomes()
        first_n_rows_bootstrap_battle_outcomes.nature_battle_outcomes = self.nature_battle_outcomes
        first_n_rows_bootstrap_battle_outcomes.num_of_bootstrap = self.num_of_bootstrap
        first_n_rows_bootstrap_battle_outcomes._bootstraped_battlecomes_dfs = [x.head(n_rows) for x in self._bootstraped_battlecomes_dfs] # list of dataframe
        return first_n_rows_bootstrap_battle_outcomes