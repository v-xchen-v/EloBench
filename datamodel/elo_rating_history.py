from __future__ import annotations

from dataclasses import dataclass, asdict
import pandas as pd
from typing import List
from datamodel.battle_outcome import BattleOutcomes
from tqdm import tqdm
from elo_rating import rating_helper
from pathlib import Path

@dataclass
class EloRatingHistoryPoint:
    rank: int
    model: str
    elo_rating: float
    num_battle: int
    
class EloRatingHistory:
    def __init__(self) -> None:
        self.elo_rating_history = []
        self.recorded_battle_num = []
        
    @classmethod
    def gen_history(self, save_dir, step=100, use_bootstrap=False, BOOTSTRAP_ROUNDS = 100, nrows=None) -> EloRatingHistory:
        # create empty history
        elo_rating_history = EloRatingHistory()
        
        # read battle outcomes
        battled_pairs = BattleOutcomes.read_csv(Path(save_dir) / 'battled_pairs.csv', nrows=nrows)

        battled_pairs_list = [asdict(obj) for obj in battled_pairs.battled_pairs_in_order]

        
        for idx_battle in tqdm(range(len(battled_pairs_list)), desc="generating elo rating history"):
                num_battle = idx_battle + 1
                if num_battle > 0 and (num_battle % step == 0 or idx_battle == len(battled_pairs_list)-1):
                    historypoint_battles_df = pd.DataFrame.from_dict(battled_pairs_list[:idx_battle+1])
                    if use_bootstrap:                     
                        historypoint_rating_df = rating_helper.get_bootstrap_medium_elo(historypoint_battles_df, K=4, BOOTSTRAP_ROUNDS=BOOTSTRAP_ROUNDS)
                    else:
                        historypoint_rating_df = rating_helper.get_elo_results_from_battles_data(historypoint_battles_df, K=4)
                    # elo_rating_history_logger.debug(historypoint_rating_df)
                    elo_rating_history.add_point(historypoint_rating_df, idx_battle+1)
        return elo_rating_history
        
    def _add_row(self, rank: int, model: str, elo_rating: float, num_battle: int):
        self.elo_rating_history.append(EloRatingHistoryPoint(rank, model, elo_rating, num_battle))
        
    def add_point(self, elo_rating_data: pd.DataFrame, num_battle: int):
        def get_rank(rating, all_ratings: list):
            """assign the rank based on the descending order of the numbers, ensuring that the largest number gets the rank 1, and the smallest number gets the rank equal to the number of unique elements in the list. Rank of numbers in a list where the same numbers have the same rank"""
            rank = 1
            for item in all_ratings:
                if rating < item:
                    rank+=1
            return rank
        
        for _, row in elo_rating_data.iterrows():
            rank = get_rank(row['elo_rating'], elo_rating_data['elo_rating'].tolist())
            self._add_row(rank, row['model'], row['elo_rating'], num_battle)
        self.recorded_battle_num.append(num_battle)
    
    def get_point(self, num_battle: int) -> pd.DataFrame:
        elo_rating_list = []
        for row in self.elo_rating_history:
            if row.num_battle == num_battle:
                 # Create a dictionary for each row and add it to the list
                elo_rating_list.append({
                    'model': row.model,
                    'elo_rating': row.elo_rating,
                    'rank': row.rank,
                })
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(elo_rating_list)

        # If 'rank' should be the index of the DataFrame
        # if not df.empty:
        #     df.set_index('rank', inplace=True)
        df.sort_values(by='rank', inplace=True)

        return df
        
    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame([asdict(point) for point in self.elo_rating_history]).sort_values(by=['num_battle', 'rank'], ascending=True)
        return df
    
    def to_csv(self, save_path) -> None:
        self.to_df().to_csv(save_path, index=False)

    @classmethod
    def read_csv(self, save_path) -> None:
        df = pd.read_csv(save_path)
        elo_rating_history = EloRatingHistory()
        for _, row in df.iterrows():
            elo_rating_history._add_row(EloRatingHistoryPoint(row['model'], row['elo_rating'], row['num_battle']))
            elo_rating_history.recorded_battle_num.append(row['num_battle'])
        elo_rating_history.recorded_battle_num = list(set(elo_rating_history.recorded_battle_num))
        return elo_rating_history
    