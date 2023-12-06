from dataclasses import dataclass, asdict
import pandas as pd
from typing import List

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
        
    def _add_row(self, rank: int, model: str, elo_rating: float, num_battle: int):
        self.elo_rating_history.append(EloRatingHistoryPoint(rank, model, elo_rating, num_battle))
        
    def add_point(self, elo_rating_data: pd.DataFrame, num_battle: int):
        for idx, row in elo_rating_data.iterrows():
            self._add_row(idx, row['model'], row['elo_rating'], num_battle)
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
    