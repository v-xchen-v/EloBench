from dataclasses import dataclass, asdict
from typing import List
import pandas as pd
from .columns import MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME, WINNER_COLUMN_NAME

@dataclass
class BattledPair:
    model_a: str
    model_b: str
    winner: str

class BattledPairs:
    def __init__(self, battled_pairs: List[BattledPair] =[]) -> None:
        self.battled_pairs_in_order = battled_pairs
        
    def add_pair(self, model_a:str, model_b: str, winner: str):
        self.battled_pairs_in_order.append(BattledPair(model_a, model_b, winner))
        
    def add_pairs(self, pairs: List[BattledPair]):
        self.battled_pairs_in_order.append(pairs)
        
    def to_csv(self, save_path):
        pd.DataFrame([list(asdict(pair).values()) for pair in self.battled_pairs_in_order], columns=[MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME, WINNER_COLUMN_NAME]).to_csv(save_path)
    
    @classmethod
    def read_csv(cls, save_path):        
        df = pd.read_csv(save_path)
        pairs = []
        for idx, row in df.iterrows():
            pairs.append(BattledPair(model_a=row[MODEL_A_COLUMN_NAME], model_b=row[MODEL_B_COLUMN_NAME], winner=row[WINNER_COLUMN_NAME]))
            
        return BattledPairs(pairs)
    
    def __repr__(self) -> str:
        return str(self.battled_pairs_in_order)