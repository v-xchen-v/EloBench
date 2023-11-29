from dataclasses import dataclass, asdict
from typing import List
import pandas as pd
from .columns import MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME, WINNER_COLUMN_NAME
from collections import defaultdict

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
    
    def clear(self):
        self.battled_pairs_in_order = []
    
    @classmethod
    def read_csv(cls, save_path):        
        df = pd.read_csv(save_path)
        pairs = []
        for idx, row in df.iterrows():
            pairs.append(BattledPair(model_a=row[MODEL_A_COLUMN_NAME], model_b=row[MODEL_B_COLUMN_NAME], winner=row[WINNER_COLUMN_NAME]))
            
        return BattledPairs(pairs)
    
    def __repr__(self) -> str:
        return str(self.battled_pairs_in_order)
    
    def frequency(self,despite_ab_order=True):
        return self.get_frequency(self.battled_pairs_in_order, despite_ab_order)
        
    def preety_print_frequency(self, frequery_dict):
        print(pd.DataFrame.from_dict(frequery_dict))
        
    @classmethod
    def get_frequency(cls, battled_pairs: List[BattledPair], despite_ab_order=True) -> dict:
        battled_pairs_dict = [asdict(obj) for obj in battled_pairs]
        battled_pairs_df = pd.DataFrame.from_dict(battled_pairs_dict)
        
        union_model_names = pd.concat([
        pd.Series(battled_pairs_df['model_a'].unique()), 
        pd.Series(battled_pairs_df['model_b'].unique())]).unique()
        
        all_counter_table = defaultdict(lambda: defaultdict(lambda: 0))
        no_tie_counter_table = defaultdict(lambda: defaultdict(lambda: 0))
        
        def get_a_battle_b_count(battles, a, b, default_as_na=False):
            valid_battles = battles[battles['winner']!='invalid']
            if a == b:
                return (pd.NA, pd.NA) if default_as_na else (0, 0)
            else:
                if despite_ab_order:
                    ab_battles = valid_battles[((valid_battles['model_a']==a) & (valid_battles['model_b']==b)) | (valid_battles['model_a']==b) & (valid_battles['model_b']==a)]
                else:
                    ab_battles = valid_battles[((valid_battles['model_a']==a) & (valid_battles['model_b']==b))]
                
                if ab_battles.shape[0] == 0 and default_as_na:
                    return pd.NA, pd.NA
                
                ab_battles.loc[:, 'winner'] = ab_battles['winner'].astype('str')
                ab_notie_battles = ab_battles[~ab_battles['winner'].str.startswith('tie')]
                return ab_battles.shape[0], ab_notie_battles.shape[0]
        
        for a in union_model_names:
            for b in union_model_names:
                if a == b:
                    continue
                
                battle_count, notie_battle_count = get_a_battle_b_count(battled_pairs_df, a, b, False)
                
                all_counter_table[a][b] = battle_count
                no_tie_counter_table[a][b] = notie_battle_count
        
        return all_counter_table, no_tie_counter_table
    