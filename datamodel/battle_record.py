from dataclasses import dataclass
from typing import List
from dataclasses import asdict
import pandas as pd
from datamodel import PairToBattle
from logger import logger

@dataclass
class BattleRecord:
    model_a: str
    model_b: str
    winner: str
    judger: str # gpt_4(_with_version) or ?
    # turn:
    # language:
    tstamp: str
    question: str
    answer_a: str
    answer_b:str
    gpt_4_response: str
    gpt_4_score: str
    is_valid: bool # exists one ans is empty or gpt-4 eval error or something like that
    
class BattleRecords:
    def __init__(self, records: List[BattleRecord]) -> None:
        self.records = records
        
        # initialized by initial records
        self.record_pair_set = set([PairToBattle(x.question, x.model_a, x.model_b) for x in records]) # A set to keep track of unique items
        self.record_map = {PairToBattle(x.question, x.model_a, x.model_b):x for x in records}
    
    def add_record(self, record: BattleRecord):
        if not self.record_exists(PairToBattle(question=record.question, model_a=record.model_a, model_b=record.model_b)):
            self.records.append(record)
            self.record_pair_set.add(PairToBattle(record.question, record.model_a, record.model_b))
            self.record_map[PairToBattle(record.question, record.model_a, record.model_b)] = record
        # else:
        #     logger.debug('battle pairs already exists in records, skip!')
            
    def add_records(self, records: List[BattleRecord]):
        for rec in records:
            self.add_record(rec)
            
    def to_csv(self, save_path, winner_only=False):
        log_dicts = [asdict(obj) for obj in self.records]
        df = pd.DataFrame.from_dict(log_dicts)
        
        if winner_only:
            inclusive_cols = ['model_a', 'model_b', 'winner']
            df = df[inclusive_cols]
        
        df.to_csv(save_path) 
        
    @classmethod
    def from_csv(cls, save_path):
        df = pd.read_csv(save_path)
        records = []
        for idx, row in df.iterrows():
            records.append(BattleRecord(model_a=row['model_a'], model_b=row['model_b'],  winner=row['winner'], judger=row['judger'], tstamp=row['tstamp'], question=row['question'], answer_a=row['answer_a'], answer_b=row['answer_b'], gpt_4_response=row['gpt_4_response'], gpt_4_score=row['gpt_4_score'], is_valid=row['is_valid']))
            
        return BattleRecords(records)
    
    def record_exists(self, pair_to_battle: PairToBattle):
        # return any(rec.question == pair_to_battle.question and rec.model_a == pair_to_battle.model_a and rec.model_b == pair_to_battle.model_b for rec in self.records)
        return pair_to_battle in self.record_pair_set
    
    def get_record(self, pair_to_battle: PairToBattle):
        # for rec in self.records:
        #     if rec.question == pair_to_battle.question and rec.model_a == pair_to_battle.model_a and rec.model_b == pair_to_battle.model_b:
        #         return rec
        # return None
        return self.record_map.get(pair_to_battle)