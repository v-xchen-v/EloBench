from dataclasses import dataclass
from typing import List
from dataclasses import asdict
import pandas as pd
from datamodel import PairToBattle
from logger import logger
from typing import Optional

# TODO: store the battle records in a database instead of a CSV file for performance considerations

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
    """
    A class representing a collection of battle records.

    Attributes:
    - records (List[BattleRecord]): A list of BattleRecord objects representing the battle records.
    - record_pair_set (set): A set to keep track of unique battle record pairs.
    - record_map (dict): A dictionary mapping PairToBattle objects to BattleRecord objects.

    Methods:
    - __init__(self, records: List[BattleRecord]) -> None: Initializes the BattleRecords object.
    - add_record(self, record: BattleRecord): Adds a new battle record to the collection.
    - add_records(self, records: List[BattleRecord]): Adds multiple battle records to the collection.
    - to_csv(self, save_path, winner_only=False): Converts the battle records to a CSV file.
    - from_csv(cls, save_path): Creates a BattleRecords object from a CSV file.
    - record_exists(self, pair_to_battle: PairToBattle) -> bool: Checks if a battle record exists in the collection.
    - get_record(self, pair_to_battle: PairToBattle) -> Optional[BattleRecord]: Retrieves a battle record from the collection.
    """

    def __init__(self, records: List[BattleRecord]) -> None:
        self.records = records
        
        # initialized by initial records
        self.record_pair_set = set([PairToBattle(x.question, x.model_a, x.model_b) for x in records]) # A set to keep track of unique items
        self.record_map = {PairToBattle(x.question, x.model_a, x.model_b):x for x in records}
    
    def add_record(self, record: BattleRecord):
        """
        Adds a new battle record to the collection.

        Args:
        - record (BattleRecord): The battle record to be added.
        """
        if not self.record_exists(PairToBattle(question=record.question, model_a=record.model_a, model_b=record.model_b)):
            self.records.append(record)
            self.record_pair_set.add(PairToBattle(record.question, record.model_a, record.model_b))
            self.record_map[PairToBattle(record.question, record.model_a, record.model_b)] = record
            
    def add_records(self, records: List[BattleRecord]):
        """
        Adds multiple battle records to the collection.

        Args:
        - records (List[BattleRecord]): The list of battle records to be added.
        """
        for rec in records:
            self.add_record(rec)
            
    def to_csv(self, save_path, winner_only=False):
        """
        Converts the battle records to a CSV file.

        Args:
        - save_path (str): The path to save the CSV file.
        - winner_only (bool): If True, only include the columns 'model_a', 'model_b', and 'winner' in the CSV file.
        """
        log_dicts = [asdict(obj) for obj in self.records]
        df = pd.DataFrame.from_dict(log_dicts)
        
        if winner_only:
            inclusive_cols = ['model_a', 'model_b', 'winner']
            df = df[inclusive_cols]
        
        df.to_csv(save_path) 
        
    @classmethod
    def from_csv(cls, save_path):
        """
        Creates a BattleRecords object from a CSV file.

        Args:
        - save_path (str): The path to the CSV file.

        Returns:
        - BattleRecords: The BattleRecords object created from the CSV file.
        """
        df = pd.read_csv(save_path)
        records = []
        for idx, row in df.iterrows():
            records.append(BattleRecord(model_a=row['model_a'], model_b=row['model_b'],  winner=row['winner'], judger=row['judger'], tstamp=row['tstamp'], question=row['question'], answer_a=row['answer_a'], answer_b=row['answer_b'], gpt_4_response=row['gpt_4_response'], gpt_4_score=row['gpt_4_score'], is_valid=row['is_valid']))
            
        return BattleRecords(records)
    
    def record_exists(self, pair_to_battle: PairToBattle) -> bool:
        """
        Checks if a battle record exists in the collection.

        Args:
        - pair_to_battle (PairToBattle): The PairToBattle object representing the battle record.

        Returns:
        - bool: True if the battle record exists, False otherwise.
        """
        return pair_to_battle in self.record_pair_set
    
    def get_record(self, pair_to_battle: PairToBattle) -> Optional[BattleRecord]:
        """
        Retrieves a battle record from the collection.

        Args:
        - pair_to_battle (PairToBattle): The PairToBattle object representing the battle record.

        Returns:
        - Optional[BattleRecord]: The battle record if found, None otherwise.
        """
        return self.record_map.get(pair_to_battle)