from typing import List, Union
import pandas as pd
from .columns import QUESTION_COLUMN_NAME
from logger import logger

class QuestionCollection:
    def __init__(self, questions: Union[List[str], None] = None):
         # Initialize the question list to store unique strings
        self.questions = []
        if questions is not None:
            self.adds(questions)
            logger.debug(f'Removed {len(questions)-len(self.questions)} repeat questions.')
    
    def add(self, question: str):
        # Add a string to the list if it's not already present
        if question not in self.questions:
            self.questions.append(question)
    
    def adds(self, questions: List[str]):
        for q in questions:
            self.add(q)
            
    def to_csv(self, save_path: str):
        # Save questions as csv.
        pd.DataFrame(self.questions, columns=[QUESTION_COLUMN_NAME]).to_csv(save_path)
    
    @classmethod
    def read_csv(cls, save_path: str):
        # Load question collection from csv file.
        return QuestionCollection(pd.read_csv(save_path)[QUESTION_COLUMN_NAME].tolist())
    
    def __repr__(self) -> str:
        # Provide a string representation of the object for debugging
        return f'QuestionCollection(num_rows: {len(self.questions)})'