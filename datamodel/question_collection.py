from typing import List, Union
import pandas as pd
from .column_names import QUESTION_COLUMN_NAME
from logger import logger

class QuestionCollection:
    """
    A collection of unique questions.

    Attributes:
        questions (List[str]): The list of unique questions.

    Methods:
        __init__(self, questions: Union[List[str], None] = None): Initializes the QuestionCollection object.
        add(self, question: str): Adds a question to the collection if it's not already present.
        adds(self, questions: List[str]): Adds multiple questions to the collection.
        to_csv(self, save_path: str): Saves the questions as a CSV file.
        read_csv(cls, save_path: str): Loads a question collection from a CSV file.
        __repr__(self) -> str: Provides a string representation of the object for debugging.
    """

    def __init__(self, questions: Union[List[str], None] = None):
        """
        Initializes the QuestionCollection object.

        Args:
            questions (Union[List[str], None], optional): List of questions to initialize the collection with. Defaults to None.
        """
        self.questions: List[str] = []
        if questions is not None:
            self.adds(questions)
            logger.debug(f'Removed {len(questions)-len(self.questions)} repeat questions.')

    def add(self, question: str):
        """
        Adds a question to the collection if it's not already present.

        Args:
            question (str): The question to add.
        """
        if question not in self.questions:
            self.questions.append(question)

    def adds(self, questions: List[str]):
        """
        Adds multiple questions to the collection.

        Args:
            questions (List[str]): The list of questions to add.
        """
        for q in questions:
            self.add(q)

    def to_csv(self, save_path: str):
        """
        Saves the questions as a CSV file.

        Args:
            save_path (str): The path to save the CSV file.
        """
        pd.DataFrame(self.questions, columns=[QUESTION_COLUMN_NAME]).to_csv(save_path)

    @classmethod
    def read_csv(cls, save_path: str):
        """
        Loads a question collection from a CSV file.

        Args:
            save_path (str): The path to the CSV file.

        Returns:
            QuestionCollection: The loaded question collection.
        """
        return QuestionCollection(pd.read_csv(save_path)[QUESTION_COLUMN_NAME].tolist())

    def __repr__(self) -> str:
        """
        Provides a string representation of the object for debugging.

        Returns:
            str: The string representation of the object.
        """
        return f'QuestionCollection(num_rows: {len(self.questions)})'
    
    def __iter__(self):
        return iter(self.questions)

    def __getitem__(self, index):
        return self.questions[index]
    
    def __len__(self):
        return len(self.questions)