from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Dict, List
from collections import defaultdict
import pandas as pd
from .columns import QUESTION_COLUMN_NAME

class LLMAnswer:
    def __init__(self, model:str, answer:str) -> None:
        self.model = model
        self.answer = answer
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, LLMAnswer):
            return False
        
        return self.model == other.model and self.answer == other.answer
            
    
class QuestionAndAnswersCollection:
    """Represents a collection of records with one question with answers of different LLM model."""
    
    def __init__(self, question_and_answers_collection: Union[Dict[str, List[LLMAnswer]], None]=None) -> None:
        if question_and_answers_collection is None:
            self.question_and_answers_collection = defaultdict(list)
        else:
            self.question_and_answers_collection = question_and_answers_collection
            
    def add_question(self, question: str):
        """Add unique question"""
        if question in self.question_and_answers_collection:
            return
        else:
            self.question_and_answers_collection[question] = []
        
    def add_questions(self, questions: List[str]):
        """Add unique questions"""
        for question in questions:
            self.add_question(question)
    
    def add_answer(self, question: str, answer: LLMAnswer):
        """Add unique answer"""
        prev_answers = self.question_and_answers_collection[question]
        if answer not in prev_answers:
            prev_answers.append(answer)
            
    def add_answers(self, question: str, answers: List[LLMAnswer]):
        """Add unique answers"""
        for ans in answers:
            self.add_answer(question, ans)
        
    
    def list_questions(self) -> List[str]:
        return list(self.question_and_answers_collection.keys())
    
    def get_answers(self, question: str) -> List[LLMAnswer]:
        return self.question_and_answers_collection[question]
    
    def get_answer(self, question: str, model: str) -> str:
        answers = self.question_and_answers_collection[question]
        for ans in answers:
            if ans.model == model:
                return ans.answer
        return None
    
    def get_question_and_answers(self, question: str) -> Dict[str, List[LLMAnswer]]:
        return { question: self.question_and_answers_collection[question] }
    
    def to_csv(self, save_path:str):
        flatten_qa_list = []
        for question, answers in self.question_and_answers_collection.items():
            flatten_qa ={}
            flatten_qa[QUESTION_COLUMN_NAME] = question
            for ans in answers:
                flatten_qa[ans.model] = ans.answer
            flatten_qa_list.append(flatten_qa)
        df = pd.DataFrame.from_dict(flatten_qa_list)
        df.to_csv(save_path)
        
    @classmethod
    def read_csv(cls, save_path) -> QuestionAndAnswersCollection:
        # TODO: optimize the loading cost
        df = pd.read_csv(save_path)
        # Convert DataFrame to dictionary
        result_dict = {}
        for key, group in df.groupby(QUESTION_COLUMN_NAME):
            # Drop the key column and convert the rest to a dictionary
            group_dicts = group.drop(QUESTION_COLUMN_NAME, axis=1).to_dict(orient='records')
            result_dict[key] = group_dicts
        return QuestionAndAnswersCollection(result_dict)
        
    def __repr__(self) -> str:
        # Provide a string representation of the object for debugging
        return f'QuestionAndAnswersCollection(num_questions: {len(self.question_and_answers_collection)})'