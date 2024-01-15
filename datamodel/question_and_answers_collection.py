from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Dict, List
from collections import defaultdict
import pandas as pd
from datamodel.column_names import QUESTION_COLUMN_NAME
import os
import numpy as np

class LLMAnswer:
    """
    Represents an answer generated by a language model.

    Attributes:
        model (str): The name of the language model.
        answer (str): The generated answer.
    """

    def __init__(self, model: str, answer: str) -> None:
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
            
        self.cache_filepath = None
          
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
    
    def get_answers(self, question: str) -> Union[None, List[LLMAnswer]]:
        if question not in self.question_and_answers_collection:
            return None

        return self.question_and_answers_collection[question]
    
    def get_answer(self, question: str, model: str) -> str:
        answers = self.get_answers(question)

        # can not find question
        if answers is None:
            return None
        
        for ans in answers:
            if ans.model == model:
             if ans.answer == 'NaN':
                 return None
             else:
                return ans.answer
            
        # can not find answer
        return None
    
    def answer_exists(self, question: str, model: str) -> bool:
        return False if self.get_answer(question, model) is None else True
    
    def get_no_ans_questions(self, model: str) -> List[str]:
        questions = self.list_questions()
        return [question for question in questions if not self.answer_exists(question, model)]
        
    def get_question_and_answers(self, question: str) -> Dict[str, List[LLMAnswer]]:
        return { question: self.question_and_answers_collection[question] }
    
    def to_csv(self, save_path:str=None):
        flatten_qa_list = []
        for question, answers in self.question_and_answers_collection.items():
            flatten_qa ={}
            flatten_qa[QUESTION_COLUMN_NAME] = question
            for ans in answers:
                flatten_qa[ans.model] = ans.answer
            flatten_qa_list.append(flatten_qa)
        df = pd.DataFrame.from_dict(flatten_qa_list)
        
        # Fixed: When saving a DataFrame to a CSV file, Pandas will, by default, write NaN values as empty fields. To represent the answer is generated as '' and not generated ans by 'NaN'(np.nan) instead of using default setting that missing value is identitical to empty string when writing to csv.
        # to_csv with na_rep='NaN' instead of na_rep=''
        # save_csv with keep_default_na=False instead of keep_default_na=True
        if save_path is not None:
            df.to_csv(save_path, na_rep='NaN')
        else:
            df.to_csv(self.cache_filepath, na_rep='NaN')
        
    @classmethod
    def read_csv(cls, save_path) -> QuestionAndAnswersCollection:
        df = pd.read_csv(save_path, index_col=0, keep_default_na=False, na_values=['NaN'], engine='python')
        # Convert DataFrame to dictionary
        result_dict = defaultdict(list)
        for key, group in df.groupby(QUESTION_COLUMN_NAME):
            # Drop the key column and convert the rest to a dictionary
            list_of_columnname_to_cell = group.drop(QUESTION_COLUMN_NAME, axis=1).to_dict(orient='records')
            result_dict[key] = [LLMAnswer(model=k, answer=v) for d in list_of_columnname_to_cell for k, v in list(d.items()) if not pd.isna(v)]
        q_and_as_collection = QuestionAndAnswersCollection(result_dict)
        q_and_as_collection.cache_filepath = save_path
        return q_and_as_collection
        
    def __repr__(self) -> str:
        # Provide a string representation of the object for debugging
        return f'QuestionAndAnswersCollection(num_questions: {len(self.question_and_answers_collection)})'