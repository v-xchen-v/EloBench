from __future__ import annotations
from typing import List, Union
import pandas as pd
from datamodel.column_names import MODEL_COLUMN_NAME
from logger import logger

class ModelCollection:
    """
    A collection of models.

    Args:
        models (Union[List[str], None], optional): List of models to initialize the collection with. Defaults to None.

    Attributes:
        models (List[str]): List of models in the collection.

    Methods:
        add(question: str) -> None: Add a model to the collection.
        adds(models: List[str]) -> None: Add multiple models to the collection.
        to_csv(save_path: str) -> None: Save the models to a CSV file.
        read_csv(save_path: str) -> ModelCollection: Read models from a CSV file and return a new ModelCollection instance.
    """

    def __init__(self, models: Union[List[str], None] = None):
        """
        Initialize the ModelCollection.

        Args:
            models (Union[List[str], None], optional): List of models to initialize the collection with. Defaults to None.
        """
        self.models: List[str] = []
        if models is not None:
            self.adds(models)
            logger.debug(f'Removed {len(models)-len(self.models)} repeat questions.')

    def add(self, question: str) -> None:
        """
        Add a model to the collection.

        Args:
            question (str): The model to add.
        """
        if question not in self.models:
            self.models.append(question)

    def adds(self, models: List[str]) -> None:
        """
        Add multiple models to the collection.

        Args:
            models (List[str]): List of models to add.
        """
        for q in models:
            self.add(q)

    def to_csv(self, save_path: str) -> None:
        """
        Save the models to a CSV file.

        Args:
            save_path (str): The path to save the CSV file.
        """
        pd.DataFrame(self.models, columns=[MODEL_COLUMN_NAME]).to_csv(save_path)

    @classmethod
    def read_csv(cls, save_path: str) -> ModelCollection:
        """
        Read models from a CSV file and return a new ModelCollection instance.

        Args:
            save_path (str): The path to the CSV file.

        Returns:
            ModelCollection: A new ModelCollection instance with the models read from the CSV file.
        """
        return ModelCollection(pd.read_csv(save_path)[MODEL_COLUMN_NAME].tolist())

    def __repr__(self) -> str:
        """
        Return a string representation of the ModelCollection.

        Returns:
            str: A string representation of the ModelCollection.
        """
        return f'ModelCollection(num_rows: {len(self.models)})'
    
    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index):
        return self.models[index]
    
    def __len__(self):
        return len(self.models)