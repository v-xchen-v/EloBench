from abc import ABC, abstractmethod

class LM(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def generate_answer(self, question, **kwargs) -> str:
        """Generate an answer for the given question.

        Args:
            question (str): The question to generate an answer for.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated answer.
        """
        pass
    
    @abstractmethod
    def batch_generate_answer(self, questions, **kwargs) -> list:
        """Generate answers for the given questions.

        For local models, this method should be overridden to provide a more efficient implementation.
        
        Args:
            questions (list): The questions to generate answers for.
            **kwargs: Additional keyword arguments.

        Returns:
            list: The generated answers.
        """
        pass