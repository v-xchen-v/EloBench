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