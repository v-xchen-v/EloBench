import unittest
from models.gpt_online_lm import GPTOnlineLM

class TestGPTOnlineLM(unittest.TestCase):
    def test_generate_answer(self):
        model = GPTOnlineLM("gpt-35-turbo")
        question = "What is the capital of France?"
        answer = model.generate_answer(question)
        self.assertIsInstance(answer, str)
        self.assertNotEqual(answer, "")