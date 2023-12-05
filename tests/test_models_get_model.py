import unittest
from models import get_model

class TestAutoCausalLM(unittest.TestCase):
    def test_generate_answer_with_get_local_hf_model(self):
        model = get_model("gpt2")
        question = "What is the capital of France?"
        answer = model.generate_answer(question)
        self.assertIsInstance(answer, str)
        self.assertNotEqual(answer, "")

    def test_generate_answer_with_get_online_gpt_model(self):
        model = get_model("gpt-35-turbo")
        question = "What is the capital of France?"
        answer = model.generate_answer(question, free_model_when_exit=False)
        self.assertIsInstance(answer, str)
        self.assertNotEqual(answer, "")