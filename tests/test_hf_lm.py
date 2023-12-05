import unittest
from models.hf_local_lm import AutoCausalLM

class TestAutoCausalLM(unittest.TestCase):
    def test_generate_answer_with_model_on_gpu(self):
        model = AutoCausalLM("gpt2")
        question = "What is the capital of France?"
        answer = model.generate_answer(question)
        self.assertIsInstance(answer, str)
        self.assertNotEqual(answer, "")

    def test_generate_answer_with_free_model_when_exit_false(self):
        model = AutoCausalLM("gpt2")
        question = "What is the capital of France?"
        answer = model.generate_answer(question, free_model_when_exit=False)
        self.assertIsInstance(answer, str)
        self.assertNotEqual(answer, "")