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
        
    def test_batch_generate_answer_is_consistent_with_nobatch(self):
        model = AutoCausalLM("meta-llama/Llama-2-7b-chat-hf")
        questions = [ \
            "Does light travel forever or does it eventually fade?",
            "What is France's record in wars?",
            "Do you think Rihanna's image has influenced the fashion industry, and if so, how?",
            "Why does the United States have higher living standards than Europe?",
            'Can you name the main actors in the film "Green Book"?',
            "What are some crazy coincidences in history ?",
            "Is anything in England actually original? Tea, for example, comes from China. Fish & chips from Portugal. Football (soccer) originates in China. Gothic style architecture is French.",
            "Have you ever tried adding exotic fruits to your overnight oats?",
            "Can you name a few controversies that have involved Jeremy Clarkson?",
            "Can you tell me about the impact of the iPad 3 on the consumer electronics industry?",
            "Who are some actors that have worked with Kevin Hart in his comedy films?",
            "How does an Andrea Bocelli concert compare to other concerts you've been to?",
            "How has Katy Perry's image evolved over the years?",
            "Hi, I'm interested in learning to play badminton. Can you explain the game to me?",
        ]
        model.batch_size = len(questions)
        answer = model.generate_answer(questions[0], free_model_when_exit=False)
        answers = [x for x in model.batch_generate_answer(questions, free_model_when_exit=False)]
        self.assertEqual(answer, answers[0])