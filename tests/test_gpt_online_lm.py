import unittest
from models.gpt_online_lm import GPTOnlineLM

class TestGPTOnlineLM(unittest.TestCase):
    def test_generate_answer(self):
        model = GPTOnlineLM("gpt-35-turbo")
        question = "What is the capital of France?"
        answer = model.generate_answer(question)
        self.assertIsInstance(answer, str)
        self.assertNotEqual(answer, "")
        
    def test_batch_generate_answer(self):
        model = GPTOnlineLM("gpt-35-turbo")
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
        answers = model.batch_generate_answer(questions)
        for idx, answer in enumerate(answers):
            print(f'Question: {questions[idx]}\nAnswer: {answer}\n\n')
            self.assertIsInstance(answer, str)
            self.assertNotEqual(answer, "")