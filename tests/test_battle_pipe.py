import unittest
from unittest.mock import MagicMock
from pipe.battle_pipe import BattlePipeline
from datamodel import ArrangementStrategy

class TestBattlePipeline(unittest.TestCase):
    def setUp(self):
        self.tempcache_dir = "tempcache/test_battle_pipe"
        self.save_dir = "results/test_battle_pipe"
        self.pipeline = BattlePipeline(self.tempcache_dir, self.save_dir)
        self.pipeline.no_cache = False
        questions = ["Give me a question about animal.", "Is Landon rainy?", 'Is Austrilia always sunny?']
        models = ['huggyllama/llama-7b', 'meta-llama/Llama-2-7b-hf', 'mosaicml/mpt-7b']
        self.pipeline.register_questions(questions)
        self.pipeline.register_models(models)
    
    def test_gen_model_answers(self):
        self.pipeline.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_CombPair, num_of_battle=2)
        self.pipeline.gen_model_answers()
        # Assert that the question and answers collection is generated
        self.assertIsNotNone(self.pipeline.question_and_answers_collection)

if __name__ == '__main__':
    unittest.main()