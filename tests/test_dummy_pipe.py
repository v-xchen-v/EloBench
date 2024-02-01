import unittest
from pipe.dummy_pipe import DummyPipeline
from datamodel.question_collection import QuestionCollection
from datamodel.model_collection import ModelCollection
from datamodel import ArrangementStrategy
import os

class TestDumpyPipeline(unittest.TestCase):
    def setUp(self):
        self.tempcache_dir = "tempcache/dummy"
        self.save_dir = "results/dummy"
        self.pipeline = DummyPipeline(self.tempcache_dir, self.save_dir)
        self.questions = ["Give me a question about animal.", "Is Landon rainy?", 'Is Austrilia always sunny?']
        self.models = ['huggyllama/llama-7b', 'meta-llama/Llama-2-7b-hf', 'mosaicml/mpt-7b']

    def tearDown(self):
        # Clean up any created files or directories
        pass

    def test_register_questions(self):
        questions = self.questions
        self.pipeline.register_questions(questions)
        question_collection = QuestionCollection.read_csv(f"{self.save_dir}/questions.csv")
        self.assertEqual(question_collection.questions, questions)

    def test_register_models(self):
        models = self.models
        self.pipeline.register_models(models)
        model_collection = ModelCollection.read_csv(f"{self.save_dir}/models.csv")
        self.assertEqual(model_collection.models, models)

    def test_arrange_battles(self):
        questions = self.questions
        models = self.models
        self.pipeline.register_questions(questions)
        self.pipeline.register_models(models)
        self.pipeline.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_CombPair, num_of_battle=2)
        # Assert that the battle arrangement file is created
        self.assertTrue(os.path.exists(f"{self.save_dir}/battle_arrangement.csv"))

    def test_gen_model_answers(self):
        questions = self.questions
        models = self.models
        self.pipeline.register_questions(questions)
        self.pipeline.register_models(models)
        self.pipeline.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_CombPair, num_of_battle=2)
        self.pipeline.gen_model_answers()
        # Assert that the question and answers collection is generated
        self.assertIsNotNone(self.pipeline.question_and_answers_collection)

    def test_battle(self):
        questions = self.questions
        models = self.models
        self.pipeline.register_questions(questions)
        self.pipeline.register_models(models)
        self.pipeline.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_CombPair, num_of_battle=2)
        self.pipeline.gen_model_answers()
        self.pipeline.battle()
        # Assert that the battle records and battled pairs are created
        self.assertTrue(os.path.exists(f"{self.save_dir}/battle_records.csv"))
        self.assertTrue(os.path.exists(f"{self.save_dir}/battled_pairs.csv"))

    def test_gen_elo(self):
        questions = self.questions
        models = self.models
        self.pipeline.register_questions(questions)
        self.pipeline.register_models(models)
        self.pipeline.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_CombPair, num_of_battle=2)
        self.pipeline.gen_model_answers()
        self.pipeline.battle()
        self.pipeline.gen_elo()
        # Assert that the Elo ratings are generated
        self.assertTrue(os.path.exists(f"{self.save_dir}/elo_rating.csv"))