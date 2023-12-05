import unittest
from datamodel.question_collection import QuestionCollection
import os

class TestQuestionCollection(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.save_path = "test_questioncollection.csv"
        
    def test_add_single_question(self):
        collection = QuestionCollection()
        collection.add("question_a")
        self.assertEqual(collection.questions, ["question_a"])

    def test_add_multiple_questions(self):
        collection = QuestionCollection()
        collection.adds(["question_a", "question_b", "question_c"])
        self.assertEqual(collection.questions, ["question_a", "question_b", "question_c"])

    def test_add_duplicate_questions(self):
        collection = QuestionCollection()
        collection.adds(["question_a", "question_b", "question_a"])
        self.assertEqual(collection.questions, ["question_a", "question_b"])

    def test_to_csv(self):
        collection = QuestionCollection()
        collection.adds(["question_a", "question_b", "question_c"])
        collection.to_csv(self.save_path)
        # Assert that the CSV file is created and contains the correct questions
        self.assertTrue(os.path.exists(self.save_path))

    def test_read_csv(self):
        collection = QuestionCollection.read_csv(self.save_path)
        self.assertEqual(collection.questions, ["question_a", "question_b", "question_c"])
        os.remove(self.save_path)

    def test_repr(self):
        collection = QuestionCollection()
        collection.adds(["question_a", "question_b", "question_c"])
        self.assertEqual(repr(collection), "QuestionCollection(num_rows: 3)")