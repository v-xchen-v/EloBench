import unittest
from datamodel.model_collection import ModelCollection
import os

class TestModelCollection(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.save_path = "test_modelcollection.csv"
        
    def test_add_single_model(self):
        collection = ModelCollection()
        collection.add("model_a")
        self.assertEqual(collection.models, ["model_a"])

    def test_add_multiple_models(self):
        collection = ModelCollection()
        collection.adds(["model_a", "model_b", "model_c"])
        self.assertEqual(collection.models, ["model_a", "model_b", "model_c"])

    def test_add_duplicate_models(self):
        collection = ModelCollection()
        collection.adds(["model_a", "model_b", "model_a"])
        self.assertEqual(collection.models, ["model_a", "model_b"])

    def test_to_csv(self):
        collection = ModelCollection()
        collection.adds(["model_a", "model_b", "model_c"])
        collection.to_csv(self.save_path)
        # Assert that the CSV file is created and contains the correct models
        self.assertTrue(os.path.exists(self.save_path))
        os.remove(self.save_path)

    def test_read_csv(self):
        collection = ModelCollection.read_csv(self.save_path)
        self.assertEqual(collection.models, ["model_a", "model_b", "model_c"])
        os.remove(self.save_path)

    def test_repr(self):
        collection = ModelCollection()
        collection.adds(["model_a", "model_b", "model_c"])
        self.assertEqual(repr(collection), "ModelCollection(num_rows: 3)")
