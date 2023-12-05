import unittest
from datamodel.pairwise_battle_arrangement import PairwiseBattleArrangement, PairToBattle
import os

class TestPairwiseBattleArrangement(unittest.TestCase):
    def setUp(self):
        self.questions = ["Question 1", "Question 2", "Question 3"]
        self.models = ["Model A", "Model B", "Model C", "Model D"]
        self.arrangement = PairwiseBattleArrangement(self.questions, self.models)
        
    def test_arrange_all_combinations(self):
        self.arrangement.arrange_all_combinations(shuffle=False)
        self.assertEqual(len(self.arrangement.battles_in_order), 18)  # 3 questions * 6 pairs
    
    def test_arrange_randomly_by_battlecountnumpercombpair(self):
        self.arrangement.arrange_randomly_by_battlecountnumpercombpair(num_of_battle=2, shuffle=False)
        self.assertEqual(len(self.arrangement.battles_in_order), 12)  # 2 battles for each pair
        
    def test_arrange_randomly_by_pairnumperquesiton(self):
        self.arrangement.arrange_randomly_by_pairnumperquesiton(num_of_pair=2, shuffle=False)
        self.assertEqual(len(self.arrangement.battles_in_order), 6)  # 3 questions * 2 pairs
        
    def test_arrange_randomly_by_pairnumpermodel(self):
        self.arrangement.arrange_randomly_by_pairnumpermodel(num_of_pair=1, shuffle=False)
        self.assertEqual(len(self.arrangement.battles_in_order), 4)  # 2 pairs for each model
        
    def test_arrange_by_existing_arrangement(self):
        existing_arrangement = PairwiseBattleArrangement(self.questions, self.models)
        existing_arrangement.arrange_all_combinations(shuffle=False)
        existing_arrangement.to_csv("existing_arrangement.csv")
        
        self.arrangement.arrange_by_existing_arrangement("existing_arrangement.csv")
        self.assertEqual(len(self.arrangement.battles_in_order), 18)  # 3 questions * 6 pairs
        
    def test_to_csv(self):
        self.arrangement.arrange_all_combinations(shuffle=False)
        self.arrangement.to_csv("test_arrangement.csv")
        
        # Assert that the CSV file is created
        self.assertTrue(os.path.exists("test_arrangement.csv"))
        os.remove("test_arrangement.csv")
        
    def test_read_csv(self):
        existing_arrangement = PairwiseBattleArrangement(self.questions, self.models)
        existing_arrangement.arrange_all_combinations(shuffle=False)
        existing_arrangement.to_csv("existing_arrangement.csv")
        
        new_arrangement = PairwiseBattleArrangement.read_csv("existing_arrangement.csv")
        self.assertEqual(len(new_arrangement.battles_in_order), 18)  # 3 questions * 6 pairs
        os.remove("existing_arrangement.csv")
        
    def test_more_battles(self):
        self.arrangement.arrange_all_combinations(shuffle=False)
        pairs = [
            PairToBattle("Question 1", "Model A", "Model B"),
            PairToBattle("Question 2", "Model C", "Model D")
        ]
        num_added = self.arrangement.more_battles(pairs)
        self.assertEqual(num_added, 0)
        self.assertEqual(len(self.arrangement.battles_in_order), 18)  # 18 + 2