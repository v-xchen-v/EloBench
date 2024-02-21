import unittest
from data import get_arena_battles_20230717_data, get_arena_elo_res_20230717,K_FACTOR
import pandas as pd
import math
import logging
from elo_rating.rating_helper import get_players_rating_and_rank_from_battles_data

logging.basicConfig(level=logging.INFO)

class TestEloRating(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # shared test data
        cls.arena_battles_data = get_arena_battles_20230717_data()
    
    # shared test logics
    def do_arena_battle(self, K):
        return get_players_rating_and_rank_from_battles_data(self.arena_battles_data, K)
    
    def assertListAlmostEqual(self, list1, list2, places=7):
        self.assertEqual(len(list1), len(list2), "Lists are of different lengths.")
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, places=places)
            
    def test_ismatch_witharenaoffical(self):
        our_res = self.do_arena_battle(K=K_FACTOR)
        arena_res = pd.DataFrame.from_dict(get_arena_elo_res_20230717())
        logging.info(our_res['Model'].tolist())
        logging.info(arena_res['Model'].tolist())
        
        self.assertListEqual(our_res['Model'].tolist(), arena_res['Model'].tolist())
        self.assertListAlmostEqual(our_res['Elo Rating'].tolist(), arena_res['Elo rating'].tolist(), places=0)
        
        
    def test_zerosum_withdifferentK(self):
        def get_rating_sum(df: pd.DataFrame):
            rating_sum = df['Elo Rating'].sum()
            return rating_sum
        
        rating_sums = [get_rating_sum(self.do_arena_battle(k)) for k in [4, 8, 16, 32]]
        logging.info('rating sum:', rating_sums)
        
        # checking zero-sum of elo rating: {sum of rating} is consistant(n*initial rating) on different Ks
        self.assertTrue(all(math.isclose(x, rating_sums[0], rel_tol=0.1) for x in rating_sums))
        
    def test_ranknotconsistent_withdifferentK(self):
        def get_ranking(df: pd.DataFrame):
            df['Rank'] = df['Elo Rating'].rank(ascending=False)
            return df
            
        rating_rankings = [get_ranking(self.do_arena_battle(k))['Model'].tolist() for k in [4, 8, 16, 32]]
        logging.info('ranking', rating_rankings)

        def all_sublists_are_same(list_of_lists):
            return all(sublist == list_of_lists[0] for sublist in list_of_lists)
        self.assertFalse(all_sublists_are_same(rating_rankings))
                