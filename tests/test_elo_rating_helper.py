import unittest
import logging
import pandas as pd

from elo_rating.rating_helper import get_players_elo_result, get_elo_results_from_battles_data
from elo_rating import LLMPlayer

logging.basicConfig(level=logging.DEBUG)

class TestEloRatingHelper(unittest.TestCase):
    def test_get_players_elo_result(self):
        player1 = LLMPlayer("model_a")
        player1.rating = 1000
        
        player2 = LLMPlayer("model_b")
        player2.rating = 800
        
        player3 = LLMPlayer("model_c")
        player3.rating = 1100
        logging.info(get_players_elo_result([player1, player2, player3]))
        
    def test_get_elo_results_from_battles_data(self):
        battles = [
            {
                "model_a": "huggyllama/llama-7b",
                "model_b": "gpt2",
                "winner": "model_a"
            },
                        {
                "model_a": "huggyllama/llama-7b",
                "model_b": "gpt2",
                "winner": "tie"
            },
            {
                "model_a": "huggyllama/llama-13b",
                "model_b": "gpt2",
                "winner": "model_a"
            },
            {
                "model_a": "huggyllama/llama-7b",
                "model_b": "huggyllama/llama-13b",
                "winner": "model_b"
            }
        ]

        logging.info(get_elo_results_from_battles_data(pd.DataFrame.from_dict(battles), K=4))