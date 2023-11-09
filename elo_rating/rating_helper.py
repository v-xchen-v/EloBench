from elo_rating.llm_player import LLMPlayer
from elo_rating.pairwise_rating_entity import PairwiseBattleScore, PairwiseRatingEntity
import pandas as pd
import numpy as np

MODEL_HEADER = "Model"
ELO_RATING_HEADER = "Elo Rating"

MODEL_A_HEADER = "model_a"
MODEL_B_HEADER = "model_b"
BATTLE_RES_HEADER = "winner"

def get_players_elo_result(llm_players: list[LLMPlayer], rating_places=0) -> pd.DataFrame:
    """Get elo ranking and rating scores of players, after players completed battles."""
    df = pd.DataFrame([[x.id, np.round(x.rating, rating_places)] for x in llm_players], columns=[MODEL_HEADER, ELO_RATING_HEADER]).sort_values(ELO_RATING_HEADER, ascending=False).reset_index(drop=True)
    df.index = df.index+1
    return df

def get_elo_results_from_battles_data(battles_data: pd.DataFrame, K: int) -> pd.DataFrame:
    """Get elo ranking and rating scores of players by the arranged order of battles."""
    battle_models = pd.concat([battles_data['model_a'], battles_data['model_b']]).unique().tolist()
    llm_players = {x: LLMPlayer(x, K=K) for x in battle_models}
    print(llm_players)

    for rd, model_a, model_b, winner in battles_data[['model_a', 'model_b', 'winner']].itertuples():
        model_a_player = llm_players[model_a]
        model_b_player = llm_players[model_b]
        
        battle_winner = None
        if winner == 'model_a':
            battle_winner = PairwiseBattleScore.WINNER_IS_A
        elif winner == 'model_b':
            battle_winner = PairwiseBattleScore.WINNER_IS_B
        else:
            battle_winner = PairwiseBattleScore.TIE
            
        PairwiseRatingEntity(model_a_player, model_b_player).battle(winner=battle_winner)
        
    return get_players_elo_result(list(llm_players.values()))

