import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from elo_rating.llm_player import LLMPlayer
from elo_rating.pairwise_rating_entity import PairwiseBattleWinner, PairwiseRatingEntity
import pandas as pd
import numpy as np

MODEL_HEADER = "Model"
ELO_RATING_HEADER = "Elo Rating"

MODEL_A_HEADER = "model_a"
MODEL_B_HEADER = "model_b"
BATTLE_RES_HEADER = "winner"

from tqdm import tqdm

def get_players_elo_result(llm_players: list[LLMPlayer], rating_places=0) -> pd.DataFrame:
    """Get elo ranking and rating scores of players, after players completed battles."""
    df = pd.DataFrame([[x.id, np.round(x.rating, rating_places)] for x in llm_players], columns=[MODEL_HEADER, ELO_RATING_HEADER]).sort_values(ELO_RATING_HEADER, ascending=False).reset_index(drop=True)
    df.index = df.index+1
    return df

def get_elo_results_from_battles_data(battles_data: pd.DataFrame, K: int) -> pd.DataFrame:
    """Get elo ranking and rating scores of players by the arranged order of battles."""
    battle_models = pd.concat([battles_data['model_a'], battles_data['model_b']]).unique().tolist()
    llm_players = {x: LLMPlayer(x, K=K) for x in battle_models}
    # print(llm_players)

    for rd, model_a, model_b, winner in battles_data[['model_a', 'model_b', 'winner']].itertuples():
        model_a_player = llm_players[model_a]
        model_b_player = llm_players[model_b]
        
        battle_winner = None
        if winner == 'model_a':
            battle_winner = PairwiseBattleWinner.WINNER_IS_A
        elif winner == 'model_b':
            battle_winner = PairwiseBattleWinner.WINNER_IS_B
        else:
            battle_winner = PairwiseBattleWinner.TIE
            
        PairwiseRatingEntity(model_a_player, model_b_player).battle(winner=battle_winner)
        
    return get_players_elo_result(list(llm_players.values()))


def get_bootstrap_result(battles, func_compute_elo, K, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rating_df = func_compute_elo(battles.sample(frac=1.0, replace=True), K)
        rating_dict = {}
        for _, row in rating_df.iterrows():
            rating_dict[row['Model']] = row['Elo Rating']
        rows.append(rating_dict)
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]

def get_bootstrap_medium_elo(battles, K=4):
    BOOTSTRAP_ROUNDS = 1000

    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(battles, get_elo_results_from_battles_data, K, BOOTSTRAP_ROUNDS)
    bootstrap_lu_median = bootstrap_elo_lu.median().reset_index().set_axis(["Model", "Elo Rating"], axis=1)
    bootstrap_lu_median["Elo Rating"] = (bootstrap_lu_median["Elo Rating"] + 0.5).astype(int)
    # print(bootstrap_lu_median)
    return bootstrap_lu_median

if __name__ == '__main__':
    battles_data = pd.read_csv(r'/elo_bench/results/quora_100_test1_shuffle_ab/battled_pairs.csv')
    print(get_bootstrap_medium_elo(battles_data))

