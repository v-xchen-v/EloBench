import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from elo_rating.llm_player import LLMPlayer
from elo_rating.pairwise_rating_entity import PairwiseBattleWinner, PairwiseRatingEntity
import pandas as pd
import numpy as np
from tqdm import tqdm

MODEL_HEADER = "model"
ELO_RATING_HEADER = "elo_rating"

MODEL_A_HEADER = "model_a"
MODEL_B_HEADER = "model_b"
BATTLE_RES_HEADER = "winner"

def get_players_elo_result(llm_players: list[LLMPlayer], rating_places=0) -> pd.DataFrame:
    """
    Get elo ranking and rating scores of players, after players completed battles.

    Parameters:
    llm_players (list[LLMPlayer]): List of LLMPlayer objects representing the players.
    rating_places (int, optional): Number of decimal places to round the rating scores to. Defaults to 0.

    Returns:
    pd.DataFrame: DataFrame containing the player IDs and their rounded elo rating scores, sorted in descending order.
    """
    df = pd.DataFrame([[x.id, np.round(x.rating, rating_places)] for x in llm_players], columns=[MODEL_HEADER, ELO_RATING_HEADER]).sort_values(ELO_RATING_HEADER, ascending=False)#.reset_index(drop=True)

    def get_rank(rating, all_ratings: list):
        """assign the rank based on the descending order of the numbers, ensuring that the largest number gets the rank 1, and the smallest number gets the rank equal to the number of unique elements in the list. Rank of numbers in a list where the same numbers have the same rank"""
        rank = 1
        for item in all_ratings:
            if rating < item:
                rank+=1
        return rank
    
    # adding rank column, same elo rating will have same rank
    # df['rank'] = df['elo_rating'].apply(lambda x: get_rank(x, df['elo_rating'].tolist()))
    df['rank'] = df[ELO_RATING_HEADER].rank(method='dense', ascending=False).astype(int)
    # df.index = df.index+1
    return df

def get_elo_results_from_battles_data(battles_data: pd.DataFrame, K: int) -> pd.DataFrame:
    """
    Get elo ranking and rating scores of players by the arranged order of battles.

    Args:
        battles_data (pd.DataFrame): DataFrame containing the battles data.
        K (int): The K-factor used in the Elo rating system.

    Returns:
        pd.DataFrame: DataFrame containing the elo results of the players.
    """
    battle_models = pd.concat([battles_data['model_a'], battles_data['model_b']]).unique().tolist()
    llm_players = {x: LLMPlayer(x, K=K) for x in battle_models}

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

def get_elo_results_from_battles_data_each_rnd(battles_data: pd.DataFrame, K: int) -> pd.DataFrame:
    """
    Get elo ranking and rating scores of players by the arranged order of battles.

    Args:
        battles_data (pd.DataFrame): DataFrame containing the battles data.
        K (int): The K-factor used in the Elo rating system.

    Returns:
        pd.DataFrame: DataFrame containing the elo results of the players.
    """
    battle_models = pd.concat([battles_data['model_a'], battles_data['model_b']]).unique().tolist()
    llm_players = {x: LLMPlayer(x, K=K) for x in battle_models}

    elo_ratings_each_rnd = {}
    notie_battle_data = battles_data[['model_a', 'model_b', 'winner']]
    for rd, model_a, model_b, winner in tqdm(notie_battle_data.itertuples(), total=len(notie_battle_data)):
        # import time
        # end_time = time.time()
        # if rd > 0:
        #     print(f"Time for segment 3: {end_time - start_time} seconds")
        model_a_player = llm_players[model_a]
        model_b_player = llm_players[model_b]

        battle_winner = None
        if winner == 'model_a':
            battle_winner = PairwiseBattleWinner.WINNER_IS_A
        elif winner == 'model_b':
            battle_winner = PairwiseBattleWinner.WINNER_IS_B
        else:
            battle_winner = PairwiseBattleWinner.TIE


        # start_time = time.time()
        PairwiseRatingEntity(model_a_player, model_b_player).battle(winner=battle_winner)
        # end_time = time.time()
        # print(f"Time for segment 1: {end_time - start_time} seconds")
        # start_time = time.time()
        elo_ratings_each_rnd[rd+1] = get_players_elo_result(llm_players.values())
        # end_time = time.time()
        # print(f"Time for segment 2: {end_time - start_time} seconds")
        # start_time = time.time()

    return elo_ratings_each_rnd


def get_bootstrap_result(battles, func_compute_elo, K, num_round):
    """
    Perform bootstrap resampling on battles data to compute Elo ratings.

    Parameters:
    - battles (DataFrame): DataFrame containing battle data.
    - func_compute_elo (function): Function to compute Elo ratings.
    - K (float): Elo rating constant.
    - num_round (int): Number of bootstrap rounds.

    Returns:
    - DataFrame: DataFrame containing Elo ratings for each model.
    """
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rating_df = func_compute_elo(battles.sample(frac=1.0, replace=True), K)
        rating_dict = {}
        for _, row in rating_df.iterrows():
            rating_dict[row['model']] = row['elo_rating']
        rows.append(rating_dict)
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]

def get_bootstrap_medium_elo(battles, K=4, BOOTSTRAP_ROUNDS=1000, with_fullset=False):
    """
    Calculate the bootstrap medium Elo rating for a given set of battles.

    Parameters:
    battles (DataFrame): The DataFrame containing the battles data.
    K (int, optional): The K-factor used in Elo rating calculation. Default is 4.

    Returns:
    DataFrame: The DataFrame containing the bootstrap medium Elo ratings for each model.
    """
    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(battles, get_elo_results_from_battles_data, K, BOOTSTRAP_ROUNDS)
    bootstrap_lu_median = bootstrap_elo_lu.median().reset_index().set_axis(["model", "elo_rating"], axis=1)
    bootstrap_lu_median["elo_rating"] = (bootstrap_lu_median["elo_rating"] + 0.5).astype(int)
    # print(bootstrap_lu_median)
    if with_fullset:
        return bootstrap_lu_median, bootstrap_elo_lu
    else:
        return bootstrap_lu_median
# if __name__ == '__main__':
#     battles_data = pd.read_csv(r'/elo_bench/results/quora_100_test1_shuffle_ab/battled_pairs.csv')
#     print(get_bootstrap_medium_elo(battles_data))

