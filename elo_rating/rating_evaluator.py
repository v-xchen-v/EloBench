import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from datamodel.elo_rating_history import EloRatingHistory, BattleOutcomes
def evaluate_rank_consistency(elo_rating_history: EloRatingHistory, num_battle1: int, num_battle2: int):
    # ! does the order of num_batlle1 and num_battle2 matter?
    # Ensure models are the same
    if not set(elo_rating_history.get_point(num_battle1)['model'].tolist()) == set(elo_rating_history.get_point(num_battle2)['model'].tolist()):
        # raise Exception("The models of elo_rating_history1 and elo_rating_history2 are not the same")
        return None
    
    elo_rating_history_point1 = elo_rating_history.get_point(num_battle1)
    elo_rating_history_point2 = elo_rating_history.get_point(num_battle2)
    
    sorted_point1 = elo_rating_history_point1.sort_values(by=['model'], inplace=False)
    sorted_point2 = elo_rating_history_point2.sort_values(by=['model'], inplace=False)
    
    res = stats.kendalltau(sorted_point1['rank'].astype(int).tolist(), sorted_point2['rank'].astype(int).tolist())
    # print(res.statistic)
    return res.correlation

def evaluate_winrate_at_historypoint(elo_rating_history: EloRatingHistory, battle_outcomes: BattleOutcomes,  num_battle: int):
    history_point = elo_rating_history.get_point(num_battle)
    actual_winrate = compute_acutal_winrate(battle_outcomes.to_df())
    predict_winrate = compute_predict_winrate(history_point)
    return evaluate_winrate(actual_winrate, predict_winrate)
    
def evaluate_winrate(actual_winrate: pd.DataFrame, predicted_winrate: pd.DataFrame):
    """
    Evaluates the mean squared error (mse) and mean absolute error (mae) between the actual winrate and predicted winrate.

    Parameters:
    actual_winrate (pd.DataFrame): DataFrame containing the actual winrate values.
    predicted_winrate (pd.DataFrame): DataFrame containing the predicted winrate values.

    Returns:
    dict: A dictionary containing the mse and mae values.
    """
    # Check if the models of actual_winrate and predicted_winrate are the same ignoring the order
    columns_same = sorted(actual_winrate.columns) == sorted(predicted_winrate.columns)
    indices_same = sorted(actual_winrate.index) == sorted(predicted_winrate.index)
    if not columns_same or not indices_same:
        raise Exception("The models of actual_winrate and predicted_winrate are not the same")
    
    # Sort columns in predict_winrate to match the order of columns in actual_winrate
    predicted_winrate = predicted_winrate[actual_winrate.columns]
    
    # Sort the row in both actual_winrate and predicted_winrate to match the order of indices in actual_winrate
    predicted_winrate = predicted_winrate.sort_index()
    actual_winrate = actual_winrate.sort_index()
    
    Y_true = np.array(actual_winrate.values.flatten().tolist())
    Y_pred = np.array(predicted_winrate.values.flatten().tolist())
    
    def mse_mae_with_nan_handling(true, pred):
        mask = ~np.isnan(true) & ~np.isnan(pred)
        mse = ((true[mask] - pred[mask]) ** 2).mean()
        mae = (np.abs(true[mask] - pred[mask])).mean()
        return mse, mae
    
    mse, mae = mse_mae_with_nan_handling(Y_true, Y_pred)
    
    return {
        "mse": mse,
        "mae": mae
    }

def compute_predict_winrate(elo_rating_data: pd.DataFrame):
    """
    Predicts the win rate between different models based on their Elo ratings.

    Parameters:
    elo_rating_data (pd.DataFrame): DataFrame containing the Elo ratings of the models.

    Returns:
    pd.DataFrame: DataFrame representing the win rates between different models.
    """
    SCALE=400
    BASE=10
    # Get all the unique models in battles_data
    all_models = sorted(elo_rating_data['model'].tolist())
    
    ratings_dict = {}
    for idx, row in elo_rating_data.iterrows():
        ratings_dict[row['model']] = row['elo_rating']
        
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in all_models:
        for b in all_models:
            ea = 1 / (1 + BASE ** ((ratings_dict[b] - ratings_dict[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea
    
    data = {
        a: [wins[a][b] if a != b else np.NAN for b in all_models]
        for a in all_models
    }
    
    df = pd.DataFrame(data, index=all_models)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T
    
def compute_acutal_winrate(battle_outcomes_data: pd.DataFrame):
    """
    Computes the actual win rate between different models based on battle outcomes data.

    Args:
        battle_outcomes_data (pd.DataFrame): DataFrame containing battle outcomes data.

    Returns:
        pd.DataFrame: DataFrame representing the win rate between different models.
    """
    # filter out no-tie battles
    battle_outcomes_notie_data = battle_outcomes_data[(battle_outcomes_data['winner']=='model_a') | (battle_outcomes_data['winner']=='model_b')].copy()
    
    # Define a function to apply
    def get_value_from_column(row):
        return row[row['winner']]

    # Apply the function
    battle_outcomes_notie_data.loc[:, 'winner_name'] = battle_outcomes_notie_data.apply(get_value_from_column, axis=1)
    def get_a_win_b_count(a, b):
        if a == b:
            return np.nan
        else:
            ab_battles = battle_outcomes_notie_data[((battle_outcomes_notie_data['model_a']==a) & (battle_outcomes_notie_data['model_b']==b)) | (battle_outcomes_notie_data['model_a']==b) & (battle_outcomes_notie_data['model_b']==a)]
            if ab_battles.shape[0] == 0:
                return np.nan
            
            ab_awin_battles = ab_battles[ab_battles['winner_name']==a]
            return ab_awin_battles.shape[0]/ab_battles.shape[0]
    
    union_model_names = pd.concat([
        pd.Series(battle_outcomes_notie_data['model_a'].unique()), 
        pd.Series(battle_outcomes_notie_data['model_b'].unique())]).unique()
    
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for name_a in union_model_names:
        for name_b in union_model_names:
            wins[name_a][name_b] = get_a_win_b_count(name_a, name_b)
    
    data = {
        a: [wins[a][b] if a != b else np.NAN for b in union_model_names]
        for a in union_model_names
    }

    # A DataFrame df is created from the data dictionary.
    # Each key from the data dictionary becomes a column in df.
    # The values in each list become the rows of the corresponding column.
    df = pd.DataFrame(data, index=union_model_names)
    row_beats_col_freq = df.T
    df.index.name = "model_a"
    df.columns.name = "model_b"
    
    # Arrange ordering according to proprition of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    model_names = list(prop_wins.keys())
    
    # if ordering is not None:
        # model_names = ordering
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col

def compute_actual_winrate_awinb(battle_outcomes_data: pd.DataFrame, model_a: str, model_b: str):
    """
    Computes the actual win rate between model_a and model_b based on battle outcomes data.

    Args:
        battle_outcomes_data (pd.DataFrame): DataFrame containing battle outcomes data.
        model_a (str): Name of model_a.
        model_b (str): Name of model_b.

    Returns:
        float: The actual win rate between model_a and model_b.
    """
    valid_winner = set(['model_a', 'model_b'])
    battle_outcomes_data_valid = battle_outcomes_data[battle_outcomes_data['winner'].isin(valid_winner)]
    if len(battle_outcomes_data_valid) == 0:
        return np.nan
    
    # filter out no-tie battles
    battle_outcomes_notie_data = battle_outcomes_data_valid[(battle_outcomes_data_valid['winner']=='model_a') | (battle_outcomes_data_valid['winner']=='model_b')].copy()
    
    # Define a function to apply
    def get_value_from_column(row):
        return row[row['winner']]

    # Apply the function
    battle_outcomes_notie_data.loc[:, 'winner_name'] = battle_outcomes_notie_data.apply(get_value_from_column, axis=1)
    
    ab_battles = battle_outcomes_notie_data[((battle_outcomes_notie_data['model_a']==model_a) & (battle_outcomes_notie_data['model_b']==model_b)) | (battle_outcomes_notie_data['model_a']==model_b) & (battle_outcomes_notie_data['model_b']==model_a)]
    if ab_battles.shape[0] == 0:
        return np.nan
    
    ab_awin_battles = ab_battles[ab_battles['winner_name']==model_a]
    return ab_awin_battles.shape[0]/ab_battles.shape[0]

def compute_predict_winrate_awinb(elo_rating_data: pd.DataFrame, model_a: str, model_b: str):
    """
    Predicts the win rate between model_a and model_b based on their Elo ratings.

    Args:
        elo_rating_data (pd.DataFrame): DataFrame containing the Elo ratings of the models.
        model_a (str): Name of model_a.
        model_b (str): Name of model_b.

    Returns:
        float: The predicted win rate between model_a and model_b.
    """
    SCALE=400
    BASE=10
    # Get all the unique models in battles_data
    all_models = sorted(elo_rating_data['model'].tolist())
    
    if model_a not in all_models or model_b not in all_models:
        return np.nan
    
    ratings_dict = {}
    for idx, row in elo_rating_data.iterrows():
        ratings_dict[row['model']] = row['elo_rating']
        
    ea = 1 / (1 + BASE ** ((ratings_dict[model_b] - ratings_dict[model_a]) / SCALE))
    # awinb = ea
    # bwina = 1-ea
    return ea


if __name__ == '__main__':
    ratings = [
        {
            'model': 'model_a',
            'elo_rating': 1000,
        },
        {
            'model': 'model_b',
            'elo_rating': 2000,
        }
    ]
    predict_wirnate_ab = compute_predict_winrate_awinb(pd.DataFrame.from_dict(ratings), 'model_a', 'model_b')
    print(predict_wirnate_ab)