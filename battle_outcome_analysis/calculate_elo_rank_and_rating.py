import os, sys
sys.path.append(r'/elo_bench')

from datamodel import BattleOutcomes
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from elo_rating.rating_helper import get_players_rating_and_rank_from_battles_data, get_elo_results_from_battles_data_each_rnd
from scipy import stats
import concurrent.futures


def get_leaderboard(battled_pairs_csv_file, FIRST_N=None):
    df = pd.read_csv(battled_pairs_csv_file, nrows=FIRST_N)
    return BattleOutcomes.read_df(df)

def get_leaderboard_history(battled_pairs_csv_file, FIRST_N=None, step=1):
    df = pd.read_csv(battled_pairs_csv_file, engine='python')
    if FIRST_N == None:
        end = len(df)
    else:
        end = FIRST_N
        
    leaderboard_all_rnd = get_elo_results_from_battles_data_each_rnd(df.head(end), K=4)
    
    leaderboards = []
    for i in tqdm(range(1, end, step)):
        leaderboard_till_now = leaderboard_all_rnd[i]
        leaderboard_till_now['num_battle'] = i
        leaderboards.append(leaderboard_till_now)
        
    leaderboards_in_one = pd.concat(leaderboards)
    return leaderboards_in_one
    
    
def get_aggregate_leaderboard_history_on_single_file(file_path, FIRST_N, step):
    # This function processes a single file and returns the result
    leaderboard_history = get_leaderboard_history(file_path, FIRST_N, step)
    return leaderboard_history

def get_aggregate_leaderboard_history(battled_pairs_csv_file_list, FIRST_N=None, step=1):
    # Create a list to store the results
    leaderboard_history_at_cur_bootstrap_list = []
    
    from functools import partial
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        # Create a new function that has FIRST_N and step filled in
        partial_process_file = partial(get_aggregate_leaderboard_history_on_single_file, FIRST_N=FIRST_N, step=step)

        # Map the function to each file in the list
        results = list(tqdm(executor.map(partial_process_file, battled_pairs_csv_file_list,), total=len(battled_pairs_csv_file_list)))
        leaderboard_history_at_cur_bootstrap_list.extend(results)
        
    # for i in tqdm(range(0, len(battled_pairs_csv_file_list))):
    #     leaderboard_history_at_cur_bootstrap = get_leaderboard_history(battled_pairs_csv_file_list[i], FIRST_N, step)
    #     # leaderboard_history_at_cur_bootstrap['rnd_bootstrap'] = leaderboard_history_at_cur_bootstrap
    #     leaderboard_history_at_cur_bootstrap_list.append(leaderboard_history_at_cur_bootstrap)
        
    concat_leaderboard_history_at_cur_bootstrap = pd.concat(leaderboard_history_at_cur_bootstrap_list)
    aggregated_leaderboard = concat_leaderboard_history_at_cur_bootstrap.groupby(['num_battle', 'model'])[['rank', 'elo_rating']].agg(['median', 'min', 'max']).reset_index()
    return aggregated_leaderboard
    
        
if __name__ == '__main__':
    import glob
    from pathlib import Path
    battle_outcome_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_1'
    analysis_data_dir = Path(battle_outcome_dir)/'.analysis'
    battled_pairs_csv_file_list = glob.glob(str(Path(analysis_data_dir)/'.bootstrap/battled_pairs_*.csv'))
    agg_df = get_aggregate_leaderboard_history(battled_pairs_csv_file_list, FIRST_N=None, step=10)
    # Renaming the "Unnamed" columns for easier access
    agg_df.columns = pd.MultiIndex.from_tuples([(x, y if 'Unnamed' not in y else '') for x, y in agg_df.columns])
    # Flatten the columns
    agg_df.columns = ['_'.join([c for c in col if c!='']).strip() for col in agg_df.columns.values]
    save_dir =  Path(battle_outcome_dir)/'output/data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    agg_df.to_csv(Path(save_dir)/'elo_rank_rating_agg_df.csv')
    