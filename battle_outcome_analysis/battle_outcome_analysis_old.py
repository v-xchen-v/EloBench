# Import necessary packages
import os, sys
sys.path.append(r'/elo_bench')
from pathlib import Path
import pandas as pd
from datamodel import BattleOutcomes, BootstrapedBattleOutcomes
from elo_rating.rating_evaluator import compute_actual_winrate_awinb
from tqdm import tqdm
from collections import defaultdict

result_dir = r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset'
NUM_BOOTSTRAP = 100


analysis_dump_dir = Path(result_dir)/'.analysis'
if not os.path.exists(analysis_dump_dir):
    os.makedirs(analysis_dump_dir)
bootstrap_n_elo_rating_file = Path(analysis_dump_dir)/'bootstrap_n_elo_rating.csv'
bootstrap_aggregate_elo_rating_file = Path(analysis_dump_dir)/'bootstrap_aggregate_elo_rating.csv'

battled_pairs_file = Path(result_dir)/'battled_pairs.csv'
# battled_pairs_file = Path(analysis_dump_dir/'battled_pairs.csv')

# setting for elo ratings
K = 4
    
# Step 1: clean battled pairs, remove None and invalid battles by gpt-4 judger not working, and drop a file with original index of nature battle pairs and battle records
cleaned_battled_pairs_file = Path(analysis_dump_dir/'battled_pairs.csv')
valid_winner = ['model_a', 'model_b', 'tie', 'tie(all bad)']
all_battled_paris = pd.read_csv(battled_pairs_file, index_col=0)
valid_battled_pairs = all_battled_paris[all_battled_paris['winner'].isin(valid_winner)]
invalid_battled_pairs = all_battled_paris[~all_battled_paris['winner'].isin(valid_winner)]
print(f'{len(invalid_battled_pairs)} invalid battled pairs removed!')
valid_battled_pairs.to_csv(cleaned_battled_pairs_file)

# # Step 2: make bootstrap battle pairs
nature_battle_outcomes = BattleOutcomes.read_csv(cleaned_battled_pairs_file)
if BootstrapedBattleOutcomes.is_cached(analysis_dump_dir, NUM_BOOTSTRAP):
    bootstrap_battle_outcomes = BootstrapedBattleOutcomes.read_csv(Path(analysis_dump_dir)/'.bootstrap')
else:
    bootstrap_battle_outcomes = BootstrapedBattleOutcomes(nature_battle_outcomes, NUM_BOOTSTRAP)
    bootstrap_battle_outcomes.to_csv(Path(analysis_dump_dir)/'.bootstrap')

# Step 3: cal elo per bootstrap
bootstrap_battle_outcomes.get_leaderboards(K=K).to_csv(bootstrap_n_elo_rating_file)

# Step 4: cal aggragate elo
# TODO: remove repeat cal elo on each bootstrap round
bootstrap_battle_outcomes.get_leaderboard(K=4).to_csv(bootstrap_aggregate_elo_rating_file)

# Step 5: cal actual winrate on no-tie battled pairs
# TODO: add nature battle round column
notie_winner = ['model_a', 'model_b']
notie_valid_battled_pairs = valid_battled_pairs[valid_battled_pairs['winner'].isin(notie_winner)]

def gen_actual_winrate_history(notie_battled_pairs_df, save_path: str):
    models = sorted(notie_battled_pairs_df['model_a'].unique())
    # info_logger.info(f'models: {models}')
    print(f'models: {models}')
    
    # swap model_a and model_b if model_a is not the first model
    def swap_model_a_b(row, models_in_order):
        if models_in_order.index(row['model_a']) > models_in_order.index(row['model_b']):
            return row
        else:
            # swap winner if swapped model_a and model_b
            winner = 'model_b' if row['winner'] == 'model_a' else 'model_a'
            return pd.Series([row['model_b'], row['model_a'], winner], index=['model_a', 'model_b', 'winner'])
        
    # Apply the function to each row
    notie_battled_pairs_df = notie_battled_pairs_df.apply(swap_model_a_b, args=(models,), axis=1)
    
    group_by_ab = notie_battled_pairs_df.groupby(['model_a', 'model_b'])
    
    # Create a dictionary to store each group as a DataFrame
    group_dfs = {}

    for (model_a, model_b), group_df in group_by_ab:
        # Create a DataFrame for each group
        group_dfs[(model_a, model_b)] = group_df.reset_index(drop=True)

    # Accessing a specific group, for example ('A', 'X')
    # specific_group_df = group_dfs.get(('gpt-35-turbo', 'Xwin-LM/Xwin-LM-7B-V0.1'), "Group not found")
    # print(specific_group_df)
    
    # iterate all the groups
    winrate_history = []
    winrate_history_dict = {}
    for (model_a, model_b), group_df in group_dfs.items():
        actual_winrate_history = []
        for i in range(1, len(group_df)):
            actual_winrate_awinb = compute_actual_winrate_awinb(group_df.head(i), model_a, model_b)
            # print(f'{i} {actual_winrate_awinb}')

            # record the actual winrate
            actual_winrate_history.append({
                'actual_winrate': actual_winrate_awinb,
                'num_battles': i,
                'models': f'{model_a} win {model_b}'
            })

        actual_winrate_history_df = pd.DataFrame(actual_winrate_history)
        if len(actual_winrate_history_df)>0:
            actual_winrate_history_df['delta_actual_winrate'] = actual_winrate_history_df['actual_winrate'].diff().abs()
            actual_winrate_history_df['model_a'] = model_a
            actual_winrate_history_df['model_b'] = model_b
        winrate_history.append(actual_winrate_history_df)
        winrate_history_dict[(model_a, model_b)] = actual_winrate_history_df
        
    # calcuated winrate history for each model pair(despite ab order)
    # self.winrate_history = winrate_history
    
    clusive_columns = ['models', 'actual_winrate', 'num_battles', 'delta_actual_winrate']
    pd.concat(winrate_history)[clusive_columns].to_csv(save_path)
gen_actual_winrate_history(notie_valid_battled_pairs, f'{analysis_dump_dir}/actual_winrate_history.csv')

# Step 6: Calculate predict winrate based on final aggregated elo ratings.
from elo_rating.rating_evaluator import compute_predict_winrate
compute_predict_winrate(pd.read_csv(bootstrap_aggregate_elo_rating_file)).to_csv(Path(analysis_dump_dir)/'predict_winrate_on_final_aggregated_elo.csv')


# # Step 6: Cal predict winrate based on aggregate elo and all n elo on each history point
# from elo_rating.rating_evaluator import compute_predict_winrate_awinb, compute_predict_winrate
# from datamodel.elo_rating_history import EloRatingHistory

# def gen_predict_winrate_history(bootstrap_battle_outcomes, valid_battled_pairs_df, save_path: str):
#     # history = EloRatingHistory.gen_history(save_path=boot, use_bootstrap=
#     #                                         False, step=1)
#     num_battles = len(bootstrap_battle_outcomes._bootstraped_battlecomes_dfs[0])
#     first_n_bootstrap_battle_outcomes_list = []
#     for i in tqdm(range(1, num_battles+1)):
#         first_n_bootstrap_battle_outcomes = bootstrap_battle_outcomes.get_first_n_rows(i)
#         first_n_bootstrap_battle_outcomes_list.append(first_n_bootstrap_battle_outcomes)
    
#     models = sorted(valid_battled_pairs_df['model_a'].unique())
#     print(f'models: {models}')
    
#     # elo_rating_history_df = history.to_df()
#     winrate_history = []
#     winrate_history_cur_bootstrap = defaultdict(lambda: [])
#     for num_battle in tqdm(range(1, num_battles+1), desc='predicting winrate'):
#         first_n_bootstrap_battle_outcomes = first_n_bootstrap_battle_outcomes_list[num_battle-1]
        
#         # bootstrap range on all round
#         first_n_bootstrap_n_elo_rating = first_n_bootstrap_battle_outcomes.get_leaderboards(K=K)
#         winrate_as_point_n = []
#         # for bootstrap_rnd in tqdm(range(1, NUM_BOOTSTRAP+1)):
#         for bootstrap_rnd in range(1, NUM_BOOTSTRAP+1):
#             first_n_bootstrap_cur_elo_rating = first_n_bootstrap_n_elo_rating[first_n_bootstrap_n_elo_rating['round_bootstrap']==bootstrap_rnd]
#             winrate_as_point_n.append(compute_predict_winrate(first_n_bootstrap_cur_elo_rating))
        
#         first_n_bootstrap_aggregate_elo_rating = first_n_bootstrap_battle_outcomes.get_leaderboard(K=K)
#         winrate_as_point = compute_predict_winrate(first_n_bootstrap_aggregate_elo_rating)
#         for model_a_i, model_a in enumerate(models):
#             for model_b in models[model_a_i+1:]:
#                 if model_a == model_b:
#                     continue
#                 if model_a not in winrate_as_point.index or model_b not in winrate_as_point.columns:
#                     predict_winrate_awinb = pd.NA
#                     continue
#                 else:
#                     predict_winrate_awinb = winrate_as_point.loc[model_a, model_b]
#                 # predict_winrate_awinb = compute_predict_winrate_awinb(history.get_point(num_battle), model_a, model_b)
                
#                 # record the actual winrate
#                 winrate_history.append({
#                     'predict_winrate': predict_winrate_awinb,
#                     'num_battles': num_battle,
#                     'models': f'{model_a} win {model_b}',
#                     'model_a': model_a,
#                     'model_b': model_b,
#                 })
                
#         # range
#         # winrate_history_n_bootstrap = []
#         for bootstrap_idx, winrate_as_point_cur in enumerate(winrate_as_point_n):
#             for model_a_i, model_a in enumerate(models):
#                 for model_b in models[model_a_i+1:]:
#                     if model_a == model_b:
#                         continue
#                     if model_a not in winrate_as_point_cur.index or model_b not in winrate_as_point_cur.columns:
#                         predict_winrate_awinb = pd.NA
#                         continue
#                     else:
#                         predict_winrate_awinb = winrate_as_point_cur.loc[model_a, model_b]
#                     # predict_winrate_awinb = compute_predict_winrate_awinb(history.get_point(num_battle), model_a, model_b)
                    
#                     # record the actual winrate
#                     winrate_history_cur_bootstrap[bootstrap_idx].append({
#                         'bootstrap_round': bootstrap_idx,
#                         'predict_winrate': predict_winrate_awinb,
#                         'num_battles': num_battle,
#                         'models': f'{model_a} win {model_b}',
#                         'model_a': model_a,
#                         'model_b': model_b,
#                     })
                    
#     for bootstrap_idx, winrate_history_cur_bootstrap in winrate_history_cur_bootstrap.items():
#         pd.DataFrame.from_dict(winrate_history_cur_bootstrap).to_csv(Path(analysis_dump_dir)/f'.bootstrap/predict_winrate_history_{bootstrap_idx}_bootstrap.csv')

#     predict_winrate_history_df = pd.DataFrame(winrate_history)
#     predict_winrate_history_df.to_csv(save_path)
        
# gen_predict_winrate_history(bootstrap_battle_outcomes, valid_battled_pairs, Path(analysis_dump_dir)/'predict_winrate_history.csv')