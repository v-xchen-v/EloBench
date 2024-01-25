import pandas as pd
import numpy as np

gt_file = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_add_mistral_notie200/.analysis/bootstrap_aggregate_elo_rating.csv'

later_n = 10 # 1, 3, 5, 10
later_register_models_files = [\
    rf'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_{later_n}/.analysis/bootstrap_aggregate_elo_rating.csv',
    rf'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_{later_n}_seed2/.analysis/bootstrap_aggregate_elo_rating.csv',
    rf'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_{later_n}_seed3/.analysis/bootstrap_aggregate_elo_rating.csv',
    rf'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_{later_n}_seed4/.analysis/bootstrap_aggregate_elo_rating.csv',
    rf'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_{later_n}_seed5/.analysis/bootstrap_aggregate_elo_rating.csv',
]

# Calculate the average delta elo rating between the original and later registered
gt_elo_rating = pd.read_csv(gt_file)[['elo_rating', 'model']]
custom_dict = {}
for idx, model in enumerate(gt_elo_rating['model']):
    custom_dict[model] = idx
    
later_elo_ratings = [pd.read_csv(file)[['elo_rating', 'model']] for file in later_register_models_files]
for later_elo_rating in later_elo_ratings:
    later_elo_rating.sort_values(by=['model'], inplace=True, key=lambda x: x.map(custom_dict))
deltas = []
for later_elo_rating in later_elo_ratings:
    delta = ((gt_elo_rating['elo_rating'] - later_elo_rating['elo_rating']).abs()).sum() / len(gt_elo_rating)
    print(f'delta elo rating: {delta}')
    deltas.append(delta)
average_delta = np.array(deltas).mean()
print(f'average delta elo rating for {later_n} later: {average_delta}')

# Calcuate the kendall tau rank correlation between the original and later registered
gt_elo_rank = pd.read_csv(gt_file)[['rank', 'model']]
later_elo_ranks = [pd.read_csv(file)[['rank', 'model']] for file in later_register_models_files]
for later_elo_rank in later_elo_ranks:
    later_elo_rank.sort_values(by=['model'], inplace=True, key=lambda x: x.map(custom_dict))
kendalltau_rank_correlations = []
from scipy.stats import kendalltau
for later_elo_rank in later_elo_ranks:
    kendalltau_rank_correlation = kendalltau(gt_elo_rank['rank'], later_elo_rank['rank']).correlation
    print(f'kendall tau rank correlation: {kendalltau_rank_correlation}')
    kendalltau_rank_correlations.append(kendalltau_rank_correlation)
average_kendalltau =  np.array(kendalltau_rank_correlations).mean()
print(f'average kendall tau rank correlation for {later_n} later: {np.round(average_kendalltau, 2)}')