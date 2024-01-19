import os, sys
sys.path.append(r'/elo_bench')
import pandas as pd
elo_rank_rating_agg_df = pd.read_csv(r'/elo_bench/battle_outcome_analysis/output/elo_rank_rating_agg_df.csv')
from elo_rating.rating_evaluator import compute_predict_winrate

num_battles = elo_rank_rating_agg_df['num_battle'].unique()
predict_winrate_list = []
for num_battle in num_battles:
    current_rating = elo_rank_rating_agg_df[elo_rank_rating_agg_df['num_battle']==num_battle]
    # rename column elo_rating_median to elo_rating
    current_rating = current_rating.rename(columns={'elo_rating_median': 'elo_rating'})
    predict_winrate = compute_predict_winrate(current_rating)
    predict_winrate['num_battle'] = num_battle
    predict_winrate_list.append(predict_winrate)
pd.concat(predict_winrate_list).to_csv(r'/elo_bench/battle_outcome_analysis/output/data/predict_winrate.csv')
    
