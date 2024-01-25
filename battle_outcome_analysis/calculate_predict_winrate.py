import os, sys
sys.path.append(r'/elo_bench')
import pandas as pd
from pathlib import Path
from elo_rating.rating_evaluator import compute_predict_winrate

battle_outcome_dir = r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_1'
elo_rank_rating_agg_df = pd.read_csv(Path(battle_outcome_dir)/'output/data/elo_rank_rating_agg_df.csv')
save_dir = Path(battle_outcome_dir)/'output/data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num_battles = elo_rank_rating_agg_df['num_battle'].unique()
predict_winrate_list = []
for num_battle in num_battles:
    current_rating = elo_rank_rating_agg_df[elo_rank_rating_agg_df['num_battle']==num_battle]
    # rename column elo_rating_median to elo_rating
    current_rating = current_rating.rename(columns={'elo_rating_median': 'elo_rating'})
    predict_winrate = compute_predict_winrate(current_rating)
    predict_winrate['num_battle'] = num_battle
    predict_winrate_list.append(predict_winrate)
pd.concat(predict_winrate_list).to_csv(Path(save_dir)/'predict_winrate.csv')
    
