import os, sys
sys.path.append(r'/elo_bench')
import glob
from pathlib import Path
from battle_outcome_analysis.calculate_elo_rank_and_rating import get_aggregate_leaderboard_history
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

battle_outcome_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset'
agg_df = pd.read_csv(Path(battle_outcome_dir)/'output/data/elo_rank_rating_agg_df.csv')
save_dir = Path(battle_outcome_dir)/'output/plot/elo_rating'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# Plot 1: elo rating history

fig = px.line(agg_df, x='num_battle', y='elo_rating_median', color='model', title='elo rank history')
fig.write_image(Path(save_dir)/'elo_rating_trend.png')

# Plot 2: elo rating kendalltau
def cal_rank_kendalltau(df: pd.DataFrame):
    nums_battle = df['num_battle'].unique().tolist()
    rank_kendalltau = []
    for idx, num_battle in enumerate(nums_battle):
        if num_battle < 2:
            continue
        cur_rank = df[df['num_battle']==nums_battle[idx]].sort_values(by=['model'])['elo_rating_median'].astype(int).tolist()
        prev_rank = df[df['num_battle']==nums_battle[idx-1]].sort_values(by=['model'])['elo_rating_median'].astype(int).tolist()
        rank_kendalltau.append(
            {
                'num_battle': num_battle,
                'elo_rating_kendalltau': stats.kendalltau(cur_rank, prev_rank).correlation
            }
        )
    return pd.DataFrame.from_dict(rank_kendalltau)

rank_kendalltau = cal_rank_kendalltau(agg_df)
fig = px.line(rank_kendalltau, x='num_battle', y='elo_rating_kendalltau', title='rank kendalltau')
fig.write_image(Path(save_dir)/'elo_rating_kandelltau.png')