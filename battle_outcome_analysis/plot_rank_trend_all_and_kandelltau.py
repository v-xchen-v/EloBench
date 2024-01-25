import os, sys
sys.path.append(r'/elo_bench')
import glob
from pathlib import Path
from battle_outcome_analysis.calculate_elo_rank_and_rating import get_leaderboard_history, get_aggregate_leaderboard_history
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

battle_outcome_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset'
analysis_data_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset/.analysis'
battled_pairs_csv_file_list = glob.glob(str(Path(analysis_data_dir)/'.bootstrap/battled_pairs_*.csv'))[:10]
# agg_df = get_aggregate_leaderboard_history(battled_pairs_csv_file_list, FIRST_N=None, step=100)
# agg_df.to_csv(r'/elo_bench/battle_outcome_analysis/output/agg_df.csv')

agg_df = pd.read_csv(Path(battle_outcome_dir)/'output/data/elo_rank_rating_agg_df.csv')

save_dir = Path(battle_outcome_dir)/'output/plot/rank'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
fig = px.line(agg_df, x='num_battle', y='rank_median', color='model', title='elo rank history')
# reverse y axis
fig.update_yaxes(autorange="reversed")
fig.write_image(Path(save_dir)/'rank_trend.png')



# fig = px.scatter(agg_df, x='num_battle', y='median', title='Plot with Error Bars Representing Bounds',color='model', error_y='error_upper', error_y_minus="error_lower")
# fig = px.scatter(agg_df, x='num_battle', y='median', title='Plot with Error Bars Representing Bounds',color='model', error_y=dict(
#         type='data',
#         symmetric=False,
#         array=agg_df['error_upper'],
#         arrayminus=agg_df['error_lower'])
#     )

# Calculate error for lower and upper bounds
# agg_df['error_lower'] = agg_df['median'] - agg_df['min']
# agg_df['error_upper'] = agg_df['max'] - agg_df['median']

def cal_rank_kendalltau(df: pd.DataFrame):
    nums_battle = df['num_battle'].unique().tolist()
    rank_kendalltau = []
    for idx, num_battle in enumerate(nums_battle):
        if num_battle < 2:
            continue
        cur_rank = df[df['num_battle']==nums_battle[idx]].sort_values(by=['model'])['rank_median'].astype(int).tolist()
        prev_rank = df[df['num_battle']==nums_battle[idx-1]].sort_values(by=['model'])['rank_median'].astype(int).tolist()
        rank_kendalltau.append(
            {
                'num_battle': num_battle,
                'rank_kendalltau': stats.kendalltau(cur_rank, prev_rank).correlation
            }
        )
    return pd.DataFrame.from_dict(rank_kendalltau)

rank_kendalltau = cal_rank_kendalltau(agg_df)
fig = px.line(rank_kendalltau, x='num_battle', y='rank_kendalltau', title='rank kendalltau')
fig.write_image(Path(save_dir)/'rank_kandelltau.png')