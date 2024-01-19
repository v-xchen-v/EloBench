import os, sys
sys.path.append(r'/elo_bench')
import glob
from pathlib import Path
from battle_outcome_analysis.calculate_elo_rank_and_rating import get_leaderboard_history, get_aggregate_leaderboard_history
import pandas as pd

battled_pairs_csv_file = r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset/battled_pairs.csv'

save_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset/.analysis'
battled_pairs_csv_file_list = glob.glob(str(Path(save_dir)/'.bootstrap/battled_pairs_*.csv'))[:10]
# agg_df = get_aggregate_leaderboard_history(battled_pairs_csv_file_list, FIRST_N=None, step=100)
# agg_df.to_csv(r'/elo_bench/battle_outcome_analysis/output/agg_df.csv')

agg_df = pd.read_csv(r'/elo_bench/battle_outcome_analysis/output/data/elo_rank_rating_agg_df.csv')

import plotly.graph_objects as go
grouped = agg_df.groupby('model')
dfs = {name: group.reset_index(drop=True) for name, group in grouped}
model_names = list(dfs.keys())
for i, model_name in enumerate(model_names):
    if i==0:
        # fig = go.Figure(data=go.Scatter(
        #         x=dfs[model_name]['num_battle'],
        #         y=dfs[model_name]['median'],
        #         error_y=dict(
        #             type='data', # value of error bar given in data coordinates
        #             array=dfs[model_name]['error_lower'],
        #             visible=True),
        #         name=model_name
        #     ))
        
        fig = go.Figure([
            go.Scatter(
                x=dfs[model_name]['num_battle'],
                y=dfs[model_name]['rank_median'],
                line=dict(color='rgb(0,100,80)'),
                mode='lines',
                name=f'{model_name} Rank'
            ),
            go.Scatter(
                x=list(dfs[model_name]['num_battle'])+list(dfs[model_name]['num_battle'][::-1]), # x, then x reversed
                y=list(dfs[model_name]['rank_max'])+list(dfs[model_name]['rank_min'][::-1]), # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            )
        ])
        
        fig.write_image(f'/elo_bench/battle_outcome_analysis/output/plot/rank/rank_trend_with_range_{model_name.replace("/", ".")}.png')
    else:
        # fig = go.Figure(data=go.Scatter(
        #         x=dfs[model_name]['num_battle'],
        #         y=dfs[model_name]['rank_median'],
        #         error_y=dict(
        #             type='data', # value of error bar given in data coordinates
        #             array=dfs[model_name]['error_lower'],
        #             visible=True),
        #         name=model_name
        #     ))
        
        fig = go.Figure([
            go.Scatter(
                x=dfs[model_name]['num_battle'],
                y=dfs[model_name]['rank_median'],
                line=dict(color='rgb(0,100,80)'),
                mode='lines',
                name=f'{model_name} Rank'
            ),
            go.Scatter(
                x=list(dfs[model_name]['num_battle'])+list(dfs[model_name]['num_battle'][::-1]), # x, then x reversed
                y=list(dfs[model_name]['rank_max'])+list(dfs[model_name]['rank_min'][::-1]), # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            )
        ])
        fig.write_image(f'/elo_bench/battle_outcome_analysis/output/plot/rank/rank_trend_with_range_{model_name.replace("/", ".")}.png')
        # fig.add_scatter(
        #     x=dfs[model_name]['num_battle'],
        #     y=dfs[model_name]['rank_median'],
        #     error_y=dict(
        #             type='data', # value of error bar given in data coordinates
        #             array=dfs[model_name]['error_lower'],
        #             visible=True),
        #     name=model_name
        # )

# fig.write_image('/elo_bench/battle_outcome_analysis/output/rank_trend_with_range_all.png')
