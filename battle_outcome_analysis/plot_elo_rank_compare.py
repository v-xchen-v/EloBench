import pandas as pd
import os
from pathlib import Path

later_register_type ='model' # 'q' or 'model'
n_later = 1
elo_rating_files = \
[
    rf'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset/.analysis/bootstrap_aggregate_elo_rating.csv',
    # rf'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_{later_register_type}_later_{n_later}_before_register/.analysis/bootstrap_aggregate_elo_rating.csv',
    # rf'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_{later_register_type}_later_{n_later}/.analysis/bootstrap_aggregate_elo_rating.csv',
    rf'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_nx1_gpt4/.analysis/bootstrap_aggregate_elo_rating.csv',
    rf'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_nx1_gpt35/.analysis/bootstrap_aggregate_elo_rating.csv',
]

def process(elo_rating_file):
    df = pd.read_csv(elo_rating_file, index_col=0)
    df['rank'] = df['elo_rating'].rank(method='dense', ascending=False)
    setting_desc = os.path.basename(os.path.dirname(os.path.dirname(elo_rating_file))).replace('google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset', '')
    if setting_desc == '':
        setting_desc = 'gt'
    # elif setting_desc.find('_before_register')>0:
    #     setting_desc = f'before_register'
    # else:
    #     setting_desc = f'after_register'
    df['setting'] = setting_desc
    return df

dfs = []
for eloraing_file in elo_rating_files:
    df = process(eloraing_file)
    dfs.append(df)
all = pd.concat(dfs, ignore_index=True)
all = all.sort_values(by='setting', key=lambda x: x.map({'gt':0, 'after_register':1, 'before_register':2}))

save_dir = r'/elo_bench/battle_outcome_analysis/output/temp'

# def plot_and_save(all, setting1, setting2):
#     elo_rating_df_across_setting = all[all['setting'].isin([setting1, setting2])]
#     import plotly.express as px
#     # Create a color mapping for each unique model
#     color_map = px.colors.qualitative.Plotly # Or choose another color sequence
#     models = elo_rating_df_across_setting['model'].unique()
#     model_colors = {model: color_map[i % len(color_map)] for i, model in enumerate(models)}

#     fig = px.line(elo_rating_df_across_setting, x='setting', y='rank', color='model', title='Elo rank over setting')
#     # add points for each model
#     for model in elo_rating_df_across_setting['model'].unique():
#         fig.add_scatter(x=elo_rating_df_across_setting[elo_rating_df_across_setting['model']==model]['setting'], y=elo_rating_df_across_setting[elo_rating_df_across_setting['model']==model]['rank'], mode='markers', name=model,marker=dict(color=model_colors[model]))
#     # reverse y axis
#     fig.update_yaxes(autorange="reversed")
#     # order legend by rank
#     fig.update_layout(legend=dict(traceorder="normal"))

#     # avoid text and legend being cut off
#     fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))
#     fig.write_image(Path(save_dir)/f'rank_compare_register_{later_register_type}_later_{n_later}_{setting1}_vs_{setting2}.png')

#     import plotly.express as px
#     fig = px.line(elo_rating_df_across_setting, x='setting', y='elo_rating', color='model', title='Elo rating over setting')
#     # add points for each model and color by model
#     for model in elo_rating_df_across_setting['model'].unique():
#         fig.add_scatter(x=elo_rating_df_across_setting[elo_rating_df_across_setting['model']==model]['setting'], y=elo_rating_df_across_setting[elo_rating_df_across_setting['model']==model]['elo_rating'], mode='markers', name=model, marker=dict(color=model_colors[model]))
#     # reverse y axis
#     # fig.update_yaxes(autorange="reversed")
#     # order legend by elo rating
#     fig.update_layout(legend=dict(traceorder="normal"))
#     # avoid text and legend being cut off
#     fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))
#     fig.write_image(Path(save_dir)/f'rating_compare_register_{later_register_type}_later_{n_later}_{setting1}_vs_{setting2}.png')
    
# plot_and_save(all, 'gt', f'after_register_{n_later}_{later_register_type}')
# plot_and_save(all, 'gt', f'before_register_{n_later}_{later_register_type}')
# # plot_and_save(all, f'before_register_{n_later}_{later_register_type}', f'after_register_{n_later}_{later_register_type}')

import plotly.express as px
# Create a color mapping for each unique model
color_map = px.colors.qualitative.Plotly # Or choose another color sequence
models = all['model'].unique()
model_colors = {model: color_map[i % len(color_map)] for i, model in enumerate(models)}
import plotly.graph_objects as go

fig = px.line(all, x='setting', y='rank', color='model', title=f'Elo rank over setting')
# add points for each model
for model in all['model'].unique():
    fig.add_scatter(x=all[all['model']==model]['setting'], y=all[all['model']==model]['rank'], mode='markers', name=model,marker=dict(color=model_colors[model]))
# reverse y axis
fig.update_yaxes(autorange="reversed")
# order legend by rank
fig.update_layout(legend=dict(traceorder="normal"))

# avoid text and legend being cut off
fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))
# fig.write_image(Path(save_dir)/f'rank_compare_register_{later_register_type}_later_{n_later}.png')
fig.write_image(Path(save_dir)/f'rank_compare_nxn_vs_nx1.png')

import plotly.express as px
fig = px.line(all, x='setting', y='elo_rating', color='model', title='Elo rating over setting')
# add points for each model and color by model
for model in all['model'].unique():
    fig.add_scatter(x=all[all['model']==model]['setting'], y=all[all['model']==model]['elo_rating'], mode='markers', name=model, marker=dict(color=model_colors[model]))
# reverse y axis
# fig.update_yaxes(autorange="reversed")
# order legend by elo rating
fig.update_layout(legend=dict(traceorder="normal"))
# avoid text and legend being cut off
fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))
# fig.update_xaxes(categoryorder='array', categoryarray= ['gt','before_register','after_register'])
# fig.write_image(Path(save_dir)/f'rating_compare_register_{later_register_type}_later_{n_later}.png')
fig.write_image(Path(save_dir)/f'rating_compare_nxn_vs_nx1.png')