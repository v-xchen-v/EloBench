from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import os
battle_outcome_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_nx1_gpt4' 
bootstrap_n_elo_rating_file = Path(battle_outcome_dir)/'.analysis/bootstrap_n_elo_rating.csv'
save_dir = Path(battle_outcome_dir)/'output/plot/elo_rating'

if not save_dir.exists():
    save_dir.mkdir(parents=True)
    
def get_elo_rating_with_condience_interval(bootstrap_n_elo_rating_file):
    # return "Elo rating with 95% confidence interval"
    title = 'default'
    
    setting_desc = os.path.basename(os.path.dirname(os.path.dirname(bootstrap_n_elo_rating_file))).replace('google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset', '')
    if setting_desc == '':
        setting_desc = 'gt'
    # elif setting_desc.find('_before_register')>0:
    #     setting_desc = f'before_register'
    # else:
    #     setting_desc = f'after_register'
    title = setting_desc
        
    df = pd.read_csv(bootstrap_n_elo_rating_file)
    # df.groupby('model').mean().sort_values('rating', ascending=False)
    df = df.groupby('model')['elo_rating'].agg(['min', 'max','median']).reset_index()
    df = df.sort_values(by=['median'], ascending=False)
    # bars = pd.DataFrame(dict(
    #     lower = df['min'],
    #     rating = df['median'],
    #     upper = df['max']))
    df['error_y'] = df['max'] - df["median"]
    df['error_y_minus'] = df['median'] - df["min"]
    df['rating_rounded'] = np.round(df['median'], 2)
    fig = px.scatter(df, x="model", y="median", error_y="error_y",
                        error_y_minus="error_y_minus", text="rating_rounded",
                        title=title)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating")
    # rotate x axis text 45 degree
    fig.update_layout(xaxis_tickangle=-45)
    # larger the image
    fig.update_layout(height=600, width=1000)
    return fig

get_elo_rating_with_condience_interval(bootstrap_n_elo_rating_file).write_image(str(save_dir/'elo_rating_with_range.png'))