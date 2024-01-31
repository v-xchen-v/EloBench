import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
from data import get_arena_battles_20230717_data, ARENA_K
from elo_rating.rating_helper import get_players_rating_and_rank_from_battles_data, get_bootstrap_medium_elo
import pandas as pd
import plotly.express as px
from collections import defaultdict
import numpy as np

def compute_pairwise_win_count(battles, ordering):
    # Define a function to apply
    def get_value_from_column(row):
        return row[row['winner']]

    # Apply the function
    battles['winner_name'] = battles.apply(get_value_from_column, axis=1)
    def get_a_win_b_count(a, b):
        if a == b:
            return pd.NA
        else:
            ab_battles = battles[((battles['model_a']==a) & (battles['model_b']==b)) | (battles['model_a']==b) & (battles['model_b']==a)]
            if ab_battles.shape[0] == 0:
                return pd.NA
            
            ab_awin_battles = ab_battles[ab_battles['winner_name']==a]
            return ab_awin_battles.shape[0]
    
    union_model_names = pd.concat([
        pd.Series(battles['model_a'].unique()), 
        pd.Series(battles['model_b'].unique())]).unique()
    
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for name_a in union_model_names:
        for name_b in union_model_names:
            wins[name_a][name_b] = get_a_win_b_count(name_a, name_b)
    
    data = {
        a: [wins[a][b] if a != b else np.NAN for b in union_model_names]
        for a in union_model_names
    }

    df = pd.DataFrame(data, index=union_model_names)
    row_beats_col_freq = df.T
    df.index.name = "model_a"
    df.columns.name = "model_b"
    
    # Arrange ordering according to proprition of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    model_names = list(prop_wins.keys())
    
    if ordering is not None:
        model_names = ordering
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col

def compute_pairwise_count(battles, ordering):
    def get_a_win_b_count(a, b):
        if a == b:
            return pd.NA
        else:
            ab_battles = battles[((battles['model_a']==a) & (battles['model_b']==b)) | (battles['model_a']==b) & (battles['model_b']==a)]
            if ab_battles.shape[0] == 0:
                return pd.NA
            
            return ab_battles.shape[0]
    
    union_model_names = pd.concat([
        pd.Series(battles['model_a'].unique()), 
        pd.Series(battles['model_b'].unique())]).unique()
    
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for name_a in union_model_names:
        for name_b in union_model_names:
            wins[name_a][name_b] = get_a_win_b_count(name_a, name_b)
    
    data = {
        a: [wins[a][b] if a != b else np.NAN for b in union_model_names]
        for a in union_model_names
    }


    df = pd.DataFrame(data, index=union_model_names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    
    row_beats_col_freq = df.T
    
    # Arrange ordering according to proprition of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    model_names = list(prop_wins.keys())
    
    if ordering is not None:
        model_names = ordering
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col


def compute_pairwise_win_fraction(battles):
    # Define a function to apply
    def get_value_from_column(row):
        return row[row['winner']]

    # Apply the function
    battles['winner_name'] = battles.apply(get_value_from_column, axis=1)
    def get_a_win_b_count(a, b):
        if a == b:
            return pd.NA
        else:
            ab_battles = battles[((battles['model_a']==a) & (battles['model_b']==b)) | (battles['model_a']==b) & (battles['model_b']==a)]
            if ab_battles.shape[0] == 0:
                return pd.NA
            
            ab_awin_battles = ab_battles[ab_battles['winner_name']==a]
            return ab_awin_battles.shape[0]/ab_battles.shape[0]
    
    union_model_names = pd.concat([
        pd.Series(battles['model_a'].unique()), 
        pd.Series(battles['model_b'].unique())]).unique()
    
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for name_a in union_model_names:
        for name_b in union_model_names:
            wins[name_a][name_b] = get_a_win_b_count(name_a, name_b)
    
    data = {
        a: [wins[a][b] if a != b else np.NAN for b in union_model_names]
        for a in union_model_names
    }

    df = pd.DataFrame(data, index=union_model_names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    
    row_beats_col_freq = df.T
    
    # Arrange ordering according to proprition of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    model_names = list(prop_wins.keys())
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col


def visualize_pairwise_win_fraction(battles, title):
    row_beats_col = compute_pairwise_win_fraction(battles)
    fig = px.imshow(row_beats_col, color_continuous_scale='RdBu',
                    text_auto=".2f", title=title)
    fig.update_layout(xaxis_title=" Model B: Loser",
                  yaxis_title="Model A: Winner",
                  xaxis_side="top", height=600, width=600,
                  title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                  "Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>")

    return fig

def visualize_pairwise_win_count(battles, title, ordering):
    row_beats_col = compute_pairwise_win_count(battles, ordering)
    fig = px.imshow(row_beats_col, color_continuous_scale='RdBu',
                    text_auto=".2f", title=title)
    fig.update_layout(xaxis_title=" Model B: Loser",
                  yaxis_title="Model A: Winner",
                  xaxis_side="top", height=600, width=600,
                  title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                  "Model A: %{y}<br>Model B: %{x}<br>Count of A Wins: %{z}<extra></extra>")

    return fig

def visualize_pairwise_count(battles, title, ordering):
    row_beats_col = compute_pairwise_count(battles, ordering)
    fig = px.imshow(row_beats_col, color_continuous_scale='RdBu',
                    text_auto=".2f", title=title)
    fig.update_layout(xaxis_title=" Model B",
                  yaxis_title="Model A",
                  xaxis_side="top", height=600, width=600,
                  title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                  "Model A: %{y}<br>Model B: %{x}<br>Count of A Battles: %{z}<extra></extra>")

    return fig

def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    # names = sorted(list(elo_ratings['Model']))
    # ratings = elo_ratings['Elo Rating']
    ratings_dict = {}
    for idx, row in elo_ratings.iterrows():
        ratings_dict[row['model']] = row['elo_rating']
        
    
    names = sorted(list(ratings_dict.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((ratings_dict[b] - ratings_dict[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.NAN for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T

def vis_predict_win_rate(win_rate):
    ordered_models = win_rate.mean(axis=1).sort_values(ascending=False).index
    fig = px.imshow(win_rate.loc[ordered_models, ordered_models],
                    color_continuous_scale='RdBu', text_auto=".2f",
                    title="Predicted Win Rate Using Elo Ratings for Model A in an A vs. B Battle")
    fig.update_layout(xaxis_title="Model B",
                    yaxis_title="Model A",
                    xaxis_side="top", height=600, width=600,
                    title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                    "Model A: %{y}<br>Model B: %{x}<br>Win Rate: %{z}<extra></extra>")
    return fig

def get_gpt4_judger_elo_on_arena(filepath=r'results/google_quora_alpaca_10629_test2/battle_records.csv', use_bootstrap=True):
    df = pd.read_csv(filepath)
    columns_to_inclusive = ['model_a', 'model_b', 'winner']
    data = df[columns_to_inclusive]
    
    # remove nan
    data = data[(data['winner'].isna()==False) & (data['winner']!='invalid')]
    
    # new_column_names = {"gpt_winner": 'winner'}
    # data.rename(columns=new_column_names, inplace=True)
    if not use_bootstrap:
        elo_result = get_players_rating_and_rank_from_battles_data(data, K=ARENA_K)
    else:
        elo_result = get_bootstrap_medium_elo(data, ARENA_K)
    data_no_ties = data[data['winner'].str.contains('tie', na=False) == False]
    fig = visualize_pairwise_win_fraction(data_no_ties,
      title = "Fraction of Model A Wins for All Non-tied A vs. B Battles")
    fig2 = vis_predict_win_rate(predict_win_rate(elo_result))
    
    model_ordering = elo_result['Model'].tolist()
    no_tie_battle_count_fig = visualize_pairwise_count(data_no_ties, title =  "Count of Model A Battles for All Non-tied A vs. B Battles", ordering=model_ordering)
    fig3 = visualize_pairwise_win_count(data_no_ties, title = "Count of Model A Wins for All Non-tied A vs. B Battles", ordering=model_ordering)
    fig5 = visualize_pairwise_count(data, title =  "Count of Model A Battles for All A vs. B Battles", ordering=model_ordering)
    return elo_result, fig, fig2, fig3, no_tie_battle_count_fig, fig5

with gr.Blocks() as demo:
    gr.Markdown('🏆Elo Bench Leadboard')
    gr.Markdown('AI Dueling Arena with GPT-4 Adjudication')
    
demo.launch()
    