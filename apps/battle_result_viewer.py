import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import plotly.express as px

import gradio as gr
from pathlib import Path
import pandas as pd

from data import ARENA_K
from elo_rating.rating_helper import get_elo_results_from_battles_data, get_bootstrap_medium_elo, get_bootstrap_result

from datamodel.elo_rating_history import EloRatingHistory, BattleOutcomes
import numpy as np
from collections import defaultdict
from elo_rating.rating_evaluator import evaluate_rank_consistency, evaluate_winrate_at_historypoint

from tqdm import tqdm

result_dir = r'results/google_quora_alpaca_10629_test3'
record_file = Path(result_dir)/'battle_records.csv'

USE_BOOTSTRAP_ON_ELO = False
USE_BOOTSTRAP_ON_HISTORY = True
FIRST_N_BATTLES = None
records_df = pd.read_csv(record_file, nrows=FIRST_N_BATTLES)

# get winner data
winner_columns_to_inclusive = ['model_a', 'model_b', 'winner']
winner_data = records_df[winner_columns_to_inclusive]
# remove nan
winner_data_valid = winner_data[(winner_data['winner'].isna()==False) & (winner_data['winner']!='invalid')]
winner_no_ties = winner_data_valid[winner_data_valid['winner'].str.contains('tie', na=False) == False]        
        
model_ordering = None

def elo_leaderboard():
    global model_ordering
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
        
    with gr.Tab('Elo Leaderboard'):
        if USE_BOOTSTRAP_ON_ELO:
            with gr.Tab("Elo rating (bootstrap=1000)"):
                elo_result, elo_result_median_all = get_bootstrap_medium_elo(winner_data_valid, ARENA_K, with_fullset=True)
                
                def visualize_bootstrap_scores(df, title):
                    bars = pd.DataFrame(dict(
                        lower = df.quantile(.025),
                        rating = df.quantile(.5),
                        upper = df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
                    bars['error_y'] = bars['upper'] - bars["rating"]
                    bars['error_y_minus'] = bars['rating'] - bars["lower"]
                    bars['rating_rounded'] = np.round(bars['rating'], 2)
                    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                                    error_y_minus="error_y_minus", text="rating_rounded",
                                    title=title)
                    fig.update_layout(xaxis_title="Model", yaxis_title="Rating")
                    return fig
            
            
                # winner_data_no_ties = winner_data_valid[winner_data_valid['winner'].str.contains('tie', na=False) == False]
                model_ordering= elo_result['model'].tolist()   
                
                elo_df = elo_result.sort_values(by=['elo_rating'], ascending=False)
                gr.Dataframe(elo_df, wrap=True)
            
                bootstrap_of_elo_fig = visualize_bootstrap_scores(elo_result_median_all, "Bootstrap of Elo Estimates")
                gr.Plot(bootstrap_of_elo_fig)
            
        
        
                actual_winrate_fig = visualize_pairwise_win_fraction(winner_no_ties,
                    title = "Fraction of Model A Wins for All Non-tied A vs. B Battles")
                predict_winrate_fig = vis_predict_win_rate(predict_win_rate(elo_result))
                
                gr.Markdown("Predict winrate Should be close to actual to actual winrate")
                
                with gr.Row():
                    gr.Plot(actual_winrate_fig)
                    gr.Plot(predict_winrate_fig)
                    
        with gr.Tab("Elo rating (without bootstrap)"):
            elo_result = get_elo_results_from_battles_data(winner_data_valid, K=ARENA_K)
            
            model_ordering= elo_result['model'].tolist()   
            
            elo_df = elo_result.sort_values(by=['elo_rating'], ascending=False)
            gr.Dataframe(elo_df, wrap=True)

            # actual_winrate_fig = visualize_pairwise_win_fraction(winner_no_ties,
            #     title = "Fraction of Model A Wins for All Non-tied A vs. B Battles")
            # predict_winrate_fig = vis_predict_win_rate(predict_win_rate(elo_result))
            
            # gr.Markdown("Predict winrate Should be close to actual to actual winrate")
            
            # with gr.Row():
            #     gr.Plot(actual_winrate_fig)
            #     gr.Plot(predict_winrate_fig)
    
def battle_outcomes():
    with gr.Tab("Battle Outcomes"):
        gr.DataFrame(winner_data_valid, wrap=True)
        
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

        def visualize_battle_count(battles, title, ordering=None):
            union_model_names = pd.concat([
                pd.Series(battles['model_a'].unique()), 
                pd.Series(battles['model_b'].unique())]).unique()
            battle_counts = {}
            for key in union_model_names:
                battle_counts[key] = {subkey: 0 for subkey in union_model_names}
            
            for model_a, model_b in zip(battles['model_a'], battles['model_b']):
                battle_counts[model_a][model_b]+=1
                    
            battle_counts_pf = pd.DataFrame.from_dict(battle_counts)
            battle_counts_despite_ab_order = battle_counts_pf + battle_counts_pf.T
            
            if ordering is None:
                ordering = battle_counts_despite_ab_order.sum().sort_values(ascending=False).index
            
            # Reindexing rows and columns
            battle_counts_despite_ab_order = battle_counts_despite_ab_order.reindex(index=ordering, columns=ordering)
            
            
            # Create a mask for the lower triangle
            mask = np.tril(np.ones(battle_counts_despite_ab_order.shape)).astype(bool)
            
            # Apply the mask to the DataFrame
            battle_counts_despite_ab_order.where(mask, np.nan, inplace=True)

            # fig = px.imshow(battle_counts.loc[ordering, ordering],
            #                 title=title, text_auto=True, width=600)
            fig = px.imshow(battle_counts_despite_ab_order.loc[ordering, ordering],
                            title=title, text_auto=True, width=600)
            fig.update_layout(xaxis_title="Model 1",
                            yaxis_title="Model 2",
                            xaxis_side="top", height=600, width=600,
                            title_y=0.07, title_x=0.5)
            fig.update_traces(hovertemplate=
                            "Model 1: %{y}<br>Model 2: %{x}<br>Count: %{z}<extra></extra>")
            return fig
        
        win_count_notie_fig = visualize_pairwise_win_count(winner_no_ties, title = "Count of Model A Wins for All Non-tied A vs. B Battles", ordering=model_ordering)
        count_notie_fig = visualize_battle_count(winner_no_ties, title =  "Count of Model Battles for All A vs B (despite A/B order) Battles", ordering=model_ordering)
        gr.Plot(count_notie_fig)
        gr.Plot(win_count_notie_fig)
        
def ab_bias():
    with gr.Tab("A/B Bias"):
        winner_df = winner_data_valid['winner'].value_counts(normalize=True).reset_index()
        gr.Dataframe(winner_df, wrap=True)
        
        gr.Markdown("Model should play as model_a as much as model_b to avoid A/B bias.")
        # Count frequencies in each column
        model_a_counts = winner_data_valid['model_a'].value_counts()
        model_b_counts = winner_data_valid['model_b'].value_counts()

        # Convert Series to DataFrame
        model_a_df = model_a_counts.reset_index().rename(columns={'model_a': 'model', 'count': 'count_as_model_a'})
        model_b_df = model_b_counts.reset_index().rename(columns={'model_b': 'model', 'count': 'count_as_model_b'})

        # Merge the two DataFrames on 'model'
        frequency_df = pd.merge(model_a_df, model_b_df, on='model', how='outer').fillna(0)

        # print(frequency_df)
        gr.Dataframe(frequency_df, wrap=True)

def elo_history():
    with gr.Tab("Elo History"):
        # Plotting
        def plot_ranking_history(df):
            ranking_fig = px.line(df, x="num_battle", y="rank", color="model", markers=True)
            ranking_fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))

            ranking_fig.update_yaxes(autorange="reversed")  # Reversing the y-axis
                        
            return ranking_fig
        
        def plot_rank_consistency(df):
            ranking_consistency_fig = px.line(df, x="num_battle", y="rank_consistency", markers=True)
            ranking_consistency_fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
            
            # limit the y-axis range from [-1, 1]
            ranking_consistency_fig.update_layout(yaxis_range=[-1,1])
            
            return ranking_consistency_fig
        
        def plot_winrate_mae(df):
            mae_fig = px.line(df, x="num_battle", y="mae", markers=True)
            mae_fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
            
            # limit the y-axis range from [-1, 1]
            # ranking_consistency_fig.update_layout(yaxis_range=[-1,1])
            
            return mae_fig
        
        def calculate_rank_consistency_metrics(history: EloRatingHistory) -> pd.DataFrame:
            rank_consistency_history = []
            for idx, battle_num in tqdm(enumerate(history.recorded_battle_num), desc='calculate ranking consistency'):
                if idx == 0:
                    continue
                
                # history_cur = history.get_point(history.recorded_battle_num[idx])
                # history_prev = history.get_point(history.recorded_battle_num[idx-1])
                rank_consistency = evaluate_rank_consistency(history, history.recorded_battle_num[idx-1], history.recorded_battle_num[idx])
                if rank_consistency is not None:
                    rank_consistency_history.append({'num_battle': battle_num, 'rank_consistency': rank_consistency})
            rank_consistency_history_pd = pd.DataFrame.from_dict(rank_consistency_history)
            return rank_consistency_history_pd
        
        def calculate_winrate_metrics(history: EloRatingHistory) -> pd.DataFrame:
            winrate_history = []
            for idx, battle_num in tqdm(enumerate(history.recorded_battle_num), desc='calculate winrate mae'):
                if idx == 0:
                    continue
                
                battle_outcomes = BattleOutcomes.read_csv(Path(result_dir) / 'battled_pairs.csv', nrows=FIRST_N_BATTLES)
                winrate_mse_mae = evaluate_winrate_at_historypoint(history, battle_outcomes, history.recorded_battle_num[idx])
                winrate_history.append({'num_battle': battle_num, 'mae': winrate_mse_mae['mae']})
            winrate_history_pd = pd.DataFrame.from_dict(winrate_history)
            return winrate_history_pd
        
        def vis_rating_history(history: EloRatingHistory):
            rating_history = []
            for idx, battle_num in tqdm(enumerate(history.recorded_battle_num), desc='loading rating history'):
                point = history.get_point(battle_num)
                for _, row in point.iterrows():
                    rating_history.append({
                        'model': row['model'],
                        'elo_rating': row['elo_rating'],
                        'num_battle': battle_num,
                    })
            rating_history_pd = pd.DataFrame.from_dict(rating_history)
            
            # plotting
            rating_history_fig = px.line(rating_history_pd, x="num_battle", y="elo_rating", color="model", markers=True)
            rating_history_fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))

            gr.Plot(rating_history_fig)   
            
        def vis_rating_delta_history(history: EloRatingHistory):
            rating_history = []
            for idx, battle_num in tqdm(enumerate(history.recorded_battle_num), desc='calculate rating delta'):
                if idx == 0:
                    continue
                point_cur = history.get_point(battle_num)
                point_prev = history.get_point(history.recorded_battle_num[idx-1])
                for _, row in point_cur.iterrows():
                    if row['model'] not in point_prev['model'].values:
                        continue
                    rating_history.append({
                        'model': row['model'],
                        'elo_rating': row['elo_rating']-point_prev[point_prev['model']==row['model']]['elo_rating'].values[0],
                        'num_battle': battle_num,
                    })
            rating_history_pd = pd.DataFrame.from_dict(rating_history)
            
            # plotting
            rating_history_fig = px.line(rating_history_pd, x="num_battle", y="elo_rating", color="model", markers=True)
            rating_history_fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))

            gr.Plot(rating_history_fig) 
            
        def vis_rating_delta_history2(history: EloRatingHistory):
            rating_history = []
            # battle_outcomes = BattleOutcomes.read_csv(Path(result_dir) / 'battled_pairs.csv', nrows=battle_num)
            # battle_outcomes_prev = BattleOutcomes.read_csv(Path(result_dir) / 'battled_pairs.csv', nrows=history.recorded_battle_num[idx-1])
            
            battled_pairs = BattleOutcomes.read_csv(Path(result_dir) / 'battled_pairs.csv').battled_pairs_in_order
            for idx, battle_num in tqdm(enumerate(history.recorded_battle_num), desc='calculate rating delta'):
                if idx == 0:
                    continue
                point_cur = history.get_point(battle_num)
                point_prev = history.get_point(history.recorded_battle_num[idx-1])
   
                for _, row in point_cur.iterrows():
                    if row['model'] not in point_prev['model'].values:
                        continue
                    battled_pairs_cur = battled_pairs[:battle_num]
                    battled_pairs_prev = battled_pairs[:history.recorded_battle_num[idx-1]]
                    n_battles_model = len([x for x in battled_pairs_cur if x.model_a == row['model'] or x.model_b == row['model']])
                    n_prev_battles_model = len([x for x in battled_pairs_prev if x.model_a == row['model'] or x.model_b == row['model']])
                    if n_battles_model - n_prev_battles_model == 0:
                        continue
                    rating_history.append({
                        'model': row['model'],
                        'elo_rating': row['elo_rating']-point_prev[point_prev['model']==row['model']]['elo_rating'].values[0],
                        'num_battle': n_battles_model,
                    })
            rating_history_pd = pd.DataFrame.from_dict(rating_history)
            
            # plotting
            rating_history_fig = px.line(rating_history_pd, x="num_battle", y="elo_rating", color="model", markers=True)
            rating_history_fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))

            gr.Plot(rating_history_fig) 
            
        history = EloRatingHistory.gen_history(result_dir, use_bootstrap=
                                               False, nrows=FIRST_N_BATTLES)
        elo_rating_history_df = history.to_df()


        # Show the bootstrap result first
        if USE_BOOTSTRAP_ON_HISTORY:
            with gr.Tab("bootstrap=100"):
                history_bootstrap = EloRatingHistory.gen_history(result_dir, use_bootstrap=USE_BOOTSTRAP_ON_HISTORY, nrows=FIRST_N_BATTLES)
                elo_rating_history_bootstrap_df = history_bootstrap.to_df()
                gr.Markdown(f"Rank History")
                ranking_history_bootstrap_fig = plot_ranking_history(elo_rating_history_bootstrap_df)
                gr.Plot(ranking_history_bootstrap_fig)

                gr.Markdown(f'Elo Rating History')
                vis_rating_history(history_bootstrap)
                
                gr.Markdown(f'Elo Rating Delta History')
                vis_rating_delta_history(history_bootstrap)
                vis_rating_delta_history2(history_bootstrap)
                
                gr.Markdown("Rank Consistency History")
                rank_consistency_bootstrap_history_pd = calculate_rank_consistency_metrics(history_bootstrap)
                # gr.Dataframe(rank_consistency_bootstrap_history_pd, wrap=True)
                bootstrap_rank_consistency_history_fig = plot_rank_consistency(rank_consistency_bootstrap_history_pd)
                gr.Plot(bootstrap_rank_consistency_history_fig)
                
                # gr.Markdown("WinRate MAE History")
                # winrate_bootstrap_history_pd = calculate_winrate_metrics(history_bootstrap)
                # # gr.DataFrame(winrate_history_pd, wrap=True)
                # gr.Plot(plot_winrate_mae(winrate_bootstrap_history_pd))
        
        with gr.Tab("without bootstrap"):
            gr.Markdown(f"Rank History")
            ranking_history_fig = plot_ranking_history(elo_rating_history_df)
            gr.Plot(ranking_history_fig)        
            
            gr.Markdown(f'Elo rating history)')
            vis_rating_history(history)
                            
            gr.Markdown(f'Elo Rating Delta History')
            vis_rating_delta_history(history)
            vis_rating_delta_history2(history)
            
            gr.Markdown("Rank Consistency History")
            rank_consistency_history_pd = calculate_rank_consistency_metrics(history)
            rank_consistency_history_fig = plot_rank_consistency(rank_consistency_history_pd)
            gr.Plot(rank_consistency_history_fig)
            # gr.Dataframe(rank_consistency_history_pd, wrap=True)
            
            # gr.Markdown("WinRate MAE History")
            # winrate_history_pd = calculate_winrate_metrics(history)
            # # gr.DataFrame(winrate_history_pd, wrap=True)
            # gr.Plot(plot_winrate_mae(winrate_history_pd))
        
        # gr.Markdown("Elo rating history board (without bootstrap)")
        # gr.Dataframe(elo_rating_history_df, wrap=True)

def question_tie_review():
    with gr.Tab("Question Review"):
        # get question with winner data
        question_columns_to_inclusive = ['question', 'model_a', 'model_b', 'winner']
        question_data = records_df[question_columns_to_inclusive]
        # remove nan
        question_data_valid = question_data[(question_data['winner'].isna()==False) & (question_data['winner']!='invalid')]
        
        def get_tie_count_and_percentage(df, winner_name='tie'):
            # Group by 'question' and count 'tie' occurrences
            tie_count = df[df['winner'] == winner_name].groupby('question').size()
            
            tie_questions = tie_count.reset_index()['question']

            # Group by 'question' and count total occurrences
            total_count = df[df['question'].isin(tie_questions)].groupby('question').size()

            # Calculate percentage
            tie_percentage = np.round((tie_count / total_count) * 100, decimals=0)

            # Combine the counts and percentages into a single DataFrame
            result = pd.DataFrame({'question': tie_questions.tolist(), 'tie_count': tie_count, 'tie_percentage': tie_percentage})
            
            result.sort_values(by=['tie_percentage'], ascending=False, inplace=True)
            
            # ignore the rows of column 'tie_count' is null or 0
            result = result[(result['tie_percentage'] > 50) & (result['tie_count']!=0)]
            
            # add index column
            result.reset_index(drop=True, inplace=True)
            
            return result
        
        gr.Markdown('Question Tie Review: Question too easy?')
        tie_question_df = get_tie_count_and_percentage(question_data_valid, 'tie')
        gr.DataFrame(tie_question_df, wrap=True)
        
        gr.Markdown('Question Tie(all bad) Review: Questio too hard?')
        tie_allbad_question_df = get_tie_count_and_percentage(question_data_valid, 'tie(all bad)')
        gr.DataFrame(tie_allbad_question_df, wrap=True)
        
        all_question_count = len(question_data_valid['question'].unique())
        tie_count = len(tie_question_df)
        tie_all_bad_count = len(tie_allbad_question_df)
        normal_question_count = all_question_count - tie_count - tie_all_bad_count
        
        fig = px.pie(names=['other','tie', 'tie(allbad)'], values=[normal_question_count, tie_count, tie_all_bad_count], title='Question Tie Percentage > 50%')
        gr.Plot(fig)
        
if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.Markdown('Result Report')
        # shows the most important information first.
        elo_leaderboard()
        elo_history()
        
        # details
        # battle_outcomes()
        # ab_bias()
        # question_tie_review()
        demo.launch()