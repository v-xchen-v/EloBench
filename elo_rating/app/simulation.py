"""Now we have some code to represent LLM as players and to simulate a match between them.
- We'll provide slider to define initial ratings of players.
- We'll provide slider to define K-factor of players.
- We'll provide slider to define the amount of players.
"""

import os, sys
sys.path.append('/elo_bench')
import gradio as gr
import elo_rating.llm_player as llm_player
import elo_rating.rating_helper as rating_helper
import plotly.express as px
import numpy as np
from typing import List
import random
import pandas as pd
from elo_rating.rating_evaluator import compute_predict_winrate, compute_acutal_winrate
from datamodel import battle_outcome

def simulate_match(llm_players: List[llm_player.LLMPlayer]):
    """Simulate matchs between the players."""
    # 100 battles random pick 2 players and random pick 1 winner
    for i in range(100):
        players = random.sample(llm_players, 2)
        winner = random.choice(players)
        loser = players[0] if players[0] != winner else players[1]
        # print(f'before: {winner.rating}, {loser.rating}')
        winner.update_rating_by_winner(loser, as_winner=True)
        loser.update_rating_by_winner(winner, as_winner=False)
        # print(f'after: {winner.rating}, {loser.rating}')
    # print(np.sum([p.rating for p in llm_players]))

# Function to process the settings and return the results
def mock_battles_data(initial_rating, k_factor, n_players):
    # Initialize the LLM players
    llm_players = [llm_player.LLMPlayer(f'LLM_{i}', initial_rating, k_factor) for i in range(n_players)]
    
    # Simulate a match between the players
    simulate_match(llm_players)
    
    # Plot the rating of each player
    fig = px.bar(x=[p.id for p in llm_players], y=[p.rating for p in llm_players])
    
    # list rating and rank of each player
    rating_and_rank = rating_helper.get_players_rating_and_rank(llm_players)
    
    predict_winrate_fig = compute_and_plot_predict_winrate(rating_and_rank)
    
    return fig, rating_and_rank, predict_winrate_fig

def upload_battles_data(file_path):
    """upload battles data in single csv file to compute elo ratings"""
    if isinstance(file_path, str):
        print(file_path)
    else:
        file_path = file_path.name
        print(file_path)
    battled_pairs = pd.read_csv(file_path) 
    rating_and_rank = rating_helper.get_players_rating_and_rank_from_battles_data(battled_pairs,K=4)
    
    # Plot the rating of each player
    fig = px.bar(x=rating_and_rank['model'], y=rating_and_rank['elo_rating'])
    
    predict_winrate_fig = compute_and_plot_predict_winrate(rating_and_rank)
    
    return fig, rating_and_rank, predict_winrate_fig

    
def compute_and_plot_predict_winrate(rating_and_rank: pd.DataFrame):
    ordered_models = rating_and_rank['model']
    predict_winrate = compute_predict_winrate(rating_and_rank)
    predict_winrate_fig = plot_winrate_matrix(predict_winrate, title="Predicted Win Rate Using Elo Ratings for Model A in an A vs. B Battle", ordered_models=ordered_models)

    return predict_winrate_fig

def plot_winrate_matrix(winrate_matrix: pd.DataFrame, title: str, ordered_models: List[str]):
    winrate_fig = px.imshow(winrate_matrix.loc[ordered_models, ordered_models],
                    color_continuous_scale='RdBu', text_auto=".2f",
                    title=title)
    winrate_fig.update_layout(xaxis_title="Model B",
                    yaxis_title="Model A",
                    xaxis_side="top", height=600, width=600,
                    title_y=0.07, title_x=0.5)
    return winrate_fig

def switch_mock(is_mock):
    if is_mock:
        # make the upload button invisible and mock button visible
        return gr.Button.update(visible=False), gr.Button.update(visible=True)
    else:
        # make the mock button invisible and upload button visible
        return gr.Button.update(visible=True), gr.Button.update(visible=False)
    
with gr.Blocks(title='Elo-based rating system for LLM evaluation') as demo:
    with gr.Accordion('Ratting settings'):
        gr.Markdown('Define the initial ratings and K-factor for LLMs:')
        INITIAL_RATING = gr.Slider(label='Initial rating:', minimum=0, maximum=2000, step=1, value=1000)
        K_FACTOR = gr.Slider(label='K-factor:', minimum=0, maximum=100, step=1, value=4)
        N_PLAYERS = gr.Slider(label='Number of players:', minimum=2, maximum=100, step=1, value=4)
    
    is_mock = gr.Checkbox(label='Mock battles', value=True) # default as going to mock battles
    
    # Button to trigger the processing of slider values
    mock_battles_button = gr.Button(value='Mock battles', ) 
    
    # upload battles data in single csv file to compute elo ratings
    upload_button = gr.UploadButton(value='Upload Battles Data', file_types=['.csv'], file_count='single', visible=False)
    
    # # Output text where results will be displayed
    # output_text = gr.Textbox(label='Mock Settings')
    # Output plot where results will be displayed
    output_plot = gr.Plot(label='Rating plot')
    output_rating_and_rank = gr.DataFrame(label='Rating and rank of each player')
    output_plot_predict_winrate = gr.Plot(label='Predicted win rate')
    
    # When the button is clicked, call process_ratings function with the slider value
    mock_battles_button.click(mock_battles_data, inputs=[INITIAL_RATING, K_FACTOR, N_PLAYERS], outputs=[output_plot, output_rating_and_rank, output_plot_predict_winrate])
    
    upload_button.upload(upload_battles_data, inputs=upload_button, outputs=[output_plot, output_rating_and_rank, output_plot_predict_winrate])
    
    is_mock.change(switch_mock, inputs=[is_mock], outputs=[upload_button, mock_battles_button])
    
    demo.launch()