import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
from data import get_arena_battles_20230717_data, ARENA_K
from elo_rating.rating_helper import get_elo_results_from_battles_data
import pandas as pd

def get_gpt4_judger_elo_on_arena(filepath=r'results\log_battle_arena_gpt4_as_judger.csv'):
    df = pd.read_csv(filepath)
    columns_to_inclusive = ['model_a', 'model_b', 'winner']
    data = df[columns_to_inclusive]
    
    # remove nan
    data = data[data['winner'].isna()==False]
    
    # new_column_names = {"gpt_winner": 'winner'}
    # data.rename(columns=new_column_names, inplace=True)
    elo_result = get_elo_results_from_battles_data(data, K=ARENA_K)
    return elo_result

with gr.Blocks() as demo:
    gr.Markdown('🏆Elo Bench Leadboard')
    gr.Markdown('AI Dueling Arena with GPT-4 Adjudication')
    
    result_data = None
    
    # # dummy data from arena
    # # TODO: add real eval result here
    # arena_battles_data = get_arena_battles_20230717_data()
    # dummy_data = get_elo_results_from_battles_data(arena_battles_data, K=ARENA_K)
    # result_data = dummy_data
    
    result_data = get_gpt4_judger_elo_on_arena()
    
    # Calculate an approximate height for the DataFrame output
    # You might need to adjust the multiplier based on your specific row height
    height_per_row = 40  # This is an approximate pixel height per row
    total_height = height_per_row * len(result_data) + 60  # Add some extra space for headers and padding


    gr.DataFrame(result_data, height=total_height)
    
demo.launch()
    