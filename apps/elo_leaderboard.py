import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
from data import get_arena_battles_20230717_data, ARENA_K
from elo_rating.rating_helper import get_elo_results_from_battles_data


with gr.Blocks() as demo:
    gr.Markdown('🏆Elo Bench Leadboard')
    gr.Markdown('AI Dueling Arena with GPT-4 Adjudication')
    
    # dummy data from arena
    # TODO: add real eval result here
    arena_battles_data = get_arena_battles_20230717_data()
    dummy_data = get_elo_results_from_battles_data(arena_battles_data, K=ARENA_K)
    
    # Calculate an approximate height for the DataFrame output
    # You might need to adjust the multiplier based on your specific row height
    height_per_row = 40  # This is an approximate pixel height per row
    total_height = height_per_row * len(dummy_data) + 60  # Add some extra space for headers and padding


    gr.DataFrame(dummy_data, height=total_height)
    
demo.launch()
    