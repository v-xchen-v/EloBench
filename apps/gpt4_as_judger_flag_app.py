import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import pandas as pd
import re
from pathlib import Path

battle_arrangement = pd.read_csv(Path('data')/'arena_data'/'chatbot_arena_conversations'/'battle_arrangement.csv')
question_and_answers = pd.read_csv(Path('data')/'arena_data'/'chatbot_arena_conversations'/'q_and_as.csv')

def pick_question(index=0):
    return battle_arrangement.iloc[index]['question']

def pick_model_a(index=0):
    return battle_arrangement.iloc[index]['model_a']

def pick_model_b(index=0):
    return battle_arrangement.iloc[index]['model_b']

def pick_answer(question: str, model_a_or_b: str):
    model_name = battle_arrangement[battle_arrangement['question'] == question].iloc[0][model_a_or_b]
    # model_name = ''
    # if model_a_or_b == 'model_a':
    #     model_name = pick_model_a()
    # elif model_a_or_b == 'model_b':
    #     model_name = pick_model_b()
    # else:
    #     raise Exception("Invalid")
    
    answer = question_and_answers[question_and_answers['question']==question].iloc[0][model_name]
    return answer

callback = gr.CSVLogger()

with gr.Blocks() as demo:
    question = gr.Text(label='question', interactive=False)
    with gr.Row():
        with gr.Column() as col1:
            model_a_name = gr.Textbox(label='model_a', interactive=False)
            model_a_answer = gr.Textbox(label='answer_a', interactive=False)
        with gr.Column() as col2:
            model_b_name = gr.Textbox(label='model_b', interactive=False)
            model_b_answer = gr.Textbox(label='answer_b', interactive=False)
    btn = gr.Button('Gen Q and As')
    
    gpt_4_reponse = gr.Textbox(label='gpt_4_response', interactive=False)
    gpt_4_score = gr.Text(label='gpt_4_score', interactive=False)
    gpt_4_winner = gr.Text(label='gpt_winner', interactive=False)
    gpt_4_button = gr.Button("GPT-4 Eval And Score")
    
    battle_index = gr.Textbox(label="battle_index", value="-1", interactive=True)
    flag_button = gr.Button("Flag")
    
    # msg = gr.Textbox()
    clear = gr.ClearButton([model_a_name, model_b_name, model_a_answer, model_b_answer, gpt_4_reponse, gpt_4_score])
    
    def response(battle_index):
        battle_index = int(battle_index)+1
        question = pick_question(battle_index)
        
        model_a_name = pick_model_a(battle_index)
        model_b_name = pick_model_b(battle_index)
        
        model_a_ans = pick_answer(question, model_a_or_b='model_a')
        model_b_ans = pick_answer(question, model_a_or_b='model_b')
        
        return battle_index, question, model_a_name, model_b_name, model_a_ans, model_b_ans
        
    # This needs to be called at some point prior to the first call to callback.flag()
    callback.setup(components=[battle_index, question, model_a_name, model_b_name, model_a_answer, model_b_answer, gpt_4_reponse, gpt_4_score, gpt_4_winner], flagging_dir="flagged_data_points")
    
    btn.click(response, inputs=[battle_index], outputs=[battle_index, question, model_a_name, model_b_name, model_a_answer, model_b_answer])
    
    gpt_4_button.click(gpt_4_eval_and_score, inputs=[question, model_a_answer, model_b_answer], outputs=[gpt_4_reponse, gpt_4_score, gpt_4_winner])
    
    flag_button.click(lambda *args: callback.flag(args), [battle_index, question, model_a_name, model_b_name, model_a_answer, model_b_answer, gpt_4_reponse, gpt_4_score, gpt_4_winner], None, preprocess=False)
    
demo.launch()