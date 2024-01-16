import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import pandas as pd
from pathlib import Path

# battle_arrangement = pd.read_csv(Path(r'results/google_quora_alpaca_10629_test3')/'battle_arrangement.csv')
# question_and_answers = pd.read_csv(Path(r'tempcache/google_quora_alpaca_10629')/'q_and_as.csv')
battle_records = pd.read_csv(Path(r'results/google_quora_alpaca_10629_test3')/'battle_records.csv', engine='python')
# inclusive_cols = ['question'm ]
target_models = ['WizardLM/WizardLM-7B-V1.0', 'chavinlo/alpaca-13b']
target_battle_records = battle_records[battle_records['model_a'].isin(target_models) & battle_records['model_b'].isin(target_models)]

pass
# battle_arrangement = pd.read_csv(Path('data')/'arena_data'/'chatbot_arena_conversations'/'battle_arrangement.csv')
# question_and_answers = pd.read_csv(Path('data')/'arena_data'/'chatbot_arena_conversations'/'q_and_as.csv')

# battle_records = pd.read_csv(r'results/quora_100_test4/battle_records.csv')


def pick_question(battle_records: pd.DataFrame, index=0):
    return battle_records.iloc[index]['question']

def pick_model_a(battle_records, index=0):
    return battle_records.iloc[index]['model_a']

def pick_model_b(battle_records, index=0):
    return battle_records.iloc[index]['model_b']

def pick_answer_a(battle_records, index=0):
    return battle_records.iloc[index]['answer_a']

def pick_answer_b(battle_records, index=0):
    return battle_records.iloc[index]['answer_b']

def pick_gpt_4_response(battle_records, index=0):
    return battle_records.iloc[index]['gpt_4_response']

def pick_gpt_4_score(battle_records, index=0):
    return battle_records.iloc[index]['gpt_4_score']

def pick_gpt_4_winner(battle_records, index=0):
    return battle_records.iloc[index]['winner']

# filtered_records = battle_records.copy()
# callback = gr.CSVLogger()


    
if __name__ == '__main__':
    with gr.Blocks() as demo:
        # with gr.Row():
        #     specific_model_a = gr.Textbox(label="filter by model_a", interactive=True)
        #     specific_model_b = gr.Textbox(label="filter by model_b", interactive=True)
        #     filter_btn = gr.Button('Filter')

        btn = gr.Button('Next')
        battle_index = gr.Textbox(label="battle_index", value="-1", interactive=True)
        question = gr.Text(label='question', interactive=False)
        with gr.Row():
            with gr.Column() as col1:
                model_a_name = gr.Textbox(label='model_a', interactive=False)
                model_a_answer = gr.Textbox(label='answer_a', interactive=False)
            with gr.Column() as col2:
                model_b_name = gr.Textbox(label='model_b', interactive=False)
                model_b_answer = gr.Textbox(label='answer_b', interactive=False)
        
        gpt_4_winner = gr.Text(label='gpt_winner', interactive=False)
        gpt_4_reponse = gr.Textbox(label='gpt_4_response', interactive=False)
        gpt_4_score = gr.Text(label='gpt_4_score', interactive=False)
        # gpt_4_button = gr.Button("GPT-4 Eval And Score")
        
        # flag_button = gr.Button("Flag")
        
        # msg = gr.Textbox()
        clear = gr.ClearButton([model_a_name, model_b_name, model_a_answer, model_b_answer, gpt_4_reponse, gpt_4_score])
        # filtered_records_df = gr.Dataframe(filtered_records)
        target_battle_records_df = gr.DataFrame(target_battle_records)
        
        # def filter_records_by_model_names(model_a, model_b, filtered_records):
        #     filtered_records = battle_records.copy()
        #     if model_a != '':
        #         filter1 = filtered_records['model_a'] == model_a
        #         filtered_records = filtered_records[filter1]
                
        #     if model_b != '':
        #         filter2 = filtered_records['model_b'] == model_b
        #         filtered_records = filtered_records[filter2]
                
        #     return filtered_records

        
        def response(records, battle_index):
            battle_index = int(battle_index)+1
            question = pick_question(records, battle_index)
            
            model_a_name = pick_model_a(records, battle_index)
            model_b_name = pick_model_b(records, battle_index)
            
            model_a_ans = pick_answer_a(records, battle_index)
            model_b_ans = pick_answer_b(records, battle_index)
            
            gpt_4_response = pick_gpt_4_response(records, battle_index) 
            gpt_4_score = pick_gpt_4_score(records, battle_index)
            gpt_4_winner = pick_gpt_4_winner(records, battle_index)
            return battle_index, question, model_a_name, model_b_name, model_a_ans, model_b_ans, gpt_4_response, gpt_4_score, gpt_4_winner
            

                
            
            
        # This needs to be called at some point prior to the first call to callback.flag()
        # callback.setup(components=[battle_index, question, model_a_name, model_b_name, model_a_answer, model_b_answer, gpt_4_reponse, gpt_4_score, gpt_4_winner], flagging_dir="flagged_data_points")
        
        btn.click(response, inputs=[target_battle_records_df, battle_index], outputs=[battle_index, question, model_a_name, model_b_name, model_a_answer, model_b_answer, gpt_4_reponse, gpt_4_score, gpt_4_winner])
        
        # filter_btn.click(filter_records_by_model_names, inputs=[specific_model_a, specific_model_b, filtered_records_df], outputs=[filtered_records_df])
        
        # gpt_4_button.click(gpt_4_eval_and_score, inputs=[question, model_a_answer, model_b_answer], outputs=[gpt_4_reponse, gpt_4_score, gpt_4_winner])
        
        # flag_button.click(lambda *args: callback.flag(args), [battle_index, question, model_a_name, model_b_name, model_a_answer, model_b_answer, gpt_4_reponse, gpt_4_score, gpt_4_winner], None, preprocess=False)
        demo.launch()