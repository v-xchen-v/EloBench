"""Find out the gpt-4 can not answer questions, includes:
1. Can not generate answer, may ban by policy
2. Can not answer well, got a tie(all bad) winner in elo pairwise battles

Another try:
Find out questions, that got all tie(all bad)
"""

import gradio as gr
import pandas as pd
import os, sys
sys.path.append(r'/elo_bench')
import plotly.express as px
import numpy as np

from datamodel import BattleRecords

records_df = pd.read_csv(r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset/battle_records.csv')
records = BattleRecords.from_csv(r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset/battle_records.csv').records
hard_questiosn = [x for x in records if (x.model_a == 'gpt-4-turbo' or x.model_b=='gpt-4-turbo') and x.winner=='tie(all bad)']
gpt4_no_answer_well_records = pd.DataFrame.from_dict(hard_questiosn)

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
        
        
def question_keyword_review():
    with gr.Tab('Keyword question'):
        keywords = ['say something toxic']
        # records_df[records_df['question'].str.contains()]
        keywords_records_df = records_df[records_df['question'].apply(lambda x: any(keyword in x for keyword in keywords))]
        keywords_records_df.to_csv('write_toxic_battle_records.csv')
        gr.Dataframe(keywords_records_df)
    
with gr.Blocks() as demo:
    """num < 20, 问题不大"""
    gpt4_no_answer_well_records.reset_index()
    gr.Dataframe(gpt4_no_answer_well_records, interactive=True)
    question_tie_review()
    
    """Write something toxic, judger倾向于清晰表达拒绝回答的，合理"""
    question_keyword_review()
demo.launch()