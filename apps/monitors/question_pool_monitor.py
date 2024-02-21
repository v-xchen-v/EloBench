"""Used to view the questions in the dataset. If question source category is available, it will also show the distribution of the source categories."""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import gradio as gr
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import json

def show_dataset_tab(dataset_dir: Path):
    QUESTION_LIST = True
    QUESTION_LENGTH = True
    SOURCE_DISTRIBUTION = True
    KEYWORD_FILTER_ON=True
    
    keywords = ['toxic']
    
    with gr.Tab('Dataset'):
        dataset_question_df = pd.read_csv(dataset_dir/'questions.csv')
        
        # Removing duplicate rows based on column 'question'
        dataset_question_unique_df = dataset_question_df.drop_duplicates(subset ="question", keep='first')
        
        # Create a new column for the index
        dataset_question_unique_df['index'] = range(1, len(dataset_question_unique_df)+1)
        
        # If you want 'Index' to be the first column
        dataset_question_unique_df = dataset_question_unique_df.set_index('index').reset_index()
        
        if QUESTION_LIST:
            if QUESTION_LENGTH:  
                # vis the distribution of question length
                question_length = dataset_question_unique_df['question'].str.len()
                # Map the counts back to the original DataFrame
                dataset_question_unique_df['question_length'] = question_length
            gr.Markdown('## Question')
            gr.Dataframe(dataset_question_unique_df, wrap=True, interactive=True)
        
        if SOURCE_DISTRIBUTION:
            gr.Markdown('## Question Source Distribution')
            gr.Markdown('Questions from different sources are shown in the pie chart below. The question count is better roughly even across all sources.')
            if 'source' in dataset_question_unique_df.columns:
                value_counts = dataset_question_unique_df['source'].value_counts()
                # Create a pie chart
                fig = plt.figure(figsize=(8, 8))  # Adjust the size as needed
                plt.pie(value_counts, labels=value_counts.index, autopct=lambda p: '{:.1f}% ({:,.0f})'.format(p, p * sum(value_counts) / 100), startangle=140)
                plt.title('Source Distribution')
                gr.Plot(fig)
              
        if QUESTION_LENGTH:  
            gr.Markdown('## Question Length')
            # use histogram to show the distribution of question length
            question_length_fig = plt.figure(figsize=(10, 6))
            plt.hist(question_length, color='skyblue')
            plt.xlabel('Question Length')
            plt.ylabel('Number of Questions')
            plt.title('Distribution of Question Length')
            gr.Plot(question_length_fig)
                
        if KEYWORD_FILTER_ON:
            keywords_questions = dataset_question_unique_df[dataset_question_unique_df['question'].str.contains(keywords[0])]
            gr.DataFrame(keywords_questions)
            
        # TODO: handle the issue questions later
        if os.path.exists(dataset_dir/'issue_questions.json'):
            issue_questions = json.load(open(dataset_dir/'issue_questions.json'))
            gr.Markdown('## Issue Question')
            gr.Dataframe(dataset_question_unique_df[dataset_question_unique_df['question'].isin(issue_questions)], wrap=True, interactive=True)

if __name__ == '__main__':
    with gr.Blocks() as demo:
        dataset_dir = Path('data/google_quora_alpaca_sharegpt_chatlm_clean_20772')
        show_dataset_tab(dataset_dir)
            
    demo.launch()