"""Used to view the questions in the dataset. If question source category is available, it will also show the distribution of the source categories."""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import pandas as pd
from pathlib import Path
from datamodel import PairwiseBattleArrangement
import plotly.express as px
import io
from matplotlib import pyplot as plt
import cv2
import numpy as np
from collections import defaultdict

def dataset_tabs(dataset_dir: Path):
    with gr.Tab('Dataset') as dataset_tab:
        with gr.Tab('Question') as dataset_question_tab:
            df = pd.read_csv(dataset_dir/'questions.csv')
            
            # Create a new column for the index
            df['index'] = range(len(df))

            # If you want 'Index' to be the first column
            df = df.set_index('index').reset_index()
            
            gr.Dataframe(df, wrap=True)
            
            if 'source' in df.columns:
                value_counts = df['source'].value_counts()
                # Create a pie chart
                fig = plt.figure(figsize=(8, 8))  # Adjust the size as needed
                plt.pie(value_counts, labels=value_counts.index, autopct=lambda p: '{:.1f}% ({:,.0f})'.format(p, p * sum(value_counts) / 100), startangle=140)
                plt.title('Source Distribution')
                gr.Plot(fig)
                
        with gr.Tab('Issue') as datset_question_issue_tab:
            df = pd.read_csv(dataset_dir/'questions.csv')
            
            # Create a new column for the index
            df['index'] = range(len(df))

            # If you want 'Index' to be the first column
            df = df.set_index('index').reset_index()
            
            # # Function to apply text color
            # # Function to apply the style
            # def highlight_nan(data, color='red'):
            #     return ['background-color: {}'.format(color) if pd.isna(v) else '' for v in data]

            # # highlight nan question rows
            # gr.Dataframe(df.style.apply(highlight_nan, subset=['question']), wrap=True)
            
            gr.Dataframe(df[df['question'].isna()])

if __name__ == '__main__':
    with gr.Blocks() as demo:
        dataset_dir = Path('data/google_quora_alpaca_10630')
        dataset_tabs(dataset_dir)
            
    demo.launch()