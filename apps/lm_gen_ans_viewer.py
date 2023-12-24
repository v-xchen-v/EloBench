import gradio as gr
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

with gr.Blocks() as demo:
    def show_missing_data(df):
        ans_null_matrix = df.applymap(lambda x: x == 'NULL')
        plt.figure(figsize=(10,6))
        sns.heatmap(ans_null_matrix, cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Heatmap of Missing Data')
        plt.tight_layout()
        
        return plt
    
    def show_empty_ans_data(df):
        ans_empty_matrix = df.applymap(lambda x: x == '')
        plt.figure(figsize=(10,6))
        sns.heatmap(ans_empty_matrix, cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Heatmap of Empty Answer Data')
        plt.tight_layout()
        
        return plt
    
    def average_ans_length(df):
        ans_null_matrix = df.applymap(lambda x: x != 'NULL')
        ans_length_df = df.applymap(lambda x: len(x))
        answer_length_fig = px.bar(ans_length_df, title="Answer Avg Length for Each Model", text_auto=True)
        answer_length_fig.update_layout(xaxis_title="model", yaxis_title="Answer Length", height=400, showlegend=False)
        plt.tight_layout()
        
        return plt
    
    ans_df = pd.read_csv(Path('tempcache/google_quora_alpaca_10629')/'q_and_as.csv', keep_default_na=False)
    gr.Dataframe(ans_df, wrap=True)
    gr.Plot(show_missing_data(ans_df))
    gr.Plot(show_empty_ans_data(ans_df))
    gr.Plot(average_ans_length(ans_df))
    # gr.Plot(show_missing_data3(ans_df))

if __name__ == '__main__':
    demo.launch()