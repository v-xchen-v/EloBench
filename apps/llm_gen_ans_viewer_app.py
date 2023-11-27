import gradio as gr
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

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
    
    # def show_missing_data2(df):
    #     missing_values = df.isnull().sum()
    #     missing_values.plot(kind='bar')
    #     plt.title('Count of Missing Values per Column')
    #     plt.ylabel('Missing Value Count')
    #     plt.xlabel('Columns')
    #     return plt
    
    # def show_missing_data3(df):
    #     sns.pairplot(df)
    #     return plt
    
    
    ans_df = pd.read_csv(Path('data')/'quora_100'/'q_and_as.csv', keep_default_na=False)
    gr.Dataframe(ans_df, wrap=True)
    gr.Plot(show_missing_data(ans_df))
    gr.Plot(show_empty_ans_data(ans_df))
    # gr.Plot(show_missing_data2(ans_df))
    # gr.Plot(show_missing_data3(ans_df))

if __name__ == '__main__':
    demo.launch()