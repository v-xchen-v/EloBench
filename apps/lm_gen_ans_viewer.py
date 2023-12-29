import gradio as gr
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

with gr.Blocks() as demo:
    def show_missing_data(df):
        # ans_null_matrix = df.applymap(lambda x: x == 'NULL')
        ans_null_matrix = df.isna()
        
        # Converting the boolean matrix to integers for visualization
        integer_matrix = ans_null_matrix.astype(int)

        # Creating a heatmap using plotly.express
        fig = px.imshow(integer_matrix, color_continuous_scale='Viridis', 
                        labels=dict(color="Boolean Value"), 
                        title="Heatmap of missing data in the answer dataframe")

        # Display the plot
        # fig.show()
        
        
        # plt.figure(figsize=(10,6))
        # sns.heatmap(ans_null_matrix, cbar=False, yticklabels=False, cmap='viridis')
        # plt.title('Heatmap of Missing Data')
        # plt.tight_layout()
        
        return fig
    
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
    # keep_default_na=False: This parameter tells Pandas not to automatically convert certain sets of strings (like the empty string '', '#N/A', '#N/A N/A', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null') to NaN. When set to False, it prevents the automatic conversion of these strings to NaN, treating them as regular strings instead.
    # na_values=['NaN']: This explicitly tells Pandas to treat only the string 'NaN' as a NaN value.
    # Putting it all together, the command reads the specified CSV file into a DataFrame (ans_df), treating only the string 'NaN' as a missing value and keeping all other potential NaN strings (like empty strings) as they are.
    ans_df = pd.read_csv(Path('tempcache/google_quora_alpaca_sharegpt_chat1m_22012')/'q_and_as.csv', keep_default_na=False, na_values=['NaN'])
    # gr.Dataframe(ans_df)
    gr.Markdown('## Missing Data')
    gr.Markdown('Also the progress of generate all questions and answers, if there are still some missing data in the answer dataframe. The missing data is shown in the heatmap below.')
    gr.Plot(show_missing_data(ans_df))
    gr.Plot(show_empty_ans_data(ans_df))
    # gr.Plot(average_ans_length(ans_df))
    # gr.Plot(show_missing_data3(ans_df))

if __name__ == '__main__':
    demo.launch()