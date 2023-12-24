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

def visualize_battle_count(battles, title):
    print(battles)
    inclusive_cols = ['model_a', 'model_b']
    
    def count_strings(col):
    # Filter out non-string values and count occurrences
        return col.apply(lambda x: x if isinstance(x, str) else None).value_counts()

    # count_model_a = count_strings(battles['model_a'])
    # count_model_b = count_strings(battles['model_b'])
    # union_index = count_model_a.index.union(count_model_b.index).sort_values()
    # reindex_count_model_a = count_model_a.reindex(union_index).fillna(0)
    # reindex_count_model_b = count_model_b.reindex(union_index).fillna(0)

    # df = pd.DataFrame(index=reindex_count_model_a, columns=reindex_count_model_b)
    # for i in df.index:
    #     for j in df.columns:
    #         df.at[i, j] = i + j
    
    union_model_names = pd.concat([
        pd.Series(battles['model_a'].unique()), 
        pd.Series(battles['model_b'].unique())]).unique()
    battle_counts = {}
    for key in union_model_names:
        battle_counts[key] = {subkey: 0 for subkey in union_model_names}
    
    for model_a, model_b in zip(battles['model_a'], battles['model_b']):
        battle_counts[model_a][model_b]+=1
            
    battle_counts_pf = pd.DataFrame.from_dict(battle_counts)
    battle_counts_despite_ab_order = battle_counts_pf + battle_counts_pf.T
    
    # print(count_strings(battles['model_a']))
    # print(count_strings(battles['model_b']))
    # battles[inclusive_cols]
    # ptbl = pd.pivot_table(count_model_despite_ab_order, index=["model_a"], columns=["model_b"], aggfunc="size", fill_value=0)
    # battle_counts = ptbl + ptbl.T
    
    ordering = battle_counts_despite_ab_order.sum().sort_values(ascending=False).index
    
    # Reindexing rows and columns
    battle_counts_despite_ab_order = battle_counts_despite_ab_order.reindex(index=ordering, columns=ordering)
    
    
    # Create a mask for the lower triangle
    mask = np.tril(np.ones(battle_counts_despite_ab_order.shape)).astype(bool)
    
    # Apply the mask to the DataFrame
    battle_counts_despite_ab_order.where(mask, np.nan, inplace=True)

    # fig = px.imshow(battle_counts.loc[ordering, ordering],
    #                 title=title, text_auto=True, width=600)
    fig = px.imshow(battle_counts_despite_ab_order.loc[ordering, ordering],
                    title=title, text_auto=True, width=600)
    fig.update_layout(xaxis_title="Model 1",
                      yaxis_title="Model 2",
                      xaxis_side="top", height=600, width=600,
                      title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                      "Model 1: %{y}<br>Model 2: %{x}<br>Count: %{z}<extra></extra>")
    return fig

def battle_set_tabs(result_dir: Path, dataset_dir: Path):
    with gr.Tab('Battle Set') as battle_set_tab:
        with gr.Tab('Question') as battle_set_question_tab:
            result_question_df = pd.read_csv(result_dir/'questions.csv')
            
            gr.Dataframe(result_question_df, wrap=True)
            
            dataset_question_df = pd.read_csv(dataset_dir/'questions.csv')
            
            if 'source' in dataset_question_df.columns:
                def get_question_source(question):
                    if question is None:
                        raise Exception("Question is None, Check the battle set.")
                    # Check if the question exists in the dataset
                    if question in dataset_question_df['question'].values:
                                return dataset_question_df[dataset_question_df['question']==question]['source'].iloc[0]
                    else:
                        return None
                
                result_question_df['source'] = result_question_df['question'].apply(get_question_source)
                
                value_counts = result_question_df['source'].value_counts()
                # Create a pie chart
                fig = plt.figure(figsize=(8, 8))  # Adjust the size as needed
                plt.pie(value_counts, labels=value_counts.index, autopct=lambda p: '{:.1f}% ({:,.0f})'.format(p, p * sum(value_counts) / 100), startangle=140)
                plt.title('Source Distribution')
                gr.Plot(fig)
                
            arrange_df = pd.read_csv(result_dir/'battle_arrangement.csv')
            # for question, num in df['question'].value_counts().iteritems()
            inclusive_cols = ['question']
            question_frequency_df = arrange_df[inclusive_cols]
            
            # draw historgram to show the distribution of question frequency
            question_frequency = arrange_df['question'].value_counts()
            # Map the counts back to the original DataFrame
            question_frequency_df['question_frequency'] = question_frequency_df['question'].map(question_frequency)
            gr.Dataframe(question_frequency_df, wrap=True)
            
            frequencies = question_frequency_df['question_frequency'].to_list()
            frequencies = np.array(frequencies)
            # remove nan value
            frequencies = frequencies[~np.isnan(frequencies)]
            
            question_frequency_fig = plt.figure(figsize=(10, 6))
            plt.hist(frequencies, color='skyblue')  # Adjust the number of bins as needed
            

            plt.xlabel('Frequency')
            plt.ylabel('Number of Questions')
            plt.title('Distribution of Question Usage Frequency')
            gr.Plot(question_frequency_fig)
                
        with gr.Tab('Model') as battle_set_model_tab:
            battle_set_model_df = pd.read_csv(result_dir/'models.csv')
            gr.Dataframe(battle_set_model_df, wrap=True)
            
            arrange_df = pd.read_csv(result_dir/'battle_arrangement.csv')
            
            models = battle_set_model_df['model'].tolist()
            model_questionnum = defaultdict(int)
            for model in models:
                questions = arrange_df[(arrange_df['model_a']==model) | (arrange_df['model_b']==model)]['question'].unique()
                model_questionnum[model] = len(questions)
                
            model_questionnum_df = pd.DataFrame.from_dict(model_questionnum, orient='index', columns=['question_num'])
            
            model_question_num_fig = px.bar(model_questionnum_df, title="Question Count for Each Model", text_auto=True)
            model_question_num_fig.update_layout(xaxis_title="model", yaxis_title="Question Count", height=400, showlegend=False)
            
            gr.Plot(model_question_num_fig)
        
        arrangement_df = pd.read_csv(result_dir/'battle_arrangement.csv')
        with gr.Tab('Model/Pair To Battle') as battle_set_to_battle_tab:
            model_to_battle_fig = px.bar(pd.concat([arrangement_df["model_a"], arrangement_df["model_b"]]).value_counts(),
                title="Battle Count for Each Model", text_auto=True)
            model_to_battle_fig.update_layout(xaxis_title="model", yaxis_title="Battle Count", height=400, showlegend=False)
            
            gr.Plot(model_to_battle_fig)

            pair_to_battle_fig2 = visualize_battle_count(arrangement_df, title="Battle Count of Each Combination of Models")
            gr.Plot(pair_to_battle_fig2)

            gr.Dataframe(arrangement_df, wrap=True)
if __name__ == '__main__':
    with gr.Blocks() as demo:
        dataset_dir = Path('data/google_quora_alpaca_10629')
        result_dir = Path('results/google_quora_alpaca_10629_test2')
        battle_set_tabs(result_dir, dataset_dir)
            
    demo.launch()