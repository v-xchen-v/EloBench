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

with gr.Blocks() as demo:
    # data_dir = Path('data/quora_100')
    # with gr.Tab('Question') as questions_tab:
        
    #     # vis input
    #     # questions need view the question and count the questions
    #     questions_df = pd.read_csv(data_dir/'questions.csv')
    #     questions_df.reset_index(inplace=True)
    #     gr.Dataframe(questions_df, wrap=True)
    data_dir = Path('results/quora_100_test4')
    
    with gr.Tab('Models') as models_tab:
        print(PairwiseBattleArrangement.read_csv(data_dir/'battle_arrangement.csv').models)
        model_df = pd.DataFrame(PairwiseBattleArrangement.read_csv(data_dir/'battle_arrangement.csv').models, columns=['model'])
        model_df.reset_index(inplace=True)
        gr.Dataframe(model_df)
                     
        
    with gr.Tab('Battle Arrangement') as arrange_tab:
        gr.Markdown('test')
        arrangement_df = pd.read_csv(data_dir/'battle_arrangement.csv')
        gr.Dataframe(arrangement_df, wrap=True)

        fig = px.bar(pd.concat([arrangement_df["model_a"], arrangement_df["model_b"]]).value_counts(),
             title="Battle Count for Each Model", text_auto=True)
        fig.update_layout(xaxis_title="model", yaxis_title="Battle Count", height=400, showlegend=False)
        
        gr.Plot(fig)
        
        fig2 = visualize_battle_count(arrangement_df, title="Battle Count of Each Combination of Models")
        gr.Plot(fig2)
demo.launch()