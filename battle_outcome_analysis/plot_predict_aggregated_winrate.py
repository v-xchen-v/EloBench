import pandas as pd
import plotly.express as px
import os
from pathlib import Path
from collections import defaultdict

models = [ \
    'gpt-4-turbo',
    'gpt-35-turbo',
    'lmsys/vicuna-7b-v1.5',
    'lmsys/vicuna-13b-v1.5',
    'lmsys/vicuna-33b-v1.3',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-2-70b-chat-hf',
    'chavinlo/alpaca-native',
    'chavinlo/alpaca-13b',
    'mosaicml/mpt-7b-chat',
    'mosaicml/mpt-30b-chat',
    'WizardLM/WizardLM-7B-V1.0',
    'WizardLM/WizardLM-13B-V1.2',
    # 'WizardLM/WizardLM-70B-V1.0',
    'Xwin-LM/Xwin-LM-7B-V0.1',
    'Xwin-LM/Xwin-LM-13B-V0.1',
    'tiiuae/falcon-7b-instruct',
    'tiiuae/falcon-40b-instruct',
    'HuggingFaceH4/zephyr-7b-beta',
    'huggyllama/llama-7b',
    'huggyllama/llama-13b',
    'huggyllama/llama-30b',
    'gemini',
]
models = sorted(models)

battle_outcome_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_1'
save_dir = Path(battle_outcome_dir)/'output/plot/predict_winrate'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
predict_winrate_pd = pd.read_csv(Path(battle_outcome_dir)/'output/data/predict_winrate.csv')

ab_winrate_delta = defaultdict(lambda: 0)
counter = 0
for idx, model_a in enumerate(models):
    for model_b in models[idx+1:]:
        counter+=1
        inclusive_cols = [model_b, 'num_battle', 'model_b'] # acutal is model_b column records modela
        winb_winrate = predict_winrate_pd[inclusive_cols]
        ab_winrate = winb_winrate[winb_winrate['model_b']==model_a]
        
        actual_winrate_pd = pd.read_csv(Path(battle_outcome_dir)/rf'output/data/actual_winrate/actual_winrate_{model_a.replace("/", ".")}_{model_b.replace("/", ".")}.csv')
        final_battle_num = actual_winrate_pd['num_battle'].max()
        awinb_actual_winrate = actual_winrate_pd[actual_winrate_pd['num_battle']==final_battle_num]['awinb_actual_winrate'].values[0]
        
        ab_winrate = ab_winrate.drop('model_b', axis=1)
        # Select all columns except 'num_battle'
        columns_to_modify = ab_winrate.columns.difference(['num_battle'])
        ab_winrate[columns_to_modify] = ab_winrate[columns_to_modify].sub(awinb_actual_winrate).abs()
        
        for idx, row in ab_winrate.iterrows():
            ab_winrate_delta[int(row['num_battle'])] += row[model_b]
        

        # ab_winrate_with_delta.append({
            
        # })
        pass
    
for num_battle in ab_winrate_delta.keys():
    ab_winrate_delta[num_battle] /= counter
    
delta_dict = []
for num_battle, delta in ab_winrate_delta.items():
    delta_dict.append(
        {
            'num_battle': num_battle,
            'abs_delta_winrate': delta,
        }
    )
ab_winrate_with_delta_df = pd.DataFrame.from_dict(delta_dict)
ab_winrate_with_delta_df.to_csv(Path(battle_outcome_dir)/'output/data/predict_vs_actual_winrate_delta.csv')
        
fig = px.line(ab_winrate_with_delta_df, x='num_battle', y='abs_delta_winrate', title=f'predict winrate delta')
# fig.show()
# plot the last point with text
fig.add_annotation(x=ab_winrate_with_delta_df['num_battle'].max(), y=ab_winrate_with_delta_df[ab_winrate_with_delta_df['num_battle']==ab_winrate_with_delta_df['num_battle'].max()]['abs_delta_winrate'].values[0],
            text=f'{ab_winrate_with_delta_df["abs_delta_winrate"].max()}',
            showarrow=True,
            arrowhead=1)
fig.write_image(Path(save_dir)/f'predict_vs_actual_winrate_delta_trend.png')
    
