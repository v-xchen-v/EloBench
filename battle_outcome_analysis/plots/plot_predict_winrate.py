import pandas as pd
import plotly.express as px
import os
from pathlib import Path
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

battle_outcome_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset'
save_dir = Path(battle_outcome_dir)/'output/plot/predict_winrate'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
predict_winrate_pd = pd.read_csv(Path(battle_outcome_dir)/'output/data/predict_winrate.csv')

for idx, model_a in enumerate(models):
    for model_b in models[idx+1:]:
        inclusive_cols = [model_b, 'num_battle', 'model_b'] # acutal is model_b column records modela
        winb_winrate = predict_winrate_pd[inclusive_cols]
        ab_winrate = winb_winrate[winb_winrate['model_b']==model_a]
        
        actual_winrate_pd = pd.read_csv(Path(battle_outcome_dir)/rf'output/data/actual_winrate/actual_winrate_{model_a.replace("/", ".")}_{model_b.replace("/", ".")}.csv')
        final_battle_num = actual_winrate_pd['num_battle'].max()
        awinb_actual_winrate = actual_winrate_pd[actual_winrate_pd['num_battle']==final_battle_num]['awinb_actual_winrate'].values[0]
        fig = px.line(ab_winrate, x='num_battle', y=model_b, title=f'{model_a} vs {model_b} predict winrate')
        fig.add_hline(y=awinb_actual_winrate, line_dash="dash", annotation_text=f'actual winrate: {awinb_actual_winrate}', annotation_position="bottom right")
        # fig.show()
        # set y range to 0-1
        fig.update_yaxes(range=[0, 1])
        fig.write_image(Path(save_dir)/f'predict_winrate_trend_{model_a.replace("/", ".")}_{model_b.replace("/", ".")}.png')
        pass
        
