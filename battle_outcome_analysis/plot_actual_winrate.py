from pathlib import Path
import pandas as pd
import plotly.express as px
import os

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

save_dir = r'battle_outcome_analysis/output/plot/actual_winrate'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for idx, model_a in enumerate(models):
    for model_b in models[idx+1:]:
        df = pd.read_csv(Path(r'battle_outcome_analysis/output/data/actual_winrate')/f'actual_winrate_{model_a.replace("/", ".")}_{model_b.replace("/", ".")}.csv')
        avg_winrate = df['awinb_actual_winrate'].mean()
        fig = px.line(df, x='num_notie_battle_on_ab', y='awinb_actual_winrate', title=f'{model_a} vs {model_b} actual winrate')
        # fig.show()
        # set y range to 0-1
        fig.update_yaxes(range=[avg_winrate-0.25, avg_winrate+0.25])
        fig.write_image(Path(save_dir)/f'actual_winrate_trend_{model_a.replace("/", ".")}_{model_b.replace("/", ".")}.png')