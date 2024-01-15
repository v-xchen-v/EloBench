
"""Select 70% question as the initial battle set, and then select 30% of the remaining as the next battle set."""
"""Set 10 as initital notie target, and then increase the target by 10 each time until the target is reached."""""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pipe.iterative_battle_pipe import IterativeBattlePipeline
from pathlib import Path
import pandas as pd
import random
from datamodel import ArrangementStrategy

if __name__ == '__main__':
    dataset_dir = Path('data')/'google_quora_alpaca_sharegpt_chatlm_clean_20772'
    iterative_battle_pipe = IterativeBattlePipeline(tempcache_dir=r'tempcache/google_quora_alpaca_sharegpt_chatlm_clean_20772', save_dir='results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset', no_cache=False, target_n_notie=5)
    
    # Register questions to battle pipeline
    all_questions = pd.read_csv(dataset_dir/'questions.csv')['question'].tolist()
    iterative_battle_pipe.register_questions(all_questions)
    # iterative_battle_pipe.register_questions(reload=True)
    
    # Regiester the initial models, register the top models on alpaca eval leaderboard later
    # Register models to battle pipeline
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
    
    iterative_battle_pipe.register_models(models)
    # iterative_battle_pipe.register_models(reload=True)


    iterative_battle_pipe.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_CombPair, num_of_battle=1)
    # iterative_battle_pipe.arrange_battles(ArrangementStrategy.Reload_Existing_Arrangement)
    iterative_battle_pipe.gen_model_answers()
    iterative_battle_pipe.battle(saving_per=50)    
    iterative_battle_pipe.iterative_battle(saving_per=50) 
