import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))))

from datamodel import PairwiseBattleArrangement, ArrangementStrategy
from battle_pipe.battle_pipe import BattlePipeline
import pandas as pd
from pathlib import Path

quora_test2_arrangement = PairwiseBattleArrangement.read_csv(r'results/quora_100_test2_shuffle_ab/battle_arrangement.csv')


battle_pipe = BattlePipeline(tempcache_dir=r'data/quora_100', save_dir=r'results/quora_100_test4')


questions = pd.read_csv(Path('data')/'quora_100'/'questions.csv')['question'].tolist()
battle_pipe.register_questions(questions)

# Register models to battle
models = [ \
    # 'gpt2',
    # 'huggyllama/llama-7b', 
    # 'huggyllama/llama-13b',
    # 'huggyllama/llama-30b',
    # 'huggyllama/llama-65b',
    # 'meta-llama/Llama-2-7b-hf',
    # 'meta-llama/Llama-2-13b-hf',
    # 'meta-llama/Llama-2-70b-hf',
    # 'lmsys/vicuna-7b-v1.5',
    # 'lmsys/vicuna-13b-v1.5',
    # 'lmsys/vicuna-33b-v1.3',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-2-70b-chat-hf',
    # 'chavinlo/alpaca-native',
    # 'chavinlo/alpaca-13b'
]
battle_pipe.register_models(models)
print(battle_pipe)


battle_pipe.arrange_battles(ArrangementStrategy.All_Combination)
print(battle_pipe.battle_arrangements.battles_in_order)

 # # Create a random boolean mask
# mask = np.random.rand(len(arrangement_df)) > 0.5

# # Shuffle using the mask
# temp = arrangement_df['model_a'][mask].copy()
# arrangement_df['model_a'][mask] = arrangement_df['model_b'][mask]
# arrangement_df['model_b'][mask] = temp

quora_test2_arrangement.more_battles(battle_pipe.battle_arrangements.battles_in_order)

quora_test2_arrangement.to_csv(r'results/quora_100_test4/battle_arrangement.csv')