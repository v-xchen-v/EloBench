import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from battle_pipe.all_from_question import AllFromQuestionsBattlePipeline
import pandas as pd
from pathlib import Path
from datamodel import ArrangementStrategy, PairwiseBattleArrangement

questions = pd.read_csv(Path('data')/'quora_100'/'questions.csv')['question'].tolist()

battle_pipe = AllFromQuestionsBattlePipeline(tempcache_dir='data/quora_100', save_dir=Path('results')/'quora_100_test4')
battle_pipe.register_questions(questions)

print(battle_pipe.question_collection)

# Register models to battle
models = [ \
    # 'gpt2',
    'huggyllama/llama-7b', 
    'huggyllama/llama-13b',
    'huggyllama/llama-30b',
    'huggyllama/llama-65b',
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-2-70b-hf',
    'lmsys/vicuna-7b-v1.5',
    'lmsys/vicuna-13b-v1.5',
    'lmsys/vicuna-33b-v1.3',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-2-70b-chat-hf',
    'chavinlo/alpaca-native',
    'chavinlo/alpaca-13b',
    'gpt-4-turbo',
    'gpt-35-turbo',
]
battle_pipe.register_models(models)
print(battle_pipe)

# Arrange battle rounds
battle_pipe.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_Model, num_of_pair=300*3)

# Generate answer
battle_pipe.gen_model_answers()
print(battle_pipe)    

battle_pipe.battle()

battle_pipe.gen_elo()
    
# initial_arrangement = PairwiseBattleArrangement.read_csv(r'results/quora_100_test4/battle_arrangement.csv')

# quora_test2_arrangement.more_battles(battle_pipe.battle_arrangements.battles_in_order)

# quora_test2_arrangement.to_csv(r'results/quora_100_test4/battle_arrangement.csv')

