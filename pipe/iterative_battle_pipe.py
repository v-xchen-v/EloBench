"""Ramdomly selected questions from given question pool for models until each model have no tied battle count > requirement"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
from pipe.dumpy_pipe import DumpyPipeline
from pipe.battle_pipe import BattlePipeline
import random
from collections import Counter
from dataclasses import asdict
import pandas as pd
from datamodel import BattledPairs, PairwiseBattleArrangement, PairToBattle, ArrangementStrategy
import numpy as np
from pathlib import Path
from logger import logger, iterate_to_no_tie_logger

# def select_by_inverted_frequency(items, K=1):
#     # Count the frequency of each item
#     freqs = Counter(items)

#     # Invert the frequencies
#     inverted_freqs = {item: 1.0 / freq for item, freq in freqs.items()}

#     # Normalize to get probabilities
#     total = sum(inverted_freqs.values())
#     probabilities = {item: freq / total for item, freq in inverted_freqs.items()}

#     # Randomly select an item based on these probabilities
#     items, probs = zip(*probabilities.items())
#     return random.choices(items, weights=probs, k=K)[0]

# class IterativeBattlePipeline(BattlePipeline):
#     def __init__(self, tempcache_dir: str, save_dir: str, start: int = 0, end: int = None, no_cache: bool = False) -> None:
#         super().__init__(tempcache_dir, save_dir, start, end, no_cache)
#         # [model_a][model_b] = no_tie_battle_count
#         self.no_tie_battle_count = defaultdict(defaultdict(0))
    
#     def _validate_battles(self):
#         """Validate battle enough"""
#         """Battle until each combination got no-tie"""
        
#         self.battled_pairs

basic_pipe = BattlePipeline(tempcache_dir=r'data/quora_100', save_dir='results/quora_100_test5_iterative_80_no_tie')
def do_initial_battle(basic_pipe):

    # Register questions to battle pipeline
    questions = pd.read_csv(Path('data')/'quora_100'/'questions.csv')['question'].tolist()
    basic_pipe.register_questions(questions)


    # Register models to battle pipeline
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
    basic_pipe.register_models(models)

    # Arrange initial battle
    basic_pipe.arrange_battles(ArrangementStrategy.Reload_Existing_Arrangement)

    # Prepare for battle
    basic_pipe.gen_model_answers()

    # Judger works
    basic_pipe.battle()

    # Gen elo
    basic_pipe.gen_elo()
    
    
do_initial_battle(basic_pipe)

# all, no_tie = BattledPairs.read_csv(r'results/quora_100_test5_iterative_80_no_tie/battled_pairs.csv').frequency(despite_ab_order=False)
all, no_tie = basic_pipe.battled_pairs.frequency(despite_ab_order=False)
arrange = basic_pipe.battle_arrangements

print(pd.DataFrame.from_dict(all, orient='index'))
print(pd.DataFrame.from_dict(no_tie, orient='index'))
    
def select_pairs_with_lower_frequency(df_frequency_with_aborder, num_need_add, no_tie_target):
    # Replace NaN values with zero (or a small value) before inversion
    # This gives NaN cells a zero probability of being selected
    masked_df = df_frequency_with_aborder.mask(df_frequency_with_aborder.isna() | ((df_frequency_with_aborder+df_frequency_with_aborder.T) >= no_tie_target))
    iterate_to_no_tie_logger.debug(masked_df.reindex(columns=masked_df.index)
)
    
    # Invert the values(assuming all values are positive)
    # inverted_values = 1 / masked_df
    inverted_values = no_tie_target - masked_df
    
    # Flatten the DataFrame and normalize the inverted values
    flatten = inverted_values.to_numpy().flatten()
    flatten_no_nan = flatten[~np.isnan(flatten)]
    
    # Normalize the inverted values to form a probability distribution
    probabilities = flatten_no_nan / flatten_no_nan.sum()
    
    new_pairs = []
    
    # if random_select:
    # Randomly select an index based on these probabilities
    if num_need_add is None:
        if len(flatten_no_nan) == 0:
            num_need_add = 0
        else:
            num_pair_despite_aborder = (len(flatten_no_nan)/2)
            num_need_add = int(no_tie_target* num_pair_despite_aborder - masked_df.sum().sum())
            iterate_to_no_tie_logger.debug(f'remaining {int(no_tie_target* num_pair_despite_aborder - masked_df.sum().sum())} need auto adding ')
    if num_need_add == 0:
        return new_pairs
    
    selected_indexs = np.random.choice(len(probabilities), p=probabilities, replace=True, size=num_need_add)

    # Convert the 1D index back to 2D row and column indices
    non_nan_indices = list(zip(*np.where(~masked_df.isna())))
    
    model_as = []
    model_bs = []
    for selected_index in selected_indexs:
        row, col = non_nan_indices[selected_index] 
        model_a_name = df_frequency_with_aborder.columns[col]
        model_b_name = df_frequency_with_aborder.index[row]
        model_as.append(model_a_name)
        model_bs.append(model_b_name)
    
        #     # Randomly decide whether to swap
        #     if random_select.choice([True, False]):
        #         model_a_name, model_b_name = model_b_name, model_a_name
    
    for model_a_name, model_b_name in zip(model_as, model_bs):
        qs = arrange.get_questions_to_arrange(model_a=model_a_name, model_b=model_b_name)
        if len(qs)>0:
            new_pairs.append(PairToBattle(random.choices(qs)[0], model_a_name, model_b_name))
    num_try_add = len(new_pairs)
    iterate_to_no_tie_logger.debug(f'trying add {len(num_try_add)} more.')        
    
    return new_pairs, num_need_add, num_try_add    
    # # return row, col, df.iloc[row, col]
    #     print(row)
    #     print(col)
    #     print(df.iloc[row, col])
        
    
def validate_enough(df, N):
    # Replace NaN values with zero (or a small value) before inversion
    # This gives NaN cells a zero probability of being selected
    masked_df = df.mask(df.isna())
    
    all_greater_than_N = (masked_df >= N).all().all()
    return all_greater_than_N
# def select_cell_by_reaching_N(df, N=80):
#     masked_df = df.mask((df>=N) | (df.isna()))
    
#     N-masked_df
def get_new_iteration_battles(no_tie_with_ab_order, NUM_NEW_BATTLES_PER_ITER, no_tie_target):
    """arrange more battles on the no-tie battle count across multiple model pairs"""

    # no_more_iteration = False
    new_pairs = select_pairs_with_lower_frequency(pd.DataFrame.from_dict(no_tie_with_ab_order, orient='index'), num_need_add=NUM_NEW_BATTLES_PER_ITER,no_tie_target=no_tie_target)
    
    return new_pairs
    
def iterative_to_n_no_tie(N, NUM_NEW_BATTLES_PER_ITER, stop_num_tryadd=1):
    # arrange num_new_battles more battles, unless no more can arrange or reach setting
    retry_counter=0
    while True:
        _, no_tie_with_aborder = basic_pipe.battled_pairs.frequency(despite_ab_order=False)
        # BattledPairs.preety_print_frequency(no_tie_with_aborder)

        
        new_pairs, num_needadd_battles, num_tryadd_battles = get_new_iteration_battles(no_tie_with_aborder, NUM_NEW_BATTLES_PER_ITER=NUM_NEW_BATTLES_PER_ITER, no_tie_target=N)
        
        if num_needadd_battles == 0:
            return True
        
        if len(new_pairs) > 0:
            num_added_battles = basic_pipe.battle_arrangements.more_battles(new_pairs)
            basic_pipe.battle_arrangements.to_csv(Path(basic_pipe.save_dir)/'battle_arrangement.csv')
            
            iterate_to_no_tie_logger.debug(f'need add {num_needadd_battles} battls\ntry add {num_tryadd_battles} battles\nacutal add {num_added_battles} battles')
            
            if num_added_battles > 0:
                basic_pipe.gen_model_answers()
                basic_pipe.battle()
                basic_pipe.gen_elo()
                iterate_to_no_tie_logger.debug(basic_pipe.elo_ratings)
                
        

        # _, new_no_tie =basic_pipe.battled_pairs.frequency(despite_ab_order=True)
        # new_no_tie_df = pd.DataFrame.from_dict(new_no_tie, orient='index')
        
        
        # iterate_to_no_tie_logger.debug(new_no_tie_df)
        # is_enough = validate_enough(new_no_tie_df, N)
        # if is_enough:
        #     break
                
        else:
            # can not find question, model_a, model_b pairs can add because no more question unused.
            iterate_to_no_tie_logger.debug(f'Retrying {retry_counter+1}th...')
            retry_counter+=1
            
        if stop_num_tryadd is not None and num_tryadd_battles <= stop_num_tryadd:
                break
        if retry_counter >= 3:
            break
    
iterative_to_n_no_tie(N=80, NUM_NEW_BATTLES_PER_ITER=None)
    
