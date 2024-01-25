"""Read the old arrangement, random select 1% used question and put the at end as the new arrangement."""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

original_battle_outcome_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset'
NUM_SELECT = 1 #int(len(used_models) * (percentage / 100.0))

BEFORE_REGISTER_ONLY = True
battle_outcome_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_1_before_register'

all_old_battled_pairs = pd.read_csv(Path(original_battle_outcome_dir)/r'battled_pairs.csv')
valid_winner = ['model_a', 'model_b', 'tie', 'tie(all bad)']
valid_old_battled_pairs = all_old_battled_pairs[all_old_battled_pairs['winner'].isin(valid_winner)]
print(f'remove invalid battled pairs: {len(all_old_battled_pairs[~all_old_battled_pairs["winner"].isin(valid_winner)])}')
used_models = pd.read_csv(Path(battle_outcome_dir)/r'models.csv')['model'].tolist()
print(len(used_models))

np.random.seed(42)
# percentage = 10.0
print(NUM_SELECT)
model_selected = np.random.choice(used_models, size=NUM_SELECT, replace=False) 
pd.DataFrame(model_selected, columns=['model']).to_csv(Path(battle_outcome_dir)/'later_register_models.csv')

# replace = False means no duplicate
remain_models = list(set(used_models) - set(model_selected))

# to put the rows with selected questions at the end
model_selected_set = set(model_selected)

mask = valid_old_battled_pairs['model_a'].isin(model_selected_set) | valid_old_battled_pairs['model_b'].isin(model_selected_set)

# Split the dataframe to two parts
initial_battled_pairs = valid_old_battled_pairs[~mask]
later_registered_battled_pairs = valid_old_battled_pairs[mask]

# Concate them in the order you want which in this case is, not in set first and then in set
expr_battled_pairs = pd.concat([initial_battled_pairs, later_registered_battled_pairs], ignore_index=True)
expr_battled_pairs.to_csv(Path(battle_outcome_dir)/'battled_pair.csv')

# shuffle the arrangement by splits
save_dir = Path(battle_outcome_dir)/'.analysis/.bootstrap'
save_path_pattern = 'battled_pairs_num_of_bootstrap.csv'
num_of_bootstrap = 100
np.random.seed(42)
bootstraped_battlecomes_dfs = []
for _ in tqdm(range(num_of_bootstrap), desc="bootstrap"):
    # performing a random shuffle of the entire DataFrame
    bootstraped_df_not_in_set = initial_battled_pairs.sample(frac=1.0, replace=False)
    if not BEFORE_REGISTER_ONLY:
        bootstraped_df_in_set = later_registered_battled_pairs.sample(frac=1.0, replace=False)
        bootstraped_battlecomes_df = pd.concat([bootstraped_df_not_in_set, bootstraped_df_in_set], ignore_index=True)
        bootstraped_battlecomes_dfs.append(bootstraped_battlecomes_df)
    else:
        bootstraped_battlecomes_dfs.append(bootstraped_df_not_in_set)
    
if not os.path.exists(Path(save_dir)):
    os.makedirs(Path(save_dir))
        
for i, battle_outcomes_df in tqdm(enumerate(bootstraped_battlecomes_dfs), desc='saving bootstrap battled pairs...'):
    save_path = Path(save_dir)/save_path_pattern.replace('num_of_bootstrap', str(i+1).zfill(5))
    battle_outcomes_df.to_csv(save_path)

