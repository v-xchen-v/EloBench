"""Read the old arrangement, random select 1% used question and put the at end as the new arrangement."""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import glob

# All settings here
NUM_SELECT = 3 #int(len(used_models) * (percentage / 100.0))
battle_outcome_dir = f'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_{NUM_SELECT}_seed3'
np.random.seed(84)

# Load the bootstraped battled pairs of original battle set outcome dir
# Note: the original bootstraped battled pairs are already cleaned out of invalid winners
original_battle_outcome_dir = r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_add_mistral_notie200'
origianl_battled_paris_filelist = battled_pairs_csv_file_list = glob.glob(str(Path(original_battle_outcome_dir)/'.analysis/.bootstrap/battled_pairs_*.csv'))
origianl_battled_paris_filelist.sort()

save_dir = Path(battle_outcome_dir)/'.analysis/.bootstrap'
if not os.path.exists(Path(save_dir)):
    os.makedirs(Path(save_dir))

# Take the random n of the models as the later registered models
used_models = pd.read_csv(Path(battle_outcome_dir)/r'models.csv')['model'].tolist()
print(len(used_models))
# percentage = 10.0
print(NUM_SELECT)
model_selected = np.random.choice(used_models, size=NUM_SELECT, replace=False) 
pd.DataFrame(model_selected, columns=['model']).to_csv(Path(battle_outcome_dir)/'later_register_models.csv')

# Then we got the remain models and selected model
# replace = False means no duplicate
remain_models = list(set(used_models) - set(model_selected))
# to put the rows with selected questions at the end
model_selected_set = set(model_selected)

# We got the first part of the battled pairs which are not in the selected set
before_register_pairs_bootstrap_list = []
for idx_bootstrap in range(0, len(origianl_battled_paris_filelist)):
    cur_bootstrap_battled_pairs = pd.read_csv(origianl_battled_paris_filelist[idx_bootstrap])
    before_register_pairs = cur_bootstrap_battled_pairs[~(cur_bootstrap_battled_pairs['model_a'].isin(model_selected_set) | cur_bootstrap_battled_pairs['model_b'].isin(model_selected_set))]
    before_register_pairs_bootstrap_list.append(before_register_pairs)
    print(f'before register pairs: {len(before_register_pairs)}')
    
BEFORE_REGISTER_ONLY = False
if not BEFORE_REGISTER_ONLY:
# We got the second part of the battled pairs which are in the selected set, and bootstrap them separately by specific random seed
    first_bootstrap_battled_pairs = pd.read_csv(origianl_battled_paris_filelist[0])
    later_register_pairs = first_bootstrap_battled_pairs[first_bootstrap_battled_pairs['model_a'].isin(model_selected_set) | first_bootstrap_battled_pairs['model_b'].isin(model_selected_set)]

    later_register_pairs_bootstrap_list = []
    
    # iterate and shuffle the second part
    for idx_bootstrap in range(0, len(origianl_battled_paris_filelist)):
        bootstrap_later_register_pairs = later_register_pairs.sample(frac=1.0, replace=False)
        later_register_pairs_bootstrap_list.append(bootstrap_later_register_pairs)
        print(f'after register pairs: {len(bootstrap_later_register_pairs)}')
        pd.concat([before_register_pairs_bootstrap_list[idx_bootstrap], bootstrap_later_register_pairs], ignore_index=True).to_csv(Path(save_dir)/f'battled_pairs_{str(idx_bootstrap).zfill(5)}.csv')
else:
    for idx_bootstrap in range(0, len(origianl_battled_paris_filelist)):
        before_register_pairs_bootstrap_list[idx_bootstrap].to_csv(Path(save_dir)/f'battled_pairs_{str(idx_bootstrap).zfill(5)}.csv')

