import os, sys
sys.path.append(r'/elo_bench')

from battle_outcome_analysis.clean_battled_pairs import clean_battled_pairs, reload_clean_battled_pairs
from bootstrap_battled_pairs import do_bootstrap, reload_bootstrap_battled_pairs

from pathlib import Path
import os

# Settings
result_dir = r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset'

# Logic
battled_pairs_file = Path(result_dir)/'battled_pairs.csv'
analysis_dump_dir = Path(result_dir)/'.analysis'
bootstrap_battled_pairs_save_dir = analysis_dump_dir/'.bootstrap'
if not os.path.exists(analysis_dump_dir):
    os.makedirs(analysis_dump_dir)
cleaned_battled_pairs_file = Path(analysis_dump_dir/'battled_pairs.csv')

# Step 1: clean battled pairs, remove None and invalid battles by gpt-4 judger not working, and drop a file with original index of nature battle pairs and battle records
# battled_pairs = clean_battled_pairs(battled_pairs_file, cleaned_battled_pairs_file)
# battled_pairs = reload_clean_battled_pairs(cleaned_battled_pairs_file)

# Step 2: make bootstrap battle pairs
# bootstrap_battled_pairs = do_bootstrap(cleaned_battled_pairs_file, bootstrap_battled_pairs_save_dir)
bootstrap_battled_pairs = reload_bootstrap_battled_pairs(bootstrap_battled_pairs_save_dir)

pass