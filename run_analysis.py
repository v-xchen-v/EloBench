"""Run analysis on the battled pairs, and save the results to a directory. Including:
1. Clean battled pairs, remove None and invalid battles by gpt-4 judger not working, and drop a file with original index of nature battle pairs and battle records
2. Make bootstrap battle pairs
3. Cal elo ratings per bootstrap
4. Cal aggragate elo
5. Save the results to a directory
"""

import os, sys
sys.path.append(r'/elo_bench')

from battle_outcome_analysis.clean_battled_pairs import clean_battled_pairs, reload_clean_battled_pairs
from battle_outcome_analysis.bootstrap_battled_pairs import do_bootstrap, reload_bootstrap_battled_pairs
from datamodel import BattleOutcomes, BootstrapedBattleOutcomes
from logger import info_logger
from pathlib import Path
import os

import argparse
parser = argparse.ArgumentParser(description='Run analysis on battled pairs with model_a, model_b, winner as columns. And save the results to a directory. Including: 1. Clean battled pairs, remove None and invalid battles by gpt-4 judger not working, and drop a file with original index of nature battle pairs and battle records 2. Make bootstrap battle pairs 3. Cal elo ratings per bootstrap 4. Cal aggragate elo')
parser.add_argument('-b', '--battle_outcome_dir', type=str, help='The battle result directory which stores the battled pairs information', required=True)
parser.add_argument('-n', '--num_bootstrap', type=int, help='The number of bootstrapping the order of battles', required=True)
args = parser.parse_args()

# args

battle_outcome_dir = Path(args.battle_outcome_dir)
NUM_BOOTSTRAP = args.num_bootstrap
# NUM_BOOTSTRAP = 100
# battle_outcome_dir = r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_21962_test1_smallset'

# settings
ELO_K_FACTOR = 4
battled_pairs_file = Path(battle_outcome_dir)/'battled_pairs.csv'
analysis_dump_dir = Path(battle_outcome_dir)/'.analysis'
bootstrap_n_elo_rating_file = Path(analysis_dump_dir)/'bootstrap_n_elo_rating.csv'
bootstrap_aggregate_elo_rating_file = Path(analysis_dump_dir)/'bootstrap_aggregate_elo_rating.csv'
cleaned_battled_pairs_file = Path(analysis_dump_dir/'battled_pairs.csv')

# create analysis dump dir if not existed
if not os.path.exists(analysis_dump_dir):
    os.makedirs(analysis_dump_dir)

# step 1: clean battled pairs, remove None and invalid battles by gpt-4 judger not working and dump a file with original index of nature battle pairs and battle records
if os.path.exists(cleaned_battled_pairs_file):
    battled_pairs = reload_clean_battled_pairs(cleaned_battled_pairs_file)
else:
    battled_pairs = clean_battled_pairs(battled_pairs_file, cleaned_battled_pairs_file)

# Step 2: make bootstrap on battle pairs
nature_battle_outcomes = BattleOutcomes.read_csv(cleaned_battled_pairs_file)
if BootstrapedBattleOutcomes.is_cached(analysis_dump_dir/'.bootstrap', NUM_BOOTSTRAP):
    bootstrap_battle_outcomes = BootstrapedBattleOutcomes.read_csv(Path(analysis_dump_dir)/'.bootstrap')
else:
    bootstrap_battle_outcomes = BootstrapedBattleOutcomes(nature_battle_outcomes, NUM_BOOTSTRAP)
    bootstrap_battle_outcomes.to_csv(Path(analysis_dump_dir)/'.bootstrap')

# Step 3: calcuate elo on each bootstrap round
bootstrap_battle_outcomes.get_leaderboards(K=ELO_K_FACTOR).to_csv(bootstrap_n_elo_rating_file)
info_logger.info(f'Elo ratings per bootstrap are saved to {bootstrap_n_elo_rating_file}')

# Step 4: calculate aggragate elo on all bootstrap rounds with agg op as median
# TODO: remove repeat cal elo on each bootstrap round
bootstrap_battle_outcomes.get_leaderboard(K=ELO_K_FACTOR).to_csv(bootstrap_aggregate_elo_rating_file)
info_logger.info(f'Aggragate elo are saved to {bootstrap_aggregate_elo_rating_file}')

info_logger.info('Done.')   