# Import necessary packages
import os, sys
sys.path.append(r'/elo_bench')
"""Loads existed bootstrap battled pairs and cal elo ratings"""

from pathlib import Path
from datamodel import BootstrapedBattleOutcomes

battle_outcome_dir = r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_10_seed5'


analysis_dump_dir = Path(battle_outcome_dir)/'.analysis'
if not os.path.exists(analysis_dump_dir):
    os.makedirs(analysis_dump_dir)
bootstrap_n_elo_rating_file = Path(analysis_dump_dir)/'bootstrap_n_elo_rating.csv'
bootstrap_aggregate_elo_rating_file = Path(analysis_dump_dir)/'bootstrap_aggregate_elo_rating.csv'

battled_pairs_file = Path(battle_outcome_dir)/'battled_pairs.csv'

# setting for elo ratings
K = 4
bootstrap_battle_outcomes = BootstrapedBattleOutcomes.read_csv(Path(analysis_dump_dir)/'.bootstrap')


# Step 3: cal elo per bootstrap
bootstrap_battle_outcomes.get_leaderboards(K=K).to_csv(bootstrap_n_elo_rating_file)

# Step 4: cal aggragate elo
# TODO: remove repeat cal elo on each bootstrap round
bootstrap_battle_outcomes.get_leaderboard(K=K).to_csv(bootstrap_aggregate_elo_rating_file)

