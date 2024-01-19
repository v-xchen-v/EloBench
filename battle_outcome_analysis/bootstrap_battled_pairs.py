"""Make bootstrap and save to file, or reload bootstrap"""

# import necessary packages
from datamodel import BattleOutcomes, BootstrapedBattleOutcomes
from pathlib import Path

# settings
NUM_BOOTSTRAP = 100

def do_bootstrap(cleaned_battled_pairs_file, save_dir):
    nature_battle_outcomes = BattleOutcomes.read_csv(cleaned_battled_pairs_file)
    bootstrap_battle_outcomes = BootstrapedBattleOutcomes(nature_battle_outcomes, NUM_BOOTSTRAP)
    bootstrap_battle_outcomes.to_csv(save_dir)
    return bootstrap_battle_outcomes

def reload_bootstrap_battled_pairs(save_dir):    
    if not BootstrapedBattleOutcomes.is_cached(save_dir, NUM_BOOTSTRAP):
        raise 'No cached bootstrap battled pairs to reload'
    bootstrap_battle_outcomes = BootstrapedBattleOutcomes.read_csv(save_dir)
    return bootstrap_battle_outcomes  