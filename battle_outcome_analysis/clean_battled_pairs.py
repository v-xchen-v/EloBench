"""clean battled pairs, remove None and invalid battles by gpt-4 judger not working, and drop a file with original index of nature battle pairs and battle records"""

# Import Necessary packages
from pathlib import Path
import os
import pandas as pd

def clean_battled_pairs(battled_pairs_file, save_path):
    valid_winner = ['model_a', 'model_b', 'tie', 'tie(all bad)']
    all_battled_paris = pd.read_csv(battled_pairs_file, index_col=0)
    valid_battled_pairs = all_battled_paris[all_battled_paris['winner'].isin(valid_winner)]
    invalid_battled_pairs = all_battled_paris[~all_battled_paris['winner'].isin(valid_winner)]
    print(f'{len(invalid_battled_pairs)} invalid battled pairs removed!')
    
    valid_battled_pairs.to_csv(save_path)
    return valid_battled_pairs

def reload_clean_battled_pairs(cleaned_battled_pairs_file):
    valid_battled_pairs = pd.read_csv(cleaned_battled_pairs_file)
    return valid_battled_pairs
