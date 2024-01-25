import os, sys
sys.path.append(r'/elo_bench')
from pathlib import Path
import pandas as pd
from datamodel import BattleOutcomes, BootstrapedBattleOutcomes
from elo_rating.rating_evaluator import compute_actual_winrate_awinb
from tqdm import tqdm
from collections import defaultdict


battle_outcome_dir = r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_model_later_1'
analysis_dump_dir = Path(battle_outcome_dir)/'.analysis'
cleaned_battled_pairs_file = Path(analysis_dump_dir/'.bootstrap/battled_pairs_00001.csv')

notie_winner = ['model_a', 'model_b']
valid_battled_pairs = pd.read_csv(cleaned_battled_pairs_file)
# notie_valid_battled_pairs = valid_battled_pairs[valid_battled_pairs['winner'].isin(notie_winner)]

        
def compute_actual_winrate(awinb_counter, ab_battle_counter, ab_notie_battle_counter, model_a, model_b, num_battle):
    return {
                'model_a': model_a,
                'model_b': model_b,
                'awinb_actual_winrate': awinb_counter[model_a][model_b] / ab_notie_battle_counter[model_a][model_b] if ab_notie_battle_counter[model_a][model_b] > 0 else pd.NA,
                'num_battle_on_ab': ab_battle_counter[model_a][model_b],
                'num_notie_battle_on_ab': ab_notie_battle_counter[model_a][model_b],
                'num_battle': num_battle
            }


models = sorted(valid_battled_pairs['model_a'].unique())
awinb_counter = defaultdict(lambda: defaultdict(lambda: 0))
ab_battle_counter = defaultdict(lambda: defaultdict(lambda: 0))
ab_notie_battle_counter = defaultdict(lambda: defaultdict(lambda: 0))
winrate_pds = []
awinb_actual_winrates = defaultdict(lambda: defaultdict(lambda: []))
for idx, row in valid_battled_pairs.iterrows():
    num_battle = idx + 1
    sorted_model_pair = sorted([row['model_a'], row['model_b']])
    switch = False
    if sorted_model_pair[0] != row['model_a']:
        switch = True
    if row['winner'] not in notie_winner:
        pass
    else:
        ab_notie_battle_counter[sorted_model_pair[0]][sorted_model_pair[1]] += 1
        ab_notie_battle_counter[sorted_model_pair[1]][sorted_model_pair[0]] += 1
        if row['winner'] == ('model_a' if not switch else 'model_b'):
            awinb_counter[sorted_model_pair[0]][sorted_model_pair[1]] += 1   
    ab_battle_counter[sorted_model_pair[0]][sorted_model_pair[1]] += 1
    ab_battle_counter[sorted_model_pair[1]][sorted_model_pair[0]] += 1
        
    winrate_dict = compute_actual_winrate(awinb_counter, ab_battle_counter, ab_notie_battle_counter, sorted_model_pair[0], sorted_model_pair[1], num_battle)
    awinb_actual_winrates[sorted_model_pair[0]][sorted_model_pair[1]].append(winrate_dict)

save_dir = Path(battle_outcome_dir)/'output/data/actual_winrate'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for model_a in awinb_actual_winrates:
    for model_b in awinb_actual_winrates[model_a]:
        pd.DataFrame.from_dict(awinb_actual_winrates[model_a][model_b]).to_csv(Path(save_dir)/f'actual_winrate_{model_a.replace("/", ".")}_{model_b.replace("/", ".")}.csv')