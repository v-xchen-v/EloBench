"""Run elo rating iterative battle pipeline for a given question set and model set"""

import argparse

# -e for experiment dir -n for notie target n -c for cache dir
parser = argparse.ArgumentParser(description='Run elo rating pipeline for a given question set and model set')
parser.add_argument('-e', '--experiment_dir', type=str, help='The dataset directory', required=True)
parser.add_argument('-c', '--cache_dir', type=str, help='The cache directory', required=True)
parser.add_argument('-n', '--notie_target_n', type=int, help='The number for no-tie battle the target n', required=True)
args = parser.parse_args()

from pipe.iterative_battle_pipe import IterativeBattlePipeline
from pathlib import Path
from datamodel import ArrangementStrategy
from logger import info_logger
import pandas as pd

project_dir = Path(args.project_dir)
target_n_note = args.notie_target_n
tempcache_dir = Path(args.cache_dir)
output_dir = project_dir

iterative_battle_pipe = IterativeBattlePipeline(tempcache_dir=tempcache_dir, save_dir=output_dir, no_cache=False, target_n_notie=target_n_note)

all_questions = pd.read_csv(project_dir/'questions.csv')['question'].tolist()
iterative_battle_pipe.register_questions(all_questions)

models = pd.read_csv(project_dir/'models.csv')['model'].tolist()
iterative_battle_pipe.register_models(models)

iterative_battle_pipe.arrange_battles(ArrangementStrategy.Random_N_BattleCount_Each_CombPair, num_of_battle=target_n_note)
iterative_battle_pipe.gen_model_answers()
iterative_battle_pipe.battle(saving_per=50)
iterative_battle_pipe.iterative_battle(saving_per=50)
iterative_battle_pipe.gen_elo()
info_logger.info('Done.')

