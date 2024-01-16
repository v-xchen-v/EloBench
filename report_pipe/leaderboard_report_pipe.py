from datamodel import BattleOutcomes, BootstrapedBattleOutcomes
from pathlib import Path
from logger import info_logger
from elo_rating.rating_helper import get_elo_results_from_battles_data

import os

class LeaderBoardReportPipe:
    def __init__(self):
        self.result_dir = None
        self.elo_K = 4
        self.nature_battle_outcomes = None
        self.leaderboard_without_bootstrap = None
        self.num_of_bootstrap_for_leaderboard = None
        self.bootstrap_battle_outcomes = None
        
    def _load_battle_outcomes(self):
        self.nature_battle_outcomes = BattleOutcomes.read_csv(Path(self.result_dir) / 'battled_pairs.csv')
        # info_logger.info(f'Loaded {len(self.nature_battle_outcomes)} battle outcomes from {self.battle_outcomes_path}')
        
    def _generate_leaderboard_without_bootstrap(self):
        self.leaderboard_without_bootstrap = self.nature_battle_outcomes.get_leaderboard(K=self.elo_K)
        
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            self.leaderboard_without_bootstrap.to_csv(Path(self.save_dir) / 'leaderboard_without_bootstrap.csv', index=False)
    
    def build(self,result_dir: str, save_dir: str=None):
        self.result_dir = result_dir
        self.save_dir = save_dir
        self._load_battle_outcomes()
        self._generate_leaderboard_without_bootstrap()
        
           
    # def __init__(self, report_pipe):
    #     self.report_pipe = report_pipe

    # def build(self):
    #     self.report_pipe.build()
    #     self.report_pipe.add_report(LeaderBoardReport())