from datamodel import BattleOutcomes, BootstrapedBattleOutcomes
from pathlib import Path
from logger import info_logger
from elo_rating.rating_helper import get_elo_results_from_battles_data
from report_pipe.leaderboard_report_pipe import LeaderBoardReportPipe
import os

class BootstrapLeaderBoardReportPipe(LeaderBoardReportPipe):
    def __init__(self, num_of_bootstrap: int):
        super().__init__()
        self.num_of_bootstrap_for_leaderboard = num_of_bootstrap
        
        self.leaderboard_with_bootstrap = None
        self.bootstrap_battle_outcomes = None

    def load_bootstrap_battle_outcomes(self):
        if BootstrapedBattleOutcomes.is_cached(self.result_dir, self.num_of_bootstrap_for_leaderboard):
            self.bootstrap_battle_outcomes = BootstrapedBattleOutcomes.read_csv(self.result_dir)
        else:
            if self.nature_battle_outcomes is None:
                self._load_battle_outcomes()
            
            self.bootstrap_battle_outcomes = BootstrapedBattleOutcomes(self.nature_battle_outcomes, num_of_bootstrap=self.num_of_bootstrap_for_leaderboard)
            self.bootstrap_battle_outcomes.to_csv(self.result_dir)
            
    def generate_leaderboard_with_bootstrap(self):
        self.leaderboard_with_bootstrap = self.bootstrap_battle_outcomes.get_leaderboard(K=self.elo_K)
        
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            self.leaderboard_with_bootstrap.to_csv(Path(self.save_dir) / f'leaderboard_{self.num_of_bootstrap_for_leaderboard}_bootstrap.csv', index=False)
    
    def build(self, result_dir: str, save_dir: str=None):
        self.result_dir = result_dir
        self.save_dir = save_dir
        self.load_bootstrap_battle_outcomes()
        self.generate_leaderboard_with_bootstrap()
        
           
    # def __init__(self, report_pipe):
    #     self.report_pipe = report_pipe

    # def build(self):
    #     self.report_pipe.build()
    #     self.report_pipe.add_report(LeaderBoardReport())