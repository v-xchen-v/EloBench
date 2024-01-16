# import pandas as pd

# df = pd.read_csv('tempcache/google_quora_alpaca_10629/q_and_as batchbug wizard13.csv', index_col=0, keep_default_na=False, engine='python')
# print(len(df))
# # Removing column 'B'
# df_no_xwin13 = df.drop('Xwin-LM/Xwin-LM-13B-V0.1', axis=1)
# df_no_xwin13.to_csv('tempcache/google_quora_alpaca_10629/q_and_as_no_xwin13.csv', na_rep='NULL')
# # df_no_wizard7b = df[(df['model_a']!='WizardLM/WizardLM-7B-V1.0') & (df['model_b']!='WizardLM/WizardLM-7B-V1.0')]
# # print(len(df_no_wizard7b))
# # df_no_wizard7b.to_csv('results/google_quora_alpaca_10629_test1/battle_arrangement_no_wizard7b.csv')
from datamodel import BattleOutcomes, BootstrapedBattleOutcomes
# from report_pipe.bootstrap_leaderboard_report_pipe import LeaderBoardReportPipe
from report_pipe.report_builder import ReportBuilder
from report_pipe.leaderboard_report_pipe import LeaderBoardReportPipe
from report_pipe.bootstrap_leaderboard_report_pipe import BootstrapLeaderBoardReportPipe
from report_pipe.notie_battle_history_report import NoTieBattleHistoryReport
                
if __name__ == '__main__':
    # battle_outcomes = BattleOutcomes.read_csv(r'/elo_bench/results/google_quora_alpaca_10629_test4/battled_pairs.csv')
    # bootstrap_battle_outcomes = BootstrapedBattleOutcomes(battle_outcomes, num_of_bootstrap=1000)
    # bootstrap_battle_outcomes.to_csv(r'/elo_bench/results/google_quora_alpaca_10629_test4') 
    
    # bootstrap_battle_outcomes = BootstrapedBattleOutcomes.read_csv(r'/elo_bench/results/google_quora_alpaca_10629_test4')
    # print(bootstrap_battle_outcomes[1])
    
    # leaderboard_report_pipe = LeaderBoardReportPipe()
    # bootstrap_leaderboard_report_pipe = BootstrapLeaderBoardReportPipe(num_of_bootstrap=10)
    experiment_name = 'google_quora_alpaca_sharegpt_chat1m_21962_test1_smallset'
    notie_battle_history_report = NoTieBattleHistoryReport(FIRST_N_BATTLES=None)
    report_builder = ReportBuilder(rf'/elo_bench/results/{experiment_name}', rf'/elo_bench/reports/{experiment_name}', report_pipes=\
        [
            # leaderboard_report_pipe,
            # bootstrap_leaderboard_report_pipe,
            notie_battle_history_report
        ])
    
    report_builder.load_data()
    report_builder.build()
    report_builder.plot()
    # report_pipe.save(r'/elo_bench/results/google_quora_alpaca_10629_test4/report')