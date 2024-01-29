import pandas as pd

nxn_arrangement = pd.read_csv(r'/elo_bench/results/rwq_200_winrate_vs_elo_on_nx1_gpt4/battle_arrangement_nxn.csv', index_col=0)
# filter out the rows that model_a or model_b is gpt-4-turbo
nx1_arrangement = nxn_arrangement[(nxn_arrangement['model_a']=='gpt-4-turbo') | (nxn_arrangement['model_b']=='gpt-4-turbo')]
print(len(nx1_arrangement))
nx1_arrangement.to_csv(r'/elo_bench/results/rwq_200_winrate_vs_elo_on_nx1_gpt4/battle_arrangement.csv')