import pandas as pd

nxn_arrangement = pd.read_csv(r'/elo_bench/results/rwq_200_winrate_vs_elo_on_nx1_llama1_13b/battle_arrangement_nxn.csv', index_col=0)
# filter out the rows that model_a or model_b is gpt-4-turbo
nx1_arrangement = nxn_arrangement[(nxn_arrangement['model_a']=='huggyllama/llama-13b') | (nxn_arrangement['model_b']=='huggyllama/llama-13b')]
print(len(nx1_arrangement))
nx1_arrangement.to_csv(r'/elo_bench/results/rwq_200_winrate_vs_elo_on_nx1_llama1_13b/battle_arrangement.csv')