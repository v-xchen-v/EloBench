import pandas as pd

df = pd.read_csv('tempcache/google_quora_alpaca_10629/q_and_as batchbug wizard13.csv', index_col=0, keep_default_na=False, engine='python')
print(len(df))
# Removing column 'B'
df_no_xwin13 = df.drop('Xwin-LM/Xwin-LM-13B-V0.1', axis=1)
df_no_xwin13.to_csv('tempcache/google_quora_alpaca_10629/q_and_as_no_xwin13.csv', na_rep='NULL')
# df_no_wizard7b = df[(df['model_a']!='WizardLM/WizardLM-7B-V1.0') & (df['model_b']!='WizardLM/WizardLM-7B-V1.0')]
# print(len(df_no_wizard7b))
# df_no_wizard7b.to_csv('results/google_quora_alpaca_10629_test1/battle_arrangement_no_wizard7b.csv')