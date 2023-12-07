import pandas as pd
data = pd.read_csv("data/google_quora_alpaca_10630/google_quora_alpaca 20231207.tsv", encoding='utf-8', sep='\t')
data=data[((data['source']=='tatsu-lab/alpaca_eval') | (data['time_sensitive']==False)) & (data['question'].isna()==False)]
inclusive_cols = ['question', 'source']
data = data[inclusive_cols]
data.to_csv('data/google_quora_alpaca_10630/question.csv', index=False)