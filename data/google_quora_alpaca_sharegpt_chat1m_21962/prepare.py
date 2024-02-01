# import pandas as pd
# sharegpt_chat1m = pd.read_csv("data/google_quora_alpaca_sharegpt_chat1m/sharegpt_chat1m_20231225.tsv", encoding='utf-8', sep='\t')
# sharegpt_chat1m=sharegpt_chat1m[(sharegpt_chat1m['time_sensitive']==False) & (sharegpt_chat1m['question'].isna()==False)]
# inclusive_cols = ['question', 'source']
# sharegpt_chat1m = sharegpt_chat1m[inclusive_cols] # 11383
# sharegpt_chat1m.to_csv('data/google_quora_alpaca_sharegpt_chat1m/questions.csv', index=False)

import pandas as pd
quesitons = pd.read_csv('data/google_quora_alpaca_sharegpt_chat1m_22012/questions.csv')
print(quesitons.shape) # (10629+11383, 2)
print(quesitons.groupby('source').count()) 
'''
                       question
source                         
Google Trends              5409
Quora                      4415
lysms_chat_1m              5969
sharegpt                   5414
tatsu-lab/alpaca_eval       805
'''
