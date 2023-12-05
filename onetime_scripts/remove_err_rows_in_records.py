# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# import pandas as pd
# from judger.gpt4_helper import extract_winner_from_response

# df = pd.read_csv(r'/elo_bench/data/quora_100/battle_records.csv', keep_default_na=False)

# non_empty_rows = df[df['gpt_4_score']=='None']
# print(non_empty_rows['gpt_4_response'])
# for idx, row in non_empty_rows.iterrows():
#     score = extract_winner_from_response(row['gpt_4_response'])
#     print(score)
# non_empty_rows.to_csv('fixed_battle_records', index=False)

# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# import pandas as pd
# from judger.gpt4_helper import extract_winner_from_response

# df = pd.read_csv(r'/elo_bench/data/quora_100/battle_records.csv', keep_default_na=False)

# non_empty_rows = df[df['gpt_4_score']!='None']
# non_empty_rows.to_csv('fixed_battle_records', index=False)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from judger.gpt_judger import _extract_winner_from_response

df = pd.read_csv(r'/elo_bench/data/quora_100/battle_records.csv', keep_default_na=False)

non_empty_rows = df[~((df['answer_a']=='NULL') | (df['answer_b']=='NULL'))]
non_empty_rows.to_csv('fixed_battle_records', index=False)