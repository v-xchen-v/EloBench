"""Read the old arrangement, random select 1% used question and put the at end as the new arrangement."""

import pandas as pd
import numpy as np

old_arrangement = pd.read_csv(r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_q_later_30/battle_arrangement_old.csv')
used_questions = old_arrangement['question'].unique()
print(len(used_questions))

np.random.seed(42)
percentage = 30.0
num_select = int(len(used_questions) * (percentage / 100.0))
print(num_select)
question_selected = np.random.choice(used_questions, size=num_select, replace=False) 
pd.DataFrame(question_selected, columns=['question']).to_csv(r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_q_later_30/later_register_questions.csv')

# replace = False means no duplicate
remain_questions = list(set(used_questions) - set(question_selected))

# to put the rows with selected questions at the end
question_selected_set = set(question_selected)

mask = old_arrangement['question'].isin(question_selected_set)

# Split the dataframe to two parts
df_not_in_set = old_arrangement[~mask]
df_in_set = old_arrangement[mask]

# Concate them in the order you want which in this case is, not in set first and then in set
new_arrangement = pd.concat([df_not_in_set, df_in_set], ignore_index=True)
new_arrangement.to_csv(r'results/google_quora_alpaca_sharegpt_chat1m_clean_20772_fullset_register_q_later_30/battle_arrangement.csv')