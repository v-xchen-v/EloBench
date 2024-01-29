# import pandas as pd

# q_and_as_file = r'/elo_bench/tempcache/google_quora_alpaca_sharegpt_chat1m_clean_20772/q_and_as.csv'
# q_and_as = pd.read_csv(q_and_as_file, index_col=0, keep_default_na=False, na_values=['NaN', 'NULL'], engine='python')
# print(q_and_as.columns)
# q_and_as.drop(columns=['mistralai/Mixtral-8x7B-v0.1'], inplace=True)
# print(q_and_as.columns)
# print(len(q_and_as.columns))

# import numpy as np

# np.random.seed(42)
# q_and_as.replace('', np.nan, inplace=True)
# filtered_df = q_and_as.dropna(how='any')
# print(len(filtered_df))
# sampled_df = filtered_df.sample(n=200)
# sampled_df.to_csv(r'/elo_bench/data/rwq_200/temp.csv')
# sampled_df[['question']].to_csv(r'/elo_bench/data/rwq_200/questions.csv')

import pandas as pd
file_to_check = r'/elo_bench/data/rwq_200/temp.csv'
df = pd.read_csv(file_to_check, keep_default_na=False, na_values=['NaN', 'NULL'], engine='python')

has_nan_or_none = df.isna().any().any()

has_empty_string = (df == '').any().any()
print(has_nan_or_none)
print(has_empty_string)