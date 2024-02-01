from datasets import load_dataset
import pandas as pd

questions = pd.read_csv(r'data/rwq/questions_20772.csv', index_col=0).sort_values(by=['source'], key=lambda x: x.map({
    'tatsu-lab/alpaca_eval': 0, 
    'lysms_chat_1m': 1, 
    'Quora': 2, 
    'Google Trends': 3, 
    'sharegpt': 4 }))

# remove duplicated questions keep first one
# before removing duplicated questions
#                        question
# source                         
# Google Trends              5409
# Quora                      4414
# lysms_chat_1m              5142
# sharegpt                   5051
# tatsu-lab/alpaca_eval       804
questions.drop_duplicates(subset=['question'], keep='first', inplace=True)
questions.reset_index(drop=True, inplace=True)
# after removing duplicated questions
# Google Trends              5409
# Quora                      4414
# lysms_chat_1m              5118
# sharegpt                   5027
# tatsu-lab/alpaca_eval       804
print(questions.groupby('source').count()) 

# questions.to_pickle(r'data/rwq/questions.pkl')
# rwq_dataset = load_dataset("pandas", data_files=r'data/rwq/questions.pkl')
# pass
questions.to_csv(r'data/rwq/questions.csv')