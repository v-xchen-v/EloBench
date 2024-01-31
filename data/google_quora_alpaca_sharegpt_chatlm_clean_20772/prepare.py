import pandas as pd
import json

# # Remove gpt-4 labeled wa questions
# bad_question_set1 = pd.read_csv(r'/elo_bench/data/google_quora_alpaca_sharegpt_chatlm_clean_20772/chat_1m_entire_question_5_all_filtered.tsv', delimiter='\t')['question'].tolist()
# bad_question_set2 = pd.read_csv(r'/elo_bench/data/google_quora_alpaca_sharegpt_chatlm_clean_20772/sharegpt_entire_question_5_all_filtered.tsv', delimiter='\t')['question'].tolist()

# # Remove 'If' labeled wa question
# bad_question_set3 = []
# with open(r'/elo_bench/data/google_quora_alpaca_sharegpt_chatlm_clean_20772/issue_questions.json') as issue_question_file:
#     bad_question_set3 = list(json.load(issue_question_file))
    
# bad_questions = list(set(bad_question_set1 + bad_question_set2 + bad_question_set3))
# print(f'{len(bad_questions)} need to be filtered out!')
# # 1630 need to be filtered out!

# initial_questions = pd.read_csv('/elo_bench/data/google_quora_alpaca_sharegpt_chat1m_22012/questions.csv')
# filtered_questions = initial_questions[~initial_questions['question'].isin(bad_questions)]
# print(filtered_questions)
# filtered_questions.to_csv(r'/elo_bench/data/google_quora_alpaca_sharegpt_chatlm_clean_20772/questions.csv')

# # remove repeat question
# custom_dict = {'tatsu-lab/alpaca_eval':0, 'Quora': 1, 'Google Trends': 2, 'lysms_chat_1m': 3, 'sharegpt':4}

# no_duplicated_filtered_questions = filtered_questions.sort_values('source',key=lambda x: x.map(custom_dict)).drop_duplicates('question')
# no_duplicated_filtered_questions.reset_index(inplace=True, drop=True)
# print(no_duplicated_filtered_questions)
# print(f'{len(filtered_questions)-len(no_duplicated_filtered_questions)} duplocated questions removed.')

# no_duplicated_filtered_questions.to_csv(r'/elo_bench/data/google_quora_alpaca_sharegpt_chatlm_clean_20772/questions.excel')

# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
quesitons = pd.read_csv('data/google_quora_alpaca_sharegpt_chatlm_clean_20772/questions.csv')
print(quesitons.shape) # (10629+11383, 2)
print(quesitons.groupby('source').count()) 
# '''
#                        question
# source                         
# Google Trends              5409
# Quora                      4415
# lysms_chat_1m              5969
# sharegpt                   5414
# tatsu-lab/alpaca_eval       805
# '''
# from datamodel import QuestionCollection
# cleaned_questions = QuestionCollection.read_csv(r'data/google_quora_alpaca_sharegpt_chat1m_xxx/questions.csv').questions
# # 22011 - 49 = 21962


                 