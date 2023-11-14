from __future__ import annotations

import sys, os
sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))

from dataclasses import dataclass
from collections import defaultdict
from datamodel import QuestionAndAnswersCollection, LLMAnswer, PairToBattle, PairwiseBattleArrangement, BattledPairs
    
from datasets import load_dataset
import pandas as pd

# question as key and QuestionAndAnswer as value
qa_list = defaultdict(list)

dataset_id = 'lmsys/chatbot_arena_conversations'
dataset_split = 'train'

# filtered single turn conversation
conversations = load_dataset(dataset_id)[dataset_split]
singleturn = conversations.filter(lambda example, idx: example['turn']==1 and example['language']=="English", with_indices=True)

def extract_conversation_question(conversation: list[dict]):
    question = [x['content'] for x in conversation if x['role']=='user'][0]
    return question
    
def extract_conversation_answer(conversation: list[dict]):
    question = [x['content'] for x in conversation if x['role']=='assistant'][0]
    return question

question_and_answers_collection = QuestionAndAnswersCollection()
for row in singleturn:
    model_a_ans = LLMAnswer(model=row['model_a'], answer=extract_conversation_answer(row['conversation_a']))
    model_b_ans = LLMAnswer(model=row['model_b'], answer=extract_conversation_answer(row['conversation_b']))
    question = extract_conversation_question(row['conversation_a'])
    question_and_answers_collection.add_answers(question, [model_a_ans, model_b_ans])

question_and_answers_collection.to_csv('chatbot_arena_q_and_as.csv')
print(question_and_answers_collection)


battle_arrangment = []
for row in singleturn:
    battle_arrangment.append([extract_conversation_question(row['conversation_a']),row['model_a'], row['model_b']])
arrangement = pd.DataFrame(battle_arrangment, columns=['question','model_a', 'model_b'])
print(arrangement)
print(arrangement)
arrangement.to_csv('battle_arrangement.csv')
ar = PairwiseBattleArrangement.read_csv('battle_arrangement.csv')
print(ar)

battle_res = BattledPairs()
for row in singleturn:
    battle_res.add_pair(row['model_a'], row['model_b'], row['winner'])

battle_res.to_csv('battle_res.csv')
battle_res = BattledPairs.read_csv('battle_res.csv')
print(battle_res)



# pd.DataFrame([ for x in qa_)])
"""                                                question                                         chatglm-6b  ...                                  claude-instant-v1 vicuna-7b
0        What is the difference between OpenCL and CUDA?    What is the difference between OpenCL and CUDA?  ...                                                NaN       NaN
1      Why did my parent not invite me to their wedding?                                                NaN  ...                                                NaN       NaN
2                       Fuji vs. Nikon, which is better?                                                NaN  ...                                                NaN       NaN
3                    How to build an arena for chatbots?                                                NaN  ...                                                NaN       NaN
4                                      When is it today?                                                NaN  ...                                                NaN       NaN
...                                                  ...                                                ...  ...                                                ...       ...
21227  If a tap is outputting 2500 ml a minute. How m...  If a tap is outputting 2500 ml a minute. How m...  ...                                                NaN       NaN
21228                 who is the president of the U.S.A?                                                NaN  ...                 who is the president of the U.S.A?       NaN
21229  how to train lora for stable diffusion? explai...                                                NaN  ...  how to train lora for stable diffusion? explai...       NaN
21230           how to evaluate a language model output?                                                NaN  ...                                                NaN       NaN
21231  generate a detailed description on how to use ...  generate a detailed description on how to use ...  ...                                                NaN       NaN

[21232 rows x 21 columns]"""
# 21 models of 21232 questions.

    
# conversation_a = singleturn_conversations['conversation_a']
# questions = set()
# for item in conversation_a:
#     for x in item:
#         if x['role'] == 'user':
#             questions.add(x['content'])
#         if x['role'] == 'assistant':
            
            
# with open('chatbot_arena_questions.txt', 'w') as f:
#     f.writelines(questions)
# # questions = [x for x in conversation_a if x['role']=='user']
# # {k: v for k, v in conversation_a.items() if k in ['a', 'b']}
# pf = pd.DataFrame(questions, columns=['questions']).to_csv('chatbot_arena_questions.csv', index=True)
# print(conversations)