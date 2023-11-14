import sys, os
sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))

from datasets import load_dataset
from datamodel import QuestionCollection

dataset_id = 'lmsys/chatbot_arena_conversations'
dataset_split = 'train'

# filtered single turn conversation
conversations = load_dataset(dataset_id)[dataset_split]
singleturn_conversations = conversations.filter(lambda example, idx: example['turn']==1 and example['language']=="English", with_indices=True)

conversation_a = singleturn_conversations['conversation_a']
questions = QuestionCollection()
for item in conversation_a:
    for x in item:
        if x['role'] == 'user':
            questions.add(x['content'])

# save as csv, not txt because the content may contains '\n'  
questions.to_csv('chatbot_arena_questions.csv')
print(questions)