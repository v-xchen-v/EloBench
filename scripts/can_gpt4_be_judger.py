# when gpt_4 acts as judger, whether the result is matched with human's judgement or not.

import pandas as pd

file_path = r"results/battle_records_arena_gpt4_as_judger_0_499.csv"
battleres_file_path = r"data/arena_data/chatbot_arena_conversations/battled_pairs_human_as_judger.csv"
battlearrangement_file_path = r"data/arena_data/chatbot_arena_conversations/battle_arrangement.csv"


df = pd.read_csv(file_path)
res_df = pd.read_csv(battleres_file_path)
arrange_df = pd.read_csv(battlearrangement_file_path)

res_with_q = pd.concat([res_df, arrange_df['question']], axis=1)

total_count = 0
match_count= 0
tie_counter = 0
for idx, row in df.iterrows():
    gpt_4_winner = row['winner']
    if  pd.isna(gpt_4_winner):
        continue
    
    # find res of human
    match_rec = res_with_q[(res_with_q['model_a']==row['model_a']) & (res_with_q['model_b']==row['model_b']) & (res_with_q['question']==row['question'])].iloc[0]
    human_winner = match_rec['winner']
    
    # if gpt_4_winner == human_winner or gpt_4_winner.startswith('tie') and human_winner.startswith('tie'):
    #     match_count+=1
    # total_count+=1
    if human_winner.startswith('tie') or gpt_4_winner.startswith('tie'):
        tie_counter+=1
        continue
    elif gpt_4_winner == human_winner:
        match_count+=1
    total_count+=1
print(tie_counter)
        
print(f'{match_count}/{total_count}')
print(f'{match_count/total_count}')
