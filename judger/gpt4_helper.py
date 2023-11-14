from .gpt4 import gpt_4_completion
import re
from .prompts import GPT4_PROMPT

def gpt_4_eval_and_score(question, model_a_ans, model_b_ans):
        gpt4_response_text = None
        gpt_4_score = None
        gpt_4_winner = None
        
        gpt4_response = gpt_4_completion(GPT4_PROMPT, question=question, model1_answer=model_a_ans, model2_answer=model_b_ans)
        
        # Compile the regular expression
        score_1_pattern = re.compile(r"- Score (?:of|for) Model 1:\s*([0-9.]+)")
        score_2_pattern = re.compile(r"- Score (?:of|for) Model 2:\s*([0-9.]+)")
        
        # Search the text
        gpt4_response_text = gpt4_response['response']
        ismatch_design_pattern1 = score_1_pattern.search(gpt4_response_text)
        ismatch_design_pattern2 = score_2_pattern.search(gpt4_response_text)

        # If a match is found, print the score
        if ismatch_design_pattern1:
            score1 = ismatch_design_pattern1.group(1)
            print(f"Model 1's Score is: {score1}")
        else:
            print("No score found for Model 1.")
            
        if ismatch_design_pattern2:
            score2 = ismatch_design_pattern2.group(1)
            print(f"Model 2's Score is: {score2}")
        else:
            print("No score found for Model 2.")
            
        if ismatch_design_pattern1 and ismatch_design_pattern2:
            gpt_4_score = {
                'model_a': score1,
                'model_b': score2
            }
            if score1 == "0" and score2 == "0":
                gpt_4_winner = 'tie(all bad)'
            elif score1 == "1" and score2 == "0":
                gpt_4_winner = 'model_a'
            elif score1 == "0.5" and score2 == "0.5":
                gpt_4_winner = 'tie'
            elif score1 == "0" and score2 == "1":
                gpt_4_winner = 'model_b'
            else:
                gpt_4_winner = 'invalid'
        print(gpt4_response)
        return gpt4_response, str(gpt_4_score), gpt_4_winner