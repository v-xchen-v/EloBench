from openai_chat import chat_completion
import re
from gpt_chat_prompts.prompts import GPT4_EVAL_AND_SCORE_PROMPT
from logger import gpt_as_judger_logger
import json
    
def _extract_winner_from_response_v1123(gpt4_response_text):
    # Compile the regular expression
    score_1_pattern = re.compile(r"-? ?Score (?:of|for) Model 1:\s*([0-9.]+)")
    score_2_pattern = re.compile(r"-? ?Score (?:of|for) Model 2:\s*([0-9.]+)")
    
    # Search the text
    ismatch_design_pattern1 = score_1_pattern.search(gpt4_response_text)
    ismatch_design_pattern2 = score_2_pattern.search(gpt4_response_text)

    # If a match is found, print the score
    if ismatch_design_pattern1:
        score1 = ismatch_design_pattern1.group(1)
        gpt_as_judger_logger.debug(f"Model 1's Score is: {score1}")
    else:
        gpt_as_judger_logger.debug("No score found for Model 1.")
        
    if ismatch_design_pattern2:
        score2 = ismatch_design_pattern2.group(1)
        gpt_as_judger_logger.debug(f"Model 2's Score is: {score2}")
    else:
        gpt_as_judger_logger.debug("No score found for Model 2.")
        
    gpt_4_winner = None
    gpt_4_score = None
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
            
    return gpt_4_winner, gpt_4_score

def _extract_winner_from_reponse_v1225(gpt4_response_text):
    gpt_4_winner = None
    gpt_4_score = None
    
    try:
        gpt4_response_json = json.loads(gpt4_response_text)
    except json.decoder.JSONDecodeError as e:
        gpt_as_judger_logger.debug(f"JSONDecodeError: {e}.")
        return 'invalid', None
    
    winner_mapping = {
        1: "model_a",
        2: "model_b",
        0: "tie",
        -1: "tie(all bad)"
    }
    
    if 'winner' not in gpt4_response_json or not gpt4_response_json['winner'] in winner_mapping.keys():
        return 'invalid', None
    
    gpt_4_winner = winner_mapping[gpt4_response_json['winner']]
    gpt_4_score = {
        'model_a': 1 if gpt4_response_json['winner'] == 1 else 0,
        'model_b': 1 if gpt4_response_json['winner'] == 2 else 0,
    }
    
    return gpt_4_winner, gpt_4_score

def _extract_winner_from_response(gpt4_response_text):
    return _extract_winner_from_reponse_v1225(gpt4_response_text)

def gpt_4_eval_and_score(question, model_a_ans, model_b_ans, judger_name):
    gpt4_response_text = None
        
    # gpt4_response = chat_completion(GPT4_EVAL_AND_SCORE_PROMPT, gpt_name=judger_name, question=question, model1_answer=model_a_ans, model2_answer=model_b_ans)
    gpt4_response = chat_completion(GPT4_EVAL_AND_SCORE_PROMPT, gpt_name=judger_name, user_submitted_question=question, llm_response_1=model_a_ans, llm_response_2=model_b_ans)
    
    gpt4_response_text = gpt4_response['response']
    
    # handle none response, return None
    if gpt4_response['is_valid']==False or gpt4_response_text is None:
        gpt_as_judger_logger.debug("No response found.")
        return gpt4_response, None, None
    
    gpt_as_judger_logger.debug(gpt4_response)
    
    gpt_4_winner, gpt_4_score = _extract_winner_from_response(gpt4_response_text)
    
    return gpt4_response, str(gpt_4_score), gpt_4_winner