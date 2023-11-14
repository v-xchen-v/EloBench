

## Models
## Elo rating system
pairwise battle between models

Interface:
- PairwiseRatingEntity.battle(winner)

## GPT_4 as judger
Given the answers to same question by two models. use GPT_4 to determine which answer is better, the results:
1 - winner: model_a
0.5 - winner: tie, tie(all_bad)
0 - winner: model_b_is better

prompt should lead to scores by required format to extract models' scores.

provided a interface 'gpt_4_completion'
```
from judger import GPT4_PROMPT
from judger.gpt4 import gpt_4_completion
gpt4_response = gpt_4_completion(GPT4_PROMPT, question=question, model1_answer=model_a_ans, model2_answer=model_b_ans)
```

## format of saving question and llm generated answers(q and as.csv)
This is presentation of question and answers pair(1 question, n models, and n1 answered, n2 not, n=n1+n2). Then need to make arrangement of battles between the answer, n models could have at most Cn1 2(n1 should >=2) combination of battle.

Intermediate format: dict
dict = 
{
    "question_1_ctx", 
    [
        {
            "model": "model_1_name",
            "answer": "model_1_answer"
        },
        ...
    ]
}


after flatten(naturally removed repeatition)..., Got:

columns: question, model_1_name, model_2_name, ....
context: question_ctx, model_1_ans, model_2_ans, ...

For example:
                                                question  ... vicuna-7b
0        What is the difference between OpenCL and CUDA?  ...       NaN
1      Why did my parent not invite me to their wedding?  ...       NaN
2                       Fuji vs. Nikon, which is better?  ...       NaN
3                    How to build an arena for chatbots?  ...       NaN
4                                      When is it today?  ...       NaN
...                                                  ...  ...       ...
21227  If a tap is outputting 2500 ml a minute. How m...  ...       NaN
21228                 who is the president of the U.S.A?  ...       NaN
21229  how to train lora for stable diffusion? explai...  ...       NaN
21230           how to evaluate a language model output?  ...       NaN
21231  generate a detailed description on how to use ...  ...       NaN

## arrange battles
given a list of question and answers, each have:
1. question
2. answers pair(1 question, n models, and n1 answered, n2 not, n=n1+n2). Then need to make arrangement of battles between the answer, n models could have at most Cn1 2(n1 should >=2) combination of battle.

Elo rating is fast, so the the most battles arrangement.

Carings:
1. battle order, default as the order of question. and C nature combination order.
2. given the battle orders, scenario: reproduce some result of others.

columns:
question, model_a, model_b

Example:
                                                question            model_a            model_b
0        What is the difference between OpenCL and CUDA?         chatglm-6b          koala-13b        
1      Why did my parent not invite me to their wedding?   oasst-pythia-12b         alpaca-13b        
2                       Fuji vs. Nikon, which is better?          koala-13b   oasst-pythia-12b        
3                    How to build an arena for chatbots?         vicuna-13b   oasst-pythia-12b        
4                                      When is it today?         vicuna-13b          koala-13b        
...                                                  ...                ...                ...        
25933  If a tap is outputting 2500 ml a minute. How m...             palm-2         chatglm-6b        
25934                 who is the president of the U.S.A?         alpaca-13b  claude-instant-v1        
25935  how to train lora for stable diffusion? explai...  claude-instant-v1        guanaco-33b        
25936           how to evaluate a language model output?        guanaco-33b          koala-13b        
25937  generate a detailed description on how to use ...         chatglm-6b       wizardlm-13b 

## get the result(winner: a, b, or tie) of arrange battles
1. by gpt_4
2. by label or others

# Example
                 model_a            model_b   winner
0             chatglm-6b          koala-13b  model_b
1       oasst-pythia-12b         alpaca-13b      tie
2              koala-13b   oasst-pythia-12b  model_b
3             vicuna-13b   oasst-pythia-12b  model_b
4             vicuna-13b          koala-13b  model_a
...                  ...                ...      ...
25933             palm-2         chatglm-6b  model_a
25934         alpaca-13b  claude-instant-v1      tie
25935  claude-instant-v1        guanaco-33b  model_a
25936        guanaco-33b          koala-13b  model_a
25937         chatglm-6b       wizardlm-13b      tie

### ablation test:
swtich the order of battle pairs, model a, model b <-> model b model a, the same results?

TODO:
1. data explration with vis
- question ans by n models
- model n question
- battle round 
- prediction level and acutally....

## 
tie(all bad) 0 0 
Example:
Model 1's Answer: Why did my parent not invite me to their wedding?

Model 2's Answer: Why did my parent not invite me to their wedding?


GPT-4 Evaluation and Scoring:

Score of Model 1: 0
Score of Model 2: 0
Brief Explanation:
Both models have simply repeated the question instead of providing an answer, which is not helpful or relevant. Therefore, neither model deserves a score.


## All columns design
model_a	model_b	winner	judge	turn		language	tstamp	question	answer_a	answer_b	gpt_4_response	gpt_4_justification	is_valid

multiple-turn qa is not inscope

### Steps to set up a dataset and get elo score
1. collect questions
3. arrange pairwise battle
2. generate answers of models by question
4. use gpt_4 as judger get winner
5. generate elo leaderboard

When add model.
1. arrange battle on questions
2. generate answer of this model
3. use gpt_4 as judger get winner
4. generate elo leaderboard

When add question
1. add new question to collection
2. arrange pairwise battle
3. generate answers of models by question
4. use gpt_4 as judger get winner
5. generate elo leaderboard

related saving format:
1. question collection to store unique quesitons
csv file
each row is a question
2. Question and answers collection to store question with llm answers
csv file
each row is a question with multiple llm answer as a column
3. Battle pairs with question
4. Battled pairs with winners

