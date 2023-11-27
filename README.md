

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


## GPT-4 as judger vs Human as judger
- 前提： GPT-4可以作为judger
1. human as judger only can label on battle pair result, but gpt-4 can at most get all combination results of one question.


TODO:
1. resume and continue when battle with gp4-4 as judger
2. different K-factor strategies(fixed k, different levels K)
3. bootstrap
4. testing on qura 100
5. to test former and laster position when compare with judger as gpt-4

### Framework design
#### Design of Battle Pipeline
Three senarios:
1. Fixed questions and models
2. + question later
3. + models later

For the fixed questions and models:


### Features:
1. Given questions and open-source models(huggingface ids), got good elo rating.

### TODOs:


### Sokved Issues
#### Empty LLM generated answer handling
- use 'Question: {question}\nAnswer: ' formating question as prompt to reduce the frequency of generating empty answer, especially for alpaca-7b/13b and vicuna-7b models.
- save missing value(not generated) as 'NULL', and empty answer as ''.
- tested on gpt-4, gpt-4 could handle the case that 1 ans is empty:
```
res = gpt_4_completion("judger/gpt4_prompts/eval_and_score_better_ans_prompt_231113.txt", question="What is the meaning of life?", model1_answer="42", model2_answer="")
print(res)
"""'- Score of Model 1: 1\n- Score of Model 2: 0\n- Brief Explanation: Model 1\'s answer, "42", is a humorous reference to "The Hitchhiker\'s Guide to the Galaxy" by Douglas Adams, where "42" is presented as the "Answer to the Ultimate Question of Life, The Universe, and Everything", according to a supercomputer named Deep Thought. Although it\'s a joke, it\'s still a response. Model 2, on the other hand, provided no answer at all, which is why it receives a score of 0.'"""
```

## RoadMap
This roadmap provides a comprehensive approach to developing an Elo system for comparing LLMs, using GPT-4 as a judger, in the context of question-answering abilities.
1. Define the Competition Rules
    - *Type of Questions*: Decide on the questions to be used(e.g., top question on quaro or google).
    - *Scoring Criteria*: Establish clear criteria for what constitutes a correct or better answer.
    - *Match Format*: Determine how models will be paired and how many rounds they will compete in.

2. Select and Prepare the LLMs
    - *Model Selection*: Choose which open-source models will compete against each other.
    - *Environment Setup*: Set up a programming environment where all models can receive questions and generate answers.

3. Implement GPT-4 as a Judger
    -*Judging Algorithm*: Define how GPT-4 will evaluate answers. Ensure GPT-4's responses are not biased towards its own 'style' of answering.
    -*Validation*: Test GPT-4's judging capabilities to ensure consistency and fairness.

4. Implement Elo Ratings
    - *Baseline Ratings*
    - *More*: different K-factor, battle ordering and so on.

5. Develop the Competition Framework
    - *Automated Questioning*: Implement a system to automatically pose questions to each model.
    - *Answer Collection*: Ensure a mechanism for collecting and organizing answers from each model.

6. Run Competitions
    - Regular Matches: Conduct regular rounds where each model answers questions.
    - Result Recording: Record each model's performace as per GPT-4's evaluation.

7. Update Elo Ratings
    - *Calculation*: After each round, calculate Elo rating changes based on the result.
    - Adjustment Mechaism: Implement a system to adjust ratings after each match.

8. Analysis and Reporting
    - *Performance Tracking*: Keep track of each model's perforance over time.
    - *Insights Generation*: Analyze rsults for insights into each model's strengths and weaknesses.

9. Iterative Improvement
    - *Feedback Loop*: Use insights to refine the judging criteria, question selection and competition format.
    - *Model Updates*: Allow for the inclusion of updated or new models over time.

10. Documentation and Transparency
    - *Public Reporting*: Regularly publish competition results and rating changes.
    - *Open Methidology*: Make the methodology of the competition and rating calculations public for transparency.

11. Community Engagement
    - *Community Feedback*: Involve the AI and research community for feedback and suggestions.
    - *Collaboration*: Collabarate with other researchers or institutions for a more robust system.

12. Legal and Ethical Considrations.
    - *Fair Use*: Ensure the use of GPT-4 and other models adheres to legal and ethical standards.
    - *Bias and Fairness*: Regularly assess the system for any biases or unfair practices.

## Considerations
- Resource Intensive: Running multiple LLMs and GPT-4 for judging can be resource-intensive. Plan for the necessary computational resources.
- Model Limitations: Be aware of the limitations of each model, including GPT-4, and how these might impact the fairness of the competition.
- Continuous Monitoring: The system should be monitored and adjusted as models evolve and improve over time.

## Features
1. *Model Integration*
    - *LLM Interface*: Interface for integrating various LLMs, including open-source models and GPT-4, to ensure smooth interaction and response handling.
    - *Model Configuration*: Allow configuration setting for each model(e.g., token limits, temperature settings for GPT-4)

2. *Question Pool Management*
    - *Question Database*:  A diverse and extensive database of questions, categorized by difficulty, type(factual, reasoning, etc), and topic
    - Randomized Question Selection: Mechanism for selection questions randomly to ensure a fair and unbais challenge for each model.

3. *Answer Assessment*
    - *Answer Evaluation Criteria*: Define a clear and objective criteria for what consititutes a correct or superior answer.
    - Scoring Algorithm: Algorithm for scoring answers, possibly with partical credits for partailly correct answer.
    - Automated Answer Judging: Using GPT-4 to evaluate answers with predefined metric for fairness and accuracy.

4. Elo Rating System
- Initial Rating Assignment: Assign initial Elo ratings to all participating models.
- Rating Update Mechanism: Algorithm to update Elo ratings based on match outcomes, ensuring fair and accurate reflection of performance.
- Rating Decay/Inflation Adjustments: Mechanisms to counteract rating inflation or decay over time.

5. Matchmaking and Competitions
- Model Matchmaking: System to pair models for competitions based on their current Elo ratings.
- Competition Scheduling: Regularly scheduled competitions or on-demand challenges.
- Round-Robin or Tournament Structures: Options for different types of competition structures.

6. Performance Tracking and Analytics
- Historical Data Tracking: Store and track the performance history of each model.
- Statistical Analysis Tools: Tools for analyzing performance trends, strengths, and weaknesses.
- Leaderboards: Display current rankings and historical performance.

7. User Interface and Reporting
- Dashboard: A user-friendly dashboard to view upcoming matches, live competitions, and Elo ratings.
- Detailed Reporting: Generate detailed reports on match outcomes, individual model performance, and rating changes.
- Data Visualization: Graphs and charts for visual representation of performance trends and ratings.

8. Feedback and Improvement Loop
- Model Feedback Integration: Integrate feedback mechanisms for model improvement.
- System Update Mechanism: Regular updates to the system based on feedback, new research, and model updates.

9. Security and Fair Use
- Model Security: Ensure the security of models and their intellectual property.
- Fair Use Compliance: Ensure that the usage of all models, especially GPT-4, is in compliance with licensing and usage terms.

10. Documentation and Community Engagement
- Comprehensive Documentation: Detailed documentation on how to use the system, methodologies used, and interpretation of results.
- Community Forum: A platform for discussion, feedback, and community engagement around the Elo system and model performances.

These features collectively provide a robust framework for an Elo rating system tailored to evaluating question-answering capabilities of LLMs. The system would need to be flexible and scalable to accommodate new models and changing technologies in the AI field.

