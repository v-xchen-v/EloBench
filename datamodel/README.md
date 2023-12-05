## Data Design

### Data Folder/File Structure
There are 3 core concept: question dataset, battle result and caching.
`data` folder stores questions from different dataset used to evaluate LLM QA skills. `cache` folders stored the common used data across experiments or multi-run of single experiment, that mainly are caching data of high-cost computation which includes LLM generated answers and GPT judgement. And `results` saved settings and results corresponds to single run of elo bench evaluation.

- data/
  - dataset1/
    - questions.csv
  - dataset2/
    - questions.csv
  - ...
- tempcache/ # could be anywhere else you want
  - default/
    - q_and_as.csv
    - battle_records.csv
  - caching1/ # other caching, stored separately becauce use different judger or other reasons.
    - q_and_as.csv
    - battle_records.csv
  - ...
- results
  - experiment1
    - questions.csv
    - models.csv
    - battle_arrangement.csv
    - battle_records.csv
    - battle_outcomes.csv
    - elo_rating.csv
  - experiment2
    - questions.csv
    - models.csv
    - battle_arrangement.csv
    - battle_records.csv
    - battle_outcomes.csv
    - elo_rating.csv
  - ...
  
### Data Flow
1. grab questions from dataset, preprocessed it and register models to battle:
   - <- data/xx/questions.csv <- a list of model ids
   - -> results/question.csv, results/model.csv
2. read or generate initial arrangement, generate LLM answers and battle with gpt as judger:
   - <-> results/xx/battle_arrangement.csv 
   - -> results/xx/battle_outcomes.csv -> results/xx/battle_records.csv -> results/xx/elo_rating.csv 
   - <-> cache/xx/q_and_as.csv <-> cache/xx/battle_records.csv
3. iterate more battle rounds to get enough observation: same as 2.

### Columns
| Column Name    | Data Type | Description                                                                                                  |
| -------------- | --------- | ------------------------------------------------------------------------------------------------------------ |
| question       | string    | The question to ask LLM.                                                                                     |
| model          | string    | The id/name of LLM.                                                                                          |
| model_a        | string    | The id/name of model 1 of pairwise LLM to battle facing another on the same question.                        |
| model_b        | string    | The id/name of model 2 of pairwise LLM to battle facing another on the same question.                        |
| winner         | string    | The winner model valued as one of `model_a, model_b, tie or tie(all bad)` as outcome of one pairwise battle. |
| judger         | string    | The gpt name with version, such as gpt-4-turbo.                                                              |
| tstamp         | string    | The time battle happens, format as `2023-11-23 02:56:34.433226`.                                             |
| answer_a       | string    | The answer of model_a.                                                                                       |
| answer_b       | string    | The answer of model_b.                                                                                       |
| gpt_4_response | string    | The reponse text of gpt-4 as judger to evaluate and score the better LLM.                                    |
| gpt_4_score    | string    | The scores of model_a and model_b with json text, e.g., `{'model_a': '0', 'model_b': '1'}`.                  |
| is_valid       | boolean   | The row is valid or not. Set to false, when gpt-4 reject the eval because of policy.                         |
| elo_rating     | float     | The elo rating score of LLM.                                                                                 |

### CSV Files
- data/
  - question.csv
    Each question in one row. May have repeatation.
    | question                                        |
    | ----------------------------------------------- |
    | What is the difference between OpenCL and CUDA? |
    | What is the difference between OpenCL and CUDA? |
- results/
  - question.csv
    Unique questions. Each row is one unique question to battle on.
    | question                                        |
    | ----------------------------------------------- |
    | What is the difference between OpenCL and CUDA? |
  - models.csv
    Models to play the battles.
    | model               |
    | ------------------- |
    | gpt-4-turbo         |
    | gpt-3.5-turbo       |
    | huggyllama/llama-7b |
  - battle_arrangement.csv
    | model_a             | model_b     | winner  |
    | ------------------- | ----------- | ------- |
    | huggyllama/llama-7b | gpt-4-turbo | model_a |
    | ...                 | ...         | ...     |
  - battle_records.csv
    The detailed battle outcomes.
    | model_a             | model_b              | winner  | judger      | tstamp                     | question                                                  | answer_a                            | answer_b                             | gpt_4_response                                                                                                                                           | gpt_4_score                        | is_valid |
    | ------------------- | -------------------- | ------- | ----------- | -------------------------- | --------------------------------------------------------- | ----------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | -------- |
    | huggyllama/llama-7b | huggyllama/llama-13b | mpdel_b | gpt-4-turbo | 2023-11-23 02:56:34.433226 | What are some of the best ways to manifest what you want? | "1. Be clear about what you want... | 1. Be clear about why you want it... | {'is_valid': True, 'response': ""- Score of Model 1: 0\n- Score of Model 2: 1\n- Brief Explanation: Model 1's response is repetitive and lacks depth...} | "{'model_a': '0', 'model_b': '1'}" | True     |
  - battle_outcomes.csv
    The clean battle outcomes. (is_valid == False removed)
    | model_a             | model_b              | winner  |
    | ------------------- | -------------------- | ------- |
    | huggyllama/llama-7b | huggyllama/llama-13b | model_b |
    | ...                 | ...                  | ...     |
- elo_rating.csv
    | model      | elo_rating |
    | ---------- | ---------- |
    | gpt4-turbo | 1235       |
    | ...        | ...        |
- tempcache/
  - q_and_as.csv
    columns names are 'question' and model_names
  - battle_records.csv
    same as file in results/.

