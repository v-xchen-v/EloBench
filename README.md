# Elo Bench

## About the Project
<!-- Briefly introduce your project. Explain the context and the problem your research addresses. You should provide a clear connection between the project and the associated research paper. -->

We provide a robust framework for an Elo rating system tailored to evaluating question-answering capabilities of LLMs. The system is flexible and scalable to accommodate new models in the AI field. We used this system to engaging 24 LLMs such as GPT-4, GPT-3.5, Google-Gemini-Pro and LLaMA-1/-2, in a two-player competitive format, with GPT-4 serving as the judge to mirror real-world usage.
And multiple-turn QA is not in scope.

## Research Paper
- *Title*: Rethinking Generative Large Language Model Evaluation for
Semantic Comprehension
<!-- - Authors: [Author Names] -->
<!-- - Published In: [Journal/Conference Name, Year] -->
- *Abstract*: Despite their sophisticated capabilities, large language models (LLMs) encounter a major hurdle in effective assessment. This paper first revisits the prevalent evaluation method—multiple choice question answering (MCQA), which allows for straightforward accuracy measurement. Through a comprehensive evaluation of 24 models across 11 benchmarks, we highlight several potential drawbacks of MCQA, for instance, the inconsistency between the MCQA evaluation and the generation of open-ended responses in practical scenarios. In response, we introduce an RWQ-Elo rating system, engaging 24 LLMs such as GPT-4, GPT-3.5, Google-Gemini-Pro and LLaMA-1/-2, in a two-player competitive format, with GPT-4 serving as the judge. Each LLM receives an Elo rating thereafter. This system is designed to mirror real-world usage, and for this purpose, we have compiled a new benchmark called Real-world questions (RWQ), comprising 20,772 authentic user inquiries. Additionally, we thoroughly analyze the characteristics of our system and compare it with prior leaderboards like AlpacaEval and MT-Bench. Our analysis reveals the stability of our RWQ-Elo system, the feasibility of registering new models, and its potential to reshape LLM leaderboards.

<!-- - Link to Paper: [URL to the paper, if available online] -->

## Getting Started
<!-- List the software, libraries, and tools needed to run your project. Provide versions and any other relevant details.
Example: -->
Requirements:

- Python 3.9+
- NumPy 1.23+
- Torch 1.13
- transformers 4.36+
- datasets 2.16+
- accelerate 0.22+

<!-- Or you can use the docker image provided: [link] -->

## Installation

Step-by-step instructions on setting up the project locally.

Example:

Clone the repo:
```
git clone https://github.com/microsoft/EloBench.git
```
Install required packages:
```
pip install -r requirements.txt
```


## Basic Usage
<!-- Provide example on how to run script or use the software. Include any necessary commands. -->

To conduct battles between models hosted on the HuggingFace Hub or Azure OpenAI Service (e.g., meta-llama/Llama-2-7b-chat-hf, HuggingFaceH4/zephyr-7b-beta, ...) on RWQ questiosn and get elo leaderboard, you can following the steps:

1. Create a workspace folder to store the battle settings and results:
    ```
    cd /elo_bench
    mkdir /results/experiment1
    mkdir /tempcache
    ```
2. Register the models to evaluate:

    Create a file named `models.csv` with a column `model`, each row is a model to register as a LLM player in ELO system.
    
    ```
    ,model
    1,lmsys/vicuna-7b-v1.5
    2,meta-llama/Llama-2-7b-chat-hf
    3,meta-llama/Llama-2-13b-chat-hf
    4,chavinlo/alpaca-native
    5,chavinlo/alpaca-13b
    6,HuggingFaceH4/zephyr-7b-beta
    7,huggyllama/llama-7b
    ```
3. Register the questions to evaluate:

    The models will generate answers for this preset questions and pairwisely compare with GPT-4 as judger.
    
    Create a file named `questions.csv` with a column `question`, each row is a question will be randomly pick up to compare the registered models' QA ability.

    ```
    ,question
    0,How does the scoring system in Wordle work?
    1,Can you explain the popularity of Wordle and why it's so addictive?
    2,What are some tips for beginners trying to master Wordle?
    3,"Is their a way to cheat on Wordle, or is it all about your search skills?"
    4,Can you tell me some interesting facts about Queen Elizabeth that most people don't know?
    5,What are some of the major events that have occurred during Queen Elizabeth's reign?

    ```
4. Run evaluation by one-line command.
    
    Command format:
    ```
    python run_experiment.py --experiment_dir your_experiment_directory --cache_dir your_cache_directory -n your_notie_battle_target_n
    ```
    e.g.,
    ```
    python run_experiment.py --experiment_dir /elo_bench/results/experiment1 --cache_dir /elo_bench/tempcache -n 500
    ```

Noted: If you have OpenAI online model to eval, set the OpenAI API key before running.
```
export OPENAI_API_KEY=[your_own_api_key]
```

To generate elo leaderboard on the battles result, run one-line command as following:

Command Format:
```
python run_analysis.py -b your_experiment_directory -n bootstrap_round_num_you_want
```
e.g.,
```
python run_analysis.py -b /elo_bench/results/experiment1 -n 100
```

## Model List
The listed 24 LLMs evaluated in our work are divided as follows: Pre-existing Gemini Answers are imported into the answer pool, while the remainder are automatically generated using our framework.

And integrate a new Hugging Face model into this evaluation framework without needing to write new code.
> When evaluating the OpenAI online model e.g, gpt-3.5, gpt-4, you could config your endpoint, version as you need in openai_chat.py.

| model |
| --- |
| gpt-4-turbo |
| gpt-35-turbo |
| lmsys/vicuna-7b-v1.5 |
| lmsys/vicuna-13b-v1.5 |
| lmsys/vicuna-33b-v1.3 |
| meta-llama/Llama-2-7b-chat-hf |
| meta-llama/Llama-2-13b-chat-hf |
| meta-llama/Llama-2-70b-chat-hf |
| chavinlo/alpaca-native |
| chavinlo/alpaca-13b |
| mosaicml/mpt-7b-chat |
| mosaicml/mpt-30b-chat |
| WizardLM/WizardLM-7B-V1.0 |
| WizardLM/WizardLM-13B-V1.2 |
| Xwin-LM/Xwin-LM-7B-V0.1 |
| Xwin-LM/Xwin-LM-13B-V0.1 |
| tiiuae/falcon-7b-instruct |
| tiiuae/falcon-40b-instruct |
| HuggingFaceH4/zephyr-7b-beta |
| huggyllama/llama-7b |
| huggyllama/llama-13b |
| huggyllama/llama-30b |
| gemini |
| mistralai/Mixtral-8x7B-Instruct-v0.1 |

## Features
1. *Model Integration*
    - *LLM Interface*: Interface for integrating various LLMs, including open-source local models and OpenAI chat online models, to ensure smooth interaction and response handling.
    - *Model Configuration*: Allow configuration setting for each model(e.g., token limits, temperature settings for GPT-4)
    - *Batch Mode*: Support batch mode for partial local models and online chat model for faster question answerng.
    - *Huge Model Inference*: Support turn on model parallel for huge model inference and answering powered by HuggingFace transformers and accelerate library.

2. *Question Pool Management*
    - *Question Database*:  A diverse and extensive database of questions.
    - Randomized Question Selection: Mechanism for selection questions randomly to ensure a fair and unbais challenge for each model.

3. *Answer Assessment*
    - *Answer Evaluation Criteria*: Define a clear and objective criteria for what consititutes a correct or superior answer.
    - Automated Answer Judging: Using GPT-4 to evaluate answers with predefined metric for fairness and accuracy.

4. Elo Rating System
    - Initial Rating Assignment: Assign initial Elo ratings to all participating models.
    - Rating Update Mechanism: Algorithm to update Elo ratings based on match outcomes, ensuring fair and accurate reflection of performance.

5. Matchmaking and Competitions
    - Model Matchmaking: System to pair models for competitions based on different strategies.
    - Competition Scheduling: Shuffle Model A/B and battle order to ensure a fair and unbais challenge.
    - Iterative Battle Arrangement: Random arrange more battle by battle frequency per player pair, until battle count of each pair reach the target number. 
    - New questions and LLM models registration: Allow new questions and new models to register later.

6. Caching
    - Caching gpt-4 judgement and LLM answer to avoid waste of time and computation.
    - resume and continue when battle with gp4-4 as judger

<!-- 7. Performance Tracking and Analytics
    - Historical Data Tracking: Store and track the performance history of each model.
    - Statistical Analysis Tools: Tools for analyzing performance trends.
    - Leaderboards: Display current rankings and historical performance.

8. User Interface and Reporting
    - Dashboard: A user-friendly dashboard to view upcoming matches, live competitions, and Elo ratings.
    - Detailed Reporting: Generate detailed reports on match outcomes, individual model performance, and rating changes.
    - Data Visualization: Graphs and charts for visual representation of performance trends and ratings. -->

These features collectively provide a robust framework for an Elo rating system tailored to evaluating question-answering capabilities of LLMs. The system is flexible and scalable to accommodate new models and changing technologies in the AI field.


## Additional Resources
<!-- Link to any additional resources like datasets, extended documentation, or related projects. -->

### datasets

RWQ-questions: [dataset_link](https://huggingface.co/datasets/rwq-elo/rwq-questions)

RWQ-answers: [dataset_link](https://huggingface.co/datasets/rwq-elo/rwq-answers)

RWQ-battle-records: [dataset_link](https://huggingface.co/datasets/rwq-elo/rwq-battle-records)


<!-- ### ablation test:
swtich the order of battle pairs, model a, model b <-> model b model a, the same results?

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
    - *More*: battle ordering and so on.

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
    - *Insights Generation*: Analyze rsults for insights into each model's performance.

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
    - *Bias and Fairness*: Regularly assess the system for any biases or unfair practices. -->

<!-- ## Considerations
- Resource Intensive: Running multiple LLMs and GPT-4 for judging can be resource-intensive. Plan for the necessary computational resources.
- Model Limitations: Be aware of the limitations of each model, including GPT-4, and how these might impact the fairness of the competition.
- Continuous Monitoring: The system should be monitored and adjusted as models evolve and improve over time.

## [Data Design](./datamodel/README.md)

### Solved Issues
#### Empty LLM generated answer handling
- use 'Question: {question}\nAnswer: ' formating question as prompt to reduce the frequency of generating empty answer, especially for alpaca-7b/13b and vicuna-7b models.
- save missing value(not generated) as 'NULL', and empty answer as ''. -->
