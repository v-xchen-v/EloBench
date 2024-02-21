## pipe design changes
initial arrangement:
arrangment -> battle

iterative battle arrangement until target condiation:
for new added players(LM models):
arrange -> battle -> arrangement -> battle

### Framework design
#### Design of Battle Pipeline
Three senarios:
1. Fixed questions and models
2. + question later
3. + models later

For the fixed questions and models:

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

### TODOs:
