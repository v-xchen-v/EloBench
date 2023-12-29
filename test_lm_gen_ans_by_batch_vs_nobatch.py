from models.hf_local_lm import AutoCausalLM

# model = AutoCausalLM("meta-llama/Llama-2-7b-chat-hf") # √
# model = AutoCausalLM("mosaicml/mpt-7b-chat") # x
# model = AutoCausalLM("tiiuae/falcon-7b-instruct") # x
# model = AutoCausalLM("msys/vicuna-7b-v1.5") # √
# model = AutoCausalLM("chavinlo/alpaca-native") # √
# model = AutoCausalLM("mosaicml/mpt-30b-chat") # x
# model = AutoCausalLM("WizardLM/WizardLM-7B-V1.0") # √
# model = AutoCausalLM("WizardLM/WizardLM-13B-V1.2") # √
# model = AutoCausalLM("Xwin-LM/Xwin-LM-7B-V0.1") # √
# model = AutoCausalLM("chavinlo/alpaca-13b") # √

questions = [ \
    "Does light travel forever or does it eventually fade?",
    "What is France's record in wars?",
    "Do you think Rihanna's image has influenced the fashion industry, and if so, how?",
    "Why does the United States have higher living standards than Europe?",
    'Can you name the main actors in the film "Green Book"?',
    "What are some crazy coincidences in history ?",
    "Is anything in England actually original? Tea, for example, comes from China. Fish & chips from Portugal. Football (soccer) originates in China. Gothic style architecture is French.",
    "Have you ever tried adding exotic fruits to your overnight oats?",
    "Can you name a few controversies that have involved Jeremy Clarkson?",
    "Can you tell me about the impact of the iPad 3 on the consumer electronics industry?",
    "Who are some actors that have worked with Kevin Hart in his comedy films?",
    "How does an Andrea Bocelli concert compare to other concerts you've been to?",
    "How has Katy Perry's image evolved over the years?",
    "Hi, I'm interested in learning to play badminton. Can you explain the game to me?",
]
model.batch_size = len(questions)
single_generate_answers = []
for question in questions:
    single_generate_answers.append(model.generate_answer(question, free_model_when_exit=False))
# answer = model.generate_answer(questions[0], free_model_when_exit=False)
batch_generation_answers = [x for x in model.batch_generate_answer(questions, free_model_when_exit=False)]
# print(answer)
# print(batch_generation_answers[0])
for i in range(len(questions)):
    print(single_generate_answers[i] == batch_generation_answers[i])
# print(single_generate_answers == batch_generation_answers)