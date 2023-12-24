import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# create a chatbot using huggingface gradio and transformers to answer questions based on the given context
import gradio as gr
from models import get_model
from transformers import AutoTokenizer, AutoModelForCausalLM

# models = [ \
#    'lmsys/vicuna-7b-v1.5',
#    'lmsys/vicuna-13b-v1.5',
#    'lmsys/vicuna-33b-v1.3',
#    'meta-llama/Llama-2-7b-chat-hf',
#    'meta-llama/Llama-2-13b-chat-hf',
#    'meta-llama/Llama-2-70b-chat-hf',
#    'chavinlo/alpaca-native',
#    'chavinlo/alpaca-13b',
#    # 'gpt-4-turbo',
#    # 'gpt-35-turbo',
#    'mosaicml/mpt-7b-chat',
#    'mosaicml/mpt-30b-chat',
#    'WizardLM/WizardLM-7B-V1.0',
#    'WizardLM/WizardLM-13B-V1.2',
#    # 'WizardLM/WizardLM-70B-V1.0',
#    'Xwin-LM/Xwin-LM-7B-V0.1',
#    'Xwin-LM/Xwin-LM-13B-V0.1',
#    'tiiuae/falcon-7b-instruct',
#    'tiiuae/falcon-40b-instruct'
#]

# for model in models:
#    tokenizer = AutoTokenizer.from_pretrained(model)
#    print(model, tokenizer.padding_side)
#    print(tokenizer.pad_token)
    
# model_name = 'meta-llama/Llama-2-7b-chat-hf'
# model_name = 'Xwin-LM/Xwin-LM-13B-V0.1'
# Can you tell me about any unique food and drink pairings you've tried at Jimmy Johns?
# What are some of the major themes explored in the TV show "Roseanne"?

# model = get_model(model_name)

#def get_ans(question: str, model_name: str) -> str:
#    # return model.generate_answer(question)
#    questions = [ \
#        "Does light travel forever or does it eventually fade?",
#        "What is France's record in wars?",
#        "Do you think Rihanna's image has influenced the fashion industry, and if so, how?",
#        "Why does the United States have higher living standards than Europe?",
#        'Can you name the main actors in the film "Green Book"?',
#        "What are some crazy coincidences in history ?",
#        "Is anything in England actually original? Tea, for example, comes from China. Fish & chips from Portugal. Football (soccer) originates in China. Gothic style architecture is French.",
#        "Have you ever tried adding exotic fruits to your overnight oats?",
#        "Can you name a few controversies that have involved Jeremy Clarkson?",
#        "Can you tell me about the impact of the iPad 3 on the consumer electronics industry?",
#        "Who are some actors that have worked with Kevin Hart in his comedy films?",
#        "How does an Andrea Bocelli concert compare to other concerts you've been to?",
#        "How has Katy Perry's image evolved over the years?",
#        "Hi, I'm interested in learning to play badminton. Can you explain the game to me?",
#    ]
#    model.batch_size=len(questions)
#    ans = []
#    for a in model.batch_generate_answer(questions):
#        ans.append(a)
#    return '\n'.join(ans)

model_name = 'meta-llama/Llama-2-7b-chat-hf'

def get_ans(question: str, model_name: str) -> str:
    model = get_model(model_name)
    return model.generate_answer(question)
    # ans = []
    # for a in model.batch_generate_answer([question]*10):
    #     ans.append(a)
    # return ans[0]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    
    def respond(message, chat_history):
        bot_message = get_ans(message, model_name)
        chat_history.append([message, bot_message])
        
        return "", chat_history
    
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    
if __name__ == '__main__':
    demo.launch()