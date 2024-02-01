"""create a chatbot to answer questions by supported online or local model based on the given context"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
from models import get_model

model_name = 'gpt2'

def get_ans(question: str, model_name: str) -> str:
    model = get_model(model_name)
    return model.generate_answer(question)

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