import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# create a chatbot using huggingface gradio and transformers to answer questions based on the given context
import gradio as gr
from models import get_model

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