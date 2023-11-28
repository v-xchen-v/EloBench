from models.huggingface import AutoCausalLM
from models.gpt_lm import GPT4LM

MODEL_REGIESTRY = {
    "gpt-4-turbo": GPT4LM,
    "gpt-35-turbo": GPT4LM,
}

def get_model(model_name):
    if model_name in MODEL_REGIESTRY:
        return MODEL_REGIESTRY[model_name](model_name)
    else:
        return AutoCausalLM(model_name)