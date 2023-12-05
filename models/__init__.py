from models.hf_local_lm import AutoCausalLM
from models.gpt_online_lm import GPTOnlineLM

# Register models here, so that they can be used by name, e.g. get_model("gpt-4-turbo")
MODEL_REGIESTRY = {
    "gpt-4-turbo": GPTOnlineLM,
    "gpt-35-turbo": GPTOnlineLM,
}

def get_model(model_name):
    if model_name in MODEL_REGIESTRY:
        return MODEL_REGIESTRY[model_name](model_name)
    else:
        return AutoCausalLM(model_name)