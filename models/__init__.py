from models.hf_local_lm import AutoCausalLM
from models.gpt_online_lm import GPTOnlineLM

# Register models here, so that they can be used by name, e.g. get_model("gpt-4-turbo")
MODEL_REGIESTRY = {
    "gpt-4-turbo": GPTOnlineLM,
    "gpt-35-turbo": GPTOnlineLM,
}

USE_MODEL_PARALLEL = {
    "meta-llama/Llama-2-70b-chat-hf": True,
}

def get_model(model_name):
    if model_name in MODEL_REGIESTRY:
        return MODEL_REGIESTRY[model_name](model_name)
    else:
        if model_name in USE_MODEL_PARALLEL:
            return AutoCausalLM(model_name, use_model_parallel=USE_MODEL_PARALLEL[model_name])
        else:
            return AutoCausalLM(model_name)