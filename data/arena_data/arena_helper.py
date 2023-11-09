# https://colab.research.google.com/drive/1RAWb22-PFNI-X1gPVzc927SGUdfr6nsR?usp=sharing#scrollTo=2IdpT27Q8IE_ where 'clean_battle_20230717.json' come from.

import pandas as pd
import os

ARENA_BATTLE_20230717 = os.path.join(os.path.dirname(__file__),'clean_battle_20230717.json')
ARENA_K=4

def get_arena_battles_20230717_data() -> pd.DataFrame:
    raw_data = pd.read_json(ARENA_BATTLE_20230717).sort_values(ascending=True, by=["tstamp"])
    # print(raw_data)

    # print("anony battle counts:", raw_data['anony'].value_counts())
    battles = raw_data[raw_data['anony']].reset_index(drop=True)
    # print(battles)
    return battles

def list_arena_battles_20230717_models() -> list[str]:
    raw_data = pd.read_json(ARENA_BATTLE_20230717).sort_values(ascending=True, by=["tstamp"])
    battles = raw_data[raw_data['anony']].reset_index(drop=True)
    models = pd.concat([battles['model_a'], battles['model_b']]).unique()
    return models

def get_arena_elo_res_20230717() -> dict:
    offcial_20230717_elo_dict = [
        {
            "Model": "claude-v1",
            "Elo rating": 1201,
        },
        {
            "Model": "gpt-4",
            "Elo rating": 1185,
        },
        {
            "Model": "gpt-3.5-turbo",
            "Elo rating": 1158,
        },
        {
            "Model": "claude-instant-v1",
            "Elo rating": 1138,
        },
        {
            "Model": "vicuna-33b",
            "Elo rating": 1088,
        },
        {
            "Model": "wizardlm-13b",
            "Elo rating": 1032,
        },
        {
            "Model": "mpt-30b-chat",
            "Elo rating": 1026,
        },
        {
            "Model": "vicuna-7b",
            "Elo rating": 1024,
        },
        {
            "Model": "guanaco-33b",
            "Elo rating": 1023,
        },
        {
            "Model": "vicuna-13b",
            "Elo rating": 1021,
        },
        {
            "Model": "palm-2",
            "Elo rating": 989,
        },
        {
            "Model": "koala-13b",
            "Elo rating": 974,
        },
        {
            "Model": "RWKV-4-Raven-14B",
            "Elo rating": 953,
        },
        {
            "Model": "mpt-7b-chat",
            "Elo rating": 947,
        },
        {
            "Model": "gpt4all-13b-snoozy",
            "Elo rating": 941,
        },
        {
            "Model": "chatglm-6b",
            "Elo rating": 928,
        },
        {
            "Model": "alpaca-13b",
            "Elo rating": 924,
        },
        {
            "Model": "oasst-pythia-12b",
            "Elo rating": 920,
        },
        {
            "Model": "stablelm-tuned-alpha-7b",
            "Elo rating": 913,
        },
        {
            "Model": "fastchat-t5-3b",
            "Elo rating": 884,
        },
        {
            "Model": "llama-13b",
            "Elo rating": 865,
        },
        {
            "Model": "dolly-v2-12b",
            "Elo rating": 864,
        },
    ]
    return offcial_20230717_elo_dict

if __name__ == '__main__':    
    print(list_arena_battles_20230717_models())