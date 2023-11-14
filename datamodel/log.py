from dataclasses import dataclass

@dataclass
class LogBattlesWithGPT4AsJudger:
    model_a: str
    model_b: str
    winner: str
    # judge:
    # turn:
    # language:
    tstamp: str
    question: str
    answer_a: str
    answer_b:str
    gpt_4_reponse: str
    gpt_4_score: str