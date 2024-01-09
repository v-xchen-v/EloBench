# from __future__ import annotations
# from dataclasses import dataclass
# from elo_rating.rating_helper import get_elo_results_from_battles_data, get_bootstrap_medium_elo
# import pandas as pd
# from datamodel import BattleOutcomes

# @dataclass
# class EloLLMLeaderBoardItem:
#     """
#     Represents an item in the Elo LLM Leaderboard.

#     Attributes:
#         model (str): The name of the model.
#         rank (int): The rank of the model in the leaderboard.
#         elo_rating (float): The Elo rating of the model.
#     """
#     model: str
#     rank: int
#     elo_rating: float
    
# class BattleOutcomesBootstrap:
#     battleout
    
# class EloLLMLeaderBoard:
#     """
#             ,model,elo_rating
#         1,gpt-4-turbo,1334.0
#         2,meta-llama/Llama-2-7b-chat-hf,1161.0
#         3,meta-llama/Llama-2-70b-chat-hf,1141.0
#         4,meta-llama/Llama-2-13b-chat-hf,1137.0
#         5,Xwin-LM/Xwin-LM-13B-V0.1,1119.0
#         6,Xwin-LM/Xwin-LM-7B-V0.1,1087.0
#         7,lmsys/vicuna-33b-v1.3,1066.0
#         8,gpt-35-turbo,1016.0
#         9,lmsys/vicuna-13b-v1.5,956.0
#         10,lmsys/vicuna-7b-v1.5,950.0
#         11,chavinlo/alpaca-native,793.0
#         12,WizardLM/WizardLM-7B-V1.0,652.0
#         13,chavinlo/alpaca-13b,589.0

#     """
#     def __init__(self):
#         self.elo_leaderboard = []
#         # self.elo_leaderboard_filepath = None
#         self.use_bootstrap = None # None for unknown, True for yes, False for no
#         self.num_of_bootstrap = None # None for unknown, 0 for no bootstrap, >0 for number of bootstrap
    
#     @classmethod 
#     def _read_df(cls, elo_df: pd.DataFrame, elo_leaderboard: EloLLMLeaderBoard):
#         elo_df = elo_df.sort_values(by=['elo_rating'], ascending=False)
#         elo_df = elo_df.reset_index(drop=True)
        
#         for i, row in elo_df.iterrows():
#             elo_leaderboard.append(EloLLMLeaderBoardItem(row['model'], i+1, row['elo_rating']))
#         return elo_leaderboard
    
#     @classmethod 
#     def read_csv(cls, save_path: str):
#         elo_leaderboard = EloLLMLeaderBoard()
#         elo_leaderboard.use_bootstrap = None
#         elo_leaderboard.num_of_bootstrap = None
        
#         elo_df = pd.read_csv(save_path)
#         cls._read_df(elo_df, elo_leaderboard)
        
#         return elo_leaderboard
        
#     @classmethod
#     def generate(cls, battled_data: pd.DataFrame, K: int, use_bootstrap: bool, num_of_bootstrap: int):
#         elo_leaderboard = EloLLMLeaderBoard()
#         elo_leaderboard.use_bootstrap = use_bootstrap
#         elo_leaderboard.num_of_bootstrap = num_of_bootstrap
        
#         elo_df = None
#         if use_bootstrap:
#             elo_df = get_bootstrap_medium_elo(battled_data, K, num_of_bootstrap)
#         else:
#             elo_df = get_elo_results_from_battles_data(battled_data, K)
            
#         cls._read_df(elo_df, elo_leaderboard)
#         return elo_leaderboard
            
#     def to_csv(self, save_path: str):
#         elo_df = pd.DataFrame(columns=['model', 'elo_rating'])
#         for item in self.elo_leaderboard:
#             elo_df = elo_df.append({'model': item.model, 'elo_rating': item.elo_rating}, ignore_index=True)
#         elo_df.to_csv(save_path, index=False)
                    
#     def get_model_rank(self, model: str):
#         for item in self.elo_leaderboard:
#             if item.model == model:
#                 return item.rank
#         return -1
    
#     def get_model_elo_rating(self, model: str):
#         for item in self.elo_leaderboard:
#             if item.model == model:
#                 return item.elo_rating
#         return -1
                
#     def get_leaderboard(self):
#         return self.elo_leaderboard
    
#     def get_leaderboard_df(self):
#         df = pd.DataFrame(columns=['model', 'rank', 'elo_rating'])
#         for item in self.elo_leaderboard:
#             df = df.append({'model': item.model, 'rank': item.rank, 'elo_rating': item.elo_rating}, ignore_index=True)
#         return df
    
#     def get_leaderboard_df_html(self):
#         return self.get_leaderboard_df().to_html()
    
#     def get_leaderboard_df_markdown(self):
#         return self.get_leaderboard_df().to_markdown()

    


    

    