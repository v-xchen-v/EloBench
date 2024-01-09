from elo_rating.rating_evaluator import compute_actual_winrate_awinb
from pathlib import Path
import pandas as pd
from logger import info_logger
import plotly.express as px
import os
from datamodel.elo_rating_history import EloRatingHistory
from elo_rating.rating_evaluator import compute_predict_winrate_awinb, compute_predict_winrate
from tqdm import tqdm

class NoTieBattleHistoryReport:
    def __init__(self, FIRST_N_BATTLES: int=None):
        self.battled_pairs_file = None
        self.battled_outcome_df = None
        self.winrate_history = None
        self.FIRST_N_BATTLES = FIRST_N_BATTLES
    
    def load_data(self, result_dir: str):
        self.battled_pairs_file = Path(result_dir)/'battled_pairs.csv'
        battled_pairs_df = pd.read_csv(self.battled_pairs_file, index_col=0)
        
        # filter out tie battles and invalid battles
        valid_winner = ['model_a', 'model_b']
        valid_battled_pairs_df = battled_pairs_df[battled_pairs_df['winner'].isin(valid_winner)]
        
        if self.FIRST_N_BATTLES is not None:
            valid_battled_pairs_df = valid_battled_pairs_df.head(self.FIRST_N_BATTLES)
            
        # the loaded data
        self.battled_outcome_df = valid_battled_pairs_df
        
        self.result_dir = result_dir
     
    def build(self):
        self.gen_actual_winrate_history()
        self.final_acutal_winrate = self.get_final_actual_winrate()
        pass
        self.gen_predict_winrate_history()
        
        
    def gen_predict_winrate_history(self):
        history = EloRatingHistory.gen_history(self.result_dir, use_bootstrap=
                                               False, nrows=self.FIRST_N_BATTLES, step=10)
        
        models = sorted(self.battled_outcome_df['model_a'].unique())
        info_logger.info(f'models: {models}')
        
        # elo_rating_history_df = history.to_df()
        winrate_history = []
        for i, num_battle in tqdm(enumerate(history.recorded_battle_num), desc='predicting winrate', total=len(history.recorded_battle_num)):
            winrate_as_point = compute_predict_winrate(history.get_point(num_battle))
            for model_a_i, model_a in enumerate(models):
                for model_b in models[model_a_i+1:]:
                    if model_a == model_b:
                        continue
                    if model_a not in winrate_as_point.index or model_b not in winrate_as_point.columns:
                        predict_winrate_awinb = pd.NA
                        continue
                    else:
                        predict_winrate_awinb = winrate_as_point.loc[model_a, model_b]
                    # predict_winrate_awinb = compute_predict_winrate_awinb(history.get_point(num_battle), model_a, model_b)
                    
                    # record the actual winrate
                    winrate_history.append({
                        'predict_winrate': predict_winrate_awinb,
                        'num_battles': num_battle,
                        'models': f'{model_a} win {model_b}',
                        'model_a': model_a,
                        'model_b': model_b,
                    })

        predict_winrate_history_df = pd.DataFrame(winrate_history)
            # winrate_history.append(predict_winrate_history_df)
        group_by_ab = predict_winrate_history_df.groupby(['model_a', 'model_b'])
           
        # Create a dictionary to store each group as a DataFrame
        group_dfs = {}

        self.predict_winrate_history = []
        for (model_a, model_b), group_df in group_by_ab:
            # Create a DataFrame for each group
            group_dfs[(model_a, model_b)] = group_df.reset_index(drop=True) 
            
            # 
            cur_dfs = group_dfs[(model_a, model_b)]
            cur_dfs['model_a'] = model_a
            cur_dfs['model_b'] = model_b
            # winrate_history.append(actual_winrate_history_df
            
            self.predict_winrate_history.append(cur_dfs)
        # # iterate all the groups
        # winrate_history = []
        # for (model_a, model_b), group_df in group_dfs.items():
        #     actual_winrate_history = []
        #     for i in range(1, len(group_df)):
        #         actual_winrate_awinb = compute_actual_winrate_awinb(group_df.head(i), model_a, model_b)
        #         # print(f'{i} {actual_winrate_awinb}')

        #         # record the actual winrate
        #         actual_winrate_history.append({
        #             'actual_winrate': actual_winrate_awinb,
        #             'num_battles': i,
        #             'models': f'{model_a} win {model_b}'
        #         })

        #     actual_winrate_history_df = pd.DataFrame(actual_winrate_history)
        #     actual_winrate_history_df['delta_actual_winrate'] = actual_winrate_history_df['actual_winrate'].diff().abs()
        #     actual_winrate_history_df['model_a'] = model_a
        #     actual_winrate_history_df['model_b'] = model_b
        #     winrate_history.append(actual_winrate_history_df)
            
        # # calcuated winrate history for each model pair(despite ab order)
        # self.winrate_history = winrate_history
       
        
        # self.predict_winrate_history = winrate_history    
        
    def gen_actual_winrate_history(self):
        models = sorted(self.battled_outcome_df['model_a'].unique())
        info_logger.info(f'models: {models}')
        
        # swap model_a and model_b if model_a is not the first model
        def swap_model_a_b(row, models_in_order):
            if models_in_order.index(row['model_a']) > models_in_order.index(row['model_b']):
                return row
            else:
                # swap winner if swapped model_a and model_b
                winner = 'model_b' if row['winner'] == 'model_a' else 'model_a'
                return pd.Series([row['model_b'], row['model_a'], winner], index=['model_a', 'model_b', 'winner'])
            
        # Apply the function to each row
        self.battled_outcome_df = self.battled_outcome_df.apply(swap_model_a_b, args=(models,), axis=1)
        
        group_by_ab = self.battled_outcome_df.groupby(['model_a', 'model_b'])
        
        # Create a dictionary to store each group as a DataFrame
        group_dfs = {}

        for (model_a, model_b), group_df in group_by_ab:
            # Create a DataFrame for each group
            group_dfs[(model_a, model_b)] = group_df.reset_index(drop=True)

        # Accessing a specific group, for example ('A', 'X')
        # specific_group_df = group_dfs.get(('gpt-35-turbo', 'Xwin-LM/Xwin-LM-7B-V0.1'), "Group not found")
        # print(specific_group_df)
        
        # iterate all the groups
        winrate_history = []
        self.winrate_history_dict = {}
        for (model_a, model_b), group_df in group_dfs.items():
            actual_winrate_history = []
            for i in range(1, len(group_df)):
                actual_winrate_awinb = compute_actual_winrate_awinb(group_df.head(i), model_a, model_b)
                # print(f'{i} {actual_winrate_awinb}')

                # record the actual winrate
                actual_winrate_history.append({
                    'actual_winrate': actual_winrate_awinb,
                    'num_battles': i,
                    'models': f'{model_a} win {model_b}'
                })

            actual_winrate_history_df = pd.DataFrame(actual_winrate_history)
            if len(actual_winrate_history_df)>0:
                actual_winrate_history_df['delta_actual_winrate'] = actual_winrate_history_df['actual_winrate'].diff().abs()
                actual_winrate_history_df['model_a'] = model_a
                actual_winrate_history_df['model_b'] = model_b
            winrate_history.append(actual_winrate_history_df)
            self.winrate_history_dict[(model_a, model_b)] = actual_winrate_history_df
            
        # calcuated winrate history for each model pair(despite ab order)
        self.winrate_history = winrate_history
    
    def get_final_actual_winrate(self):
        last_winrates = []
        for item in self.winrate_history:
            max_num_battles = item['num_battles'].max()
            # filter out the last point
            last_winrate = item[item['num_battles']==max_num_battles]
            last_winrates.append(last_winrate)
            
        last_winrate_df = pd.concat(last_winrates)
        return last_winrate_df     
    
    def plot(self, save_dir: str):
        self.plot_actual_winrate(save_dir)
        self.plot_predict_winrate(save_dir)
        
    def plot_predict_winrate(self, save_dir: str):
        save_dir = Path(save_dir)/'battle_history/predict_winrate'
        for his_index, his in enumerate(self.predict_winrate_history):
        # print(his)

            # plot the actual winrate
            fig = px.line(his, x="num_battles", y="predict_winrate", title=f'{his.iloc[0]["models"]}')
            # Updating line color to black
            fig.update_traces(line=dict(color='black'))
            # plot the points of samples points of actual winrate
            fig.add_scatter(x=his['num_battles'], y=his['predict_winrate'], mode='markers', name='predict_winrate')
            
            # # plot line using px.plne with specific color
            # px.line(his, x="num_battles", y="delta_actual_winrate", title=f'Delta of actual winrate for {his.iloc[0]["models"]}', color='red', add_trace=True)
            # fig.add_scatter(x=his['num_battles'], y=his['delta_actual_winrate'], mode='lines', name='delta_actual_winrate')
            
            # plot a horizontal line at 0.5
            # get actual_winrate of model_
            model_a = his.iloc[0]['model_a']
            model_b = his.iloc[0]['model_b']
            
            ab_swapped = False
            winrate_records = self.final_acutal_winrate[(self.final_acutal_winrate['model_a']==model_a) & (self.final_acutal_winrate['model_b']==model_b)]
            if len(winrate_records) == 0:
                ab_swapped = True
                winrate_records = self.final_acutal_winrate[(self.final_acutal_winrate['model_a']==model_b) & (self.final_acutal_winrate['model_b']==model_a)]

            if len(winrate_records) == 0:
                print("no actual winrate found")
            else:
                final_actual_winrate = winrate_records.iloc[0]['actual_winrate']
                if ab_swapped:
                    final_actual_winrate = 1-final_actual_winrate
                fig.add_hline(y=final_actual_winrate, line_dash="dot", annotation_text="actual winrate", annotation_position="bottom right")
            # pair_key = (model_a, model_b)
            # if (model_a, model_b) not in self.winrate_history_dict:
            #     pair_key = (model_b, model_a)
                
            #     actual_winrate = self.winrate_history_dict[pair_key].sort_values(by='num_battles').iloc[-1]['actual_winrate']
            #     actual_winrate = 1-actual_winrate
            #     fig.add_hline(y=actual_winrate, line_dash="dot", annotation_text="actual winrate", annotation_position="bottom right")
            # else:
            #     if pair_key in self.winrate_history_dict:
            #         actual_winrate = self.winrate_history_dict[pair_key].sort_values(by='num_battles').iloc[-1]['actual_winrate']
            #         fig.add_hline(y=actual_winrate, line_dash="dot", annotation_text="actual winrate", annotation_position="bottom right")
            #     else:
            #         print("no actual winrate found")
                
            
            # y range to [0, 1]
            fig.update_yaxes(range=[0, 1])
            
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
            fig.write_image(f'{save_dir}/predict_winrate_his_{his_index}.png')
            fig.write_html(f'{save_dir}/predict_winrate_his_{his_index}.html')

        
    def plot_actual_winrate(self, save_dir: str):
        save_dir = Path(save_dir)/'battle_history/actual_winrate'
        for his_index, his in enumerate(self.winrate_history):
        # print(his)
            if len(his) == 0:
                continue
            # plot the actual winrate
            fig = px.line(his, x="num_battles", y="actual_winrate", title=f'{his.iloc[0]["models"]}')
            # Updating line color to black
            fig.update_traces(line=dict(color='black'))
            
            # # plot line using px.plne with specific color
            # px.line(his, x="num_battles", y="delta_actual_winrate", title=f'Delta of actual winrate for {his.iloc[0]["models"]}', color='red', add_trace=True)
            fig.add_scatter(x=his['num_battles'], y=his['delta_actual_winrate'], mode='lines', name='delta_actual_winrate')
            
            # # plot the moving std
            # fig.add_scatter(x=his['num_battles'], y=his['moving_std'], mode='lines', name='moving_std')
            
            # # plot the moving average
            # fig.add_scatter(x=his['num_battles'], y=his['moving_average'], mode='lines', name='moving_average')
            
            # # plot the delta of moving average
            # fig.add_scatter(x=his['num_battles'], y=his['delta_moving_average'], mode='lines', name='delta_moving_average')
            
            # # plot filling between moving average + moving std and moving average - moving std
            # from matplotlib import pyplot as plt
            # plt.fill_between(his['num_battles'], his['moving_average']-his['moving_std'], his['moving_average']+his['moving_std'], alpha=0.2)
            
            
            # # find the stable point
            # if len(his[his['is_stable']])>0:
            #     stable_point = his[his['is_stable']].iloc[-1]
            #     # treat the found stable point as the last point of a stable range, find the first point of the stable range
            #     for i in range(stable_point['num_battles']-1, 0, -1):
            #         if his.iloc[i]['is_stable'] == False:
            #             stable_point = his.iloc[i+1]
            #             break
            #     # print(f'stable point found at {stable_point}')
            #     fig.add_scatter(x=[stable_point['num_battles']], y=[stable_point['actual_winrate']], mode='markers', name='stable_point')
            # else:
            #     print(f'no stable point found for {his.iloc[0]["models"]}')
            # fig.show()
            # save_dir = r'/elo_bench/results/google_quora_alpaca_10629_test3/plots'
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
            fig.write_image(f'{save_dir}/actual_winrate_his_{his_index}.png')
            fig.write_html(f'{save_dir}/actual_winrate_his_{his_index}.html')
