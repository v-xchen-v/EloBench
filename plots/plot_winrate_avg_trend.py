import plotly.express as px
import pandas as pd


df = pd.read_csv(r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_21962_test1_smallset/actual_winrate_history.csv')

# add column 'model_a' and 'model_b' to dataframe
# df['model_a'] = df['models'].apply(lambda x: x.split(' win ')[0])
# df['model_b'] = df['models'].apply(lambda x: x.split(' win ')[1])
# df['models'] = df['model_a'] + ' win ' + df['model_b']

# calcuate the average of delta winrate across models with same num_battles
df_new = df[['delta_actual_winrate', 'num_battles']].groupby(['num_battles']).mean().reset_index()


fig = px.line(df_new, y='delta_actual_winrate', x='num_battles', height=600, width=600)
# set the y axis title
fig.update_yaxes(title_text='Average delta winrate')

fig.show()
fig.write_image('delta_winrate_trend.pdf')