import plotly.express as px
import pandas as pd


df = pd.read_csv(r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_21962_test1_smallset/actual_winrate_history.csv')

# add column 'model_a' and 'model_b' to dataframe
df['model_a'] = df['models'].apply(lambda x: x.split(' win ')[0])
df['model_b'] = df['models'].apply(lambda x: x.split(' win ')[1])
# df['models'] = df['model_a'] + ' win ' + df['model_b']

fig = px.line(df,
              x='num_battles', y ='actual_winrate', 
              color='models',
              facet_row="model_a", facet_col='model_b',
              facet_col_wrap=7,
              facet_row_spacing=0.04, # default is 0.07 when facet_col_wrap is used
              facet_col_spacing=0.04, # default is 0.03
              height=1200, width=1200)

# smaller the line size
fig.update_traces(line=dict(width=1))

# Adjust the font size of facet subplot titles
for annotation in fig.layout.annotations:
    annotation.font.size = 8 # Adjust the font size as needed

# # Hide individual y-axis titles
# for axis in fig.layout:
#     if axis.startswith('yaxis') or axis.startswith('xaxis'):
#         fig.layout[axis].title.text = ''
        
# Update the font size of the legend title and items
fig.update_layout(legend_title_font_size=12, legend_font_size=10)

# Update the font size for the x and y-axis titles
fig.update_xaxes(title_font=dict(size=12))
fig.update_yaxes(title_font=dict(size=12))

# hide the legend
fig.update_layout(showlegend=False)

fig.show()

fig.write_image('winrate_trend.pdf')
