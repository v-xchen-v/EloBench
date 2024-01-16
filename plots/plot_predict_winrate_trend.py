import plotly.express as px
import pandas as pd


df = pd.read_csv(r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_21962_test1_smallset/predict_winrate_history.csv')
df2 = pd.read_csv(r'/elo_bench/results/google_quora_alpaca_sharegpt_chat1m_21962_test1_smallset/actual_winrate_history.csv')


# add column 'model_a' and 'model_b' to dataframe
# df['model_a'] = df['models'].apply(lambda x: x.split(' win ')[0])
# df['model_b'] = df['models'].apply(lambda x: x.split(' win ')[1])
# df['models'] = df['model_a'] + ' win ' + df['model_b']

num_row = df['model_a'].unique()
num_col = df['model_b'].unique()

fig = px.line(df,
              x='num_battles', y ='predict_winrate', 
              color='models',
              facet_row="model_a", facet_col='model_b',
              facet_col_wrap=7,
              facet_row_spacing=0.04, # default is 0.07 when facet_col_wrap is used
              facet_col_spacing=0.04, # default is 0.03
              height=1200, width=1200)

# fig.add_vline()

# add vertical line for each facet sub plot
# iterate over each facet sub plot
# fig.add_hline(y=means, line_dash="dot", row=idx[0], col=idx[1])

# for annotation in fig.layout.annotations:
#     print(annotation.text)
    
for i in range(0, len(fig.data), 1):
    models_name = fig.data[i]['name']
    model_a = models_name.split(' win ')[0]
    model_b = models_name.split(' win ')[1]
    actual = df2[(df2['models']==models_name)]['actual_winrate']#.iloc[-1]
    if len(actual) != 0:
        actual_val = actual.iloc[-1]
    else:
        actual = df2[df2['models']==(model_b + ' win ' + model_a)]['actual_winrate']#.iloc[-1]
        if len(actual) != 0:
            actual_val = 1 - actual.iloc[-1]
        else: 
            actual_val = None
    print(f'actual: {actual_val} of {models_name}')
    
    loc_idx = int(fig.data[i].xaxis[1:])
    row_index = (loc_idx-1)//len(num_row)+1
    col_index = ((loc_idx-1)%len(num_col))+1
    # print(f'{models_name} at {row_index} {col_index}')
    
    # if i >3:
    #     break
    if actual_val is not None:
        fig.add_hline(y=actual_val, line_width=1, line_dash="dash", line_color="red", row=row_index, col=col_index)
    # print(fig.data)
# for i in range(1, len(fig.data), 2):
#     fig.add_hline(y=fig.data[i]['y'][-1], line_width=1, line_dash="dash", line_color="red", row=i//2+1, col=1)
fig.add_hline(y=0.1, line_dash="dot",
              annotation_text="final actual winrate",
              annotation_position="bottom right")

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

fig.write_image('predict_winrate_trend.pdf')
