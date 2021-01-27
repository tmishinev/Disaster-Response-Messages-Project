import pandas as pd
import plotly.express as px  # (version 4.7.0)

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pickle
import plotly
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from custom.custom_tokens import tokenize, CustomUnpickler, top_words
from  flask_caching import Cache
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.LUX])

server = app.server
#cache = Cache(app.server, config={
    #'CACHE_TYPE': 'filesystem',
    #'CACHE_DIR': 'cache-directory'
#})

#TIMEOUT = 60

# ---------- Import and clean data (importing csv into pandas)


# load data
#@cache.memoize(timeout=TIMEOUT)
def load_data():
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterMessageETL', engine)
    df.drop(['child_alone'], inplace = True, axis = 1)
    return df


def load_model():
    # load model
    model = CustomUnpickler(open('models/classifier.pkl', 'rb')).load()
    return model

df = load_data()
model = load_model()
words_disaster = top_words(df[df['related']==1]).head(100)
words_nondisaster = top_words(df[df['related']==0]).head(100)

y = df.iloc[:, 4:]

#paper color const
BGCOLOR = "rgb(240, 240, 240)"


def distribution_charts():
    #positive labels per category 
    pos_labels = y.sum().sort_values(ascending = False)
    
    #genre distribution
    pie = df[['genre', 'related', 'id']].groupby(['genre', 'related']).count().reset_index()
    di = {0: "related", 1: "not related"}
    pie['category'] = pie['genre'] + ' - ' + pie['related'].map(di)
    pie.sort_values(by=['category'], inplace = True)

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]], column_widths=[0.54, 0.46], subplot_titles=("Percentage positive labels per category", "Message Distribution"))
    fig.add_trace(
        go.Bar(
            x=pos_labels.index,
            y=round(pos_labels/len(y)*100,1),
            opacity = 0.8
        ),
        row=1, col=1
        )

    fig.add_trace(
        go.Pie(
            values = pie['id'],
            labels = pie['category'],
            opacity = 0.8,
            hole = 0.4,
            text = pie['genre']
        ),
        row=1, col=2
        )
    fig.update_layout(plot_bgcolor = BGCOLOR, paper_bgcolor = BGCOLOR)

    return fig




# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div(children=[

    dbc.Row(dbc.Col([
                    html.Hr(),
                    html.H3( "Disaster Response Message Classification", style={'text-align': 'center', 'size' : 10}),
                    html.Hr()
                    ], width = {'size': 6, 'offset': 3 }
                    ),
            ),
    dbc.Row(
        dbc.Col(dbc.Tabs(id='tabs-example',  children=[
            dbc.Tab(label='Predict message', children = [
                            dbc.Row(
                                    dbc.Col([   html.Hr(),
                                                html.H5( "Enter message for classification", style={'text-align': 'left'}),
                                                dbc.Input(
                                                    id="message_input",
                                                    value = '',
                                                    type="text", 
                                                    debounce=True
                                                ),
                                                html.Hr()
                                            ],
                                                width = {'size': 5, 'offset': 1 }
                                            ),
                                            

                                    ),

                            dbc.Row(dbc.Col([
                                                dcc.Graph(id='my_bar_chart', figure={}, style={'backgroundColor' : 'grey'}),
                                                html.Hr()
                                            ], width = {'size': 10, 'offset': 1 }
                                            ),
                                    ),

                            ]),

            dbc.Tab(label='Explore the Dataset', children = [
                
                 dbc.Row(dbc.Col([
                                html.Hr(),
                                html.H5( "Select number of words: ", style={'text-align': 'left'}),
                                
                                dcc.Slider(
                                    id='words_slider',
                                    min=0,
                                    max=100,
                                    step=2,
                                    value=20,
                                    ),
                                    html.Hr()
                                    ],
                                    width = {'size': 5, 'offset': 1 }
                                ),
                        ),


                 dbc.Row(dbc.Col([
                                    dcc.Graph(id='bar_disaster_words', figure={}, style = {'height' : 500}), 
                                    dcc.Graph(id='bar_positive_labels', figure=distribution_charts(), style = {'height' : 600}),
                                    html.Hr()
                                ],
                                width = {'size': 10, 'offset': 1 },
                                ),
                        ),
                    
    
            ]),
            ])),
    ),
 

])

@app.callback(
    Output(component_id='bar_disaster_words', component_property='figure'),
    [Input(component_id='words_slider', component_property='value')]
)
def bar_top_words(number):

    words_dis = words_disaster.head(number)
    words_non = words_nondisaster.head(number)

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "bar"}]], column_widths=[0.5, 0.5],subplot_titles=("Top Disaster Related Words", "Top Non-Disaster Related Words"))
    fig.add_trace(
        go.Bar(
            x=words_dis['Word'],
            y=words_dis['Count'],
            opacity = 0.8
        ),
        row=1, col=1
        )

    fig.add_trace(
        go.Bar(
            x=words_non['Word'],
            y=words_non['Count'],
            opacity = 0.8
        ),
        row=1, col=2
        )
    fig.update_layout(plot_bgcolor = BGCOLOR, paper_bgcolor = BGCOLOR)
    return fig

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    Output(component_id='my_bar_chart', component_property='figure'),
    [Input(component_id='message_input', component_property='value')]
)
def update_graph(input_text):
    
    classification_label = model.predict([input_text])[0]
    print('predicting...')
        
    # Plotly Express
    fig = px.bar(
        x =y.columns,
        y=classification_label,
        labels={
                    "x": "Predicted categories",
                    "y": ""
                },
        title="Message Categories: ", height = 600)
    

    fig.update_traces(marker_color='darkgrey')
    fig.update_layout(plot_bgcolor = BGCOLOR, paper_bgcolor = BGCOLOR)
    fig.update_yaxes(showticklabels=False)
    return fig




    


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)