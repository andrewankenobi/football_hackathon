import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from google.cloud import bigquery
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import datetime

# Connect to BigQuery and fetch data
client = bigquery.Client()
query = """
SELECT player_name, ml_generate_embedding_result
FROM `statsbomb.player_embeddings`
"""
df = client.query(query).to_dataframe()

# Extract embeddings
embeddings = np.array(df['ml_generate_embedding_result'].tolist())

# Initialize the Dash app
app = dash.Dash(__name__, title="Player TSNE")

current_year = datetime.datetime.now().year

# Define the layout
app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Bungee&display=swap'
    ),
    html.Div([
        html.H1("Player Embeddings t-SNE Visualization", style={
            'font-family': 'Bungee', 
            'color': '#4285F4', 
            'text-align': 'center'
        }),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Perplexity:", style={
                        'font-family': 'Inter', 
                        'font-weight': '500', 
                        'margin-right': '10px',
                        'display': 'inline-block',
                        'width': '80px'
                    }),
                    html.Div(dcc.Slider(
                        id='perplexity-slider',
                        min=5,
                        max=100,
                        step=5,
                        value=45,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode='drag',
                        className='custom-slider'
                    ), style={'display': 'inline-block', 'width': 'calc(100% - 90px)'}),
                ], style={
                    'background-color': '#f1f3f4',
                    'padding': '10px',
                    'border-radius': '8px',
                    'margin-bottom': '10px',
                }),
                
                html.Div([
                    html.Label("Iterations:", style={
                        'font-family': 'Inter', 
                        'font-weight': '500', 
                        'margin-right': '10px',
                        'display': 'inline-block',
                        'width': '80px'
                    }),
                    html.Div(dcc.Slider(
                        id='iterations-slider',
                        min=100,
                        max=1000,
                        step=100,
                        value=700,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode='drag',
                        className='custom-slider'
                    ), style={'display': 'inline-block', 'width': 'calc(100% - 90px)'}),
                ], style={
                    'background-color': '#f1f3f4',
                    'padding': '10px',
                    'border-radius': '8px',
                    'margin-bottom': '10px',
                }),
            ], style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top'}),
            
            html.Div([
                html.Div([
                    html.Label("Search Player:", style={
                        'font-family': 'Inter', 
                        'font-weight': '500', 
                        'margin-bottom': '10px',
                        'display': 'block'
                    }),
                    dcc.Input(id='player-search', type='text', placeholder='Enter player name', style={
                        'font-family': 'Inter',
                        'border': '1px solid #dadce0',
                        'border-radius': '4px',
                        'padding': '8px',
                        'width': 'calc(100% - 18px)',
                        'margin-bottom': '10px'
                    }),
                    html.Button('Search', id='search-button', n_clicks=0, style={
                        'font-family': 'Inter',
                        'background-color': '#1a73e8',
                        'color': 'white',
                        'border': 'none',
                        'border-radius': '4px',
                        'padding': '8px 16px',
                        'cursor': 'pointer',
                        'width': '100%'
                    }),
                ], style={
                    'background-color': '#f1f3f4',
                    'padding': '10px',
                    'border-radius': '8px',
                }),
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '5%'}),
        ], style={'margin-bottom': '20px'}),
        
        dcc.Graph(id='tsne-plot'),
        
        html.Div([
            html.P(f"© {current_year} Made by andrewankenobi@google.com", style={
                'font-family': 'Inter',
                'color': '#5f6368',
                'font-size': '12px',
                'text-align': 'center',
                'margin-top': '20px'
            })
        ])
    ], style={
        'max-width': '1200px',
        'margin': '0 auto',
        'padding': '20px',
        'font-family': 'Inter, sans-serif',
        'background-color': '#ffffff',
        'box-shadow': '0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15)',
        'border-radius': '8px'
    })
], style={
    'background-color': '#CEDAEC',
    'min-height': '100vh',
    'padding': '20px'
})

# Add this to your app.py file, after the layout definition
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

# Add these custom styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .custom-slider .rc-slider-track {
                background-color: #B8C7E0;
            }
            .custom-slider .rc-slider-handle {
                border-color: #A4B7D7;
                background-color: #A4B7D7;
            }
            .custom-slider .rc-slider-rail {
                background-color: #DFE5F0;
            }
            .custom-slider .rc-slider-handle:hover {
                border-color: #8FA8CC;
            }
            .custom-slider .rc-slider-handle:active {
                border-color: #8FA8CC;
                box-shadow: 0 0 5px #8FA8CC;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Callback to update the plot
@app.callback(
    Output('tsne-plot', 'figure'),
    [Input('perplexity-slider', 'value'),
     Input('iterations-slider', 'value'),
     Input('search-button', 'n_clicks')],
    [State('player-search', 'value')]
)
def update_plot(perplexity, n_iter, n_clicks, search_player):
    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    df['x'] = tsne_results[:, 0]
    df['y'] = tsne_results[:, 1]
    df['z'] = tsne_results[:, 2]
    
    fig = px.scatter_3d(df, x='x', y='y', z='z', hover_name='player_name')
    fig.update_traces(marker=dict(size=5, color='#4285F4'))  # Google Blue
    fig.update_layout(
        height=750,
        font_family='Inter',
        title_font_family='Bungee',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#CEDAEC',
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    if search_player:
        search_player_lower = search_player.lower()
        player_data = df[df['player_name'].str.lower().str.contains(search_player_lower, na=False)]
        if not player_data.empty:
            highlight_trace = px.scatter_3d(player_data, x='x', y='y', z='z', hover_name='player_name',
                                            color_discrete_sequence=['#EA4335']).data[0]  # Google Red
            highlight_trace.update(marker=dict(size=10))
            fig.add_trace(highlight_trace)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
