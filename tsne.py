import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from google.cloud import bigquery
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import datetime
import json

# Connect to BigQuery and fetch data
client = bigquery.Client()

# Team query
team_query = """
WITH player_team_mapping AS (
    SELECT DISTINCT team.name AS team_name, player.name AS player_name
    FROM `statsbomb.events`
),
player_embeddings AS (
    SELECT player_name, ml_generate_embedding_result AS embedding
    FROM `statsbomb.player_embeddings`
),
team_player_embeddings AS (
    SELECT 
      ptm.team_name,
      pe.embedding
    FROM player_team_mapping ptm
    JOIN player_embeddings pe ON ptm.player_name = pe.player_name
)
SELECT 
    team_name,
    ARRAY(
      SELECT SUM(e)
      FROM team_player_embeddings tpe,
      UNNEST(tpe.embedding) e WITH OFFSET pos
      WHERE tpe.team_name = team_player_embeddings.team_name
      GROUP BY pos
    ) AS team_embedding
FROM team_player_embeddings
GROUP BY team_name
"""

# Player query
player_query = """
WITH player_info AS (
    SELECT DISTINCT team.name AS team_name, player.name as player_name, position.name as position
    FROM `statsbomb.events`
)
SELECT pe.player_name, pe.ml_generate_embedding_result AS embedding, pi.team_name, pi.position
FROM `statsbomb.player_embeddings` pe
JOIN player_info pi ON pe.player_name = pi.player_name
"""

team_df = client.query(team_query).to_dataframe()
player_df = client.query(player_query).to_dataframe()

# Extract embeddings
team_df['team_embedding'] = team_df['team_embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
team_embeddings = np.array(team_df['team_embedding'].tolist())
player_embeddings = np.array(player_df['embedding'].tolist())

# Calculate the number of teams
n_teams = len(team_df)

# Set the maximum perplexity for teams
max_perplexity_teams = min(30, n_teams - 1)

# Initialize the Dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, title="Football t-SNE", external_stylesheets=external_stylesheets)

current_year = datetime.datetime.now().year

# Define the layout
app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Montserrat:wght@700&display=swap'
    ),
    html.Div([
        html.Div([
            html.H1("Embeddings t-SNE Visualization", style={
                'font-family': 'Montserrat', 
                'color': 'white',
                'text-align': 'center',
                'font-weight': '700',
                'margin': '0',
                'padding': '20px',
            })
        ], style={
            'background-color': '#4285F4',
            'border-radius': '20px',
            'margin-bottom': '20px',
        }),
        
        html.Div([
            # Team View
            html.Div([
                html.H2("Team View", style={'font-family': 'Montserrat', 'color': '#5B9BFF', 'font-weight': '700'}),
                html.Div([
                    html.Div([
                        html.Label("Perplexity:", style={'font-family': 'Inter', 'font-weight': '500', 'margin-right': '10px'}),
                        dcc.Slider(
                            id='team-perplexity-slider',
                            min=5,
                            max=max_perplexity_teams,
                            step=1,
                            value=min(30, max_perplexity_teams),
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='custom-slider'
                        ),
                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                    html.Div([
                        html.Label("Iterations:", style={'font-family': 'Inter', 'font-weight': '500', 'margin-right': '10px'}),
                        dcc.Slider(
                            id='team-iterations-slider',
                            min=100,
                            max=1000,
                            step=100,
                            value=700,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='custom-slider'
                        ),
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={
                    'backgroundColor': '#f1f3f4',
                    'padding': '10px',
                    'borderRadius': '20px',
                    'marginBottom': '10px'
                }),
                html.Div([
                    dcc.Input(id='team-search', type='text', placeholder='Search team', style={'margin-right': '10px'}),
                    html.Button('Search', id='team-search-button', n_clicks=0),
                ], style={'margin-bottom': '10px'}),
                html.Div([
                    dcc.Dropdown(
                        id='team-select',
                        options=[{'label': team, 'value': team} for team in sorted(team_df['team_name'].unique())],
                        multi=True,  # Allow multiple selections
                        placeholder="Select team(s)"
                    ),
                ], style={'margin-bottom': '10px'}),
                dcc.Loading(
                    id="loading-team",
                    type="default",
                    children=[dcc.Graph(id='team-tsne-plot')]
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

            # Player View
            html.Div([
                html.H2("Player View", style={'font-family': 'Montserrat', 'color': '#5B9BFF', 'font-weight': '700'}),
                html.Div([
                    html.Div([
                        html.Label("Perplexity:", style={'font-family': 'Inter', 'font-weight': '500', 'margin-right': '10px'}),
                        dcc.Slider(
                            id='player-perplexity-slider',
                            min=5,
                            max=100,
                            step=5,
                            value=45,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='custom-slider'
                        ),
                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                    html.Div([
                        html.Label("Iterations:", style={'font-family': 'Inter', 'font-weight': '500', 'margin-right': '10px'}),
                        dcc.Slider(
                            id='player-iterations-slider',
                            min=100,
                            max=1000,
                            step=100,
                            value=700,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='custom-slider'
                        ),
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={
                    'backgroundColor': '#f1f3f4',
                    'padding': '10px',
                    'borderRadius': '20px',
                    'marginBottom': '10px'
                }),
                html.Div([
                    dcc.Input(id='player-search', type='text', placeholder='Search player', style={'margin-right': '10px'}),
                    html.Button('Search', id='player-search-button', n_clicks=0),
                ], style={'margin-bottom': '10px'}),
                html.Div([
                    dcc.Dropdown(
                        id='position-filter',
                        options=[{'label': pos, 'value': pos} for pos in sorted(player_df['position'].unique())],
                        multi=True,
                        placeholder="Select positions"
                    ),
                ], style={'margin-bottom': '10px'}),
                html.Div([
                    dcc.Checklist(
                        id='show-selected-team-only',
                        options=[{'label': 'Show selected team players only', 'value': 'show_only'}],
                        value=[]
                    ),
                ], style={'margin-bottom': '10px'}),
                dcc.Loading(
                    id="loading-player",
                    type="default",
                    children=[dcc.Graph(id='player-tsne-plot')]
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '4%'}),
        ], style={'display': 'flex', 'alignItems': 'flex-start'}),  # Add this style to align the two views
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
        'max-width': '1800px',
        'margin': '0 auto',
        'padding': '20px',
        'font-family': 'Inter, sans-serif',
        'background-color': '#ffffff',
        'box-shadow': '0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15)',
        'border-radius': '20px'
    })
], style={
    'background-color': '#CEDAEC',
    'min-height': '100vh',
    'padding': '20px'
})

# Callback to update the checkbox when a team is selected
@app.callback(
    Output('show-selected-team-only', 'value'),
    [Input('team-select', 'value'),
     Input('team-tsne-plot', 'clickData')]
)
def update_checkbox(selected_teams, click_data):
    if selected_teams or (click_data and 'points' in click_data and len(click_data['points']) > 0):
        return ['show_only']
    return []

# Callback for team plot
@app.callback(
    Output('team-tsne-plot', 'figure'),
    [Input('team-perplexity-slider', 'value'),
     Input('team-iterations-slider', 'value'),
     Input('team-search-button', 'n_clicks'),
     Input('team-select', 'value')],
    [State('team-search', 'value')]
)
def update_team_plot(perplexity, n_iter, n_clicks, selected_teams, search_team):
    tsne = TSNE(n_components=3, perplexity=min(perplexity, n_teams - 1), n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(team_embeddings)
    
    team_df['x'] = tsne_results[:, 0]
    team_df['y'] = tsne_results[:, 1]
    team_df['z'] = tsne_results[:, 2]
    
    fig = px.scatter_3d(team_df, x='x', y='y', z='z', hover_name='team_name')
    fig.update_traces(marker=dict(size=5, color='#4285F4'))  # Google Blue
    fig.update_layout(
        height=750,
        font_family='Inter',
        title_font_family='Montserrat',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#CEDAEC',
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    if search_team:
        search_team_lower = search_team.lower()
        team_data = team_df[team_df['team_name'].str.lower().str.contains(search_team_lower, na=False)]
        if not team_data.empty:
            highlight_trace = px.scatter_3d(team_data, x='x', y='y', z='z', hover_name='team_name',
                                            color_discrete_sequence=['#EA4335']).data[0]  # Google Red
            highlight_trace.update(marker=dict(size=10))
            fig.add_trace(highlight_trace)
    
    if selected_teams:
        team_data = team_df[team_df['team_name'].isin(selected_teams)]
        highlight_trace = px.scatter_3d(team_data, x='x', y='y', z='z', hover_name='team_name',
                                        color_discrete_sequence=['#EA4335']).data[0]  # Google Red
        highlight_trace.update(marker=dict(size=10))
        fig.add_trace(highlight_trace)
    
    return fig

# Callback for player plot
@app.callback(
    Output('player-tsne-plot', 'figure'),
    [Input('player-perplexity-slider', 'value'),
     Input('player-iterations-slider', 'value'),
     Input('player-search-button', 'n_clicks'),
     Input('position-filter', 'value'),
     Input('team-tsne-plot', 'clickData'),
     Input('show-selected-team-only', 'value'),
     Input('team-select', 'value')],
    [State('player-search', 'value')]
)
def update_player_plot(perplexity, n_iter, n_clicks, positions, click_data, show_selected_team_only, selected_teams, search_player):
    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(player_embeddings)
    
    player_df['x'] = tsne_results[:, 0]
    player_df['y'] = tsne_results[:, 1]
    player_df['z'] = tsne_results[:, 2]
    
    # Filter by position if specified
    if positions:
        filtered_df = player_df[player_df['position'].isin(positions)]
    else:
        filtered_df = player_df
    
    # Filter by selected teams if checkbox is checked and teams are selected
    if click_data:
        clicked_team = click_data['points'][0]['hovertext']
        if clicked_team not in selected_teams:
            selected_teams = selected_teams + [clicked_team] if selected_teams else [clicked_team]
    
    if selected_teams and show_selected_team_only and 'show_only' in show_selected_team_only:
        filtered_df = filtered_df[filtered_df['team_name'].isin(selected_teams)]
    
    fig = px.scatter_3d(filtered_df, x='x', y='y', z='z', 
                        color='position',  # Color by position
                        hover_name='player_name', 
                        hover_data=['team_name', 'position'],
                        color_discrete_sequence=px.colors.qualitative.Set1)  # Use a qualitative color scheme
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        height=750,
        font_family='Inter',
        title_font_family='Montserrat',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#CEDAEC',
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Highlight players from the selected teams
    if selected_teams and not (show_selected_team_only and 'show_only' in show_selected_team_only):
        team_players = filtered_df[filtered_df['team_name'].isin(selected_teams)]
        highlight_trace = px.scatter_3d(team_players, x='x', y='y', z='z', 
                                        color='position',  # Keep color by position
                                        hover_name='player_name',
                                        hover_data=['team_name', 'position'],
                                        color_discrete_sequence=px.colors.qualitative.Set1).data[0]
        highlight_trace.update(marker=dict(size=7, line=dict(width=2, color='DarkSlateGrey')))  # Add outline to highlight
        fig.add_trace(highlight_trace)
    
    # Highlight searched player
    if search_player:
        search_player_lower = search_player.lower()
        player_data = filtered_df[filtered_df['player_name'].str.lower().str.contains(search_player_lower, na=False)]
        if not player_data.empty:
            search_trace = px.scatter_3d(player_data, x='x', y='y', z='z', 
                                         hover_name='player_name',
                                         hover_data=['team_name', 'position'],
                                         color_discrete_sequence=['#FBBC05']).data[0]  # Google Yellow
            search_trace.update(marker=dict(size=10, symbol='star'))
            fig.add_trace(search_trace)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
