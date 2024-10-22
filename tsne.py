import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from google.cloud import bigquery
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import datetime
import json
from scipy.stats import spearmanr
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import callback_context as ctx
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import dash_bootstrap_components as dbc
import time

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
      AVG(e) AS avg_embedding
    FROM player_team_mapping ptm
    JOIN player_embeddings pe ON ptm.player_name = pe.player_name,
    UNNEST(pe.embedding) AS e WITH OFFSET
    GROUP BY ptm.team_name, OFFSET
),
team_stats AS (
    SELECT
        team.name AS team_name,
        AVG(CASE WHEN type.name = 'Pass' THEN 1 ELSE 0 END) AS pass_ratio,
        AVG(CASE WHEN type.name = 'Shot' THEN 1 ELSE 0 END) AS shot_ratio,
        AVG(CASE WHEN type.name = 'Ball Recovery' THEN 1 ELSE 0 END) AS ball_recovery_ratio,
        AVG(CASE WHEN type.name = 'Duel' THEN 1 ELSE 0 END) AS duel_ratio,
        AVG(CASE WHEN type.name = 'Interception' THEN 1 ELSE 0 END) AS interception_ratio,
        AVG(CASE WHEN shot.outcome.name = 'Goal' THEN 1 ELSE 0 END) AS goal_ratio,
        AVG(CASE WHEN type.name = 'Pressure' THEN 1 ELSE 0 END) AS pressure_ratio,
        AVG(CASE WHEN type.name = 'Dribble' THEN 1 ELSE 0 END) AS dribble_ratio,
        AVG(CASE WHEN type.name = 'Foul Committed' THEN 1 ELSE 0 END) AS foul_committed_ratio,
        AVG(CASE WHEN type.name = 'Foul Won' THEN 1 ELSE 0 END) AS foul_won_ratio,
        AVG(CASE WHEN type.name = 'Carry' THEN 1 ELSE 0 END) AS carry_ratio,
        AVG(CASE WHEN type.name = 'Dispossessed' THEN 1 ELSE 0 END) AS dispossessed_ratio,
        AVG(CASE WHEN type.name = 'Clearance' THEN 1 ELSE 0 END) AS clearance_ratio,
        AVG(CASE WHEN type.name = 'Block' THEN 1 ELSE 0 END) AS block_ratio
    FROM `statsbomb.events`
    GROUP BY team.name
)
SELECT 
    tpe.team_name,
    ARRAY_AGG(tpe.avg_embedding) AS team_embedding,
    ts.pass_ratio,
    ts.shot_ratio,
    ts.ball_recovery_ratio,
    ts.duel_ratio,
    ts.interception_ratio,
    ts.goal_ratio,
    ts.pressure_ratio,
    ts.dribble_ratio,
    ts.foul_committed_ratio,
    ts.foul_won_ratio,
    ts.carry_ratio,
    ts.dispossessed_ratio,
    ts.clearance_ratio,
    ts.block_ratio
FROM team_player_embeddings tpe
JOIN team_stats ts ON tpe.team_name = ts.team_name
GROUP BY 
    tpe.team_name,
    ts.pass_ratio,
    ts.shot_ratio,
    ts.ball_recovery_ratio,
    ts.duel_ratio,
    ts.interception_ratio,
    ts.goal_ratio,
    ts.pressure_ratio,
    ts.dribble_ratio,
    ts.foul_committed_ratio,
    ts.foul_won_ratio,
    ts.carry_ratio,
    ts.dispossessed_ratio,
    ts.clearance_ratio,
    ts.block_ratio
"""

# Player query
player_query = """
WITH player_info AS (
    SELECT DISTINCT team.name AS team_name, player.name as player_name, position.name as position
    FROM `statsbomb.events`
),
player_stats AS (
    SELECT 
        player_name,
        pass_ratio,
        shot_ratio,
        ball_recovery_ratio,
        duel_ratio,
        interception_ratio,
        goal_ratio,
        pressure_ratio,
        dribble_ratio,
        foul_committed_ratio,
        foul_won_ratio,
        carry_ratio,
        dispossessed_ratio,
        clearance_ratio,
        block_ratio
    FROM `statsbomb.vw_player_stats_for_embeddings`
)
SELECT 
    pe.player_name, 
    pe.ml_generate_embedding_result AS embedding, 
    pi.team_name, 
    pi.position,
    ps.*
FROM `statsbomb.player_embeddings` pe
JOIN player_info pi ON pe.player_name = pi.player_name
JOIN player_stats ps ON pe.player_name = ps.player_name
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
app = dash.Dash(__name__, title="Football t-SNE", external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

current_year = datetime.datetime.now().year

# Define the layout
app.layout = html.Div([
    dcc.Store(id='filtered-data'),  # Add this line
    dcc.Store(id='filtered-team-data'),  # Add this line
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Montserrat:wght@700&display=swap'
    ),
    html.Div([
        html.Div([
            html.H1("Visualizing Embeddings using a t-SNE map", style={
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
                    html.Details([
                        html.Summary("Adjust t-SNE map settings", style={
                            'cursor': 'pointer',
                            'font-family': 'Montserrat',
                            'font-weight': '700',
                            'color': '#4285F4',
                            'margin-bottom': '10px',
                            'font-size': '18px'
                        }),
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
                        ], style={'margin-bottom': '10px'}),
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
                        ])
                    ], style={
                        'backgroundColor': '#f1f3f4',
                        'padding': '10px',
                        'borderRadius': '20px',
                        'marginBottom': '10px'
                    })
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
                html.Div([
                    html.Details([
                        html.Summary("Team Feature Importance", style={
                            'cursor': 'pointer',
                            'font-family': 'Montserrat',
                            'font-weight': '700',
                            'color': '#4285F4',
                            'margin-bottom': '10px',
                            'font-size': '18px'
                        }),
                        html.Div(id='team-feature-importance')
                    ], style={
                        'margin-top': '20px',
                        'backgroundColor': '#f1f3f4',
                        'padding': '10px',
                        'borderRadius': '20px',
                        'marginBottom': '10px'
                    })
                ], className='feature-importance-box'),
                dcc.Loading(
                    id="loading-team-insights",
                    type="default",
                    children=[html.Div(id='team-insights', style={'margin-top': '20px'})]
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

            # Player View
            html.Div([
                html.H2("Player View", style={'font-family': 'Montserrat', 'color': '#5B9BFF', 'font-weight': '700'}),
                html.Div([
                    html.Details([
                        html.Summary("Adjust t-SNE map settings", style={
                            'cursor': 'pointer',
                            'font-family': 'Montserrat',
                            'font-weight': '700',
                            'color': '#4285F4',
                            'margin-bottom': '10px',
                            'font-size': '18px'
                        }),
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
                        ], style={'margin-bottom': '10px'}),
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
                        ])
                    ], style={
                        'backgroundColor': '#f1f3f4',
                        'padding': '10px',
                        'borderRadius': '20px',
                        'marginBottom': '10px'
                    })
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
                dcc.Loading(
                    id="loading-player",
                    type="default",
                    children=[dcc.Graph(id='player-tsne-plot')]
                ),
                html.Div([
                    html.Details([
                        html.Summary("Player Feature Importance", style={
                            'cursor': 'pointer',
                            'font-family': 'Montserrat',
                            'font-weight': '700',
                            'color': '#4285F4',
                            'margin-bottom': '10px',
                            'font-size': '18px'
                        }),
                        html.Div(id='feature-importance')
                    ], style={
                        'margin-top': '20px',
                        'backgroundColor': '#f1f3f4',
                        'padding': '10px',
                        'borderRadius': '20px',
                        'marginBottom': '10px'
                    })
                ], className='feature-importance-box'),
                dcc.Loading(
                    id="loading-player-insights",
                    type="default",
                    children=[html.Div(id='player-insights', style={'margin-top': '20px'})]
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

# Define feature columns
feature_columns = ['pass_ratio', 'shot_ratio', 'ball_recovery_ratio', 'duel_ratio', 'interception_ratio',
                   'goal_ratio', 'pressure_ratio', 'dribble_ratio', 'foul_committed_ratio', 'foul_won_ratio',
                   'carry_ratio', 'dispossessed_ratio', 'clearance_ratio', 'block_ratio']

# Modify the team plot callback
@app.callback(
    [Output('team-tsne-plot', 'figure'),
     Output('filtered-team-data', 'data'),
     Output('team-insights', 'children', allow_duplicate=True)],
    [Input('team-perplexity-slider', 'value'),
     Input('team-iterations-slider', 'value'),
     Input('team-search-button', 'n_clicks'),
     Input('team-select', 'value')],
    [State('team-search', 'value')],
    prevent_initial_call='initial_duplicate'
)
def update_team_plot(perplexity, n_iter, n_clicks, selected_teams, search_team):
    global feature_columns
    
    tsne = TSNE(n_components=3, perplexity=min(perplexity, n_teams - 1), n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(team_embeddings)
    
    team_df['x'] = tsne_results[:, 0]
    team_df['y'] = tsne_results[:, 1]
    team_df['z'] = tsne_results[:, 2]
    
    # Create 3D scatter plot
    scatter_fig = px.scatter_3d(team_df, x='x', y='y', z='z', hover_name='team_name')
    scatter_fig.update_traces(marker=dict(size=5, color='#4285F4'))  # Google Blue
    scatter_fig.update_layout(
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
            scatter_fig.add_trace(highlight_trace)
    
    if selected_teams:
        team_data = team_df[team_df['team_name'].isin(selected_teams)]
        highlight_trace = px.scatter_3d(team_data, x='x', y='y', z='z', hover_name='team_name',
                                        color_discrete_sequence=['#EA4335']).data[0]  # Google Red
        highlight_trace.update(marker=dict(size=10))
        scatter_fig.add_trace(highlight_trace)
    
    # Generate insights
    insights_data = team_df[['team_name', 'x', 'y', 'z'] + feature_columns].to_dict('records')
    insights = generate_insights(insights_data, selected_teams, "team", perplexity, n_iter)
    insights_div = html.Div([
        
        dcc.Markdown(insights, dangerously_allow_html=True)
    ])
    
    return scatter_fig, team_df.to_dict('records'), insights_div

# Modify the player plot callback
@app.callback(
    [Output('player-tsne-plot', 'figure'),
     Output('filtered-data', 'data'),
     Output('player-insights', 'children', allow_duplicate=True)],
    [Input('player-perplexity-slider', 'value'),
     Input('player-iterations-slider', 'value'),
     Input('player-search-button', 'n_clicks'),
     Input('position-filter', 'value'),
     Input('team-tsne-plot', 'clickData'),
     Input('team-select', 'value')],
    [State('player-search', 'value')],
    prevent_initial_call=True
)
def update_player_plot(perplexity, n_iter, n_clicks, positions, click_data, selected_teams, search_player):
    if not selected_teams:
        return go.Figure(), [], html.Div(
            "Please select one or more teams to view player insights.",
            style={
                'font-family': 'Montserrat',
                'font-size': '18px',
                'text-align': 'center',
                'color': '#4285F4',
                'margin-top': '20px'
            }
        )

    global feature_columns
    
    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(player_embeddings)
    
    player_df['x'] = tsne_results[:, 0]
    player_df['y'] = tsne_results[:, 1]
    player_df['z'] = tsne_results[:, 2]
    
    # Filter by position if specified
    filtered_df = player_df[player_df['position'].isin(positions)] if positions else player_df
    
    # Filter by selected teams
    if click_data:
        clicked_team = click_data['points'][0]['hovertext']
        if clicked_team not in selected_teams:
            selected_teams = selected_teams + [clicked_team] if selected_teams else [clicked_team]
    
    if selected_teams:
        filtered_df = filtered_df[filtered_df['team_name'].isin(selected_teams)]
    
    fig = px.scatter_3d(filtered_df, x='x', y='y', z='z', 
                        color='position',
                        hover_name='player_name', 
                        hover_data=['team_name', 'position'],
                        color_discrete_sequence=px.colors.qualitative.Set1)
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
    
    # Generate insights
    insights_data = filtered_df[['player_name', 'team_name', 'position', 'x', 'y', 'z'] + feature_columns].to_dict('records')
    insights = generate_insights(insights_data, positions, "player", perplexity, n_iter)
    insights_div = html.Div([
        
        dcc.Markdown(insights, dangerously_allow_html=True)
    ])
    
    return fig, filtered_df[['player_name', 'team_name', 'position', 'x', 'y', 'z'] + feature_columns].to_dict('records'), insights_div

# Modify the feature importance callback
@app.callback(
    Output('feature-importance', 'children'),
    [Input('filtered-data', 'data')]
)
def update_feature_importance(filtered_data):
    global feature_columns
    # Convert the filtered data back to a dataframe
    filtered_df = pd.DataFrame(filtered_data)

    if filtered_df.empty:
        return html.Div("No data available for feature importance calculation.")

    # Extract t-SNE coordinates from the filtered dataframe
    x = filtered_df['x']
    y = filtered_df['y']
    z = filtered_df['z']

    # Calculate correlations between original features and t-SNE coordinates
    feature_columns = ['pass_ratio', 'shot_ratio', 'ball_recovery_ratio', 'duel_ratio', 'interception_ratio',
                       'goal_ratio', 'pressure_ratio', 'dribble_ratio', 'foul_committed_ratio', 'foul_won_ratio',
                       'carry_ratio', 'dispossessed_ratio', 'clearance_ratio', 'block_ratio']
    
    correlations = []
    for feature in feature_columns:
        corr_x, _ = spearmanr(filtered_df[feature], x)
        corr_y, _ = spearmanr(filtered_df[feature], y)
        corr_z, _ = spearmanr(filtered_df[feature], z)
        correlations.append((feature, max(abs(corr_x), abs(corr_y), abs(corr_z))))

    # Sort correlations by absolute value
    correlations.sort(key=lambda x: x[1], reverse=True)

    # Create a table of feature importances
    table_data = [{'Feature': feature, 'Importance': f"{importance:.3f}"} for feature, importance in correlations]

    return html.Div([
        html.H3("Feature Importance", style={'font-family': 'Montserrat', 'color': '#5B9BFF', 'font-weight': '700'}),
        html.P("The table below shows the correlation between each feature and the t-SNE coordinates. Higher values indicate stronger influence on player positioning."),
        dash_table.DataTable(
            data=table_data,
            columns=[{'name': i, 'id': i} for i in ['Feature', 'Importance']],
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ])

# Add a new callback for team feature importance
@app.callback(
    Output('team-feature-importance', 'children'),
    [Input('filtered-team-data', 'data')]
)
def update_team_feature_importance(filtered_data):
    global feature_columns
    # Convert the filtered data back to a dataframe
    filtered_df = pd.DataFrame(filtered_data)

    if filtered_df.empty:
        return html.Div("No data available for team feature importance calculation.")

    # Extract t-SNE coordinates from the filtered dataframe
    x = filtered_df['x']
    y = filtered_df['y']
    z = filtered_df['z']

    # Calculate correlations between original features and t-SNE coordinates
    feature_columns = ['pass_ratio', 'shot_ratio', 'ball_recovery_ratio', 'duel_ratio', 'interception_ratio',
                       'goal_ratio', 'pressure_ratio', 'dribble_ratio', 'foul_committed_ratio', 'foul_won_ratio',
                       'carry_ratio', 'dispossessed_ratio', 'clearance_ratio', 'block_ratio']
    
    correlations = []
    for feature in feature_columns:
        corr_x, _ = spearmanr(filtered_df[feature], x)
        corr_y, _ = spearmanr(filtered_df[feature], y)
        corr_z, _ = spearmanr(filtered_df[feature], z)
        correlations.append((feature, max(abs(corr_x), abs(corr_y), abs(corr_z))))

    # Sort correlations by absolute value
    correlations.sort(key=lambda x: x[1], reverse=True)

    # Create a table of feature importances
    table_data = [{'Feature': feature, 'Importance': f"{importance:.3f}"} for feature, importance in correlations]

    return html.Div([
        html.H3("Team Feature Importance", style={'font-family': 'Montserrat', 'color': '#5B9BFF', 'font-weight': '700'}),
        html.P("The table below shows the correlation between each team feature and the t-SNE coordinates. Higher values indicate stronger influence on team positioning."),
        dash_table.DataTable(
            data=table_data,
            columns=[{'name': i, 'id': i} for i in ['Feature', 'Importance']],
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ])

# Initialize Vertex AI
vertexai.init(project="awesome-advice-420021", location="us-central1")
model = GenerativeModel("gemini-1.5-pro")

# Generation config for Gemini
generation_config = {
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
}

def generate_insights(data, selected_items=None, item_type="team", perplexity=30, n_iter=1000):
    # Prepare the data for Gemini
    data_str = json.dumps(data, indent=2)
    
    context = f"""
    This data represents {item_type}s in a 3D space generated by t-SNE (t-Distributed Stochastic Neighbor Embedding).
    The t-SNE algorithm was run with perplexity {perplexity} and {n_iter} iterations.
    
    The x, y, and z coordinates represent the position of each {item_type} in this 3D space.
    {item_type.capitalize()}s that are closer together in this space have more similar playing styles based on the provided metrics.
    
    The other metrics are ratios representing the proportion of a {item_type}'s actions dedicated to that specific activity.
    For example, pass_ratio = passes made / total actions.
    
    Please analyze this data considering the following:
    1. Identify any clusters of {item_type}s with similar playing styles.
    2. Highlight any outliers and explain what makes them unique.
    3. Interpret the importance of different metrics in determining the {item_type}s' positions in the 3D space.
    4. If specific {item_type}s are selected, focus on explaining their position relative to others.

    Format your response in HTML, using appropriate tags for headings, paragraphs, lists, and emphasis.
    Use <h2 style="font-family: 'Montserrat'; color: '#5B9BFF';"> for main sections, <h3 style="font-family: 'Montserrat'; color: '#4285F4';"> for subsections, <p> for paragraphs, <ul> and <li> for unordered lists, and <strong> for emphasis.
    """
    
    if selected_items:
        selected_str = ", ".join(selected_items)
        prompt = f"{context}\n\nFocus on explaining the differences between the selected {item_type}s: {selected_str}. What makes them stand out from others?"
    else:
        prompt = f"{context}\n\nExplain why some {item_type}s are clustered together while others are far apart. Highlight any interesting patterns or outliers."
    
    prompt += f"\n\nData:\n{data_str}"
    
    # Generate insights using Gemini
    response = model.generate_content(prompt, generation_config=generation_config)
    
    # Add a small delay to show the loading animation
    time.sleep(1)
    
    return response.text

if __name__ == '__main__':
    app.run_server(debug=True)
