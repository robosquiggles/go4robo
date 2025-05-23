import sys
import os

import dash
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import dash_bootstrap_components as dbc

from . import bot_3d_problem, bot_3d_rep

import webbrowser

import plotly.express as px
import pandas as pd

# import bot_3d_problem, bot_3d_rep
import glob
import os
import subprocess
import dill

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go

import base64
from io import BytesIO, StringIO
import sys

matplotlib.use('agg')
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'Arial' 

df = pd.DataFrame()

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, 
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css],
           )
url = "http://127.0.0.1:8050/"
server = app.server

img_width = 400

dash_df_page_size = 25

@app.callback(
    Output('pop-df-table', 'data'),
    Output('pop-df-table', 'columns'),
    Output('pop-df-table', 'style_data_conditional'),
    Output('tradespace-plot', 'figure'),
    Output('selected-bot-plot', 'figure'),
    Input('pop-df-store', 'data'),
    Input('pop-df-table', 'active_cell'),
    State('problem-store', 'data')
)
def update_table_and_plot(
        pop_df_json, 
        active_cell, 
        problem_json
    ):
    # If no population yet, return all empty/placeholder figures
    if pop_df_json is None:
        empty_fig = go.Figure()
        return [], [], [], empty_fig, empty_fig

    # Reconstruct the DataFrame from JSON store
    df = pd.read_json(StringIO(pop_df_json), orient='split')

    # Build the DataTable data & columns as before
    data = df.to_dict('records')
    columns = [{"name": col, "id": col} for col in df.columns]

    # Highlighting style
    style_data_conditional = []
    
    # Highlight the row based on active_cell or clickData
    selected_name:str = None
    row_id = None
    if active_cell:
        row_id = active_cell['row_id'] #THIS IS THE ACTUAL INDEX IN THE DF (not effected by pagination)
        row_index = active_cell['row']  # This is the index in the DataTable (effected by pagination)
    
    selected_name = df.iloc[int(row_id)].to_dict()['Name'] if row_id is not None else None

    # Highlight the selected row in the DataTable
    if selected_name is not None:
        style_data_conditional = [
            {
                'if': {'row_index': row_index},
                'backgroundColor': '#00a3df',
                'color': 'black',
            }
        ]
        
    # Re-generate tradespace plot with or without the selected index highlighted
    tradespace_fig = bot_3d_problem.plot_tradespace(df, selected_name=selected_name, show=False)

    # Generate the selected bot plot
    selected_fig = go.Figure()
    if row_id is not None:
        row_dict = df[df["id"] == row_id].iloc[0].to_dict()
        problem = bot_3d_problem.SensorPkgOptimization.from_json(problem_json)
        bot = problem.convert_1D_to_bot(row_dict)
        selected_fig = bot.plot_bot_3d(
            perception_space=problem.perception_space,
            show=False,
            width=600,
        )

    return data, columns, style_data_conditional, tradespace_fig, selected_fig

@app.callback(
    Output('prior-bot-plot', 'figure'),
    Input('prior-bot-store', 'data'),  # Trigger when prior-bot-store is updated
    Input('problem-store', 'data'),  # Trigger when problem-store is updated
)
def update_prior_bot_plot(prior_bot_json, problem_json):
    if prior_bot_json is None:
        # Return an empty figure if no data is available
        return go.Figure()

    # Deserialize the JSON data into a Bot3D object
    prior_bot:bot_3d_rep.Bot3D = bot_3d_rep.Bot3D.from_json(prior_bot_json)

    if problem_json is not None:
        problem:bot_3d_problem.SensorPkgOptimization = bot_3d_problem.SensorPkgOptimization.from_json(problem_json)

        # Generate the 3D plot
        prior_bot_figure = prior_bot.plot_bot_3d(
            perception_space=problem.perception_space, 
            show=False,
            width=800,
        )
    else:
        # Generate the 3D plot without perception space
        prior_bot_figure = prior_bot.plot_bot_3d(
            perception_space=None,
            show=False,
            width=800,
        )

    # TODO: Add code to update the design variable table

    return prior_bot_figure

########################### Download callback ##########################
@app.callback(
    Output("download", "data"),
    Input("btn_csv", "n_clicks"),
    State("dropdown", "value"),
    State("pop-df-store", "data"),
    prevent_initial_call=True,
)
def download_results(n_clicks_btn, download_type, pop_df_json):
    if pop_df_json is None:
        return None  # No data to download

    # Deserialize the DataFrame from the JSON store
    df = pd.read_json(StringIO(pop_df_json), orient='split')

    # Convert the DataFrame to the requested format
    filename = "generated_designs"
    match download_type:
        case "csv":
            return dcc.send_data_frame(df.to_csv, f"{filename}.csv")
        case "excel":
            return dcc.send_data_frame(df.to_excel, f"{filename}.xlsx")
        case "json":
            return dcc.send_data_frame(df.to_json, f"{filename}.json")
        case "npy":
            return dcc.send_data_frame(df.to_numpy, f"{filename}.npy")
        case "pickle":
            return dcc.send_data_frame(df.to_pickle, f"{filename}.pkl")
    

def build_layout():
    return html.Div([
        dbc.Container([
            html.H1("Generation and Selection of Sensor Packages for Mobile Robots"),
            html.Hr(className="my-2"),
            html.P([
                html.A("Rachael Putnam", href="https://www.linkedin.com/in/robosquiggles/"),
                html.P("MIT Thesis, Copyright 2025")
            ], className="lead"),
            html.P("The goal of this thesis is to generate, select, and optimize sensor packages for mobile robots."),
        ], className="h-100 p-4 bg-light text-dark border rounded-3",),
        html.Hr(className="my-2"),
        dbc.Container([
            html.H2("Problem Definition"),
            dcc.Store(id='problem-store', data=None),  # Store for the problem
            dcc.Store(id='prior-bot-store', data=None),  # Store for the bot
            dbc.Row([
                dbc.Col([
                    html.H3("Problem Description"),
                    html.P("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."),
                ], width=4),  # Right column for the problem description
                dbc.Col([
                    html.H3("Prior Bot Design"),
                    dcc.Graph(
                        id='prior-bot-plot',
                        figure=go.Figure()  # Placeholder figure for the prior bot design
                    ),
                ], width=8),  # Left column for the prior bot design
            ]),
        ], className="h-100 p-4 bg-light text-dark border rounded-3"),
        dbc.Container([
            html.H2("Generated Designs"),
            html.H3("Tradespace"),
                    dcc.Graph(
                        id='tradespace-plot',
                        figure=go.Figure()
                    ),
            dbc.Row([
                dbc.Col([
                ], width=6),  # Right column for the tradespace
                dbc.Col([
                    html.H3("Selected Bot Design"),
                    # dcc.Store(id='selected-bot-store', data=None),  # Store for the selected bot
                    dcc.Graph(
                        id='selected-bot-plot',
                        figure=go.Figure()  # Placeholder figure
                    ),
                ], width=6),  # Left column for the bot plot
            ]),
            html.H3("Design Population"),
            dcc.Store(id='pop-df-store', data=None),  # Store for the DataFrame
            dash_table.DataTable(
                id='pop-df-table',
                page_size=dash_df_page_size,
                style_table={'overflowX': 'auto'},
                sort_action='native',  # Enable native sorting
                sort_mode='multi',  # Allow multi-column sorting
            ),
            dcc.Download(id="download"),
            dbc.Col([
                dcc.Dropdown(options=[
                                {"label": "Numpy file", "value": "npy"},
                                {"label": "JSON file", "value": "json"},
                                {"label": "CSV file", "value": "csv"},
                            ],
                            id="dropdown",
                            placeholder="Choose download file type. Default is CSV format!",
                )]),
            dbc.Col([
                dbc.Button(
                    "Download Data", id="btn_csv"
                )
            ], width=3)
        ], className="h-100 p-4 bg-light text-dark border rounded-3",)
    ])

if __name__ == '__main__':
    print("Starting Dash app...")
    app.layout = build_layout()
    app.run_server(debug=True, use_reloader=False)
    # webbrowser.open(url, new=2, autoraise=True)