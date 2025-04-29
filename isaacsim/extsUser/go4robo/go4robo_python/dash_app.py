import sys
import os

import dash
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import dash_bootstrap_components as dbc

from . import bot_3d_problem

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

@app.callback(
    Output('pop-df-table', 'data'),
    Output('pop-df-table', 'columns'),
    Output('tradespace-plot', 'figure'),
    Input('pop-df-store', 'data')
)
def update_results(pop_df_json):
    if pop_df_json is None:
        return [], [], go.Figure()
    
    # Deserialize the DataFrame
    df = pd.read_json(StringIO(pop_df_json), orient='split')
    
    # Update DataTable
    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict('records')
    
    # Update Tradespace Plot
    figure = bot_3d_problem.plot_tradespace(df)
    
    return data, columns, figure

# TODO: Add a callback to update the design variable table

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
    return  html.Div([
        dbc.Container([
            # html.Div([
            #     html.Img(
            #         src="data:image/png;base64,{}".format(base64.b64encode(open(f"../data/icon.png", 'rb').read()).decode('ascii')), 
            #         style={"height": 100, "marginRight": "10px"}
            #     ),
            #     html.H1("Generation and Selection of Sensor Packages for Mobile Robots"),
            # ], style={"display": "flex", "alignItems": "center"}),
            html.H1("Generation and Selection of Sensor Packages for Mobile Robots"),
            html.Hr(className="my-2"),
            html.P([
                html.A("Rachael Putnam", href="https://www.linkedin.com/in/robosquiggles/"), 
                html.P("MIT Thesis, Copyright 2025")], className="lead"),
            html.P("The goal of this thesis is to generate, select, and optimize sensor packages for mobile robots."),
        ], className="h-100 p-4 bg-light text-dark border rounded-3",),
        dbc.Container([
            html.H1("Source Robot Visualized"),
            dcc.Graph(
                id='source-bot-plot',
                figure=px.scatter_3d(pd.DataFrame({'x':[0.0], 'y':[0.0], 'z':[0.0], 'name':'PLACEHOLDER DOT'}), 
                                     x='x', 
                                     y='y', 
                                      z='z', 
                                     hover_name='name')
                ),
            html.Hr(className="my-2"),
        ]),
        dbc.Container([
            html.H2("Problem Definition"),
            html.P("The problem is defined as a multi-objective optimization problem with the following objectives:"),
            html.Ul([
                html.Li("Minimize Cost"),
                html.Li("Maximize Coverage & Detection (Minimize Perception Entropy)"),
            ]),
            html.Hr(className="my-2"),
            html.H3("Design Variables"),
            dbc.Col([
                html.H4("Sensor Types"),
                dcc.Store(id='dv-df-store', data=None),  # Store for the DataFrame
                dash_table.DataTable(
                    id=('dv-df-table'),
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                ),
                # html.H4("Sensor Config"),
                # html.P("Each of N sensors can be placed anywhere in the 3D constraint space."),
                # dash_table.DataTable(
                #     id=('dv-df-table'),
                #     page_size=10,
                #     style_table={'overflowX': 'auto'},
                # ),
            ]),
            html.Hr(className="my-2"),

        ], className="h-100 p-4 bg-light text-dark border rounded-3",),
        dbc.Container([
            html.H1("Optimization Results"),
            html.H2("Tradespace"),
            dcc.Graph(
                id='tradespace-plot',
                figure=go.Figure()
            ),
            html.H2("Design Population"),
            dcc.Store(id='pop-df-store', data=None),  # Store for the DataFrame
            dash_table.DataTable(
                id='pop-df-table',
                # columns=[{"name": i, "id": i} for i in pop_df.columns] if pop_df is not None else [],
                # data=pop_df.to_dict('records') if pop_df is not None else [],
                page_size=25,
                style_table={'overflowX': 'auto'},
            ),
            dcc.Download(id="download"),
            dbc.Col([
                dcc.Dropdown(options=[
                                {"label": "Numpy file", "value": "npy"},
                                # {"label": "Pickle file", "value": "pickle"},
                                {"label": "JSON file", "value": "json"},
                                # {"label": "Excel file", "value": "xlsx"},
                                {"label": "CSV file", "value": "csv"},
                            ],
                            id="dropdown",
                            placeholder="Choose download file type. Default is CSV format!",
                ),
                dbc.Button(
                            "Download Data", id="btn_csv"
                        ),
            ])
        #     dbc.Accordion(
        #     [
        #         dbc.AccordionItem(
        #             [create_abstract_section()], title=html.H2("Abstract")
        #         ),
        #         dbc.AccordionItem(
        #             [create_motivation_section()], title=html.H2("Motivation")
        #         ),
        #         dbc.AccordionItem(
        #             [create_methodology_section()], title=html.Span([html.H2("Methodology"), dbc.Badge("Video!", "99+", color="primary", pill=True, className="position-absolute top-0 start-100 translate-middle")])
        #         ),
        #         dbc.AccordionItem(
        #             [create_results_section()], title=html.Span([html.H2("Results"), dbc.Badge("Interactive!", "99+", color="primary", pill=True, className="position-absolute top-0 start-100 translate-middle")])
        #         ),
        #     ],
        #     start_collapsed=True,
        # ),
        ], className="h-100 p-4 bg-light text-dark border rounded-3",)
    ])

if __name__ == '__main__':
    print("Starting Dash app...")
    app.layout = build_layout()
    app.run_server(debug=True, use_reloader=False)
    # webbrowser.open(url, new=2, autoraise=True)