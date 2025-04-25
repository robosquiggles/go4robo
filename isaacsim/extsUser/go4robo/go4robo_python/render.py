import sys
import os

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc


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
import base64
from io import BytesIO
import sys

matplotlib.use('agg')
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'Arial' 

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, 
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css],
           )
url = "http://127.0.0.1:8050/"
server = app.server

img_width = 400

app.layout = html.Div([
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
            html.P("MIT Thesis, 2025")], className="lead"),
        html.P("The goal of this project was to generate, select, and optimize sensor packages for mobile robots."),
    ], className="h-100 p-4 bg-light text-dark border rounded-3",)
    # dbc.Container([
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
    # ])
])


if __name__ == '__main__':
    print("Starting Dash app...")
    app.run_server(debug=True, use_reloader=False)
    # webbrowser.open(url, new=2, autoraise=True)