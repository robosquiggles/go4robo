from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

import plotly.express as px
import pandas as pd

import bot_2d_problem
import bot_2d_rep
import glob
import os
import dill

import matplotlib
import matplotlib.pyplot as plt
import base64
from io import BytesIO

matplotlib.use('agg')
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'Arial' 

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
server = app.server

def get_latest_timestamp(path):
    files = glob.glob(path)
    if len(files) == 0:
        return None
    latest_file = max(files, key=os.path.getctime)
    timestamp = latest_file.split('_')[-1].split('.pkl')[0]
    return timestamp

def get_latest_timestamps_in_folders(base_path):
    folders = [f.path for f in os.scandir(base_path) if f.is_dir()]
    timestamps = {}
    for folder in folders:
        timestamp = get_latest_timestamp(os.path.join(folder, 'df_opt_*.pkl'))
        if timestamp:
            timestamps[os.path.basename(folder)] = timestamp
    return timestamps

timestamps_dict = get_latest_timestamps_in_folders('./_output/')
timestamp = timestamps_dict["spot"]

def load_bot_images(folder, timestamp):
    images = {}
    image_files = glob.glob(f'./_output/{folder}/botcompare_*_{timestamp}.png')
    for image_file in image_files:
        idx = int(image_file.split('_')[-2])
        with open(image_file, 'rb') as file:
            images[idx] = base64.b64encode(file.read()).decode('ascii')
    return images

bot_images = {}
for folder, timestamp in timestamps_dict.items():
    bot_images[folder] = load_bot_images(folder, timestamp)

#build all of the results into containers so that I can display them in separate tabs
results_containers = []
for folder, timestamp in timestamps_dict.items():
    unopt_df = pd.read_pickle(f'./_output/{folder}/df_unopt_{timestamp}.pkl')
    opt_df = pd.read_pickle(f'./_output/{folder}/df_opt_{timestamp}.pkl')
    with open(f'./_output/{folder}/problem_{timestamp}.pkl', 'rb') as file:
        problem = dill.load(file)

    combined_df = pd.concat([unopt_df, opt_df])

    container = dbc.Container([
                    dcc.Graph(id=f'tradespace_{folder}',figure=bot_2d_problem.plot_tradespace(combined_df, unopt_df.shape[0], width=800, height=600)),
                    html.Img(id=f'bot_plot_{folder}', width=800),
                    ])
    
    results_containers.append(container)
    
    @app.callback(
        Output(component_id=f'bot_plot_{folder}', component_property='src'),
        Input(f'tradespace_{folder}', 'hoverData')
    )
    def update_bots(hoverData):
        if hoverData is None:
            return None
        point_index = hoverData['points'][0]['pointIndex']
        return 'data:image/png;base64,{}'.format(bot_images[folder][point_index])

app.layout = html.Div([
    dbc.Container([
        html.H1("Generation and Selection of Sensor Packages for Mobile Robots"),
        html.P(
                "Rachael Putnam - MIT 2.156 Final Project",
                className="lead",
            ),
        html.Hr(className="my-2"),
        html.P("The goal of this project was to generate, select, and optimize sensor packages for mobile robots."),
    ], className="h-100 p-4 bg-light text-dark border rounded-3",),
    dbc.Container([
        dbc.Accordion(
        [
            dbc.AccordionItem(
                [html.P("Meaningful applications of Mobile Robotics tend to require exteroceptive sensing (perception) capable of observing complex environments. Firms that design robots for these applications face complex tradeoffs early on in their development process. Selection of (1) appropriate sensors for the environment, and (2) how and where to mount those sensors are architectural decisions which must be made very early in the design process, but also have immense impact on the downstream capabilities of the robot. This causes firms to partake in manual iteration, which is costly both in time and resources."),
                 html.P("While there seem to be a variety of standard tools and methods for simulation (though often these tools are difficult to set up, or are expensive), the actual exploration and selection process is left to subject matter experts. Additionally, there seems to be little-to-no widely-used tooling for optimization of sensor poses."),
                ], title="Motivation"
            ),
            dbc.AccordionItem(
                "This is the content of the second section", title="Approach"
            ),
            dbc.AccordionItem([
                dbc.Tabs([
                    dbc.Tab(results_containers[i], label=folder.capitalize(), tab_id=folder) for i, folder in enumerate(timestamps_dict.keys())
                ]),
            ], title="Results"
            ),
        ],
        start_collapsed=True,
    ),
    ])
])


if __name__ == '__main__':
    app.run(debug=True)