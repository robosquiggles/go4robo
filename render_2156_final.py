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
import sys

matplotlib.use('agg')
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'Arial' 

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
server = app.server

img_width = 400

def get_latest_timestamp(path):
    files = glob.glob(path)
    if len(files) == 0:
        return None
    latest_file = max(files, key=os.path.getctime)
    timestamp = latest_file.split('/')[-1].split('_')[0]
    return timestamp

def get_latest_timestamps_in_folders(base_path):
    folders = [f.path for f in os.scandir(base_path) if f.is_dir()]
    timestamps = {}
    for folder in folders:
        timestamp = get_latest_timestamp(os.path.join(folder, '*_df_opt.pkl'))
        if timestamp:
            timestamps[os.path.basename(folder)] = timestamp
    return timestamps

timestamps_dict = get_latest_timestamps_in_folders('./_output/')
timestamp = timestamps_dict["spot"]

def load_bot_images(folder, timestamp):
    images = {}
    image_files = glob.glob(f'./_output/{folder}/{timestamp}_botcompare_*.png')
    for image_file in image_files:
        pkg = image_file.split('_')[-1].split('.')[0]
        idx = int(pkg) if pkg != 'none' else pkg
        with open(image_file, 'rb') as file:
            images[idx] = base64.b64encode(file.read()).decode('ascii')
    return images

def create_abstract_section():
    return html.Div([
        html.P(["Mobile robots use “exteroceptive” sensors to perceive and interact with the world. As such it is vital that engineers design their sensor packages carefully. This project served as a preliminary investigation into AI and classical methods for generation and optimization of sensor packages for mobile robots. Results suggest that even simple Multi-Objective Optimization AI algorithms are effectively adapted for generating and optimizing sensor systems for mobile robots, serving as a strong motivating basis for future research."], className="mb-4")
    ], className="max-w-4xl mx-auto my-8")

def create_motivation_section():
    return html.Div([
        # Main content container with flex layout
        html.Div([
            # Text content
            html.Div([
                html.P([
                    html.B("Mobile robots provide immense value when operating in complex, harsh environments. ",
                            className="font-bold"),
                    "Perception system performance in a robot's operational design domain (ODD) drive its operational safety, and how quickly and completely it can achieve its task."
                ], className="mb-4"),
            ], className="w-2/3 pr-4"),

            # Image
            html.Div([
                html.Img(
                    src="data:image/jpeg;base64,{}".format(base64.b64encode(open('./_artifacts/robot_fog.jpg', 'rb').read()).decode('ascii')),
                    alt="Mobile robot in harsh environment",
                    width=img_width/2,
                    className="rounded-lg"
                ),
                html.P([dcc.Link("AnyMAL Robot in a foggy scene", href="https://researchfeatures.com/anymal-unique-quadruped-robot-conquering-harsh-environments/")])
            ], className="w-1/3", style={'textAlign': 'center'}),

            html.Div([
                html.P("Design of perception systems for mobile robotics can be challenging due to:",
                      className="mb-2"),
                
                # Numbered list of challenges
                html.Ol([
                    html.Li("Complexity of the operating domain (ODD)"),
                    html.Li("Highly constrained use of known, discrete sensor options"),
                    html.Li("Infinite options for 3D, sometimes dynamic sensor placement")
                ], className="list-decimal ml-6 mb-4"),
                
                # Industry standard description
                html.P([
                    "The prevailing industry standard sensor selection process is ",
                    html.Span("manual, time consuming, and expensive. ", className="font-bold"),
                    "While there exist a variety of tools and methods for simulation (though difficult to set up, or expensive), the exploration and selection process is left to subject matter experts. Additionally, there seems to be little-to-no widely-used tooling for optimization of sensor poses."
                ]),
            ], className="w-2/3 pr-4"),
            
            # Image
            html.Div([
                html.Img(
                    src="data:image/png;base64,{}".format(base64.b64encode(open('./_artifacts/rviz.png', 'rb').read()).decode('ascii')),
                    alt="Mobile robot in harsh environment",
                    width=img_width/2,
                    className="rounded-lg"
                ),
                html.P([dcc.Link("Simulation of PR2 using Gazebo in Rviz", href="https://docs.fetchrobotics.com/gazebo.html")])
            ], className="w-1/3", style={'textAlign': 'center'}),
            
        ], className="flex items-start bg-white p-6 rounded-lg shadow-md"),
        
    ], className="max-w-4xl mx-auto my-8")

def create_methodology_section():
    return html.Div([
        # Main content container with flex layout
        html.Div([
            # Text content
            html.Div([
                html.P([
                    "The robot is decomposed into its relevant component parts, and its ODD analyzed. Then, the components and target ODD are used as inputs to the ",
                    html.B("Multi-Objective Optimization", className="font-bold"),
                    "Process, where the cost is minimized, and perception coverage is maximized."
                ], className="mb-4"),
            ], className="w-2/3 pr-4"),

            # Image
            html.Div([
                html.Img(
                    src="data:image/png;base64,{}".format(base64.b64encode(open('./_artifacts/inputs_outputs.png', 'rb').read()).decode('ascii')),
                    alt="Inputs & Outputs of the Multi-Objective Optimization Process",
                    width=img_width/2,
                    className="rounded-lg"
                ),
                html.P("Inputs and Outputs of the Multi-Objective Optimization Process")
            ], className="w-1/3", style={'textAlign': 'center'}),

        ], className="flex items-start bg-white p-6 rounded-lg shadow-md"),

        html.P(["Using Custom ",
                html.B("Mixed-Variable Sensor Package Sampling", className="font-bold"),
                f", and ",
                html.B("Mixed Variable GA", className="font-bold"),
                f" the process generated optimized sets of sensor package options (shown in orange in the tradespaces, below)."
                ]),
        html.P([html.B("Constrained Pose Optimization", className="font-bold"),
                " is then used to optimize individual sensor placements for each concept on the pareto front, pushing them upward toward the global optimum.",
        ]),
        html.Div([
            html.P("Each Constrained Pose Optimization optimization generates the following history of sensor coverage per optimization iteration. The orange points denote invalid sensor poses (i.e., the sensors intersect with eachother, or they lie partially outside the green area on the bot). Because Constrained Pose Optimization cannot guarantee that the final iteration is valid, the best option is chosen from the optimization history. Note that some packages are already optimally placed, and Constrained Pose Optimization produces no change.", className="mb-2 text-xl font-semibold text-gray-700"),
            html.Div([
                html.Img(
                    src="data:image/png;base64,{}".format(base64.b64encode(open('./_artifacts/optimization.png', 'rb').read()).decode('ascii')),
                    alt="Optimization Diagram",
                    width=img_width,
                    className="rounded-lg"
                ),html.P("Example of Constrained Optimization Process History")
            ], className="w-1/3", style={'textAlign': 'center'}),
            html.P("The following video is an example of a single sensor package generated for Spot using constrained sensor pose optimization. Note that the sensor poses are constrained to stay to within the green areas on the bot.", className="mb-2 text-xl font-semibold text-gray-700"),
            html.Video(
                src="data:video/mp4;base64,{}".format(base64.b64encode(open('./_artifacts/optimization_animation.mp4', 'rb').read()).decode('ascii')),
                controls=True,
                style={'width': '100%', 'height': 'auto'}
            )
        ], className="my-4")
    ], className="max-w-4xl mx-auto my-8")

def create_results_section():

    def make_callback(folder):
            def update_bots(hoverData):
                # print(hoverData)
                if hoverData is not None:
                    for point in hoverData['points']:
                        if 'customdata' in point:
                            pkg = point['customdata'][0]
                            try:
                                return 'data:image/png;base64,{}'.format(bot_images[folder][pkg])
                            except KeyError:
                                print(f"KeyError: {pkg}")
                return 'data:image/png;base64,{}'.format(bot_images[folder]['none'])
            return update_bots

    bot_images = {}
    for folder, timestamp in timestamps_dict.items():
        bot_images[folder] = load_bot_images(folder, timestamp)

    results_containers = []
    for folder, timestamp in timestamps_dict.items():

        unopt_df = pd.read_pickle(f'./_output/{folder}/{timestamp}_df_unopt.pkl')
        opt_df = pd.read_pickle(f'./_output/{folder}/{timestamp}_df_opt.pkl')

        combined_df = pd.concat([unopt_df, opt_df])

        hv_unoptimized = bot_2d_problem.get_hypervolume(unopt_df, [20000, 0], x='Cost', y='Perception Coverage')
        hv_combined = bot_2d_problem.get_hypervolume(opt_df, [20000, 0], x='Cost', y='Perception Coverage')
        hv_improvement = hv_combined - hv_unoptimized

        results_containers.append(dbc.Container([
            dbc.Container([
                        html.H2(f"{folder.capitalize()} Problem"),
                        html.P(f"The {folder.capitalize()} problem is defined with the following robot and sensors as inputs:"),
                        dbc.Col([
                            html.Img(id=f'inputs_{folder}', 
                                    src="data:image/png;base64,{}".format(base64.b64encode(open(f'./_output/{folder}/{timestamp}_inputs.png', 'rb').read()).decode('ascii')), 
                                    width=img_width
                                    ),
                        ], xs=12, lg=6, md=6, style={'text-align': 'center'}),
                ]),
            dbc.Container([
                        html.H3(f"{folder.capitalize()} Results"),
                        html.P([html.B(f"Hypervolume Unoptimized: ", className="font-bold"), f"{hv_unoptimized:.2f}"]),
                        html.P([html.B(f"Hypervolume Optimized:   ", className="font-bold"), f"{hv_combined:.2f}"]),
                        html.P([html.B(f"Hypervolume Improvement: ", className="font-bold"), f"{hv_improvement:.2f} = +{hv_improvement/hv_unoptimized *100:.2f}%"]),
                        html.P("Hover over the tradespace (tap on mobile) to see the robot comparison plots for each concept.", className="mb-4"),
                        dbc.Col([
                            dcc.Graph(id=f'tradespace_{folder}',figure=bot_2d_problem.plot_tradespace(combined_df, unopt_df.shape[0], width=800, height=600, title=f"Tradespace of Optimal Sensor Packages")),
                        ], xs=12, lg=6, md=6, style={'text-align': 'center'}),
                        dbc.Col([
                            html.Img(id=f'bot_plot_{folder}', 
                                    width=img_width
                                    ),
                        ], xs=12, lg=6, md=6, style={'text-align': 'center'})
                    ])
                ])
        )

        # Create & register the callback for the bot plot
        update_bots = make_callback(folder)
        app.callback(
            Output(f'bot_plot_{folder}', 'src'),
            Input(f'tradespace_{folder}', 'hoverData')
        )(update_bots)
        
    return dbc.Container([
        dbc.Tabs([
            dbc.Tab(results_containers[i], label=folder.capitalize(), tab_id=folder) for i, folder in enumerate(timestamps_dict.keys())
        ])
    ])




app.layout = html.Div([
    dbc.Container([
        html.Div([
            html.Img(
                src="data:image/png;base64,{}".format(base64.b64encode(open(f"./_artifacts/go4r.png", 'rb').read()).decode('ascii')), 
                style={"height": 100, "marginRight": "10px"}
            ),
            html.H1("Generation and Selection of Sensor Packages for Mobile Robots"),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Hr(className="my-2"),
        html.P([
            html.A("Rachael Putnam", href="https://www.linkedin.com/in/robosquiggles/"), 
            html.P("MIT 2.156 Final Project")],
            className="lead",
            ),
        html.P("The goal of this project was to generate, select, and optimize sensor packages for mobile robots."),
    ], className="h-100 p-4 bg-light text-dark border rounded-3",),
    dbc.Container([
        dbc.Accordion(
        [
            dbc.AccordionItem(
                [create_abstract_section()], title=html.H2("Abstract")
            ),
            dbc.AccordionItem(
                [create_motivation_section()], title=html.H2("Motivation")
            ),
            dbc.AccordionItem(
                [create_methodology_section()], title=html.Span([html.H2("Methodology"), dbc.Badge("Video!", "99+", color="primary", pill=True, className="position-absolute top-0 start-100 translate-middle")])
            ),
            dbc.AccordionItem(
                [create_results_section()], title=html.Span([html.H2("Results"), dbc.Badge("Interactive!", "99+", color="primary", pill=True, className="position-absolute top-0 start-100 translate-middle")])
            ),
        ],
        start_collapsed=True,
    ),
    ])
])


if __name__ == '__main__':
    app.run(debug=True)