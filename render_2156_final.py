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
# server = app.server

def get_latest_timestamp(path):
    files = glob.glob(path)
    if len(files) == 0:
        return None
    latest_file = max(files, key=os.path.getctime)
    timestamp = latest_file.split('_')[-1].split('.pkl')[0]
    return timestamp

timestamp = get_latest_timestamp('./_output/df_opt_*.pkl')

unopt_df = pd.read_pickle(f'./_output/df_unopt_{timestamp}.pkl')
opt_df = pd.read_pickle(f'./_output/df_opt_{timestamp}.pkl')
with open(f'./_output/problem_{timestamp}.pkl', 'rb') as file:
    problem = dill.load(file)

combined_df = pd.concat([unopt_df, opt_df])

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
                dcc.Graph(id='tradespace',figure=bot_2d_problem.plot_tradespace(combined_df, unopt_df.shape[0])),
                html.Img(id='bot_plot')
            ], title="Results"
            ),
        ],
        start_collapsed=True,
    ),
    ])
])



@app.callback(
    Output(component_id='bot_plot', component_property='src'),
    Input('tradespace', 'hoverData')
)
def update_bots(hoverData):
    
    matplotlib.pyplot.close()

    if hoverData is None:
        return {}
    if len(hoverData['points']) < 2:
        return {}
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ub_idx = hoverData['points'][0]['pointIndex']
    ub = problem.convert_1D_to_bot(combined_df.iloc[ub_idx]['X'])
    ub.plot_bot(title="Pre-Optimization", ax=axes[0])

    ob_idx = hoverData['points'][1]['pointIndex']
    ob = problem.convert_1D_to_bot(combined_df.iloc[ob_idx]['X'])
    ob.plot_bot(title="Optimized", ax=axes[1])
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_matplotlib = f'data:image/png;base64,{fig_data}'
    
    return fig_matplotlib


if __name__ == '__main__':
    app.run(debug=True)