from bot_2d_rep import *

import numpy as np
import pandas as pd

import plotly.express as px

import copy

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import Sampling, FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

class SensorPkgOptimization(ElementwiseProblem):

    def __init__(self, bot:SimpleBot2d, sensor_options:list[FOV2D|None], max_n_sensors:int=10, **kwargs):
        """
        Initializes the sensor package optimization problem.

        Design Variables (each of N sensors):
            type :      int (sensor object enumerated)
            x :         float (meters)
            y :         float (meters)
            rotation :  float (0-360 deg)
        """

        # BOT
        self.bot = copy.deepcopy(bot)
        self.bot.clear_sensors()

        # SENSORS
        if None not in sensor_options:
            sensor_options.insert(0, None)
        self.sensor_options = dict(enumerate(sensor_options))
        self.max_n_sensors = max_n_sensors

        # VARIABLES
        variables = dict()
        s_bounds = np.array(bot.sensor_pose_constraint.bounds)
        for i in range(self.max_n_sensors):
            variables[f"s{i}_type"] = Integer(bounds=(0,len(self.sensor_options)-1))
            variables[f"s{i}_x"] = Real(bounds=(s_bounds[0], s_bounds[2]))
            variables[f"s{i}_y"] = Real(bounds=(s_bounds[1], s_bounds[3]))
            variables[f"s{i}_rotation"] = Real(bounds=(0.0, 360.0))
        self.n_var = len(variables)

        super().__init__(vars=variables, n_obj=2, **kwargs)

    def convert_sensor_to_1D(self, sensor:FOV2D|None, idx:int, dtype=np.ndarray):
        """
        Converts a 2D sensor object to a 1D representation.
        Parameters:
        sensor (FOV2D): The 2D sensor object to be converted.
        idx (int): The index of the sensor.
        Returns:
        dict: A dictionary containing the 1D representation of the sensor with keys:
            - 's{idx}_type': The type of the sensor.
            - 's{idx}_x': The x-coordinate of the sensor's focal point.
            - 's{idx}_y': The y-coordinate of the sensor's focal point.
            - 's{idx}_rotation': The rotation of the sensor.
        Raises:
        KeyError: If the sensor is not found in the sensor_options.
        """

        def get_sensor_key(sensor):
            for key, s in self.sensor_options.items():
                if s == sensor:
                    return key
            raise KeyError(f"Sensor: {sensor} not found in options: {self.sensor_options}")
        
        if sensor is not None:
            x = {
                f"s{idx}_type": get_sensor_key(sensor),
                f"s{idx}_x": sensor.focal_point[0],
                f"s{idx}_y": sensor.focal_point[1],
                f"s{idx}_rotation": sensor.rotation
            }
        else:
            x = {
                f"s{idx}_type": 0,
                f"s{idx}_x": 0,
                f"s{idx}_y": 0,
                f"s{idx}_rotation": 0
            }

        if dtype == dict:
            return x
        elif dtype == np.ndarray or dtype == np.array or dtype == list:
            return np.array(list(x.values()))
        else:
            raise ValueError("Invalid dtype:", dtype)
    
    def convert_1D_to_sensor(self, x:dict|np.ndarray|list, idx:int, verbose=False):
        """
        Converts a 1D representation of a sensor to a sensor object.
        Args:
            x (dict): A dictionary containing sensor parameters.
            idx (int): The index of the sensor in the dictionary.
        Returns:
            Sensor: A deep copy of the sensor object with updated translation and rotation,
                    or None if the sensor type is not available.
        """
        if verbose:
            print("Convert 1D->sensor X:", x)
        if type(x) is not dict:
            x = {
                f"s{idx}_type": x[0],
                f"s{idx}_x": x[1],
                f"s{idx}_y": x[2],
                f"s{idx}_rotation": x[3]
            }

        if self.sensor_options[x[f"s{idx}_type"]] is None:
            return None
        else:
            sensor = copy.deepcopy(self.sensor_options[x[f"s{idx}_type"]])
            sensor.set_translation(x[f"s{idx}_x"], x[f"s{idx}_y"])
            sensor.set_rotation(x[f"s{idx}_rotation"])
            return sensor

    def convert_bot_to_1D(self, bot, verbose=False, dtype=np.ndarray):
        """
        Converts a bot object with 2D sensor data into a 1D numpy array.
        Parameters:
            bot (object): The bot object containing sensors with 2D data.
        Returns:
            numpy.ndarray: A 1D numpy array containing the converted sensor data.
        """
        
        if dtype == dict:
            x = dict()
            for i in range(self.max_n_sensors):
                if i < len(bot.sensors):
                    sensor = bot.sensors[i]
                else:
                    sensor = None
                x.update(self.convert_sensor_to_1D(sensor, i, dtype=dict))
            if verbose:
                print("Convert bot->1d X (dict):", x)
            return x
        else:
            x = np.ndarray((self.max_n_sensors, self.n_var / self.max_n_sensors))
            for i in range(self.max_n_sensors):
                if i < len(bot.sensors):
                    sensor = bot.sensors[i]
                else:
                    sensor = None
                x[i] = self.convert_sensor_to_1D(sensor, i, dtype=np.ndarray)
            if verbose:
                print("Convert bot->1d X (array-like):", x)
            return x.flatten()
        

    def convert_1D_to_bot(self, x, verbose=False):
        """
        Converts a 1D dictionary of sensor data into a bot object with sensor attributes.
        Args:
            x (1D np array): A 1D array specifying sensor information.
        Returns:
            Bot: A bot object with its sensors populated based on the input dictionary.
        Example:
            Given a dictionary `x` with keys like 's0_param1', 's1_param2', etc., this method
            will split the dictionary into separate sensor dictionaries and assign them to the bot's sensors.
        """
        if verbose:
            print("Convert 1d->bot X:", x)
        bot = copy.deepcopy(self.bot)
        if type(x) is not dict:
            xs = x.reshape(self.max_n_sensors, -1)
        else:
            xs = [{k: v for k, v in x.items() if k.startswith(f"s{i}_")} for i in range(0, self.max_n_sensors)]
        if verbose:
            print("Convert 1d->bot (xs):", xs)
        bot.add_sensors_2d([self.convert_1D_to_sensor(x, i) for i, x in enumerate(xs)])
        return bot


    def _evaluate(self, x, out, *args, **kwargs):
        # print("In EVALUATE, eavulating:", x)
        bot = self.convert_1D_to_bot(x)
        if bot.is_valid_pkg():
            out["F"] = [
                1 - bot.get_sensor_coverage(),  # maximize sensor coverage, so subtract from 1
                bot.get_pkg_cost()              # minimize cost as is
                ]
        else:
            out["F"] = [
                np.inf,
                np.inf
                ]
    
class CustomSensorPkgRandomSampling(Sampling):
    def __init__(self, p=None, **kwargs):
        self.p = p
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        # print("In Custom Random Sampling")
        xl, xu = problem.bounds()
        xl = list(xl.values())
        xu = list(xu.values())
        assert np.all(xu >= xl)

        if self.p is None:
            p = [1 / len(problem.sensor_options)] * len(problem.sensor_options)
        else:
            p = self.p
            assert len(p) == len(problem.sensor_options)
            assert np.sum(p) == 1

        X = []
        for _ in range(n_samples):
            bot = copy.deepcopy(problem.bot)
            bot.clear_sensors()
            for i in range(problem.max_n_sensors):
                sensor = problem.convert_1D_to_sensor({
                    f"s{i}_type": np.random.choice(range(0,len(problem.sensor_options)), p=p),
                    f"s{i}_x": 0,
                    f"s{i}_y": 0,
                    f"s{i}_rotation": 0
                }, i)
                if sensor is not None:
                    bot.add_sensor_valid_pose(sensor)
            X.append(problem.convert_bot_to_1D(bot, dtype=dict))
        # print("Sampled X shape:", X.shape)
        return X
        

def get_pareto_front(df, x='Cost', y='Perception Coverage'):
    # Extract the relevant columns for the Pareto front
    points = df[[x, y]].values
    
    # Sort the points by the first objective (Perception Coverage)
    sorted_points = points[np.argsort(points[:, 0])]
    
    # Initialize the Pareto front with the first point
    pareto_front = [sorted_points[0]]
    indices = [0]
    
    # Iterate through the sorted points and add to Pareto front if it dominates the previous point
    for point in sorted_points[1:]:
        if point[1] > pareto_front[-1][1]:
            pareto_front.append(point)
            indices.append(df.loc[df[[x, y]].eq(point).all(axis=1)].index[0])
    
    return np.array(pareto_front), indices


def get_hypervolume(df, ref_point, x='Cost', y='Perception Coverage', x_minimize=True, y_minimize=False):
    pareto, idx = get_pareto_front(df, x=x, y=y)
    if not x_minimize:
        pareto[:, 0] = -pareto[:, 0]
    if not y_minimize:
        pareto[:, 1] = -pareto[:, 1]
    ref_point = np.array(ref_point)
    
    # Calculate the hypervolume
    hv = HV(ref_point=ref_point)
    hypervolume = hv(pareto)
    
    return hypervolume


def plot_tradespace(combined_df:pd.DataFrame, num_results, show_pareto=True, show=False, panzoom=False, **kwargs):
    """
    Plot the trade space of concepts based on Cost and Perception Coverage.
    This function creates a scatter plot visualizing the trade-offs between Cost and Perception Coverage for different concepts.
    Each point represents a concept, colored based on its optimization status. An ideal point is also marked on the plot.
    The plot can be displayed interactively with optional pan and zoom capabilities.
    Parameters:
        combined_df (pd.DataFrame): DataFrame containing the data to plot, with columns 'Cost', 'Perception Coverage',
                                     'Optimized', and 'Name'.
        num_results (int): The number of top concepts to include in the title of the plot.
        show (bool, optional): If True, display the plot. Defaults to False.
        panzoom (bool, optional): If False, disables panning and zooming by fixing the axis ranges. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the plotly express scatter function.
            height (int, optional): The height of the plot in pixels. Defaults to 600.
            width (int, optional): The width of the plot in pixels. Defaults to 600.
            opacity (float, optional): The opacity of the points on the plot (0 to 1). Defaults to 0.9.
            title (str, optional): The title of the plot. Defaults to "Objective Space (best of {num_results} concepts)".
    Returns:
        plotly.graph_objs._figure.Figure: The generated Plotly figure object.
    """
    height = 600 if 'height' not in kwargs else kwargs['height']
    width = 600 if 'width' not in kwargs else kwargs['width']
    opacity = 0.9 if 'opacity' not in kwargs else kwargs['opacity']
    title = f"Objective Space (best of {num_results} concepts)" if 'title' not in kwargs else kwargs['title']
    
    fig = px.scatter(combined_df, x='Cost', y='Perception Coverage', 
                     color='Optimized', 
                     color_discrete_sequence=['#fc7114', '#1276a4'], 
                     opacity=opacity,
                     title=title, 
                     template="plotly_white", 
                     labels={'Cost': 'Cost ($)', 'Perception Coverage': 'Perception Coverage (%)'},
                     hover_name='Name',
                     hover_data=['Cost', 'Perception Coverage'],
                     custom_data=['Index'])
    
    fig.update_traces(marker=dict(size=5*(width/600)),
                      hovertemplate="<br>".join([
                            "Pkg: %{customdata[0]}",
                            "Cost: $%{x:.2f}",
                            "Perception Coverage: %{y:.2f}%",
                            ])
                      )

    fig.add_scatter(x=[0], 
                    y=[100], 
                    mode='markers', 
                    marker=dict(symbol='star', size=12*(width/600), color='gold'), 
                    name='Ideal',
                    hoverinfo='none',  # Disable hover data
                    )
    
    if show_pareto:
        pareto, idx = get_pareto_front(combined_df, x="Cost", y="Perception Coverage")
        fig.add_scatter(x=pareto[:, 0],
                        y=pareto[:, 1],
                        mode='lines+markers', 
                        line=dict(color='grey', width=1*(width/600)), 
                        marker=dict(size=10*(width/600), color='grey', symbol='circle-open'),
                        name='Pareto Front',
                        hoverinfo='none',  # Disable hover data
                        )
    
    if not panzoom:
        fig.update_layout(
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True)
        )
    
    fig.update_layout(
        hovermode='x unified',
        height=height, width=width,
        legend=dict(
            # orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="right",
            x=1
        ),
        yaxis=dict(range=[0, 110])
    )
    if show:
        fig.show()
    
    return fig