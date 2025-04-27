from .bot_3d_rep import *

import numpy as np

import sys

import plotly.express as px

import copy

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import Sampling, FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX #, MixedVariableTwoPointsCrossover
from pymoo.operators.mutation.pm import PolynomialMutation #, MixedVariableGaussianMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV


from tqdm import tqdm

class ProgressBar:
    def __init__(self, n_gen, update_fn=None):
        self.pbar = tqdm(
            total=n_gen,
            desc="MOO Progress",
            unit="gen",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout  # Ensure tqdm writes to stdout
        )
        self.update_fn = update_fn  # Callback function to update the Isaac Sim progress bar

    def notify(self, algorithm, problem):
        self.pbar.update(1)
        self.pbar.set_postfix({
            "Designs": problem.n_evals,
            "Best Cost": algorithm.pop.get("F").min(axis=0)[1],
            "Best PE": algorithm.pop.get("F").min(axis=0)[0]
        })
        self.pbar.refresh()  # Force immediate update to the terminal

        # Update the Isaac Sim progress bar
        if self.update_fn:
            progress = self.pbar.n / self.pbar.total  # Calculate progress as a fraction
            self.update_fn(progress)

    def close(self):
        self.pbar.close()

import pandas as pd

class DesignRecorder:
    def __init__(self):
        self.records = []

    def notify(self, algorithm):
        # Get the current population's variables and objectives
        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")
        G = algorithm.pop.get("G") if algorithm.pop.has("G") else None
        for x, f, g in zip(X, F, G):
            self.notify_single(x, f, g)
    
    def notify_single(self, x, f, g=None):
        # Store as dict for easy DataFrame conversion
        self.records.append({
            "design": x.copy(),
            "Cost": float(f[1]),
            "Perception Entropy": float(f[0])
        })

    def to_dataframe(self):
        # Convert to DataFrame for analysis/plotting
        df = pd.DataFrame(self.records)
        df = self.expand_designs_into_df(df)
        df["Index"] = df.index
        df["Name"] = df["Index"].map(lambda x: f"Design {x}")
        # Reorder columns to put Index and Name at the beginning
        cols = ["Index", "Name"] + [col for col in df.columns if col not in ["Index", "Name"]]
        df = df[cols]
        return df
    
    def expand_designs_into_df(self, df):

        def expand_design(row):
            # Convert the design dict into a list of values
            # For each sensor, we have a type and a tf (3x4 matrix)

            # For now just show the number of sensor in the design
            
            return [row["design"][f"s{i}_type"] for i in range(n_sensors)]

        # Expand the design dict into separate columns
        a_design_dict = df["design"].iloc[0]
        n_sensors = int(len(a_design_dict.keys()) / 13)
        for i in range(1, n_sensors + 1):
            df[f"Sensor {i}"] = df.apply(lambda row: row["design"].get(f"s{i-1}_type", None), axis=1)
            # for j in range(12):
            #     df[f"Sensor {i} tf_{j}"] = df.apply(lambda row: row["design"].get(f"s{i-1}_tf_{j}", None), axis=1)
        # Drop the original design column
        df = df.drop(columns=["design"])
        return df


class SensorPkgOptimization(ElementwiseProblem):

    def __init__(self, bot:Bot3D, sensor_options:list[Sensor3D|None], perception_space:PerceptionSpace, max_n_sensors:int=5, **kwargs):
        """
        Initializes the sensor package optimization problem.

        Design Variables (each of N sensors):
            type : int (sensor object enumerated)
            tf:    np.ndarray (shape (4,4) transformation matrix of the sensor instance sensor in 3D space)
                   where tf = [ [r1, r2, r3, x],
                                [r4, r5, r6, y],
                                [r7, r8, r9, z],
                                [0,  0,  0,  1]]
                   We simply chop off the last row (containing only the scaling information) to get a 3x4 matrix, and then flatten it.
        Parameters:
            bot (Bot3D): The robot object to be optimized.
            sensor_options (list): List of sensor objects to be used in the optimization.
            max_n_sensors (int): Maximum number of sensor instances to be used in the optimization.
            **kwargs: Additional keyword arguments for the parent class.
        """

        # BOT
        self.bot = copy.deepcopy(bot)
        self.bot.clear_sensors()

        # SENSORS
        if isinstance(sensor_options, set):
            sensor_options = list(sensor_options)
        if None not in sensor_options:
            sensor_options.insert(0, None)                    # This makes sure that we have a None option for the sensor
        self.sensor_options = dict(enumerate(sensor_options)) # This is the mapping of sensor type (enum) to sensor object
        self.max_n_sensors = max_n_sensors

        # PERCEPTION SPACE
        assert isinstance(perception_space, PerceptionSpace), "perception_space must be a PerceptionSpace object"
        self.perception_space = perception_space

        # SENSOR POSE CONSTRAINTS
        if bot.sensor_pose_constraint is None:
            print("WARNING: No sensor pose constraint provided. Defaulting to a 1m cube.")
            self.s_bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]) # Default to a 1m cube
        else:
            if isinstance(bot.sensor_pose_constraint.bounds, list):
                self.s_bounds = np.array(bot.sensor_pose_constraint.bounds)
            elif isinstance(bot.sensor_pose_constraint.bounds, tuple):
                self.s_bounds = np.array(bot.sensor_pose_constraint.bounds)
            elif isinstance(bot.sensor_pose_constraint.bounds, np.ndarray):
                self.s_bounds = bot.sensor_pose_constraint.bounds
            elif isinstance(bot.sensor_pose_constraint.bounds, UsdGeom.Mesh):
                self.s_bounds = np.array(bot.sensor_pose_constraint.bounds.GetExtent())
            else:
                raise ValueError("Invalid sensor pose constraint bounds type:", type(bot.sensor_pose_constraint.bounds))


        # PROBLEM VARIABLES
        variables = dict()
        for i in range(self.max_n_sensors):
            variables[f"s{i}_type"] = Integer(bounds=(0,len(self.sensor_options)-1))
            for j in range(12):  # 3*4 = 12
                variables[f"s{i}_tf_{j}"] = Real(bounds=(0, 1))
        self.n_var = len(variables)

        # ETCETERA
        self.n_evals = 0

        super().__init__(vars=variables, n_obj=2, **kwargs)

    def convert_4dtf_to_1D(self, tf:np.ndarray|list|tuple, dtype=np.ndarray):
        """
        Converts a 4x4 transformation matrix to a 1D representation. 1D representation is a flattened version of the matrix.
        The last row (containing only the scaling information) is chopped off to get a 3x4 matrix, and then flattened.
        The translations are normalized to the bounds of the sensor pose constraint.
        The rotation portion of the matrix is normalized using SVD.
        Parameters:
            tf (np.ndarray|list|tuple): The 4x4 transformation matrix to be converted.
            dtype (type): The desired output type (e.g., np.ndarray, list).
        Returns:
            a 1D representation of the transformation matrix. We simply chop off the last row 
            (containing only the scaling information) to get a 3x4 matrix, and then flatten it.
        """
        normalized_tf = TF.normalize_svd(tf, bounds=self.s_bounds)
        flat = TF.flatten_matrix(normalized_tf)
        flat = flat[:12]  # Take only the first 12 elements (3x4 matrix)
        
        if dtype == dict:
            return {f"tf_{i}": tf[i] for i in range(12)}
        elif dtype == np.ndarray or dtype == np.array or dtype == list:
            return tf
        else:
            raise ValueError("Invalid dtype:", dtype)
        
    def convert_1D_to_4dtf(self, tf_1D:np.ndarray|list|tuple):
        """
        Converts a 1D representation of a transformation matrix back to a 4x4 matrix.
        The translation portion of the matrix is denormalized using the bounds of the sensor pose constraint.
        The rotation portion of the matrix is kept as is because the SVD normalization is equivalent equivalent.
        The last row (containing only the scaling information) is added back to get a 4x4 matrix.
        Parameters:
            tf_1D (np.ndarray|list|tuple): The 1D representation of the transformation matrix.
        Returns:
            A 4x4 transformation matrix.
        """
        if isinstance(tf_1D, dict):
            tf_1D = np.array(list(tf_1D.values()))
        assert len(tf_1D) == 12, "tf_1D must be of length 12"
        return TF.unflatten_matrix(tf_1D)

    def convert_sensor_instance_to_1D(self, sensor_instance:Sensor3D_Instance|None, idx:int, dtype=np.ndarray, verbose=False):
        """
        Converts a 2D sensor object to a 1D representation.
        Parameters:
        sensor (Sensor3D_Instance): The 2D sensor object to be converted.
        idx (int): The index of the sensor.
        Returns:
            dict: A dictionary containing the 1D representation of the sensor with keys:
                - 's{idx}_type': The type of the sensor.
                - 's{idx}_tf': The transformation matrix of the sensor, flattened.
        Raises:
            KeyError: If the sensor is not found in the sensor_options.
        """

        assert sensor_instance is None or isinstance(sensor_instance, Sensor3D_Instance), "sensor_instance must be a Sensor3D_Instance or None"
        assert idx < self.max_n_sensors, "idx must be less than max_n_sensors"
        assert sensor_instance is None or sensor_instance.sensor in self.sensor_options.values(), "sensor_instance must be in sensor_options"
        assert sensor_instance is None or sensor_instance.get_transform() is not None, "sensor_instance must have a transform"

        if verbose:
            print("Convert sensor->1d X:", sensor_instance)

        def get_sensor_key(sensor_instance:Sensor3D_Instance|None):
            for key, s in self.sensor_options.items():
                if s == sensor_instance.sensor:
                    return key
            raise KeyError(f"Sensor: {sensor_instance} not found in options: {self.sensor_options}")
        
        if sensor_instance is not None:
            x = {
                f"s{idx}_type": get_sensor_key(sensor_instance)
            }

            tf_to_1D = self.convert_4dtf_to_1D(sensor_instance.get_transform(), dtype=dict)
            for k, v in tf_to_1D.items():
                x[f"s{idx}_{k}"] = v
        else:
            x = {
                f"s{idx}_type": 0
            }
            for j in range(12):
                x[f"s{idx}_tf_{j}"] = 0.0

        if dtype == dict:
            return x
        elif dtype == np.ndarray or dtype == np.array or dtype == list:
            return np.array(list(x.values()))
        else:
            raise ValueError("Invalid dtype:", dtype)
    
    def convert_1D_to_sensor_instance(self, x:dict|np.ndarray|list, idx:int, verbose=False) -> Sensor3D_Instance|None:
        """
        Converts a 1D representation of a sensor to a sensor object.
        Args:
            x (dict): A dictionary containing sensor parameters.
            idx (int): The index of the sensor in the dictionary.
        Returns:
            Sensor: A Sensor3D_Instance object with updated translation and rotation,
                    or None if the sensor type is not available.
        """
        assert idx < self.max_n_sensors, "idx must be less than max_n_sensors"
        assert x is not None, "x must not be None"
        assert type(x) is dict or type(x) is np.ndarray or type(x) is list, "x must be a dict or np.ndarray or list"

        if type(x) is dict:
            sensor_type = x[f"s{idx}_type"]
            tf = {k: v for k, v in x.items() if k.startswith(f"s{idx}_tf_")}
        else:
            sensor_type = x[idx * 2]
            tf = x[idx * 2 + 1]

        if verbose:
            print("Convert 1d->sensor X:", x)
        
        if sensor_type > 0 and sensor_type in self.sensor_options: # 0 is None
            sensor = self.sensor_options[sensor_type]
            sensor_instance = Sensor3D_Instance(sensor=sensor,
                                                name=f"sensor_{idx}",
                                                tf=self.convert_1D_to_4dtf(tf),
                                                path=None)
            return sensor_instance
        else:
            return None

    def convert_bot_to_1D(self, bot:Bot3D, verbose=False, dtype=np.ndarray) -> np.ndarray|dict:
        """
        Converts a Bot3D object (perception system design) into a 1D numpy array.
        Parameters:
            bot (Bot3D): The bot object containing sensors with 2D data.
        Returns:
            numpy.ndarray: A 1D numpy array containing the converted sensor data.
        """

        assert isinstance(bot, Bot3D), "bot must be a Bot3D object"
        assert len(bot.sensors) <= self.max_n_sensors, "bot must have less than max_n_sensors sensors"
        
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
        bot.sensors = [self.convert_1D_to_sensor_instance(x, i) for i, x in enumerate(xs)]
        while None in bot.sensors:
            bot.sensors.remove(None)
        return bot


    def _evaluate(self, x, out, *args, **kwargs):
        # print("In EVALUATE, eavulating:", x)
        start_time = time.time()
        bot = self.convert_1D_to_bot(x)
        bot.name = f"Design {self.n_evals}"
        if bot.get_design_validity():
            pe, cov = bot.calculate_perception_entropy(self.perception_space)
            out["F"] = [
                pe,                  # minimize perception entropy
                bot.calculate_cost() # minimize cost as is
                ]
        else:
            out["F"] = [
                np.inf,
                np.inf
                ]
        if hasattr(self, "recorder_callback"):
            self.recorder_callback.notify_single(x, out["F"], out["G"] if "G" in out else None)
        self.n_evals += 1

        if 'verbose' in kwargs and kwargs['verbose']:
            print(f"{bot.name} eval took {time.time() - start_time:.2f} sec. PE: {bot.perception_entropy:.3f}, Cov: {bot.perception_coverage_percentage:.3f}")
    
class CustomSensorPkgRandomSampling(Sampling):
    def __init__(self, p=None, **kwargs):
        self.p = p
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        # Get bounds for translation (assuming 3D box)
        xl, xu = problem.bounds()
        xl = np.array(list(xl.values()))
        xu = np.array(list(xu.values()))
        assert np.all(xu >= xl)

        n_sensors = problem.max_n_sensors
        n_types = len(problem.sensor_options)

        # Sensor type probabilities
        if self.p is None:
            p = np.ones(n_types) / n_types
        else:
            p = np.array(self.p)
            assert len(p) == n_types
            assert np.isclose(np.sum(p), 1)

        # Vectorized sampling of sensor types for all samples and all sensors
        sensor_types = np.random.choice(n_types, size=(n_samples, n_sensors), p=p)

        # Vectorized sampling of transforms (here: uniform in [xl, xu])
        # For each sensor: 3x4 matrix flattened
        # We'll use normalized [0,1] for each tf param, then scale/shift as needed
        sensor_tfs = np.random.rand(n_samples, n_sensors, 12)  # shape: (n_samples, n_sensors, 3, 4)

        # Prepare X as a list of dicts (or arrays) for batch conversion
        X = []
        for i in range(n_samples):
            sample_dict = {}
            for j in range(n_sensors):
                sample_dict[f"s{j}_type"] = sensor_types[i, j]
                for k in range(12):
                    sample_dict[f"s{j}_tf_{k}"] = sensor_tfs[i, j, k]
            X.append(sample_dict)

        # TODO batch convert to bots and filter down to valid ones
        # Use joblib or multiprocessing here
        return X
        

def get_pareto_front(df, x='Cost', y='Perception Entropy'):
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

def run_moo(problem:SensorPkgOptimization,
            num_generations:int=10, 
            num_offsprings:int=5,
            population_size:int=10, 
            # mutation_rate:float=0.1, 
            # crossover_rate:float=0.5,
            verbose:bool=False) -> Tuple[OptimizeResult, pd.DataFrame]:
    """Run the mixed-variable multi-objective optimization algorithm on the bot.
    Args:
        num_generations (int): The number of generations to run.
        population_size (int): The size of the population.
        mutation_rate (float): The mutation rate.
        crossover_rate (float): The crossover rate.
        verbose (bool): If True, print debug information.
    Returns:
        Tuple[OptimizeResult, pd.DataFrame]: The optimization result and the design space DataFrame, including all the generated designs.
    """

    progress_callback = ProgressBar(num_generations)
    problem.recorder_callback = DesignRecorder()

    algorithm = MixedVariableGA(
        pop_size=population_size,
        n_offsprings=num_offsprings,
        sampling=CustomSensorPkgRandomSampling(),
        survival=RankAndCrowdingSurvival(),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )

    res = minimize( problem,
                    algorithm,
                    ('n_gen', num_generations),
                    seed=1,
                    callback=lambda algo: [progress_callback.notify(algo, problem), problem.recorder_callback.notify(algo)],
                    verbose=verbose)

    progress_callback.close()

    all_bots_df = problem.recorder_callback.to_dataframe()

    return res, all_bots_df

def plot_tradespace(combined_df:pd.DataFrame, 
                    x=('Cost', '$'), 
                    y=('Perception Entropy', '-'), 
                    hover_name='Name',
                    show_pareto=True, 
                    show=False, 
                    panzoom=True, 
                    **kwargs) -> px.scatter:
    """
    Plot the trade space of concepts based on Cost and Perception Entropy.
    Each point represents a concept, colored based on its optimization status. An ideal point is also marked on the plot.
    The plot can be displayed interactively with optional pan and zoom capabilities.
    Parameters:
        combined_df (pd.DataFrame): DataFrame containing the data to plot, with at least the columns 'Name', 'Cost',
                                    'Perception Entropy'.
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

    num_results = len(combined_df)

    height = 800 if 'height' not in kwargs else kwargs['height']
    width = 800 if 'width' not in kwargs else kwargs['width']
    opacity = 0.9 if 'opacity' not in kwargs else kwargs['opacity']
    title = f"Objective Space ({num_results} concepts)" if 'title' not in kwargs else kwargs['title']
    
    y_min = min(combined_df[y[0]])
    y_max = max(combined_df[y[0]])
    y_range = y_max - y_min
    x_min = min(combined_df[x[0]])
    x_max = max(combined_df[x[0]])
    x_range = x_max - x_min

    fig = px.scatter(combined_df, x=x[0], y=y[0], 
                    #  color='Optimized', 
                     color_discrete_sequence=['#1276a4', '#fc7114'], 
                     opacity=opacity,
                     title=title, 
                     template="plotly_white", 
                     labels={x[0]: f'{x[0]} [{x[1]}]', y[0]: f'{y[0]} [{y[1]}]'},
                     hover_name=hover_name,
                     hover_data=[x[0], y[0]],
                     custom_data=[hover_name])
    
    fig.update_traces(marker=dict(size=5*(width/600)),
                      hovertemplate="<br>".join([
                            "%{customdata[0]}",
                            f"{x[0]} [{x[1]}]: "+"%{x:.2f}",
                            f"{y[0]} [{y[1]}]: "+"%{y:.2f} ",
                            ])
                      )

    fig.add_scatter(x=[0], 
                    y=[min(combined_df[y[0]])], 
                    mode='markers', 
                    marker=dict(symbol='star', size=12*(width/600), color='gold'), 
                    name='Ideal',
                    hoverinfo='none',  # Disable hover data
                    )
    
    if show_pareto:
        pareto, idx = get_pareto_front(combined_df, x="Cost", y="Perception Entropy")
        fig.add_scatter(x=pareto[:, 0],
                        y=pareto[:, 1],
                        mode='lines+markers', 
                        line=dict(color='orange', width=1*(width/600)), 
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
            yanchor="Top",
            y=0,
            xanchor="right",
            x=1
        ),
        yaxis=dict(range=[y_min-(0.1*y_range), y_max+(0.1*y_range)]),
        xaxis=dict(range=[x_min-(0.1*x_range), x_max+(0.1*x_range)]),
    )

    fig.update_layout(clickmode='event+select')
    if show:
        fig.show()
    
    return fig