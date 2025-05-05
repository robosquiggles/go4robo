from .bot_3d_rep import *

import numpy as np

import sys
import traceback

try:
    PLOTLY_MODE = True
    import plotly.express as px
except (ImportError, ModuleNotFoundError):
    PLOTLY_MODE = False
    print("Plotly is not installed in this environment, Plotly visualizations will not work.")
    # traceback.print_exc()
    # sys.exit(1)

try:
    SNS_MODE = True
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1)
except (ImportError, ModuleNotFoundError):
    SNS_MODE = False
    print("Seaborn is not installed in this environment, Seaborn visualizations will not work.")
    # traceback.print_exc()
    # sys.exit(1)

import copy

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.sampling import Sampling
from pymoo.core.mating import Mating
from pymoo.core.repair import Repair
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import Sampling, FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX #, MixedVariableTwoPointsCrossover
from pymoo.operators.mutation.pm import PolynomialMutation #, MixedVariableGaussianMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm

color_scale_blue = ['#94b8c9', '#5a91aa','#2e6985' ,'#216483' , '#004666']

class ProgressBar:
    def __init__(self, n_gen, update_fn=None):
        self.pbar = tqdm(
            total=n_gen,
            desc="MOO",
            unit="gen",
            dynamic_ncols=True,
            leave=True,
            colour='#005b97',
            ncols=80,
            file=sys.stdout  # Ensure tqdm writes to stdout
        )
        self.update_fn = update_fn  # Callback function to update the Isaac Sim progress bar

    def notify(self, algorithm, problem):
        self.pbar.update(1)
        self.pbar.set_postfix({
            "designs": problem.n_designs_generated,
            "min cost": algorithm.pop.get("F").min(axis=0)[1],
            "min PE": algorithm.pop.get("F").min(axis=0)[0]
        })
        self.pbar.refresh()  # Force immediate update to the terminal

        # Update the Isaac Sim progress bar
        if self.update_fn:
            progress = self.pbar.n / self.pbar.total  # Calculate progress as a fraction
            self.update_fn(progress)

    def close(self):
        self.pbar.close()

import pandas as pd
import re

class DesignRecorder:
    def __init__(
            self, 
            max_sensors:int, 
            save_file=None, ):
        assert isinstance(max_sensors, int), "max_sensors must be an int"
        assert max_sensors > 0, "max_sensors must be greater than 0"
        self.records = []
        self.max_sensors = max_sensors
        self.n_notifs = 0
        self.save_file = save_file
        self.last_saved_index = 0
        self.n_designs = 0

        variables = [
            [f"s{i}_type",
             f"s{i}_x",
             f"s{i}_y",
             f"s{i}_z",
             f"s{i}_qw",
             f"s{i}_qx",
             f"s{i}_qy",
             f"s{i}_qz"] for i in range(max_sensors)
               ]
        self.df_column_order = [
            "id",
            "Name",
            "Generation", 
            "Cost", 
            "Perception Entropy", 
            "Perception Coverage"
        ] + [var for var_set in variables for var in var_set]  # Flatten the list of lists

    def notify(self, algorithm):
        # Get the current population's variables and objectives
        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")
        G = algorithm.pop.get("G") if algorithm.pop.has("G") else None
        for x, f, g in zip(X, F, G):
            self.notify_single(x, f, g=g)
        self.n_notifs += 1
    
    def notify_single(self, x, f, g=None):
        # Store as dict for easy DataFrame conversion
        x_dict = {
            "design": x.copy(),
            "id": self.n_designs,
            "Name": f"Design {self.n_designs}",
            "Generation": self.n_notifs,
            "Cost": float(f[1]),
            "Perception Entropy": float(f[0]),
            "Perception Coverage": float(g[0]) if g is not None else None,
        }
        self.records.append(x_dict)
        if self.save_file:
            x_dict_cpy = x_dict.copy()
            # Expand the design dict into separate columns
            for dv in x_dict_cpy["design"].keys():
                # Create new columns for each design variable
                x_dict_cpy[dv] = x_dict_cpy["design"].get(f"{dv}", None)
            # Drop the original design column
            x_dict_cpy.pop("design", None)
            # Reorder columns to put Index and Name at the beginning
            df = pd.DataFrame([x_dict_cpy])
            df = df[self.df_column_order]
            # Save the new records to the CSV file in append mode
            df.to_csv(
                self.save_file, 
                mode='a' if self.n_notifs > 0 else 'w', # this creates the file if it doesn't exist 
                header=False if self.n_notifs > 0 else True, 
                index=False
                )
        self.n_designs += 1

    def notify_init(self, x, f, g=None):
        # Store as dict for easy DataFrame conversion
        self.notify_single(x, f, g)
        self.n_notifs += 1
    
    def to_dataframe(self):
        # Convert to DataFrame for analysis/plotting
        df = pd.DataFrame(self.records)
        df["id"] = df.index
        df["Name"] = df["id"].map(lambda x: f"Design {x}")
        # Rename the "Generation 0" design to "Prior"
        df.loc[df["Generation"] == 0, "Name"] = "Prior Design"

        # Reorder columns to put Index and Name at the beginning
        cols = ["id", "Name"] + [col for col in df.columns if col not in ["id", "Name"]]
        df = df[cols]
        df = self.expand_designs_into_df(df)
        return df
    
    def expand_designs_into_df(self, df, sensor_list=None):

        # Expand the design dict into separate columns
        for dv in df["design"].iloc[0].keys():
            # Create new columns for each design variable
            df[dv] = df["design"].apply(lambda x: x.get(f"{dv}", None))
            
        # Drop the original design column
        df = df.drop(columns=["design"])
        return df
    
    def squash_expanded_df_into_designs(self, df:pd.DataFrame):
        """From a DF, create a dict of designs.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the expanded designs.
            
        Returns:
            (dict, int): A tuple containing (1) the dictionary containing the designs, and the max number of sensors.
            """
        design_dicts = []
        # Use regex to match any key that starts with "s{i}_"
        pattern = re.compile(rf"^s(\d+)_")
        max_sensors = 0
        for i, row in df.iterrows():
            design_dict = {}
            for k, v in row.items():
                match = pattern.match(k)
                if match:
                    sensor_idx = int(match.group(1))
                    max_sensors = max(max_sensors, sensor_idx + 1)
                    design_dict.update({k: v})

    def sensor_options_to_df(self, sensor_options:list[Sensor3D|None]):
        # Convert the sensor options to a DataFrame
        data = []
        for sensor in sensor_options:
            if sensor is None:
                data.append({"Type": "None"})
            else:
                sensor_data = {"Type": type(sensor).__name__}
                sensor_data.update(sensor.get_properties_dict())
                data.append(sensor_data)

        df = pd.DataFrame(data)
        return df

class SensorPkgOptimization(ElementwiseProblem):

    def __init__(
            self, 
            bot:Bot3D, 
            max_n_sensors:int, 
            sensor_options:list[Sensor3D|None], 
            perception_space:PerceptionSpace, 
            sensor_pose_bounds:np.ndarray|list|tuple|None=None,
            **kwargs
            ):
        """
        Initializes the sensor package optimization problem.

        Design Variables (each of N sensors):
            type : int (sensor object enumerated)
            pos  : float (x, y, z) (3D position of the sensor)
            rot  : float (qw, qx, qy, qz) (quaternion rotation of the sensor)
        Parameters:
            bot (Bot3D): The robot object to be optimized.
            sensor_options (list): List of sensor objects to be used in the optimization.
            max_n_sensors (int): Maximum number of sensor instances to be used in the optimization.
            perception_space (PerceptionSpace): The perception space for the optimization.
            sensor_pose_bounds (np.ndarray|list|tuple|None): The bounds for the sensor poses. If None, comes from the robot.
            **kwargs: Additional keyword arguments for the parent class.
        """
        
        # BOT
        self.prior_bot = copy.deepcopy(bot) # Create a deep copy of the bot to avoid modifying the original
        self.bot = copy.deepcopy(bot) # Create a deep copy of the bot to avoid modifying the original
        self.bot.clear_sensors() # Clear any existing sensors
        self.bot.name = "Design" # Set the name of the bot to "Design"

        # SENSORS
        if isinstance(sensor_options, (set, np.ndarray)):
            sensor_options = list(sensor_options)

        # Remove the none sensors from the list. We'll add them back.
        while None in sensor_options:
            sensor_options.remove(None)
        
        sensor_options.sort(key=lambda s: (type(s).__name__, s.name)) # Sort sensor options by type and name
        sensor_options.insert(0, None) # Make sure that we have a single None option at the beginning

        self.sensor_options = {i:s for i, s in enumerate(sensor_options)} # This is the mapping of int to sensor object
        self.max_n_sensors = max_n_sensors

        # PERCEPTION SPACE
        # assert isinstance(perception_space, PerceptionSpace), f"perception_space must be a PerceptionSpace object, but is a {type(perception_space)}"
        self.perception_space = perception_space

        # SENSOR POSE CONSTRAINTS
        if sensor_pose_bounds is None:
            if bot.sensor_pose_constraint is None:
                print("WARNING: No sensor pose constraint provided. Defaulting to a 1m cube.")
                self.s_bounds = np.array([[0,1],
                                        [0,1],
                                        [0,1]]) # Default to a 1m cube
            else:
                if isinstance(bot.sensor_pose_constraint, list):
                    self.s_bounds = np.array(bot.sensor_pose_constraint)
                elif isinstance(bot.sensor_pose_constraint, tuple):
                    self.s_bounds = np.array(bot.sensor_pose_constraint)
                elif isinstance(bot.sensor_pose_constraint, np.ndarray):
                    self.s_bounds = bot.sensor_pose_constraint
                else:
                    raise ValueError("Invalid sensor pose constraint bounds type:", type(bot.sensor_pose_constraint.bounds))
        else:
            if isinstance(sensor_pose_bounds, list):
                self.s_bounds = np.array(sensor_pose_bounds)
            elif isinstance(sensor_pose_bounds, tuple):
                self.s_bounds = np.array(sensor_pose_bounds)
            elif isinstance(sensor_pose_bounds, np.ndarray):
                self.s_bounds = sensor_pose_bounds
            else:
                raise ValueError("Invalid sensor pose constraint bounds type:", type(sensor_pose_bounds))
        assert self.s_bounds.shape == (3, 2), "sensor_pose_constraint must be a 3x2 array of bounds, [[x0,x1], [y0,y1], [z0,z1]]"

        x_bounds = tuple(self.s_bounds[0])
        y_bounds = tuple(self.s_bounds[1])
        z_bounds = tuple(self.s_bounds[2])

        # PROBLEM VARIABLES
        variables = dict()
        for i in range(self.max_n_sensors):
            variables[f"s{i}_type"] = Integer(bounds=(min(self.sensor_options.keys()), max(self.sensor_options.keys())))         # Sensor name (string). 'none' means no sensor
            variables[f"s{i}_x"] = Real(bounds=x_bounds)                                 # Translation x
            variables[f"s{i}_y"] = Real(bounds=y_bounds)                                 # Translation y
            variables[f"s{i}_z"] = Real(bounds=z_bounds)                                 # Translation z
            variables[f"s{i}_qw"] = Real(bounds=(-1, 1))                                 # Quaternion qw
            variables[f"s{i}_qx"] = Real(bounds=(-1, 1))                                 # Quaternion qx
            variables[f"s{i}_qy"] = Real(bounds=(-1, 1))                                 # Quaternion qy
            variables[f"s{i}_qz"] = Real(bounds=(-1, 1))                                 # Quaternion qz
        self.n_var = len(variables)

        # ETCETERA
        self.n_evals = 0
        self.n_designs_generated = 0

        super().__init__(vars=variables, n_obj=2, n_ieq_constr=1, **kwargs)

    def to_json(self, file_path=None):
        """
        Serialize the problem definition to a JSON-compatible dictionary.

        Args:
            file_path (str, optional): Path to save the JSON file. If None, returns the dictionary.

        Returns:
            dict: A dictionary representation of the problem.
        """

        problem_dict = {
            "prior_bot": self.prior_bot.to_json(), # Use the prior bot that has all the sensors!
            "sensor_options": {key: (sensor.to_json() if sensor else "None") for key, sensor in self.sensor_options.items()},
            "max_n_sensors": self.max_n_sensors,
            "sensor_pose_bounds": self.s_bounds.tolist(),  # Convert numpy array to list
            "perception_space": self.perception_space.to_json(),  # Serialize perception space as a string or ID
        }

        if file_path:
            import json
            with open(file_path, "w") as f:
                json.dump(problem_dict, f, indent=4)
        else:
            return problem_dict
        
    def from_json(json_dict):
        """
        Deserialize the problem definition from a JSON-compatible dictionary.

        Args:
            json_dict (dict): A dictionary representation of the problem.

        Returns:
            SensorPkgOptimization: An instance of the optimization problem.
        """

        return SensorPkgOptimization(
            bot=Bot3D.from_json(json_dict["prior_bot"]),
            sensor_options=[Sensor3D.from_json(sensor_dict) for sensor_dict in json_dict["sensor_options"].values()], # This should handle the None case, StereoCamera, or a MonoCamera or Lidar
            max_n_sensors=json_dict["max_n_sensors"],
            perception_space=PerceptionSpace.from_json(json_dict["perception_space"]),
            sensor_pose_bounds=np.array(json_dict["sensor_pose_bounds"]),
        )

    def new_bot_design(self, incr=True) -> Bot3D:
        """
        Creates a new bot object with the same properties as the original bot.
        Returns:
            Bot3D: A new bot object with the same properties as the original bot.
        """
        new_bot = copy.deepcopy(self.bot)
        new_bot.name = f"Design {self.n_designs_generated}"
        if incr:
            self.n_designs_generated += 1
        return new_bot

    def get_sensor_option_key(self, sensor_instance:Sensor3D_Instance|None):
        for key, s in self.sensor_options.items():
            if s == sensor_instance.sensor:
                return key
        raise KeyError(f"Sensor: {sensor_instance} not found in options: {self.sensor_options}")

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

        assert sensor_instance is None or isinstance(sensor_instance, (Sensor3D_Instance, None)), "sensor_instance must be a Sensor3D_Instance or None"
        assert idx < self.max_n_sensors, "idx must be less than max_n_sensors"
        assert sensor_instance is None or sensor_instance.sensor in self.sensor_options.values(), "sensor_instance must be in sensor_options"
        assert sensor_instance is None or sensor_instance.translation is not None, "sensor_instance must have a translation"
        assert sensor_instance is None or sensor_instance.quat_rotation is not None, "sensor_instance must have a quaternion rotation"

        if verbose:
            print("Convert sensor->1d X:", sensor_instance)
        
        if sensor_instance is not None:
            x = {
                f"s{idx}_type": self.get_sensor_option_key(sensor_instance),
                f"s{idx}_x": sensor_instance.translation[0],
                f"s{idx}_y": sensor_instance.translation[1],
                f"s{idx}_z": sensor_instance.translation[2],
                f"s{idx}_qw": sensor_instance.quat_rotation[0],
                f"s{idx}_qx": sensor_instance.quat_rotation[1],
                f"s{idx}_qy": sensor_instance.quat_rotation[2],
                f"s{idx}_qz": sensor_instance.quat_rotation[3],
            }
            
        else:
            x = {
                f"s{idx}_type": 0,
                f"s{idx}_x": 0.0,
                f"s{idx}_y": 0.0,
                f"s{idx}_z": 0.0,
                f"s{idx}_qw": 0.0,
                f"s{idx}_qx": 0.0,
                f"s{idx}_qy": 0.0,
                f"s{idx}_qz": 0.0,
            }

        if dtype == dict:
            return x
        elif dtype == np.ndarray or dtype == np.array or dtype == list:
            return np.array(list(x.values()))
        else:
            raise ValueError("Invalid dtype:", dtype)
    
    def convert_1D_to_sensor_instance(self, x:dict, idx:int, verbose=False) -> Sensor3D_Instance|None:
        """
        Converts a 1D representation of a sensor to a Sensor3D_Instance object.
        Args:
            x (dict): A dictionary containing sensor parameters.
            idx (int): The index of the sensor in the dictionary.
        Returns:
            Sensor: A Sensor3D_Instance object with updated translation and rotation,
                    or None if the sensor type is not available.
        """
        assert idx < self.max_n_sensors, "idx must be less than max_n_sensors"
        assert x is not None, "x must not be None"
        assert type(x) is dict, "x must be a dictionary"

        
        sensor_type = x[f"s{idx}_type"]
        quat = (
            x[f"s{idx}_qw"],
            x[f"s{idx}_qx"],
            x[f"s{idx}_qy"],
            x[f"s{idx}_qz"]
        )
        pos = (
            x[f"s{idx}_x"],
            x[f"s{idx}_y"],
            x[f"s{idx}_z"]
        )
        
        if sensor_type > 0 and sensor_type in self.sensor_options: # 0 is None
            print(f"Found sensor type {sensor_type} in problem sensor options.")
            sensor = self.sensor_options[sensor_type]
            sensor_instance = Sensor3D_Instance(sensor=sensor,
                                                name=f"sensor_{idx}",
                                                tf=(pos, quat),
                                                path='')
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
        
        dict_1D = {}
        for i in range(self.max_n_sensors):
            if i < len(bot.sensors):
                dict_1D.update(self.convert_sensor_instance_to_1D(bot.sensors[i], i, dtype=dict, verbose=verbose))
            else:
                dict_1D.update(self.convert_sensor_instance_to_1D(None, i, dtype=dict, verbose=verbose))
        
        if dtype == dict:
            return dict_1D
        elif dtype == np.ndarray or dtype == np.array:
            return np.array(list(dict_1D.values()))
        elif dtype == list:
            return list(dict_1D.values())
        else:
            raise ValueError("Invalid dtype:", dtype)
                
        

    def convert_1D_to_bot(self, x:dict, verbose=False):
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
        assert isinstance(x, dict), "x must be a dictionary"

        if verbose:
            print("Convert 1d->bot X:", x)

        sensor_instances = []
        for i in range(self.max_n_sensors):
            sensor_instance = self.convert_1D_to_sensor_instance(x, i, verbose=verbose)
            if sensor_instance is not None:
                sensor_instances.append(sensor_instance)

        bot = self.new_bot_design(incr=False)
        bot.name=x["Name"]
        bot.sensors = sensor_instances

        return bot


    def convert_1D_to_spq_tensors(self, X:dict, device=None) -> tuple[list, torch.Tensor, torch.Tensor]:
        """Convert a 1D dictionary of sensor data into separate tensors for sensor types, positions, and quaternions.
        
        Args:
            X (dict): A 1D dictionary containing sensor data.
            device (torch.device, optional): The device to which the tensors should be moved. Defaults to None.
            
        Returns:
            tuple: A tuple containing:
                - sensor_types (list): A list of sensor types.
                - positions_tensor (torch.Tensor): A tensor of shape (N, 3) containing sensor positions.
                - quaternions_tensor (torch.Tensor): A tensor of shape (N, 4) containing sensor rotation quaternions."""
        
        assert isinstance(X, dict), "X must be a dictionary or a list"
        
        sensor_types = []
        positions = []
        quaternions = []

        for i in range(self.max_n_sensors):
            sensor_type = self.sensor_options[X[f"s{i}_type"]]
            if sensor_type == None or sensor_type == 0 or sensor_type == "None":
                continue  # Skip inactive sensors

            sensor_types.append(sensor_type)
            positions.append([
                X[f"s{i}_x"],
                X[f"s{i}_y"],
                X[f"s{i}_z"]
            ])
            quaternions.append([
                X[f"s{i}_qw"],
                X[f"s{i}_qx"],
                X[f"s{i}_qy"],
                X[f"s{i}_qz"]
            ])

        # Convert to tensors
        positions_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
        quaternions_tensor = torch.tensor(quaternions, dtype=torch.float32, device=device)

        # Ensure the right shapes
        if positions_tensor.ndim == 1:
            positions_tensor = positions_tensor.unsqueeze(0)
        if quaternions_tensor.ndim == 1:
            quaternions_tensor = quaternions_tensor.unsqueeze(0)

        # Normalize quaternions
        quaternions_tensor = torch.nn.functional.normalize(quaternions_tensor, p=2, dim=1, eps=1e-12)

        return sensor_types, positions_tensor, quaternions_tensor
    
    def get_normalized_quats_from_1Ds_as_tensor(self, X:np.ndarray, device=None) -> torch.Tensor:
        """Convert a 1D dictionary of sensor data into a tensor of quaternions."""
        assert isinstance(X, np.ndarray), "X must be a numpy array of dicts"
        assert isinstance(X[0], dict), "X must be a numpy array of dicts"
        # Extract quaternions from the 1D dictionaries
        for x_dict in X:
            quats = []  # Use a list to collect quaternions initially
            for i in range(self.max_n_sensors):
                quats.append([
                    x_dict[f"s{i}_qw"],
                    x_dict[f"s{i}_qx"],
                    x_dict[f"s{i}_qy"],
                    x_dict[f"s{i}_qz"]
                ])
    
        if len(quats) == 0:
            # Return an empty tensor if no quaternions are found
            return torch.tensor([], dtype=torch.float32, device=device)

        # Convert the list of quaternions to a tensor
        quats = torch.tensor(quats, dtype=torch.float32, device=device)

        # Normalize quaternions
        quats = torch.nn.functional.normalize(quats, p=2, dim=1, eps=1e-10)

        return quats
    
    def convert_quats_tensor_to_1D_dicts(self, quats:torch.tensor) -> dict:
        """Convert a tensor of quaternions into a 1D dictionary."""
        n_sensors = self.max_n_sensors
        n_quats = quats.shape[0]
        n_xs = int(n_quats/n_sensors)
    
        X = []
        for bot in range(n_xs):
            quats_1D_dict = {}
            for i in range(n_sensors):
                if i < n_quats:
                    quats_1D_dict[f"s{i}_qw"] = quats[i, 0].item()
                    quats_1D_dict[f"s{i}_qx"] = quats[i, 1].item()
                    quats_1D_dict[f"s{i}_qy"] = quats[i, 2].item()
                    quats_1D_dict[f"s{i}_qz"] = quats[i, 3].item()
                else:
                    quats_1D_dict[f"s{i}_qw"] = 0.0
                    quats_1D_dict[f"s{i}_qx"] = 0.0
                    quats_1D_dict[f"s{i}_qy"] = 0.0
                    quats_1D_dict[f"s{i}_qz"] = 0.0
            X.append(quats_1D_dict)
        return np.array(X)
    
    def normalize_quats_in_1D(self, X:dict) -> dict:
        """Normalize the quaternions in a 1D dictionary. Keep the sensor types and translations in the 1D dict the same."""
        quats = self.get_normalized_quats_from_1Ds_as_tensor(X, device=device)
        quats_dicts = self.convert_quats_tensor_to_1D_dicts(quats)
        for i, q_dict in enumerate(quats_dicts):
            X[i].update(q_dict)  # Update the original dictionary with normalized quaternions
        return X

    
    def _eval_bot_obj(self, bot:Bot3D, out=None, *args, **kwargs):
        """Evaluate the design variables and calculate the objectives."""
        start_time = time.time()
        torch.cuda.empty_cache()  # Clear GPU memory

        if out is None:
            out = {}

        if bot.get_design_validity():
            pe, cov = bot.calculate_perception_entropy(self.perception_space)
            cost = bot.calculate_cost()

            out["F"] = [
                pe,                  # Minimize perception entropy
                cost                 # Minimize cost
            ]
            out["G"] = [         # Note G is usually used as a constraint
                cov                 # Track coverage (extra info)
            ]
        else:
            out["F"] = [
                np.inf,             # Inf perception entropy
                np.inf              # Inf cost
            ]
            out["G"] = [
                0.0                 # No coverage
            ]

        if 'verbose' in kwargs and kwargs['verbose']:
            print(f"{bot.name} eval took {time.time() - start_time:.2f} sec. PE: {pe:.3f}, Cov: {cov:.3f}")

        return out


    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate the design variables and calculate the objectives."""
        verbose = kwargs.get('verbose', False)
        if verbose:
            start_time = time.time()

        sensor_types, positions_tensor, quaternions_tensor = self.convert_1D_to_spq_tensors(X, device=device)

        if not sensor_types:
            out["F"] = [np.inf, np.inf]
            out["G"] = [0.0]
            return

        bot:Bot3D = self.new_bot_design()
        bot.add_sensors_batch_quat(sensor_types, positions_tensor, quaternions_tensor)

        print(f"Evaluating bot: {bot.name} with {len(bot.sensors)}") if verbose else None

        self._eval_bot_obj(bot, out, *args, **kwargs)
    
class SensorPkgRandomSampling(Sampling):
    def __init__(self, p=None, **kwargs):
        """
        Custom random sampling for sensor package optimization.

        Args:
            p (list or None): Probabilities for each sensor type. If None, uniform probabilities are used.
        """
        self.p = p
        super().__init__()

    def _do(self, problem:SensorPkgOptimization, n_samples:int, **kwargs):
        """
        Generate random samples for the sensor package optimization problem.

        Args:
            problem (SensorPkgOptimization): The optimization problem instance.
            n_samples (int): Number of samples to generate.

        Returns:
            list[dict]: A list of dictionaries representing the sampled designs.
        """
        # Get the number of sensors and sensor types
        n_sensors = problem.max_n_sensors
        n_types = len(problem.sensor_options)

        # Get bounds for translation and rotation
        xl, xu = problem.bounds()
        xl = np.array(list(xl.values())).reshape(n_sensors,-1)
        xu = np.array(list(xu.values())).reshape(n_sensors,-1)
        assert np.all(xu >= xl), "Upper bounds must be greater than or equal to lower bounds."

        # Sensor type probabilities
        if self.p is None:
            p = np.ones(n_types) / n_types # Uniform distribution
        else:
            p = np.array(self.p)
            
        assert len(p) == n_types, "Probabilities must match the number of sensor types."
        assert np.isclose(np.sum(p), 1), "Probabilities must sum to 1."

        # print(f"Batch sampling {n_samples} designs with {n_sensors} sensors, out of {n_types} sensor options with probabilities {p}")

        # The first sensor cannot be None
        p_first = p.copy()
        p_first[0] = 0.0
        p_first /= np.sum(p_first)  # Normalize to sum to 1

        # Vectorized sampling of sensor types for all samples and all sensors
        first_sensor_type = np.random.choice(list(problem.sensor_options.keys()), size=(n_samples,1), p=p_first)
        other_sensor_types = np.random.choice(list(problem.sensor_options.keys()), size=(n_samples, n_sensors-1), p=p)
        sensor_types = np.concatenate((first_sensor_type, other_sensor_types), axis=1)  # shape (n_samples, n_sensors)

        # Sort sensor types numerically for each sample, with all 0s at the end
        sorted_indices = np.argsort(sensor_types == 0, axis=1)  # Prioritize non-zero values
        sensor_types = np.take_along_axis(sensor_types, sorted_indices, axis=1)
        
        # Vectorized sampling of translations and rotations
        lower_bounds = xl[:,1:4] # shape (n_sensors, 3)
        upper_bounds = xu[:,1:4] # shape (n_sensors, 3)
        lower_bounds = np.tile(lower_bounds, (n_samples, 1, 1))  # shape (n_samples, n_sensors, 3)
        upper_bounds = np.tile(upper_bounds, (n_samples, 1, 1))  # shape (n_samples, n_sensors, 3)
        translations = np.random.uniform(lower_bounds, upper_bounds, size=(n_samples, n_sensors, 3))  # shape (n_samples, n_sensors, 3)
        quaternions = TF.batch_random_quaternions(n_samples * n_sensors, device=device).reshape(n_samples, n_sensors, 4).cpu().numpy()  # qw, qx, qy, qz; shape (n_samples, n_sensors, 4). These are already normalized.

        # Prepare X as a list of dicts for batch conversion
        X = []
        for i in range(n_samples):
            sample_dict = {}
            for j in range(n_sensors):
                if sensor_types[i, j] != 0:
                    sample_dict[f"s{j}_type"] = sensor_types[i, j]
                    sample_dict[f"s{j}_x"] = translations[i, j, 0]
                    sample_dict[f"s{j}_y"] = translations[i, j, 1]
                    sample_dict[f"s{j}_z"] = translations[i, j, 2]
                    sample_dict[f"s{j}_qw"] = quaternions[i, j, 0]
                    sample_dict[f"s{j}_qx"] = quaternions[i, j, 1]
                    sample_dict[f"s{j}_qy"] = quaternions[i, j, 2]
                    sample_dict[f"s{j}_qz"] = quaternions[i, j, 3]
                else:
                    sample_dict[f"s{j}_type"] = 0
                    sample_dict[f"s{j}_x"] = 0.0
                    sample_dict[f"s{j}_y"] = 0.0
                    sample_dict[f"s{j}_z"] = 0.0
                    sample_dict[f"s{j}_qw"] = 0.0
                    sample_dict[f"s{j}_qx"] = 0.0
                    sample_dict[f"s{j}_qy"] = 0.0
                    sample_dict[f"s{j}_qz"] = 0.0

            X.append(sample_dict)

        return X
    

class SensorPkgFlatCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)  # two parents → two children

    def _do(self, problem, X, **kwargs):
        #Print some warnings if anything fishy is going on
        if np.any(np.isnan(X)): print("X-O WARN: NaN values found in X")
        if np.any(np.isinf(X)): print("X-O WARN: Inf values found in X")
        if not np.all(X >= problem.bounds()[0]): print("X-O WARN: Values in X are below the lower bounds")
        if not np.all(X <= problem.bounds()[1]): print("X-O WARN: Values in X exceed the upper bounds")

        # X: shape (2, n_matings, n_var)
        n_parents, n_matings, n_var = X.shape
        Y = X.copy()
        
        # Get flat list of variable names and build index maps
        var_names = list(problem.vars.keys())
        max_n = problem.max_n_sensors

        # Pre-compute all indices
        idx = {}
        for i in range(max_n):
            idx[f"type{i}"] = var_names.index(f"s{i}_type")
            idx[f"x{i}"]    = var_names.index(f"s{i}_x")
            idx[f"y{i}"]    = var_names.index(f"s{i}_y")
            idx[f"z{i}"]    = var_names.index(f"s{i}_z")
            idx[f"qw{i}"]   = var_names.index(f"s{i}_qw")
            idx[f"qx{i}"]   = var_names.index(f"s{i}_qx")
            idx[f"qy{i}"]   = var_names.index(f"s{i}_qy")
            idx[f"qz{i}"]   = var_names.index(f"s{i}_qz")

        # For each mating pair
        for k in range(n_matings):
            p1 = X[0, k]  # parent1 vector of length n_var
            p2 = X[1, k]

            c1 = p1.copy()
            c2 = p2.copy()

            for i in range(max_n):
                # 1) Type: swap with 50% chance
                if np.random.rand() < 0.5:
                    c1[idx[f"type{i}"]], c2[idx[f"type{i}"]] = p2[idx[f"type{i}"]], p1[idx[f"type{i}"]]

                # 2) Position: simple blend
                alpha = np.random.rand()
                for axis in ("x", "y", "z"):
                    ia = idx[f"{axis}{i}"]
                    c1[ia] = alpha * p1[ia] + (1-alpha)*p2[ia]
                    c2[ia] = (1-alpha)*p1[ia] + alpha * p2[ia]

                # 3) Quaternion LERP + normalize
                q1 = np.array([p1[idx[f"qw{i}"]],
                               p1[idx[f"qx{i}"]],
                               p1[idx[f"qy{i}"]],
                               p1[idx[f"qz{i}"]]])
                q2 = np.array([p2[idx[f"qw{i}"]],
                               p2[idx[f"qx{i}"]],
                               p2[idx[f"qy{i}"]],
                               p2[idx[f"qz{i}"]]])
                qt1 = alpha*q1 + (1-alpha)*q2
                qt2 = (1-alpha)*q1 + alpha*q2

                # Normalize
                nt1 = np.linalg.norm(qt1)
                nt2 = np.linalg.norm(qt2)
                if nt1 > 1e-12 and p1[idx[f"type{i}"]] != 0:
                    qt1 /= nt1
                else:
                    qt1[:] = 0  # zero‐out if inactive or degenerate
                if nt2 > 1e-12 and p2[idx[f"type{i}"]] != 0:
                    qt2 /= nt2
                else:
                    qt2[:] = 0

                # Write back
                c1[idx[f"qw{i}"]:]   = qt1
                c2[idx[f"qw{i}"]:]   = qt2

            Y[0, k] = c1
            Y[1, k] = c2

        # Check to make sure all the sensor types are integers
        for i in range(max_n):
            if not isinstance(Y[:, idx[f"type{i}"]], int):
                raise ValueError(f"Sensor type {Y[:, idx[f'type{i}']]} is not an integer after CROSSOVER.")

        return Y
    

class SensorPkgFlatMutation(Mutation):
    def __init__(self, type_prob=0.1, pos_sigma=0.02, quat_sigma=0.02):
        super().__init__()
        self.type_prob  = type_prob
        self.pos_sigma  = pos_sigma
        self.quat_sigma = quat_sigma

    def _do(self, problem, X, **kwargs):
        #Print some warnings if anything fishy is going on
        if np.any(np.isnan(X)): print("MUT WARN: NaN values found in X")
        if np.any(np.isinf(X)): print("MUT WARN: Inf values found in X")
        if not np.all(X >= problem.bounds()[0]): print("MUT WARN: Values in X are below the lower bounds")
        if not np.all(X <= problem.bounds()[1]): print("MUT WARN: Values in X exceed the upper bounds")

        # X: shape (pop_size, n_var) or (offsprings, n_var)
        pop, n_var = X.shape
        Y = X.copy()

        var_names = list(problem.vars.keys())
        max_n = problem.max_n_sensors
        n_types = len(problem.sensor_options)

        # Pre-compute indices
        idx = { }
        for i in range(max_n):
            idx[f"type{i}"] = var_names.index(f"s{i}_type")
            idx[f"x{i}"]    = var_names.index(f"s{i}_x")
            idx[f"y{i}"]    = var_names.index(f"s{i}_y")
            idx[f"z{i}"]    = var_names.index(f"s{i}_z")
            idx[f"qw{i}"]   = var_names.index(f"s{i}_qw")
            idx[f"qx{i}"]   = var_names.index(f"s{i}_qx")
            idx[f"qy{i}"]   = var_names.index(f"s{i}_qy")
            idx[f"qz{i}"]   = var_names.index(f"s{i}_qz")

        # Apply mutation for each individual
        for p in range(pop):
            for i in range(max_n):
                # 1) Type flip
                if np.random.rand() < self.type_prob:
                    Y[p, idx[f"type{i}"]] = np.random.randint(0, n_types)

                # 2) Position noise + clip
                for axis in ("x", "y", "z"):
                    ia = idx[f"{axis}{i}"]
                    lb, ub = problem.s_bounds[("x","y","z").index(axis)]
                    Y[p, ia] += np.random.normal(0, self.pos_sigma)
                    Y[p, ia] = np.clip(Y[p, ia], lb, ub)

                # 3) Quaternion noise + normalize or zero-out
                qs = np.array([Y[p, idx[f"qw{i}"]],
                               Y[p, idx[f"qx{i}"]],
                               Y[p, idx[f"qy{i}"]],
                               Y[p, idx[f"qz{i}"]]])
                qs += np.random.normal(0, self.quat_sigma, 4)
                norm = np.linalg.norm(qs)
                if norm > 1e-12 and Y[p, idx[f"type{i}"]] != 0:
                    qs /= norm
                else:
                    qs[:] = 0
                Y[p, idx[f"qw{i}"]:idx[f"qw{i}"]+4] = qs

        # Check to make sure all the sensor types are integers
        for i in range(max_n):
            if not isinstance(Y[:, idx[f"type{i}"]], int):
                raise ValueError(f"Sensor type {Y[:, idx[f'type{i}']]} is not an integer after MUTATION.")

        return Y
    

class SensorPkgQuaternionRepairTorch(Repair):
    """
    Repair operator that normalizes quaternion variables stored in X.
    X is expected to be a 1D array of dictionaries.
    The quaternions are extracted from the dicts, tensorized, normalized, and placed back in the dicts.
    """
    def _do(self, problem:SensorPkgOptimization, X, **kwargs):
        eps = 1e-10 # Small value to avoid division by zero
        for x in X:
            for i in range(problem.max_n_sensors):
                if not x[f"s{i}_type"] in problem.sensor_options.keys():
                    raise ValueError(f"Sensor type {x[f's{i}_type']} must be one of {problem.sensor_options.keys()}.")
                if x[f"s{i}_type"] == 0:  # If sensor type is 0, set everything to zero
                    x[f"s{i}_qw"] = 0.0
                    x[f"s{i}_qx"] = 0.0
                    x[f"s{i}_qy"] = 0.0
                    x[f"s{i}_qz"] = 0.0
                    x[f"s{i}_x"] = 0.0
                    x[f"s{i}_y"] = 0.0
                    x[f"s{i}_z"] = 0.0
                else:
                    # Normalize the quaternion
                    quat = torch.tensor([
                        x[f"s{i}_qw"],
                        x[f"s{i}_qx"],
                        x[f"s{i}_qy"],
                        x[f"s{i}_qz"]
                    ], dtype=torch.float32)
                    # Correct for zero norm
                    if quat.norm() < eps:
                        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                    quat = torch.nn.functional.normalize(quat, p=2, dim=0, eps=eps)
                    x[f"s{i}_qw"], x[f"s{i}_qx"], x[f"s{i}_qy"], x[f"s{i}_qz"] = quat.tolist()
        return X

        
        

def get_pareto_front(df, x='Cost', y='Perception Entropy',
                     x_minimize=True, y_minimize=True):
    """
    Get the non-dominated Pareto front for a DataFrame of bi-objective designs.

    Args:
        df (pd.DataFrame): DataFrame containing at least the columns x and y.
        x (str): Name of the first objective column.
        y (str): Name of the second objective column.
        x_minimize (bool): If True, smaller x is better; if False, larger x is better.
        y_minimize (bool): If True, smaller y is better; if False, larger y is better.

    Returns:
        pareto_points (np.ndarray): Array of shape (k,2) of the Pareto designs' (x,y).
        pareto_idx   (List[int]):  Indices in `df` corresponding to those designs.
        utopia_point (tuple):      The “best possible” utopia corner for these objectives.
    """
    # 1) Extract raw points
    raw = df[[x, y]].values.astype(float)  # shape (n,2)
    pts = raw.copy()

    # 2) Flip signs if we’re maximizing
    if not x_minimize:
        pts[:, 0] = -pts[:, 0]
    if not y_minimize:
        pts[:, 1] = -pts[:, 1]

    n = pts.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    # 3) Pairwise check: i is dominated if any j beats it on both dims,
    #    and strictly better in at least one.
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            # j dominates i?
            if (pts[j] <= pts[i]).all() and (pts[j] < pts[i]).any():
                is_dominated[i] = True
                break

    # 4) Collect non-dominated
    pareto_idx = np.where(~is_dominated)[0]
    pareto_pts = pts[pareto_idx, :]

    # 5) Restore original signs
    restored = pareto_pts.copy()
    if not x_minimize:
        restored[:, 0] = -restored[:, 0]
    if not y_minimize:
        restored[:, 1] = -restored[:, 1]

    # 6) Sort points for continuity
    restored = restored[np.argsort(restored[:, 0])]

    # 7) Compute utopia point (ideal corner)
    ux = restored[:, 0].min() if x_minimize else restored[:, 0].max()
    uy = restored[:, 1].min() if y_minimize else restored[:, 1].max()
    utopia = (ux, uy)

    return restored, df.index[pareto_idx].tolist(), utopia


def get_hypervolume(df, ref_point, x='Cost', y='Perception Coverage', x_minimize=True, y_minimize=False):
    pareto, idx, u = get_pareto_front(df, x=x, y=y, x_minimize=x_minimize, y_minimize=y_minimize)
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
            verbose:bool=False,
            prior_bot:Bot3D=None,
            progress_callback=None,
            save_dir:str=os.path.join(os.path.dirname(__file__), 'results'),
        ) -> Tuple[OptimizeResult, pd.DataFrame]:
    """Run the mixed-variable multi-objective optimization algorithm on the bot.
    Args:
        problem (SensorPkgOptimization): The optimization problem instance.
        num_generations (int): Number of generations to run the optimization.
        num_offsprings (int): Number of offsprings to generate in each generation.
        population_size (int): Size of the population for the optimization algorithm.
        verbose (bool): If True, print detailed information during the optimization process.
        prior_bot (Bot3D): A prior design to initialize the optimization with.
        progress_callback: A callback function to update progress during optimization.
    Returns:
        Tuple[OptimizeResult, pd.DataFrame]: The optimization result and the design space DataFrame, including all the generated designs.
    """
    assert isinstance(problem, SensorPkgOptimization), "problem must be an instance of SensorPkgOptimization"
    assert isinstance(num_generations, int), "num_generations must be an integer"
    assert isinstance(num_offsprings, int), "num_offsprings must be an integer"
    assert isinstance(population_size, int), "population_size must be an integer"

    if progress_callback is None:
        progress_callback = ProgressBar(num_generations)

    # First figure out where to save the problem and designs
    if save_dir is not None:
        # Generate a unique file name for the results
        import datetime
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_df_file_name = os.path.join(save_dir, f"designs_{problem.prior_bot.name}_{now_str}.csv")
        unique_problem_file_name = os.path.join(save_dir, f"problem_{problem.prior_bot.name}_{now_str}.json")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        problem.to_json(unique_problem_file_name)

    algorithm = MixedVariableGA(
        pop_size=population_size,
        n_offsprings=num_offsprings,
        sampling=SensorPkgRandomSampling(),
        mating=MixedVariableMating(
            repair=SensorPkgQuaternionRepairTorch(), 
            eliminate_duplicates=MixedVariableDuplicateElimination(),
            ),
        survival=RankAndCrowdingSurvival(),
        repair=SensorPkgQuaternionRepairTorch(),
        crossover=SensorPkgFlatCrossover(),
        mutation=SensorPkgFlatMutation(),
    )

    problem.recorder_callback = DesignRecorder(
        problem.max_n_sensors, 
        save_file=unique_df_file_name,
        )
    
    if prior_bot is not None:
        first_bot_out = problem._eval_bot_obj(prior_bot)  # Evaluate the first design
        problem.recorder_callback.notify_init(problem.convert_bot_to_1D(prior_bot, dtype=dict), first_bot_out["F"], first_bot_out["G"])  # Initialize the recorder with the first design

    try:
        res = minimize( problem,
                        algorithm,
                        ('n_gen', num_generations),
                        seed=1,
                        callback=lambda algo: [progress_callback.notify(algo, problem), problem.recorder_callback.notify(algo)],
                        verbose=verbose)
    except Exception as e:
        print('\033[91m' + f"ERROR DURING run_moo()!! Returing res(=None) and df anywho for introspection" + '\033[0m')
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        res = None

    progress_callback.close()

    all_bots_df = problem.recorder_callback.to_dataframe()

    return res, all_bots_df

def plot_tradespace(combined_df:pd.DataFrame, 
                    selected_name=None,
                    x=('Cost', '$'), 
                    y=('Perception Entropy', '-'), 
                    x_minimize=True,
                    y_minimize=True,
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

    assert PLOTLY_MODE, "Plotly is not available in this environment. Please install plotly to use this function."

    num_results = combined_df.shape[0]

    height = 800 if 'height' not in kwargs else kwargs['height']
    width = 800 if 'width' not in kwargs else kwargs['width']
    opacity = 0.8 if 'opacity' not in kwargs else kwargs['opacity']
    title = f"Objective Space ({num_results} designs)" if 'title' not in kwargs else kwargs['title']
    
    y_min = min(combined_df[y[0]]+ [0])
    y_max = max(combined_df[y[0]])
    y_range = y_max - y_min
    x_min = min(combined_df[x[0]]+ [0])
    x_max = max(combined_df[x[0]])
    x_range = x_max - x_min

    # Find the pareto front
    pareto, idx, ut = get_pareto_front(combined_df, x="Cost", y="Perception Entropy", x_minimize=x_minimize, y_minimize=y_minimize)

    # Add a column to the dataframe for pareto
    combined_df['Pareto Optimal'] = ''
    combined_df.loc[idx, 'Pareto Optimal'] = 'Pareto Optimal'

    # Split the "Prior Design" from the rest of the designs as a df
    prior_df = combined_df[combined_df['Name'] == 'Design 0']
    generated_df = combined_df[combined_df['Name'] != 'Design 0']

    # Plot the population of generated designs
    fig = px.scatter(
        generated_df, x=x[0], y=y[0], 
        color='Generation', 
        color_continuous_scale=color_scale_blue, 
        opacity=opacity,
        title=title, 
        template="plotly_white", 
        labels={x[0]: f'{x[0]} [{x[1]}]', y[0]: f'{y[0]} [{y[1]}]'},
        # name='Generated Design',
        hover_name=hover_name,
        hover_data=[x[0], y[0]],
        custom_data=[hover_name]
        )
    
    # Plot the prior/original design
    fig.add_scatter(
        x=prior_df[x[0]].values,
        y=prior_df[y[0]].values,
        mode='markers',
        opacity=opacity,
        marker=dict(symbol='square', size=18 * (width / 600), color='#dd6b00'),
        name='Prior Design',
        hoverinfo='text',
        text=prior_df[hover_name],
        # hover_name=hover_name,
        # hover_data=[x[0], y[0]],
        customdata=[hover_name]
    )
    
    # Set the hover template for the designs
    fig.update_traces(
        marker=dict(size=5*(width/600)),
        hovertemplate="<br>".join([
        "%{customdata[0]}",
        f"{x[0]} [{x[1]}]: "+"%{x:.2f}",
        f"{y[0]} [{y[1]}]: "+"%{y:.2f} ",
        ])
    )

    # Plot the utopia point
    fig.add_scatter(
        x=[ut[0]], 
        y=[ut[1]], 
        mode='markers', 
        marker=dict(symbol='star', size=12*(width/600), color='gold'), 
        name='Ideal',
        hoverinfo='none',  # Disable hover data
    )

    # Plot the Pareto front
    if show_pareto:
        fig.add_scatter(
            x=pareto[:, 0],
            y=pareto[:, 1],
            mode='lines+markers', 
            line=dict(color='orange', width=1*(width/600)), 
            marker=dict(size=10*(width/600), color='orange', symbol='circle-open'),
            name='Pareto Front',
            hoverinfo='none',  # Disable hover data
        )

    # Finally draw a circle around the selected design, if there is one
    if selected_name is not None:
        selected_df = combined_df[combined_df['Name'] == selected_name]
        fig.add_scatter(
            x=selected_df[x[0]].values,
            y=selected_df[y[0]].values,
            mode='markers',
            marker=dict(symbol='circle-open', size=15*(width/600), color='#00729b', line=dict(width=2*(width/600))),
            name='Selected Design',
            hoverinfo='none',  # Disable hover data
        )
    
    
    if not panzoom:
        fig.update_layout(
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True)
        )
    
    fig.update_layout(
        hovermode='closest',
        height=height, width=width,
        legend=dict(
            # orientation="h",
            yanchor="top",
            y=1,
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

def plot_pairplot(df:pd.DataFrame,
                  paired_axes:list[str]=["Cost", "Perception Entropy"],
                  hue:str="Generation",
                  ):
    
    assert SNS_MODE, "Seaborn is not available. Please install seaborn to use this function."

    axes = paired_axes + [hue]
    pair_plot_df = df[axes]
    plot = sns.pairplot(pair_plot_df, hue=hue, diag_kind="kde")
    return plot