import time

from typing import Type

import PIL
import PIL.ImageColor

import copy
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import Bounds, OptimizeResult, NonlinearConstraint, LinearConstraint

try:
    from pxr import UsdGeom, Gf, Sdf, Usd
    print("USD found; USD-specific features will work.")
    USD_MODE = True
except ImportError:
    print("USD not found; USD-specific features will not work.")
    USD_MODE = False

try:
    import open3d as o3d
    print("Open3D found; Open3D-specific features will work.")
    OPEN3D_MODE = True
except ImportError:
    print("Open3D not found; Open3D-specific features will not work.")
    OPEN3D_MODE = False
    # Stub classes so references like o3d.geometry.TriangleMesh won't break outside Open3D
    class o3d:
        class geometry:
            class TriangleMesh:
                pass
        class utility:
            class Vector3dVector:
                pass


class TF:
    def translation_matrix(tx, ty, tz):
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

    # Rotation matrix around X-axis
    def rotation_x_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
    
    # Rotation matrix around Y-axis
    def rotation_y_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ])
    
    # Rotation matrix around Z-axis
    def rotation_z_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])
    
    def inverse_matrix(tf_matrix):
        return np.linalg.inv(tf_matrix)


class Sensor3D:
    def __init__(self, 
                 name:str,
                 type:str,
                 h_fov:float=None, 
                 h_res:int=None,
                 v_fov:float=None,
                 v_res:int=None,
                 max_range:float=None,
                 min_range:float=None,
                 cost:float=None,
                 body:UsdGeom.Mesh=None, 
                 focal_point:tuple[float, float, float]=(0.0, 0.0, 0.0), 
                 ):
        """
        Initialize a new instance of the class.
        Args:
            name (str): The name of the sensor.
            type (str): The type of the sensor.
            h_fov (float): The horizontal field of view *in radians*.
            h_res (int): The horizontal resolution of the sensor.
            v_fov (float): The vertical field of view *in radians*.
            v_res (int): The vertical resolution of the sensor.
            distance (float): The distance that the sensor can sense in meters.
            cost (float): The cost of the sensor.
            body (USDGeom.Mesh): The body of the sensor.
            focal_point (tuple[float]): The focal point of the sensor (relative to the body geometry).
        """
        self.h_fov = h_fov
        self.h_res = h_res
        self.v_fov = v_fov
        self.v_res = v_res
        self.max_range = max_range
        self.min_range = min_range
        self.cost = cost
        self.name = name
        self.type = type
        self.body = body
        if isinstance(focal_point, (list, tuple)):
            self.focal_point = np.array([[1, 0, 0, focal_point[0]],
                                         [0, 1, 0, focal_point[1]],
                                         [0, 0, 1, focal_point[2]],
                                         [0,0,0,1]])
        else:
            self.focal_point = focal_point

    def get_properties_dict(self):
        properties = {}
        for key, value in self.__dict__.items():
            if not key.startswith("__") and not callable(value):
                properties[key] = value
        return properties


class MonoCamera3D(Sensor3D):
    def __init__(self,
                 name:str,
                 focal_length:float=None,
                 h_aperture:float=None,
                 v_aperture:float=None,
                 aspect_ratio:float=None,
                 h_res:int=None,
                 v_res:int=None,
                 body:UsdGeom.Mesh=None,
                 cost:float=None,
                 focal_point:tuple[float, float, float]=(0.0, 0.0, 0.0), 
                 ):

        self.h_aperture = h_aperture
        self.v_aperture = v_aperture
        self.aspect_ratio = aspect_ratio
        self.h_res = h_res
        self.v_res = v_res
        self.body = body
        self.cost = cost
        self.focal_point = focal_point

        self.h_fov = 2 * np.arctan(h_aperture / (2 * focal_length))
        self.v_fov = 2 * np.arctan(v_aperture / (2 * focal_length))

        self.max_range = 100.0 # TODO: This should be clipping distance?
        self.min_range = 0.0 # TODO: This should be clipping distance?

        super().__init__(name, "MonoCamera", self.h_fov, self.h_res, self.v_fov, v_res, self.max_range, self.min_range, self.cost, self.body, self.focal_point)
        

class StereoCamera3D(Sensor3D):
    def __init__(self,
                 name:str,
                 camera1:MonoCamera3D,
                 camera2:MonoCamera3D,
                 tf_camera1:tuple[Gf.Vec3d, Gf.Matrix3d],
                 tf_camera2:tuple[Gf.Vec3d, Gf.Matrix3d],
                 cost:float=None,
                 body:UsdGeom.Mesh=None,
                 ):

        self.camera = camera1
        self.camera2 = camera2
        self.tf_camera1 = tf_camera1
        self.tf_camera2 = tf_camera2
        self.cost = cost
        self.body = body
        self.base_line = np.linalg.norm(tf_camera1[0] - tf_camera2[0])
        self.h_fov = camera1.h_fov
        self.v_fov = camera1.v_fov
        self.h_res = camera1.h_res
        self.v_res = camera1.v_res
        self.max_range = camera1.max_range
        self.min_range = camera1.min_range

        super().__init__(name, "StereoCamera", name=name)
        

class Lidar3D(Sensor3D):
    def __init__(self, 
                 name:str,
                 h_fov:float, 
                 h_res:int,
                 v_fov:float,
                 v_res:int,
                 max_range:float,
                 min_range:float,
                 cost:float,
                 body:UsdGeom.Mesh, 
                 focal_point:tuple[float, float, float]=(0.0, 0.0, 0.0), 
                 ):
        """
        Initialize a new instance of the class.
        Args:
            name (str): The name of the sensor.
            h_fov (float): The horizontal field of view *in radians*.
            h_res (int): The horizontal resolution of the sensor.
            v_fov (float): The vertical field of view *in radians*.
            v_res (int): The vertical resolution of the sensor.
            distance (float): The distance that the sensor can sense in meters.
            cost (float): The cost of the sensor.
            body (USDGeom.Mesh): The body of the sensor.
            focal_point (tuple[float]): The focal point of the sensor (relative to the body geometry).
        """
        super().__init__(name, "Lidar", h_fov, h_res, v_fov, v_res, max_range, min_range, cost, body, focal_point)


class Sensor3D_Instance:
    def __init__(self,
                 sensor:Sensor3D,
                 path:str,
                 tf:tuple[Gf.Vec3d, Gf.Matrix3d],
                 name:str|None=None
                 ):
        self.name = name
        self.sensor = sensor
        self.path = path
        self.tf = tf

    def get_position(self):
        return self.tf[0]
    
    def get_rotation(self):
        return self.tf[1]
    
    def get_transform(self):
        return self.tf
    
    def set_position(self, position:Gf.Vec3d|list[float]):
        if isinstance(position, list):
            position = Gf.Vec3d(position)
        else:
            self.tf[0] = position

    def set_rotation(self, rotation:Gf.Matrix3d):
        self.tf[1] = rotation

    def set_transform(self, tf:tuple[Gf.Vec3d, Gf.Matrix3d]):
        self.tf = tf
    
    def translate(self, translation:Gf.Vec3d|list[float]):
        if isinstance(translation, list):
            translation = Gf.Vec3d(translation)
        self.tf[0] += translation
        return self
    
    def rotate(self, rotation:Gf.Matrix3d):
        self.tf[1] = self.tf[1] * rotation
        return self
    
    def transform(self, tf_matrix):
        self.tf[0] = self.tf[0] * tf_matrix
        self.tf[1] = self.tf[1] * tf_matrix
        return self
    
    def contained_in(self, mesh:o3d.geometry.TriangleMesh):
        """Returns whether or not the sensor body is within the given mesh volume."""
        raise NotImplementedError("This method is not yet implemented.")
    

class Bot3D:
    def __init__(self, 
                 name:str,
                 body:list[UsdGeom.Mesh]=None,
                 path:str=None,
                 sensor_coverage_requirement:list[UsdGeom.Mesh]=None,
                 sensor_pose_constraint:list[UsdGeom.Mesh]=None, 
                 sensors:list[Sensor3D_Instance]=[]):
        """
        Initialize a bot representation with a given shape, sensor coverage requirements, and optional color and sensor pose constraints.
        Args:
            body (open3d.geometry): The mesh body of the bot.
            sensor_coverage_requirement (list[open3d.geometry]): The required coverage area of the sensors.
            color (str): The color of the bot.
            sensor_pose_constraint (list[open3d.geometry]): The constraints on the sensor pose.
            occlusions (list[open3d.geometry]): The occlusions that the sensors must avoid.
        """
        self.name = name
        self.path = path
        self.body = body
        self.sensors = sensors
            
        self.sensor_coverage_requirement = sensor_coverage_requirement
        self.sensor_pose_constraint = sensor_pose_constraint

        # TODO Remove self.body from any of the sensor_coverage_requirement meshes

    def get_sensors_by_type(self, sensor_type:Type[Sensor3D]) -> list[Sensor3D_Instance]:
        sensor_instances = []
        for sensor_instance in self.sensors:
            if isinstance(sensor_instance.sensor, sensor_type):
                sensor_instances.append(sensor_instance.sensor)
        return sensor_instances

    def add_sensor_3d(self, sensor:Sensor3D|list[Sensor3D]|None):
        """
        Adds a 3D sensor to the list of sensors. Only adds a sensor if it is not None.
        Parameters:
            sensor (Sensor3D|None): The 3D sensor to be added (or None).
        Returns:
            bool: True if the sensor was added successfully, False otherwise.
        """
        if sensor is not None:
            if sensor is list:
                self.sensors.extend(sensor)
            else:
                self.sensors.append(sensor)
            return True
        return False

#     def add_sensor_valid_pose(self, sensor:FOV3D, max_tries:int=25, verbose=False):
#         """
#         Adds a sensor to a valid location within the defined constraints.
#         This method generates random points within the bounding box of the 
#         sensor pose constraints and translates the sensor to these points. 
#         It checks if the new sensor pose is valid and, if so, adds the sensor 
#         to the list of sensors.
#         Args:
#             sensor (FOV3D): The sensor to be added, which will be translated 
#                     to a valid location within the constraints.
#         """
#         for i in range(max_tries):
#             x, y = pointpats.random.poisson(self.sensor_pose_constraint, size=1)
#             rotation = -np.degrees(np.arctan2(x, y))

#             sensor.set_translation(x, y)
#             sensor.set_rotation(rotation) #this isn't quite right but good enough
            
#             if self.is_valid_sensor_pose(sensor):
#                 self.add_sensor_2d(sensor)
#                 break
#             if i == max_tries and verbose:
#                 print(f"Did not find a valid sensor pose in {max_tries} tries. Quitting!")
#         return sensor

#     def remove_sensor_by_index(self, index):
#         """Removes a sensor from the sensors list by its index."""
#         del self.sensors[index]

#     def clear_sensors(self):
#         """Removes all sensors from the sensors list."""
#         self.sensors = []

#     def add_sensors_2d(self, sensors:list[FOV2D]):
#         """
#         Adds a list of 2D sensors to the current object.
#         Args:
#             sensors (list[FOV2D]): A list of FOV2D sensor objects to be added.
#         """
#         for sensor in sensors:
#             self.add_sensor_2d(sensor)

#     def is_valid_sensor_pose(self, sensor:FOV2D, verbose=False):
#         """
#         Verifies if the sensor's position is within the defined 
#         sensor pose constraints and does not intersect with any existing sensors.

#         Parameters:
#         sensor (FOV2D): The sensor object whose pose needs to be validated.
#         verbose (bool): If True, prints detailed information about why a sensor 
#                         pose is invalid. Default is False.

#         Returns:
#         bool: True if the sensor pose is valid, False otherwise.
#         """

#         # Check if the sensor is within the sensor pose constraint
#         if not self.sensor_pose_constraint.contains(sensor.bounds):
#             if verbose:
#                 print(f"A Sensor at {sensor.focal_point} is invalid because it is outside of physical constraints.")
#             return False

#         # Check if the sensor does not intersect with any existing sensors
#         for existing_sensor in self.sensors:
#             if sensor.bounds.intersects(existing_sensor.bounds):
#                 if verbose:
#                     print(f"A Sensor at {sensor.focal_point} is invalid because it intersects with the sensor at {existing_sensor.focal_point}.")
#                 return False

#         return True
    
#     def get_package_validity(self, verbose=False):
#         """
#         Check how valid the current configuration of sensors is. Validity is a 
#         measure of how much of the sensor body IS within the sensor pose constraints, 
#         and how much of the sensor body IS NOT intersecting with other sensors.
        
#         Returns:
#             float: A value between -2 and 0 representing the validity of the sensor
#               package, where -2 is completely invalid (all sensors intersecting and
#               outside of the bounds) and 0 is completely valid (all sensors inside
#               the bounds and none intersecting).
#         """
#         if not self.sensors or len(self.sensors) == 0:
#             return 0
#         total_sensor_area = sum(sensor.bounds.area for sensor in self.sensors if sensor is not None)
#         total_sensor_area_invalid = sum(sensor.bounds.difference(self.sensor_pose_constraint).area for sensor in self.sensors if sensor is not None)
#         total_intersection_area = 0.0
#         for i, sensor1 in enumerate(self.sensors):
#             for j, sensor2 in enumerate(self.sensors):
#                 if i != j and sensor1.bounds.intersects(sensor2.bounds):
#                     intersection_area = sensor1.bounds.intersection(sensor2.bounds).area
#                     total_intersection_area += intersection_area
        
#         if verbose:
#             print("Total Sensor Area:", total_sensor_area)
#             print("Total Sensor Area Invalid:", total_sensor_area_invalid)
#             print("Total Intersection Area:", total_intersection_area)

#         return -(total_sensor_area_invalid + total_intersection_area) / total_sensor_area

#     def is_valid_pkg(self, verbose=False):
#         """
#         Check if the current configuration of sensors is valid.
#         This method performs two checks:
#         1. Ensures that all sensors are within the defined sensor pose constraints.
#         2. Ensures that no two sensors intersect with each other.
#         Returns:
#             bool: True if the configuration is valid, False otherwise.
#         """

#         valid = True

#         # Check if all sensors are within the sensor pose constraint
#         for sensor in self.sensors:
#             if verbose:
#                 print("Checking validity of", sensor, " in ", self.sensor_pose_constraint)
#             if not self.sensor_pose_constraint.contains(sensor.bounds):
#                 valid = False
#                 if verbose:
#                     print("Bot Sensor Package is invalid because sensor is outside of physical constraints.")
#                 break

#         # Check if sensors do not touch each other
#         for i, sensor1 in enumerate(self.sensors):
#             for j, sensor2 in enumerate(self.sensors):
#                 if i != j and sensor1.bounds.intersects(sensor2.bounds):
#                     valid = False
#                     if verbose:
#                         print("Sensor Package is invalid because sensors intersect.")
#                     break
#             if not valid:
#                 break
#         if valid and verbose:
#             print("Bot Sensor Package is Valid!") 
#         return valid
    

#     def get_sensor_coverage(self, occluded=True):
#         """
#         Calculate the coverage percentage of the sensors based on the required coverage area.
#         This method computes the total area covered by all sensors and compares it to the required 
#         coverage area. It returns the ratio of the covered area to the required area as a percentage.
#         Returns:
#             float: The coverage percentage of the sensors. Returns 0.0 if there is no coverage requirement.
#         """

#         if not self.sensor_coverage_requirement:
#             return 0.0

#         total_coverage = shapely.geometry.Polygon()
#         for sensor in self.sensors:
#             if sensor is not None:
#                 fov = sensor.fov if not occluded else sensor.get_occluded_fov(self.occlusions)
#                 total_coverage = total_coverage.union(fov)
#         total_coverage = total_coverage.intersection(self.sensor_coverage_requirement)
#         coverage_area = total_coverage.area
#         requirement_area = self.sensor_coverage_requirement.area

#         return (coverage_area / requirement_area)
    
    def get_pkg_cost(self):
        return sum([sensor.cost for sensor in self.sensors if sensor is not None])
    
#     def optimize_sensor_placement(self, method='trust-constr', plot=False, ax=None, plot_title=None, animate=False, anim_interval:int=100, verbose=False):

#         results_hist = {"fun":[],
#                         "x":[],
#                         "validity":[]}
        
#         # Get the bounds of the perception area for normalization
#         largest_dimension = max(*self.sensor_coverage_requirement.bounds)
#         unnorm_bounds = Bounds(lb=[-largest_dimension, -largest_dimension, 0] * len(self.sensors), ub=[largest_dimension, largest_dimension, 360] * len(self.sensors))
        
#         def normalize(params):
#             """
#             Normalize the parameters to the range [0, 1].
#             Args:
#                 params (list): List of parameters [x1, y1, rotation1, x2, y2, rotation2, ...].
#             Returns:
#                 list: Normalized parameters.
#             """
#             lb, ub = unnorm_bounds.lb, unnorm_bounds.ub
#             return [(p - l) / (u - l) for p, l, u in zip(params, lb, ub)]

#         def denormalize(params):
#             """
#             Denormalize the parameters from the range [0, 1] to their original scale.
#             Args:
#                 params (list): List of normalized parameters [x1, y1, rotation1, x2, y2, rotation2, ...].
#             Returns:
#                 list: Denormalized parameters.
#             """
#             lb, ub = unnorm_bounds.lb, unnorm_bounds.ub
#             return [p * (u - l) + l for p, l, u in zip(params, lb, ub)]
        
#         def update_sensors_from_normalized_params(params):
#             for i, sensor in enumerate(self.sensors):
#                 x, y, rotation = denormalize(params)[i*3:(i+1)*3]
#                 sensor.set_translation(x, y)
#                 sensor.set_rotation(rotation)

#         def objective(params):
#             """
#             Objective function to minimize (negative coverage).
#             Args:
#                 params (list): List of parameters [x1, y1, rotation1, x2, y2, rotation2, ...].
#             Returns:
#                 float: Negative of the sensor coverage.
#             """
#             update_sensors_from_normalized_params(params)
#             if verbose:
#                 print(" Objective:", -self.get_sensor_coverage())
#             return -self.get_sensor_coverage()
        
#         def constraint_ineq(params):
#             """
#             Adjusts the translation and rotation of each sensor based on the provided parameters
#             and checks if the package configuration is valid.
#             Args:
#                 params (list): A list of parameters where each set of three consecutive values
#                                represents the x, y translation and rotation for a sensor.
#             Returns:
#                 int: Returns 1 if the package configuration is valid, otherwise returns -1.
#             """
#             update_sensors_from_normalized_params(params)
#             validity = self.get_package_validity()
#             if verbose:
#                 print(" Constraint:", validity)
#             return validity
        
#         def track_history(intermediate_result:OptimizeResult|np.ndarray):
#             """
#             Tracks the history of the optimization process.
#             Args:
#                 xk (list): The current set of parameters.
#             """
#             if isinstance(intermediate_result, np.ndarray):
#                 intermediate_result = OptimizeResult({'fun': -self.get_sensor_coverage(), 'x': intermediate_result})
#             if verbose:
#                 print(" Callback (norm):", intermediate_result)
#             results_hist["fun"].append(intermediate_result.fun)
#             results_hist["x"].append(intermediate_result.x)
#             results_hist["validity"].append(self.get_package_validity())

#         def optimize_coverage():
#             """
#             Optimize the placement of sensors using gradient descent to maximize coverage.
#             Args:
#                 method (str): Optimization method to use. Default is "scipy_gradient_descent".
#             """

#             #PARAMS: x, y, rotation  <-- NORMALIZE
#             initial_params = []
#             for sensor in self.sensors:
#                 initial_params.extend([sensor.focal_point[0], sensor.focal_point[1], sensor.rotation])
#             initial_params = normalize(initial_params)
            
#             if verbose:
#                 print("================== STARTING OPTIMIZATION ==================")
#                 print("Initial Params:", initial_params)

#             #CONSTRAINTS
#             constraints = [NonlinearConstraint(constraint_ineq, 0, np.inf)]
            
#             #RESULTS HISTORY
#             results_hist["fun"] = [-self.get_sensor_coverage()]
#             results_hist["x"] = [initial_params]
#             results_hist["validity"] = [0]
            
#             #OPTIMIZE!
#             result = scipy_minimize(objective, initial_params, method=method, constraints=constraints, callback=track_history)
#             optimized_params = result.x

#             if verbose:
#                 print("Optimized Params (denorm):", denormalize(optimized_params))
#                 print("Optimized Coverage:", -result.fun)
        
#         def plot_coverage_optimization(results:dict, best_valid_iter=None, ax=None):
#             """
#             Plots the convergence of the sensor coverage over time.
#             Args:
#                 results (dict): List of tuples containing result.fun.
#             """
#             iterations = list(range(len(results["fun"])))
#             coverages = -1 * np.array(results["fun"])
#             labels = ['Valid' if v==0 else 'Invalid' for v in results["validity"]]

#             unique_labels = list(set(labels))

#             fig = go.Figure()

#             for label in unique_labels:
#                 label_indices = [i for i, lbl in enumerate(labels) if lbl == label]
#                 fig.add_trace(go.Scatter(
#                     x=[iterations[i] for i in label_indices],
#                     y=[coverages[i] for i in label_indices],
#                     mode='markers',
#                     marker=dict(color='teal' if label == 'Valid' else 'orange'),
#                     name=label
#                 ))

#             if best_valid_iter is not None:
#                 fig.add_trace(go.Scatter(
#                     x=[best_valid_iter],
#                     y=[coverages[best_valid_iter]],
#                     mode='markers',
#                     marker=dict(color='blue', size=12, symbol='circle-open'),
#                     name='Best Valid'
#                 ))

#             fig.update_layout(
#                 title='Convergence of Sensor Coverage Over Time' if plot_title is None else plot_title,
#                 xaxis_title='Optimization Iteration',
#                 yaxis_title='Sensor Coverage',
#                 legend_title='Legend',
#                 template='plotly_white'
#             )

#             if ax is None:
#                 fig.show()
            
#             return fig

#         def animate_optimization(results:dict, interval:int=anim_interval):
#             """
#             Animates the optimization process by plotting the bot at each iteration.
#             Args:
#             results (dict): Dictionary containing the optimization history.
#             interval (int): Time interval between frames in milliseconds.
#             """

#             fig, ax = plt.subplots()
#             bot_copy = copy.deepcopy(self)

#             def update(frame):
#                 ax.clear()
#                 params = results["x"][frame]
#                 for i, sensor in enumerate(bot_copy.sensors):
#                     x, y, rotation = denormalize(params)[i*3:(i+1)*3]
#                     sensor.set_translation(x, y)
#                     sensor.set_rotation(rotation)
#                 bot_copy.plot_bot(ax=ax)
#                 ax.set_title(f"Optimization Iteration {frame}\nCoverage: {-results['fun'][frame]*100:.2f}%\nValidity: {results['validity'][frame]:.5f}", loc='left')

#             ani = FuncAnimation(fig, update, frames=len(results["x"]), interval=interval, repeat=False)
#             return ani
        
#         if not self.sensors or self.get_sensor_coverage() == 1.0:
#             print("No sensors to optimize or already optimal coverage.")
#         else:
#             optimize_coverage()

#         # Find the best valid point from the optimization history
#         best_valid_iter = None
#         best_valid_coverage = -np.inf

#         for i, v in enumerate(results_hist["validity"]):
#             obj = - results_hist["fun"][i]
#             if v==0 and obj > best_valid_coverage:
#                 best_valid_coverage = obj
#                 best_valid_iter = i

#         if best_valid_iter is not None:
#             best_params = results_hist["x"][best_valid_iter]
#             update_sensors_from_normalized_params(best_params)
#             if verbose:
#                 print("Best Valid Iteration:", best_valid_iter)
#                 print("Best Valid Coverage:", best_valid_coverage)
#                 print("Best Valid Params (denorm):", denormalize(best_params))
#         else:
#             if verbose:
#                 print("No valid configuration found in the optimization history, using original.")

#         if plot:
#             if ax is None:
#                 fig, ax = plt.subplots()
#             else:
#                 fig = ax.figure
#             plot_coverage_optimization(results_hist, best_valid_iter=best_valid_iter)
        
#         if animate:
#             ani = animate_optimization(results_hist)
#             return ani
        
#         return results_hist