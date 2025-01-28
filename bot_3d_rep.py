import time

import PIL
import PIL.ImageColor

import open3d as o3d
try:
    import bpy
    import bpy.types
    BLENDER_MODE = True
except ImportError:
    print("Running outside Blender; Blender-specific features will not work.")
    BLENDER_MODE = False
    # Stub classes so references like bpy.types.Object won't break outside Blender
    class bpy:
        class types:
            class Object:
                pass
import bmesh

from matplotlib.path import Path
from matplotlib.patches import PathPatch
import plotly.graph_objects as go

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import copy

import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import Bounds, OptimizeResult, NonlinearConstraint, LinearConstraint
from matplotlib.animation import FuncAnimation


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
    

class Mesh:
    """
    A class to represent a 3D mesh object. Allows for easy operation and conversion between Open3D and Blender mesh objects."""
    
    def __init__(self, mesh:o3d.geometry.TriangleMesh|bpy.types.Object, name:str=None):
        """
        Initialize a new instance of the class.
        Args:
            mesh: The mesh to be converted.
            name (str): The name of the mesh.
        """
        self.name = name
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            self.omesh = mesh
            self.bmesh = self.o3d_to_blender(mesh, name)
        elif isinstance(mesh, bpy.types.Object):
            self.bmesh = mesh
            self.omesh = self.blender_to_o3d(mesh, name)
        else:
            raise ValueError(f"Invalid mesh type {type(mesh)}. Must be either an Open3D TriangleMesh or a Blender Object.")


    def transform(self, tf_matrix):
        """
        Transforms the mesh by the given matrix.
        Args:
            tf_matrix (np.array): The transformation matrix.
        """
        self.omesh.transform(tf_matrix)
        self.bmesh = self.o3d_to_blender(self.omesh, name=self.name)
        return self
    
    def color(self, color:np.array, alpha:float=1.0):
        """
        Colors the mesh with the given color.
        Args:
            color (np.array): The color to apply to the mesh.
        """
        # BMESH
        self.bmesh.data.materials.clear()
        mat = bpy.data.materials.new(name=f"{self.name}_mat")
        mat.diffuse_color = np.append(color, [alpha])
        self.bmesh.data.materials.append(mat)

        #OMESH
        self.omesh.paint_uniform_color(color)
        return self

    def o3d_to_blender(self, o3d_mesh, name):
        """
        Converts an Open3D mesh to a Blender mesh.
        Args:
            o3d_mesh (o3d.geometry.TriangleMesh): The Open3D mesh to convert.
            name (str): The name of the mesh.
        Returns:
            bpy.types.Object: The Blender mesh object.
        """

        # Create a new mesh
        blender_mesh = bpy.data.meshes.new(name=f"{name}_mesh")
        blender_object = bpy.data.objects.new(name=f"{name}_obj", object_data=blender_mesh)
        bpy.context.collection.objects.link(blender_object)

        # Get vertices and faces from Open3D mesh
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)

        # Create a new bmesh
        bm = bmesh.new()

        # Add vertices
        for v in vertices:
            bm.verts.new(v)
        bm.verts.ensure_lookup_table()

        # Add faces
        for f in faces:
            bm.faces.new([bm.verts[i] for i in f])
        bm.faces.ensure_lookup_table()

        # Write the bmesh to the Blender mesh
        bm.to_mesh(blender_mesh)
        bm.free()

        return blender_object
    
    def o3d_show(self):
        o3d.visualization.draw_geometries([self.omesh])

    def blender_to_o3d(self, b_mesh, name):
        """
        Converts a Blender bmesh to an Open3D TriangleMesh.
        Args:
            b_mesh (bmesh.types.BMesh): The Blender bmesh to convert.
            name (str): The name of the mesh.
        Returns:
            o3d.geometry.TriangleMesh: The Open3D TriangleMesh object.
        """
        vertices = []
        faces = []

        for v in b_mesh.verts:
            vertices.append([v.co.x, v.co.y, v.co.z])

        for f in b_mesh.faces:
            faces.append([v.index for v in f.verts])

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        return o3d_mesh
    
    def blender_show(self):
        bpy.context.view_layer.objects.active = self.bmesh
        self.bmesh.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.view3d.view_selected()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        self.bmesh.select_set(False)
    
    
    
class FOV3D:
    def __init__(self, 
                 h_fov:float, 
                 v_fov:float,
                 distance:float,
                 cost:float,
                 body:Mesh=None, 
                 focal_point:tuple[float, float, float]=(0, 0, 0), 
                 tf_matrix: np.array = np.eye(4), 
                 name=None,
                 color: str = 'purple',
                 coord_size: float = 0.2
                 ):
        """
        Initialize a new instance of the class.
        Args:
            h_fov (float): The horizontal field of view *in radians*.
            v_fov (float): The vertical field of view *in radians*.
            distance (float): The distance that the sensor can sense in meters.
            body (open3d.geometry): The body of the sensor.
            focal_point (tuple[float]): The focal point of the sensor (relative to the body geometry).
            tf_matrix (float): The initial tf_matrix of the sensor.
            name (str): The name of the sensor.
            color (str): The color of the sensor.
        """
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.distance = distance
        self.cost = cost
        self.name = name
        self.body = Mesh(body) if not isinstance(body, Mesh) else body
        self.coord_mesh = Mesh(o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_size, origin=focal_point))
        if isinstance(focal_point, (list, tuple)):
            self.focal_point = np.array([[1, 0, 0, focal_point[0]],
                                         [0, 1, 0, focal_point[1]],
                                         [0, 0, 1, focal_point[2]],
                                         [0,0,0,1]])
        else:
            self.focal_point = focal_point
        self.tf_matrix = np.eye(4)
        self.set_transformation(tf_matrix)
        if isinstance(color, str):
            self.color = np.array(PIL.ImageColor.getrgb(color), dtype=np.float64)/255
        elif isinstance(color, tuple):
            self.color = np.array(color, dtype=np.float64)/255
        self.body.color(self.color)
    
    def get_fov_mesh(self, obstacles:list[Mesh]=None):
        """
        Returns a mesh representing the field of view of the sensor for visualization. 
        If obstacles are passed in, resultant FOV mesh will be occluded (calculated using ray casting).
        """
        # TODO occlusions
        return None

    def get_viz_meshes(self, viz_body=True, viz_coord=True, viz_fov=True, obstacles=None, show_now=True) -> list[Mesh]:
        """
        Plots the field of view (FOV) of the object.
        Parameters:
            viz_body (bool): Whether to visualize the body of the sensor.
            viz_coord (bool): Whether to visualize the coordinate frame of the sensor.
            viz_fov (bool): Whether to visualize the field of view of the sensor.
            obstacles (open3d.geometry): The obstacles to consider for occlusion.
        Returns:
            a list of open3d.geometry objects representing the visualization meshes.
        """
        meshes = []

        if viz_body:
            if self.body is not None:
                meshes.append(self.body)
            else:
                print(f"Sensor {self.name} has no body to visualize.")
        
        if viz_coord:
            meshes.append(self.coord_mesh)

        if viz_fov:
            meshes.append(self.get_fov_mesh(obstacles))

        if show_now:
            for mesh in meshes:
                mesh.o3d_show()

        return meshes

    def transform(self, tf_matrix):
        """Transforms the FOV by the given matrix. Also returns the self (FOV3D object) for quick use."""
        print(f"Transforming sensor {self.name} by the tf matrix:\n{tf_matrix}")
        self.set_transformation(self.tf_matrix @ tf_matrix)
        print(f" New TF:\n{self.tf_matrix}")
        print(f" New focal pt:\n{self.focal_point}")
        return self
    
    def set_transformation(self, tf_matrix):
        """Sets the absolute pose of the focal point of the sensor."""
        print(f"Transforming sensor {self.name} to:\n{tf_matrix}")
        # Send the sensor back to the origin
        self.body.transform(TF.inverse_matrix(self.tf_matrix))
        self.coord_mesh.transform(TF.inverse_matrix(self.tf_matrix))
        self.tf_matrix = np.eye(4)

        # Transform the sensor to the new pose
        self.body.transform(tf_matrix)
        self.coord_mesh.transform(tf_matrix)
        self.tf_matrix = tf_matrix
        self.focal_point = self.focal_point @ tf_matrix
        print(f" New TF:\n{self.tf_matrix}")
        print(f" New focal pt:\n{self.focal_point}")
        return self
    
    def contained_in(self, mesh:o3d.geometry.TriangleMesh):
        """Returns whether or not the sensor body is within the given polygon."""
        raise NotImplementedError("This method is not yet implemented.")
    

class Bot3d:
    def __init__(self, 
                 body:o3d.geometry.TriangleMesh,
                 sensor_coverage_requirement:list[o3d.geometry.TriangleMesh],
                 color:str="blue",
                 sensor_pose_constraint:list[o3d.geometry.TriangleMesh]=None, 
                 occlusions:list[o3d.geometry.TriangleMesh]=None,
                 sensors:list[FOV3D]=[]):
        """
        Initialize a bot representation with a given shape, sensor coverage requirements, and optional color and sensor pose constraints.
        Args:
            body (open3d.geometry): The mesh body of the bot.
            sensor_coverage_requirement (list[open3d.geometry]): The required coverage area of the sensors.
            color (str): The color of the bot.
            sensor_pose_constraint (list[open3d.geometry]): The constraints on the sensor pose.
            occlusions (list[open3d.geometry]): The occlusions that the sensors must avoid.
        """
        self.body = body
        self.color = color
        self.sensors = []
        self.add_sensor_3d(sensors)
            
        self.sensor_coverage_requirement = sensor_coverage_requirement
        self.sensor_pose_constraint = sensor_pose_constraint
        self.occlusions = occlusions

        # TODO Remove self.body from any of the sensor_coverage_requirement meshes

    def add_sensor_3d(self, sensor:FOV3D|list[FOV3D]|None):
        """
        Adds a 3D sensor to the list of sensors. Only adds a sensor if it is not None.
        Parameters:
            sensor (FOV3D|None): The 3D sensor to be added (or None).
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

    def show_bot_blender(self, show_constraint=True, show_coverage_requirement=True, show_sensors=True, show_sensor_fovs=True, show_occlusions=True, title=None, ax=None):
        """
        Open Blender showing (optionally, as specified) the robot's shape, sensor constraints, coverage requirements, and sensors.
        
        Parameters:
        -----------
        show_constraint : bool, optional
            If True, plots the sensor pose constraints (default is True).
        show_coverage_requirement : bool, optional
            If True, plots the sensor coverage requirements (default is True).
        show_sensors : bool, optional
            If True, plots the sensors' fields of view (default is True).
        title : str, optional
            The title of the plot (default is None).
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. If None, a new figure and axes are created (default is None).
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.
        """

        # Set the bounds to just beyond the bounds of any of the shapes in the plot
        all_shapes = [self.shape] + [self.sensor_pose_constraint] + [self.sensor_coverage_requirement] + [sensor.fov for sensor in self.sensors]
        min_x = min(shape.bounds[0] for shape in all_shapes)
        min_y = min(shape.bounds[1] for shape in all_shapes)
        max_x = max(shape.bounds[2] for shape in all_shapes)
        max_y = max(shape.bounds[3] for shape in all_shapes)
        ax.set_xlim(min_x - 1, max_x + 1)
        ax.set_ylim(min_y - 1, max_y + 1)
        ax.set_aspect('equal', adjustable='box')

        if title is not None:
            ax.set_title(title)
        
    
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