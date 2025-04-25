import time

from typing import Type

import PIL
import PIL.ImageColor

import os
import copy
import random
import numpy as np
import math

from typing import List, Dict, Tuple, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go

import torch

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import Bounds, OptimizeResult, NonlinearConstraint, LinearConstraint

import omni
import omni.physx as physx
import omni.isaac.core.utils.prims as prim_utils


try:
    from pxr import UsdGeom, Gf, Sdf, Usd, PhysicsSchemaTools
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

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


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
        """Calculate the inverse of a transformation matrix."""
        # Cast to a np.ndarray if not already
        if not isinstance(tf_matrix, np.ndarray):
            tf_matrix_np = np.array(tf_matrix)
        else:
            tf_matrix_np = tf_matrix
        return np.linalg.inv(tf_matrix_np)


class PerceptionSpace:

    class VoxelGroup:
        def __init__(self, 
                    name:str,
                    voxels:list[str]|None,
                    voxel_centers:torch.Tensor|None,
                    voxel_sizes:float|torch.Tensor|None):
            """
            Initialize a new instance of the class.
            Args:
                name (str): The name of the voxel group.
                voxels (list[str]): List of paths to voxels in the stage.
                voxel_centers (torch.Tensor): The centers of the voxels.
                voxel_sizes (torch.Tensor): The sizes of the voxels.
            """
            self.name = name
            self.voxels = voxels if voxels is not None else []
            self.voxel_centers = voxel_centers if voxel_centers is not None else torch.tensor([])
            if voxel_sizes is float:
                self.voxel_sizes = torch.tensor([voxel_sizes] * len(voxels))
            else:
                self.voxel_sizes = voxel_sizes if voxel_sizes is not None else torch.tensor([])

            # all the tensors must be the same size
            if len(voxels) != voxel_centers.shape[0] or len(voxels) != voxel_sizes.shape[0]:
                raise ValueError("All tensors must be the same size.")

        def add_voxel(self, voxel:str, center:Gf.Vec3d, size:float):
            """
            Add a voxel to the voxel group.
            Args:
                voxel (str): The voxel to be added.
                center (Gf.Vec3d): The center of the voxel.
                size (float): The size of the voxel.
            """
            self.voxels = self.voxels + [voxel]
            self.voxel_centers = torch.cat((self.voxel_centers, torch.tensor([center])))
            self.voxel_sizes = torch.cat((self.voxel_sizes, torch.tensor([size])))

        def remove_voxel(self, voxel:str):
            """
            Remove a voxel from the voxel group.
            Args:
                voxel (str): The voxel to be removed.
            """
            indices = [i for i, v in enumerate(self.voxels) if v == voxel]
            if len(indices) > 0:
                self.voxels = [v for i, v in enumerate(self.voxels) if i not in indices]
                self.voxel_centers = torch.cat((self.voxel_centers[:indices], self.voxel_centers[indices+1:]))
                self.voxel_sizes = torch.cat((self.voxel_sizes[:indices], self.voxel_sizes[indices+1:]))

        def remove_voxel_by_center(self, center:Gf.Vec3d):
            """
            Remove a voxel from the voxel group by its center.
            Args:
                center (Gf.Vec3d): The center of the voxel to be removed.
            """
            index = torch.where(self.voxel_centers == center)[0]
            if index.size(0) > 0:
                self.voxels = [v for i, v in enumerate(self.voxels) if i != index]
                self.voxel_centers = torch.cat((self.voxel_centers[:index], self.voxel_centers[index+1:]))
                self.voxel_sizes = torch.cat((self.voxel_sizes[:index], self.voxel_sizes[index+1:]))


    def __init__(self,
                 usd_context:omni.usd.UsdContext,
                 voxel_groups:list[VoxelGroup]|np.ndarray[VoxelGroup]=None, 
                 weights:list[float]|np.ndarray[float]=None):
        """Initialize a new instance of the class.
        Args:
            usd_context (omni.usd.UsdContext): The USD context.
            voxel_groups (torch.Tensor[VoxelGroup]): The voxel groups.
            weights (torch.Tensor[float]): The weights of the voxel groups.
        """
        self.usd_context = usd_context
        self.stage = self.usd_context.get_stage()
        self.voxel_groups = voxel_groups if voxel_groups is not None else np.array([])
        self.weights = weights if weights is not None else np.array([])

    def add_voxel_group(self, voxel_group:VoxelGroup, weight:float):
        """
        Add a voxel group to the perception space.
        Args:
            voxel_group (VoxelGroup): The voxel group to be added.
            weight (float): The weight of the voxel group.
        """
        
        self.voxel_groups = np.append(self.voxel_groups, voxel_group)
        self.weights = np.append(self.weights, weight)

    def remove_voxel_group(self, voxel_group:VoxelGroup):
        """
        Remove a voxel group from the perception space.
        Args:
            voxel_group (VoxelGroup): The voxel group to be removed.
        """
        index = np.where(self.voxel_groups == voxel_group)[0]
        if index.size(0) > 0:
            self.voxel_groups = np.delete(self.voxel_groups, index)
            self.weights = np.delete(self.weights, index)

    def get_voxel_group(self, name:str) -> VoxelGroup:
        """
        Get a voxel group by its name.
        Args:
            name (str): The name of the voxel group.
        Returns:
            VoxelGroup: The voxel group.
        """
        index = np.where(self.voxel_groups == name)[0]
        if index.size(0) > 0:
            return self.voxel_groups[index]
        else:
            raise ValueError(f"Voxel group {name} not found.")
        
    def get_group_names(self) -> list[str]:
        """
        Get the names of the voxel groups.
        Returns:
            list[str]: The names of the voxel groups.
        """
        group_names = []
        for voxel_group in self.voxel_groups:
            group_names.append(voxel_group.name)
        return group_names
        
    def get_voxel_paths(self) -> list[str]:
        """
        Get the paths of the voxels in the voxel groups.
        Returns:
            list[str]: The paths of the voxels.
        """
        voxel_paths = []
        for voxel_group in self.voxel_groups:
            voxel_paths.append(voxel_group.voxels)
        return voxel_paths
    
    def get_voxel_weights(self) -> torch.Tensor:
        """
        Get the weights of the voxel groups.
        Returns:
            torch.Tensor[float]: The weights of the voxel groups, shape (N,) where N is the number of voxels.
        """
        voxel_weights = torch.tensor([])
        for i, w in enumerate(self.weights):
            ws_tensor = torch.tensor([w]).repeat(len(self.voxel_groups[i].voxels))
            voxel_weights = torch.cat((voxel_weights, ws_tensor), dim=0)
        return voxel_weights
    
    def get_voxel_centers(self) -> torch.Tensor:
        """
        Get the centers of the voxels in the voxel groups.
        Returns:
            torch.Tensor[Gf.Vec3d]: The centers of the voxels.
        """
        voxel_centers = []
        for voxel_group in self.voxel_groups:
            voxel_centers.append(voxel_group.voxel_centers)
        return torch.cat(voxel_centers)
    
    def get_voxel_sizes(self) -> torch.Tensor:
        """
        Get the sizes of the voxels in the voxel groups.
        Returns:
            torch.Tensor[float]: The sizes of the voxels.
        """
        voxel_sizes = []
        for voxel_group in self.voxel_groups:
            voxel_sizes.append(voxel_group.voxel_sizes)
        return torch.cat(voxel_sizes)
    
    def get_voxel_mins(self) -> torch.Tensor:
        """
        Get the minimum coordinates of the voxels in the voxel groups.
        Returns:
            torch.Tensor[Gf.Vec3d]: The minimum coordinates of the voxels.
        """
        voxel_centers = self.get_voxel_centers()
        voxel_sizes = self.get_voxel_sizes()
        return voxel_centers - (voxel_sizes / 2)
    
    def get_voxel_maxs(self) -> torch.Tensor:
        """
        Get the maximum coordinates of the voxels in the voxel groups.
        Returns:
            torch.Tensor[Gf.Vec3d]: The maximum coordinates of the voxels.
        """
        voxel_centers = self.get_voxel_centers()
        voxel_sizes = self.get_voxel_sizes()
        return voxel_centers + (voxel_sizes / 2)
    
    def get_vozel_meshes_from_stage(self, voxel_group_name:str|None) -> List[UsdGeom.Mesh]:
        """
        Get the meshes of the voxels in the voxel group. If voxel_group_name is None, return all the voxels from the stage.
        Returns:
            List[UsdGeom]: The meshes of the voxels.
        """
        if voxel_group_name is None:
            # Get all the meshes in the stage
            meshes = []
            for prim in self.stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    meshes.append(prim)
            return meshes
        else:
            # Get the meshes in the voxel group
            voxel_group = self.get_voxel_group(voxel_group_name)
            meshes = []
            for voxel in voxel_group.voxels:
                mesh = self.stage.GetPrimAtPath(voxel)
                if mesh.IsA(UsdGeom.Mesh):
                    meshes.append(mesh)
            return meshes
        
    def set_voxel_group_weight(self, voxel_group_name:str, weight:float):
        """
        Set the weight of the voxel group.
        Args:
            voxel_group_name (str): The name of the voxel group.
            weight (float): The weight of the voxel group.
        """
        index = torch.where(self.voxel_groups == voxel_group_name)[0]
        if index.size(0) > 0:
            self.weights[index] = weight
        else:
            raise ValueError(f"Voxel group {voxel_group_name} not found.")
        
    
    def batch_ray_voxel_intersections(self, ray_origins:torch.Tensor, ray_directions:torch.Tensor, batch_size:int=1000) -> torch.Tensor:
        """
        Check if the rays intersect with the voxels in the voxel groups.
        Args:
            ray_origins (torch.Tensor): The origins of the rays.
            ray_directions (torch.Tensor): The directions of the rays.
        Returns:
            torch.Tensor: A tensor of shape (N,) where N is the number of voxels. Each element is the number of rays that intersect with the voxel.
        """
        start_time = time.time()
        # Use the GPU if available to make this quick
        
        ray_origins = ray_origins                     # Shape: (R, 3)
        ray_directions = ray_directions               # Shape: (R, 3)
        voxel_mins = self.get_voxel_mins().to(device) # Shape: (N, 3)
        voxel_maxs = self.get_voxel_maxs().to(device) # Shape: (N, 3)

        num_rays = ray_origins.shape[0]
        num_voxels = voxel_mins.shape[0]

        v_hits = torch.zeros(num_voxels)

        for start in range(0, num_rays, batch_size):
            end = min(start + batch_size, num_rays)
            # Expand rays and voxels for broadcasting
            rays_o = ray_origins[start:end].unsqueeze(1).expand(-1, num_voxels, -1)     # Shape: (R, N, 3)
            rays_d = ray_directions[start:end].unsqueeze(1).expand(-1, num_voxels, -1)  # Shape: (R, N, 3)
            boxes_min = voxel_mins.unsqueeze(0).expand(end-start, -1, -1)               # Shape: (R, N, 3)
            boxes_max = voxel_maxs.unsqueeze(0).expand(end-start, -1, -1)               # Shape: (R, N, 3)

            inv_dir = 1.0 / rays_d
            tmin = (boxes_min - rays_o) * inv_dir
            tmax = (boxes_max - rays_o) * inv_dir

            t1 = torch.minimum(tmin, tmax)
            t2 = torch.maximum(tmin, tmax)

            t_enter = torch.max(t1, dim=2).values
            t_exit = torch.min(t2, dim=2).values

            # A ray intersects a voxel if t_enter <= t_exit and t_exit >= 0
            hits = (t_enter <= t_exit) & (t_exit >= 0)
            # Shape: (R, N) where R is the number of rays and N is the number of voxels. 
            # Each element is True if the ray intersects with the voxel, False otherwise.

            # To get the number of rays that intersect with each voxel, we can sum along the first dimension
            v_hits += hits.sum(dim=0)

            # v_hits.cpu() # Move back to CPU if needed

        print(f"Batch ray voxel intersectiontraversal took {time.time() - start_time:.2f} seconds for {num_rays} rays and {num_voxels} voxels.")

        return v_hits  # Shape: (N,), hits for each voxel


class Sensor3D:
    def __init__(self, 
                 name:str,
                 h_fov:float=None, 
                 h_res:int=None,
                 v_fov:float=None,
                 v_res:int=None,
                 max_range:float=None,
                 min_range:float=None,
                 cost:float=None,
                 body:UsdGeom.Mesh=None, 
                 focal_point:tuple[float, float, float]=(0.0, 0.0, 0.0), 
                 ap_constants:dict = {
                        "a": 0.055,  # coefficient from the paper for camera
                        "b": 0.155   # coefficient from the paper for camera
                    }
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

        self.ap_constants = ap_constants

    def get_properties_dict(self):
        properties = {}
        for key, value in self.__dict__.items():
            if not key.startswith("__") and not callable(value):
                properties[key] = value
        return properties
    
    def calculate_ap(self, pixel_count: int) -> float:
        """Calculate Average Precision (AP) based on pixel count using the paper's formula"""
        if pixel_count <= 0:
            return 0.001  # Minimal AP for numerical stability
        
        # Using the formula from the paper: AP ≈ a * ln(m) + b
        
        ap = self.ap_constants['a'] * math.log(pixel_count) + self.ap_constants['b']
        
        # Clamp AP to valid range
        ap = max(0.001, min(0.999, ap))
        
        return ap
    
    def calculate_ap_sigma(self, pixel_count: int) -> float:
        ap = self.calculate_ap(pixel_count)
        sigma = (1 / ap) - 1
        return sigma
    
    def calculate_gaussian_entropy(self, sigma: float) -> float:
        """Calculate the entropy of a 2D Gaussian distribution with given standard deviation"""
        # Using the formula from the paper: H(S|m, q) = 2*ln(σ) + 1 + ln(2π)
        sigma = self.calculate_ap_sigma(sigma)
        entropy = 2 * math.log(sigma) + 1 + math.log(2 * math.pi)
        return entropy



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
                 ap_constants:dict = {
                        "a": 0.055,  # coefficient from the paper for camera
                        "b": 0.155   # coefficient from the paper for camera
                    }
                 ):
        """
        Initialize a new instance of the class.
        Args:
            name (str): The name of the sensor.
            focal_length (float): The focal length of the camera.
            h_aperture (float): The horizontal aperture of the camera.
            v_aperture (float): The vertical aperture of the camera.
            aspect_ratio (float): The aspect ratio of the camera.
            h_res (int): The horizontal resolution of the camera.
            v_res (int): The vertical resolution of the camera.
            body (USDGeom.Mesh): The body of the sensor.
            cost (float): The cost of the sensor.
            focal_point (tuple[float]): The focal point of the sensor (relative to the body geometry).
        """

        self.name = name
        self.h_aperture = h_aperture
        self.v_aperture = v_aperture
        self.aspect_ratio = aspect_ratio
        self.body = body
        self.cost = cost
        if isinstance(focal_point, (list, tuple)):
            self.focal_point = np.array([[1, 0, 0, focal_point[0]],
                                         [0, 1, 0, focal_point[1]],
                                         [0, 0, 1, focal_point[2]],
                                         [0,0,0,1]])
        else:
            self.focal_point = focal_point

        self.h_fov = np.rad2deg(2 * np.arctan(h_aperture / (2 * focal_length)))
        self.v_fov = np.rad2deg(2 * np.arctan(v_aperture / (2 * focal_length)))

        self.h_res = self.h_fov/h_res # number of degrees between pixels. It is the way it is for isaac sim ray casting, don't ask me why
        self.v_res = self.v_fov/v_res # number of degrees between pixels. It is the way it is for isaac sim ray casting, don't ask me why

        self.max_range = 100.0 # TODO: This should be clipping distance?
        self.min_range = 0.0 # TODO: This should be clipping distance?

        self.ap_constants = ap_constants
        

class StereoCamera3D(Sensor3D):
    def __init__(self,
                 name:str,
                 sensor1:MonoCamera3D,
                 sensor2:MonoCamera3D,
                 tf_sensor1:tuple[Gf.Vec3d, Gf.Matrix3d],
                 tf_sensor2:tuple[Gf.Vec3d, Gf.Matrix3d],
                 cost:float=None,
                 body:UsdGeom.Mesh=None,
                 ap_constants:dict = {
                        "a": 0.055,  # coefficient from the paper for camera
                        "b": 0.155   # coefficient from the paper for camera
                    }
                 ):
        """
        Initialize a new instance of the class.
        Args:
            name (str): The name of the sensor.
            sensor1 (MonoCamera3D): The first camera sensor.
            sensor2 (MonoCamera3D): The second camera sensor.
            tf_sensor1 (tuple[Gf.Vec3d, Gf.Matrix3d]): The transformation matrix for the first camera.
            tf_sensor2 (tuple[Gf.Vec3d, Gf.Matrix3d]): The transformation matrix for the second camera.
            cost (float): The cost of the sensor.
            body (USDGeom.Mesh): The body of the sensor.
        """

        self.name = name
        self.sensor1 = sensor1
        self.sensor2 = sensor2
        self.tf_1 = tf_sensor1
        self.tf_2 = tf_sensor2
        self.cost = cost
        self.body = body
        self.base_line = np.linalg.norm(tf_sensor1[0] - tf_sensor2[0])
        self.h_fov = sensor1.h_fov
        self.v_fov = sensor1.v_fov
        self.h_res = sensor1.h_res
        self.v_res = sensor1.v_res
        self.max_range = sensor1.max_range
        self.min_range = sensor1.min_range
        self.ap_constants = ap_constants


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
                 ap_constants = {
                        "a": 0.152,  # coefficient from the paper for lidar
                        "b": 0.659   # coefficient from the paper for lidar
                    }
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
        super().__init__(name, h_fov, h_res, v_fov, v_res, max_range, min_range, cost, body, focal_point, ap_constants=ap_constants)


class Sensor3D_Instance:
    def __init__(self,
                 sensor:Sensor3D,
                 path:str,
                 usd_context:omni.usd.UsdContext,
                 tf:tuple[Gf.Vec3d, Gf.Matrix3d],
                 name:str|None=None,
                 ):
        """Initialize a new instance of the class.
        Args:
            sensor (Sensor3D): The sensor object.
            path (str): The path to the sensor in the USD stage.
            tf (tuple[Gf.Vec3d, Gf.Matrix3d]): The transformation matrix for the sensor.
            name (str|None): The name of the sensor instance. If None, use the sensor's name.
        """
        self.name = name
        self.sensor = sensor
        self.tf = tf

        self.usd_context = usd_context
        self.stage = self.usd_context.get_stage()
        self.path = path

        self.ray_casters = []
        self.ray_casters = self.create_ray_casters()
        self.body = self.create_sensor_body(sensor.body)
        

    def create_sensor_body(self, body:UsdGeom.Mesh):
        """Create the sensor body in the USD stage. Returns the created sensor body."""
        # First check the stage for the sensor body
        if self.stage.GetPrimAtPath(self.path).IsValid():
            # print(f"Sensor body {self.path} already exists in stage, adding it to Sensor3D_Instance: {self.name}.")
            return prim_utils.get_prim_at_path(self.path)
        else:
            # If the sensor body does not exist, create it
            # Copy the body to the sensor path on the stage
            result, sensor_body = omni.kit.commands.execute('CopyPrim',
                                                           path_from=body.GetPath(),
                                                           path_to=self.path,
                                                           exclusive_select=False,
                                                           copy_to_introducing_layer=False
                                                           )
            if result:
                return sensor_body
            else:
                print(f"Failed to create sensor body for {self.name} at {self.path}. Skipping!")
                return None


    def create_ray_casters(self, disable=False):
        """Check if the ray casters have been created in the stage. If not, create them. Sets self.ray_casters to the created ray casters. Returns the created ray casters in a list."""
        import omni.kit.commands
        import isaacsim.core.utils.transformations as tf_utils
        import isaacsim.core.utils.xforms as xforms_utils

        def tree_search(prim, name):
            """Search the tree for the ray caster with name: f'/GO4R_RAYCASTER_{sensor.name}'"""
            if str(prim.GetPath()).endswith(name):
                return prim
            for child in prim.GetChildren():
                result = tree_search(child, name)
                if result:
                    return result
        

        if self.ray_casters == []: # No ray casters are loaded
            if isinstance(self.sensor, StereoCamera3D):
                # If the sensor is a stereo camera, the ray casters were already created for the mono cameras
                # We just have to find them and add them to the list
                sensors = [self.sensor.sensor1, self.sensor.sensor2]
                for sensor in sensors:
                    # Search the tree for the ray caster with name: f'/GO4R_RAYCASTER_{sensor.name}'
                    ray_caster_path_end = f'/GO4R_RAYCASTER_{sensor.name}'
                    highest_path = self.get_ancestor_path(1)
                    ray_caster = tree_search(self.stage.GetPrimAtPath(highest_path), ray_caster_path_end)
                    if ray_caster:
                        # print(f"Ray caster {ray_caster_path_end} already exists in stage, adding it to Sensor3D_Instance: {self.name}.")
                        self.ray_casters.append(ray_caster)


            else:
                sensor = self.sensor

                parent_path = self.get_ancestor_path(1)
                ray_caster_path = parent_path + f'/GO4R_RAYCASTER_{sensor.name}'
                # First check the stage for the ray caster
                if self.stage.GetPrimAtPath(ray_caster_path).IsValid():
                    # print(f"Ray caster {ray_caster_path} already exists in stage, adding it to Sensor3D_Instance: {self.name}.")
                    self.ray_casters.append(prim_utils.get_prim_at_path(ray_caster_path))

                # If the ray caster does not exist, create it
                else:
                    # First get the path that is currently selected in the stage
                    # This is the path that the ray caster will be parented to

                    # Then select the prim from the stage
                    omni.kit.commands.execute('SelectPrims',
                                              old_selected_paths=[self.usd_context.get_selection().get_selected_prim_paths()],
                                              new_selected_paths=[self.get_ancestor_path(1)])
                    #Then create the ray caster at the selection
                    result, rc_prim = omni.kit.commands.execute('RangeSensorCreateLidar',
                        path=f'/GO4R_RAYCASTER_{sensor.name}',
                        parent=parent_path,
                        min_range=sensor.min_range,
                        max_range=sensor.max_range,
                        draw_points=True, # Turn this off for simulation performance
                        draw_lines=False, # Turn this off for simulation performance
                        horizontal_fov=sensor.h_fov,
                        vertical_fov=sensor.v_fov,
                        horizontal_resolution=sensor.h_res,
                        vertical_resolution=sensor.v_res,
                        rotation_rate=0.0, # Generate all points at once!
                        high_lod=True, # Generate all points at once!
                        yaw_offset=0.0,
                        enable_semantics=True)
                    
                    if disable:
                        omni.kit.commands.execute('ToggleActivePrims',
                                    prim_paths=[ray_caster_path],
                                    active=False,
                                    stage_or_context=self.stage)
                    
                    #If the ray caster is a MonoCamera3D, set the transform to match the Camera (which is rotated 90 about x, and -90 about y)
                    if isinstance(sensor, MonoCamera3D):

                        cam_prim = prim_utils.get_prim_at_path(self.path)
                        parent_prim = prim_utils.get_prim_at_path(parent_path)

                        # Get the original/current rc local transform
                        rc_local_transform_orig = Gf.Matrix4d(tf_utils.get_relative_transform(parent_prim, rc_prim))

                        # Get the camera local transform
                        cam_local_pose = xforms_utils.get_local_pose(str(cam_prim.GetPath()))
                        cam_local_tf_from_pose = tf_utils.tf_matrix_from_pose(*cam_local_pose)

                        cam_to_rc_tf = TF.rotation_y_matrix(np.pi/2) @ TF.rotation_x_matrix(-np.pi/2)
                        cam_to_rc_tf = TF.inverse_matrix(np.array([[ 0.0, -1.0,  0.0,  0.0],
                                                                   [ 0.0,  0.0,  1.0,  0.0],
                                                                   [-1.0,  0.0,  0.0,  0.0],
                                                                   [ 0.0,  0.0,  0.0,  1.0]]))
                        

                        # Get the new ray caster local transform
                        rc_local_transform = Gf.Matrix4d(cam_to_rc_tf @ cam_local_tf_from_pose.T) #TODO Try to use Torch/Tensors?

                        # Transform the ray caster to the correct camera transform
                        omni.kit.commands.execute('TransformPrimCommand',
                            path=rc_prim.GetPath(),
                            old_transform_matrix=rc_local_transform_orig,
                            new_transform_matrix=rc_local_transform,
                            time_code=Usd.TimeCode(),
                            had_transform_at_key=False)

                        if result:
                            self.ray_casters.append(rc_prim)
                        else:
                            print(f"Failed to create ray caster for {sensor.name} at {self.path}. Skipping!")
        else:
            print(f"Ray casters already exist for {self.name}. Skipping creation.")

        return self.ray_casters
    
    def get_prim(self):
        """Get the USD prim for the sensor"""
        return self.stage.GetPrimAtPath(self.path)
    
    def get_world_transform(self) -> Gf.Matrix4d:
        """Get the world transform of a prim"""
        xform = UsdGeom.Xformable(self.get_prim())
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Extract position and rotation
        # position = Gf.Vec3d(world_transform.ExtractTranslation())
        # rotation = world_transform.ExtractRotationMatrix()
        
        return world_transform

    def get_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (ray_origins, ray_directions) for this sensor instance.

        Returns
        -------
            ray_origins: (N, 3) numpy array of world-space ray origins.
            ray_directions: (N, 3) numpy array of world-space ray directions.
        """
        # TODO: Write this with torch with cuda for performance

        start_time = time.time()

        ray_origins = np.zeros((0, 3))
        ray_directions = np.zeros((0, 3))
        # ray_distances = np.zeros((0,))

        for ray_caster in self.ray_casters:
            hfov = ray_caster.GetAttribute("horizontalFov").Get() #degrees
            vfov = ray_caster.GetAttribute("verticalFov").Get() #degrees
            hres = int(hfov / ray_caster.GetAttribute("horizontalResolution").Get())
            vres = int(vfov / ray_caster.GetAttribute("verticalResolution").Get())
            world_tf = self.get_world_transform()
            position = np.array(world_tf.ExtractTranslation())
            rotation = np.array(world_tf.ExtractRotationMatrix())
            
            # The ray distances are the same for all rays, just max_range
            # ray_distances = np.append(ray_distances, np.full((hres*vres,), max_range), axis=0)
            # The origin is the World space origin of the ray caster
            ray_origins = np.append(ray_origins, np.tile(position, (hres*vres, 1)), axis=0)

            # The direction is the ray direction in world space are a bit more complicated
            h_angles = np.linspace(-hfov/2, hfov/2, hres)
            v_angles = np.linspace(-vfov/2, vfov/2, vres)
            rays = []
            for v in v_angles:
                for h in h_angles:
                    # Spherical coordinates to direction vector
                    x = np.cos(v) * np.sin(h)
                    y = np.sin(v)
                    z = np.cos(v) * np.cos(h)
                    dir_vec = np.array([x, y, z])
                    dir_vec = dir_vec / np.linalg.norm(dir_vec)
                    rays.append(dir_vec)
            rotated_directions = (rotation @ np.array(rays).T).T
            ray_directions = np.append(ray_directions, rotated_directions, axis=0)

        print(f"Ray origins and directions for {self.name} calculated in {time.time() - start_time:.2f} seconds.")

        return ray_origins, ray_directions

    
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
    
    def get_ancestor_path(self, level:int=1):
        """Returns the path to the ancestor of the sensor instance at the given level. 1 is the direct parent, 2 is the grandparent, etc."""
        path = self.path
        for i in range(level):
            path = str(path).rsplit('/', 1)[0]
        return path
    
    def contained_in(self, mesh:o3d.geometry.TriangleMesh):
        """Returns whether or not the sensor body is within the given mesh volume."""
        raise NotImplementedError("This method is not yet implemented.")
    
    def overlaps(self):
        """Returns whether or not the sensor body overlaps with the given mesh volume."""

        prim_path = prim_utils.get_prim_path(self.body)
        prim_found = False

        def report_overlap(overlap):
            if overlap.rigid_body == prim_path:
                global prim_found
                prim_found = True
                # Now that we have found our prim, return False to abort further search.
                return False
            return True
        
        physx.get_physx_scene_query_interface().overlap_mesh(PhysicsSchemaTools.sdfPathToInt(prim_path),
                                                             report_overlap,
                                                             anyHit=False)
        if prim_found:
            print("Prim overlapped mesh!")
            return True
        return False
    
    def overlaps_any(self, stage):
        """Returns whether or not the sensor body overlaps with anything in the given stage."""
        raise NotImplementedError("This method is not yet implemented.")
    

class Bot3D:
    def __init__(self, 
                 name:str,
                 usd_context:omni.usd.UsdContext,
                 body:list[UsdGeom.Mesh]=None,
                 path:str=None,
                 sensor_coverage_requirement:list[UsdGeom.Mesh]=None,
                 sensor_pose_constraint:list[UsdGeom.Mesh]=None, 
                 sensors:list[Sensor3D_Instance]=[]):
        """
        Initialize a bot representation with a given shape, sensor coverage requirements, and optional color and sensor pose constraints.
        Args:
            name (str): The name of the bot.
            body (list[UsdGeom.Mesh]): The body of the bot.
            path (str): The path to the bot in the USD stage.
            sensor_coverage_requirement (list[UsdGeom.Mesh]): The required coverage area for the sensors.
            sensor_pose_constraint (list[UsdGeom.Mesh]): The constraints for the sensor poses.
            sensors (list[Sensor3D_Instance]): A list of sensor instances attached to the bot.
        """
        self.name = name
        self.path = path
        self.body = body
        self.sensors = sensors

        self.usd_context = usd_context
        self.stage = self.usd_context.get_stage()
            
        self.sensor_coverage_requirement = sensor_coverage_requirement
        self.sensor_pose_constraint = sensor_pose_constraint

        self.perception_entropy = 0.0
        self.perception_coverage_percentage = 0.0

        # set the device to GPU if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using {self.device} for all calucations for robot: {self.name}.")
        

        # TODO Remove self.body from any of the sensor_coverage_requirement meshes

    def get_sensors_by_type(self, sensor_type:Type[Sensor3D]) -> list[Sensor3D_Instance]:
        sensor_instances = []
        for sensor_instance in self.sensors:
            if isinstance(sensor_instance.sensor, sensor_type):
                sensor_instances.append(sensor_instance)
        return sensor_instances
    
    def get_sensors_by_name(self, name:str) -> list[Sensor3D_Instance]:
        sensor_instances = []
        for sensor_instance in self.sensors:
            if sensor_instance.sensor.name == name:
                sensor_instances.append(sensor_instance)
        return sensor_instances
    
    def get_ray_casters(self):
        """Returns a list of all the ray casters in the bot."""
        ray_casters = []
        for sensor_instance in self.sensors:
            if sensor_instance.ray_casters != []:
                ray_casters.extend(sensor_instance.ray_casters)
        return ray_casters

    def add_sensor_3d_at(self, sensor:Sensor3D|list[Sensor3D]|None, tf:Gf.Matrix4d, name:str=None, path:str=None):
        """
        Adds a 3D sensor to the list of sensors. Only adds a sensor if it is not None.
        Parameters:
            sensor (Sensor3D|list[Sensor3D]|None): The sensor to be added.
            tf (Gf.Matrix4d): The transformation matrix for the sensor.
            name (str|None): The name of the sensor. If None, use the sensor's name.
            path (str|None): The path to the sensor in the USD stage. If None, use the default path.
        Returns:
            bool: True if the sensor was added successfully, False otherwise.
        """
        if (sensor is not None) and (tf is not None):
            sensor_instance = Sensor3D_Instance(sensor=sensor, 
                                                tf=tf,
                                                name=name if name is not None else sensor.name,
                                                path=path if path is not None else f"{self.path}/{sensor.name}/{sensor.name}",
                                                usd_context=self.usd_context)
            self.sensors.append(sensor_instance)
        else:
            print("Sensor or Transform is None, not adding to bot.")
            return False
        
    def get_prim(self):
        """Get the USD prim for the bot"""
        return self.stage.GetPrimAtPath(self.path)

    def get_world_transform(self) -> Tuple[Gf.Vec3d, Gf.Rotation]:
        """Get the world transform (position and rotation) of a prim"""
        xform = UsdGeom.Xformable(self.get_prim())
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        return world_transform
    
    def calculate_perception_entropy(self, perception_space:PerceptionSpace):
        """
        Calculate the perception entropy of the bot based on the sensors and the perception space.
        Args:
            perception_space (PerceptionSpace): The perception space to calculate the entropy for.

        Returns:
            float: The perception entropy of the bot. Lower entropy indicates better coverage.
        """

        def _apply_early_fusion(sensor_measurements: torch.Tensor) -> torch.Tensor:
            """Apply early fusion strategy to combine measurements of the same sensor per voxel.
            This is just a sum of the measurements on the voxel.
            
            Parameters
            ----------
            measurements : torch.Tensor
                A tensor of shape (N, S) N is the number of voxels, and S is the sensors.

            Returns
            -------
            torch.Tensor
                A tensor of shape (N,) where N is the number of voxels.
            """
            if sensor_measurements.ndim == 1:
                # If the tensor is 1D, it means we have a single sensor measurement per voxel
                return sensor_measurements
            elif sensor_measurements.ndim == 2:
                # If the tensor is 2D, we need to sum along the first axis (axis 0)
                # This will give us a tensor of shape (N,) where N is the number of voxels.
                return torch.sum(sensor_measurements, dim=0)
        
        def _apply_late_fusion(uncertainties: torch.Tensor) -> torch.Tensor:
            """Apply late fusion strategy to combine uncertainties (σ's)from different sensor types per voxel
            This is based on the formula from the paper Ma et al. 2020 "Perception Entropy..."
            
            σ_fused = sqrt(1 / Σ(1/σ_i²))
            
            Parameters
            ----------
            measurements : torch.Tensor
                The tensor of uncertainties (σ_i's) for each sensor type, where (N, S) where N is the number of voxels, and S is the number of sensors.

            Returns
            -------
            torch.Tensor
                A tensor of shape (N, S) where N is the number of voxels and S is the number of sensors.
            """

            # Calculate the fused uncertainty using the formula
            # σ_fused = sqrt(1 / Σ(1/σ_i²))
            fused_uncertainty = torch.sqrt(1 / torch.sum(1 / (uncertainties ** 2), dim=0))
            
            return fused_uncertainty
        
        def _calc_aps(measurements:torch.Tensor, a:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
            """Calculate the sensor AP for each voxel, where AP = a ln(m) + b
            This is based on the formula from the paper Ma et al. 2020 "Perception Entropy..."
            
            Parameters
            ----------
            measurements : torch.Tensor
                The tensor of measurements (m_i's) for each sensor type, of shape (N, S) where N is the number of voxels, and S is the number of sensors.
            a : torch.Tensor
                The tensor of sensor AP coefficient a, of shape (S) where S is the number of sensors.
            b : torch.Tensor
                The tensor of sensor AP coefficient b, of shape (S) where S is the number of sensors.

            Returns
            -------
            torch.Tensor
                A tensor of shape (N, S) where N is the number of voxels, and S is the number of sensors.
            """
            # Calculate the AP for each voxel using the formula
            # AP = a ln(m) + b
            shape = measurements.shape
            if len(shape) == 1:
                N = shape[0]
                S = 1
            elif len(shape) == 2:
                N = shape[0]
                S = shape[1]
            else:
                raise ValueError(f"Invalid shape for measurements: {measurements.shape}. Should be (N,) or (N, S).")
            print(f"Calculating AP for N={N} voxels and S={S} sensors.")
            print(f"  measurements.shape: {measurements.shape}, should be ({N}, {S})")
            print(f"  a.shape: {a.shape}, should be ({S},)")
            print(f"  b.shape: {b.shape}, should be ({S},)")

            bs_t = b.repeat(measurements.shape[0], 1).T.to(device) # a is (S,), so we need to repeat it for each voxel. Also make sure to move to device
            as_t = a.repeat(measurements.shape[0], 1).T.to(device) # b is (S,), so we need to repeat it for each voxel. Also make sure to move to device
        
            # ap = a * ln_m + b
            ap = as_t * torch.log(measurements) + bs_t

            # Transpose and clamp AP to valid range
            ap = torch.clamp(ap, min=0.001, max=0.999)
            print(f"  ap.shape: {ap.shape}, should be ({N}, {S})")
            
            return ap
        
        def _calc_uncertainties(aps: torch.Tensor) -> torch.Tensor:
            """Calculate the uncertainties (σ_i's) for each sensor type, where σ_i = (1 / AP) - 1
            This is based on the formula from the paper Ma et al. 2020 "Perception Entropy..."
            
            Parameters
            ----------
            aps : torch.Tensor
                The tensor of sensor AP for each voxel, of shape (N,) where N is the number of voxels.

            Returns
            -------
            torch.Tensor
                A tensor of shape (N,) where N is the number of voxels.
            """
            # Calculate the uncertainty for each voxel using the formula
            # σ_i = (1 / AP) - 1
            
            uncertainties = (1 / aps) - 1
            
            return uncertainties
        
        def _calc_entropies(uncertainties: torch.Tensor) -> torch.Tensor:
            """Calculate the entropy for each voxel, where H(S|m,q) = 2ln(σ) + 1 + ln(2pi)
            This is based on the formula from the paper Ma et al. 2020 "Perception Entropy..."
            
            Parameters
            ----------
            uncertainties : torch.Tensor
                The tensor of uncertainties (σ's) for each voxel, shape (N,) where N is the number of voxels.

            Returns
            -------
            torch.Tensor
                A tensor of shape (N,) where N is the number of voxels.
            """
            # Calculate the entropy for each voxel using the formula
            # H(S|m,q) = 2ln(σ) + 1 + ln(2pi)
            # H(S|m,q) = 2 * torch.log(uncertainties) + 1 + torch.log(torch.tensor(2 * np.pi))

            ln2pi_p_1 = torch.tensor(1 + 2 * np.log(np.pi)).to(device)
            entropy = 2 * torch.log(uncertainties) + ln2pi_p_1.repeat(uncertainties.shape[0], 1).T.to(device) # repeat for each voxel

            # Reshape the entropy tensor to (N,) where N is the number of voxels
            entropy = entropy.view(-1)

            # Normalize the entropy to be between 0 and 1
            entropy = (entropy - torch.min(entropy)) / (torch.max(entropy) - torch.min(entropy))
            
            return entropy
            
        ###################################################################################################
        
        # Create one tensor per sensor type (R, N, S) where R is the number of rays, 
        # N is the number of voxels, and S is the number of sensors.
        start_time = time.time()

        sensor_ms = {}
        for sensor_inst in self.sensors:
            o,d = sensor_inst.get_rays()

            # This is a tensor of shape (R, N) where R is the number of rays and N is the number of voxels. 
            # Each element is True if the ray intersects with the voxel, False otherwise.
            sensor_m = perception_space.batch_ray_voxel_intersections(torch.Tensor(o).to(device), torch.Tensor(d).to(device))
            print(f"sensor_m.shape: {sensor_m.shape}, should be (N,)")

            # Add the sensor measurements to the tensor for the sensor type
            if sensor_inst.sensor not in sensor_ms.keys():
                sensor_ms.update({sensor_inst.sensor: torch.Tensor([]).to(device)})
            sensor_ms[sensor_inst.sensor] = torch.cat((sensor_ms[sensor_inst.sensor], sensor_m), dim=0)

        # Apply early fusion to combine measurements of the same sensor per voxel
        early_fusion_ms = torch.Tensor([]).to(device)
        ap_as = torch.Tensor([])
        ap_bs = torch.Tensor([])
        for sensor, sensor_m_tensor in sensor_ms.items():
            m = _apply_early_fusion(sensor_m_tensor).to(device)
            print(f"m.shape: {m.shape}, should be (N,)")

            early_fusion_ms = torch.cat((early_fusion_ms, m), dim=0)
            ap_as = torch.cat((ap_as, torch.Tensor([sensor.ap_constants['a']])), dim=0)
            ap_bs = torch.cat((ap_bs, torch.Tensor([sensor.ap_constants['b']])), dim=0)
            # This is a tensor of shape (N, S) where N is the number of voxels and S is the number of sensors.
            # Each element is the number of rays that intersect with the voxel for that sensor type.
        print(f"early_fusion_ms.shape: {early_fusion_ms.shape}, should be (N, S)")
        
        # Calculate the sensor AP for each voxel, where AP = a ln(m) + b
        # This is a tensor of shape (N, S) where N is the number of voxels, and S is the number of sensors.
        aps = _calc_aps(early_fusion_ms, ap_as, ap_bs)
        print(f"aps.shape: {aps.shape}, should be (N,S)")
        print(f"  max of aps: {aps.max()} (should be <=0.999)")
        print(f"  min of aps: {aps.min()} (should be >=0.001)")

        us = _calc_uncertainties(aps)
        
        # Apply late fusion to combine uncertainties (σ's) from different sensor types per voxel
        late_fusion_Us = _apply_late_fusion(us)

        # Calculate the entropies for each voxel, where H(S|m,q) = 2ln(σ) + 1 + ln(2pi)
        # This is a tensor of shape (N) where N is the number of voxels.
        entropies = _calc_entropies(late_fusion_Us)

        # Calculate the weighted average entropy for the bot
        weights = perception_space.get_voxel_weights()
        sum_weights = torch.sum(weights, dim=0) #this should be a float
        weights = weights / sum_weights #normalize the weights
        entropy = torch.sum(entropies * weights, dim=0)

        print(f"{self.name} perception entropy calculated in {time.time() - start_time:.2f} seconds!!")

        return entropy
        
    

    # def add_sensor_valid_pose(self, sensor:Sensor3D, max_tries:int=25, verbose=False):
    #     """
    #     Adds a sensor to a valid location within the defined constraints.
    #     This method generates random points within the bounding box of the 
    #     sensor pose constraints and translates the sensor to these points. 
    #     It checks if the new sensor pose is valid and, if so, adds the sensor 
    #     to the list of sensors.
    #     Args:
    #         sensor (FOV3D): The sensor to be added, which will be translated 
    #                 to a valid location within the constraints.
    #     """
    #     for i in range(max_tries):
    #         x, y = pointpats.random.poisson(self.sensor_pose_constraint, size=1)
    #         rotation = -np.degrees(np.arctan2(x, y))

    #         sensor.set_translation(x, y)
    #         sensor.set_rotation(rotation) #this isn't quite right but good enough
            
    #         if self.is_valid_sensor_pose(sensor):
    #             self.add_sensor_2d(sensor)
    #             break
    #         if i == max_tries and verbose:
    #             print(f"Did not find a valid sensor pose in {max_tries} tries. Quitting!")
    #     return sensor

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