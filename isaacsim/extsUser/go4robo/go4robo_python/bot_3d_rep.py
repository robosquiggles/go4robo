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
import hashlib

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px

import torch

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import Bounds, OptimizeResult, NonlinearConstraint, LinearConstraint

from scipy.spatial.transform import Rotation as R

try:
    import omni
    import omni.physx as physx
    import omni.isaac.core.utils.prims as prim_utils
    import isaacsim.core.utils.transformations as tf_utils
    print("Isaac Sim found; Isaac Sim-specific features will work.")
    ISAAC_SIM_MODE = True
except ImportError:
    print("Isaac Sim not found; Isaac Sim-specific features will not work.")
    ISAAC_SIM_MODE = False

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
        """Create a translation matrix from the given translation vector."""
        assert isinstance(tx, (int, float)), "Translation values must be int or float."
        assert isinstance(ty, (int, float)), "Translation values must be int or float."
        assert isinstance(tz, (int, float)), "Translation values must be int or float."
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

    # Rotation matrix around X-axis
    def rotation_x_matrix(theta):
        """Create a rotation matrix around the X-axis. Theta is in radians."""
        assert isinstance(theta, (int, float)), "Rotation angle must be int or float."

        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
    
    # Rotation matrix around Y-axis
    def rotation_y_matrix(theta):
        """Create a rotation matrix around the Y-axis. Theta is in radians."""
        assert isinstance(theta, (int, float)), "Rotation angle must be int or float."
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
            
        ])
    
    # Rotation matrix around Z-axis
    def rotation_z_matrix(theta):
        """Create a rotation matrix around the Z-axis. Theta is in radians."""
        assert isinstance(theta, (int, float)), "Rotation angle must be int or float."
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])
    
    def inverse_matrix(tf_matrix):
        """Calculate the inverse of a transformation matrix."""
        assert tf_matrix.shape == (4, 4), "Input matrix must be a 4x4 transformation matrix."
        # Cast to a np.ndarray if not already
        if not isinstance(tf_matrix, np.ndarray):
            tf_matrix_np = np.array(tf_matrix)
        else:
            tf_matrix_np = tf_matrix
        return np.linalg.inv(tf_matrix_np)

    def normalize_matrix_svd(tf_matrix:np.ndarray, bounds:None|list[tuple]=None, majority='column'):
        """
        Normalize the rotation part of a 4x4 transformation matrix.
        If bounds are provided, the translation part is also normalized to be from 0 to 1 in all directions.

        Parameters:
            matrix_4x4 (np.ndarray): A 4x4 transformation matrix.
            bounds (list(tuple), optional): A list of tuples specifying the min and max bounds for each axis.
                If provided, the translation part will be normalized to be from 0 to 1 in all directions.
                Order of the bounds should be ((min_x, max_x), (min_y, max_y), (min_z, max_z)).
            majority (str): Specify whether to extract the translation from the last 'column' or 'row' of the matrix.

        Returns:
            np.ndarray: A 4x4 transformation matrix with a normalized rotation component.
        """

        assert tf_matrix.shape == (4, 4), "Input matrix must be a 4x4 transformation matrix."

        if majority == 'column':
            translation = tf_matrix[:, 3]  # Extract the translation component (last column)
        elif majority == 'row':
            translation = tf_matrix[3, :]  # Extract the translation component (last row)
        
        # Extract the translation component (last column)
        
        if bounds is not None:
            # Normalize the translation component to be from 0 to 1 in all directions
            for i in range(3):
                min_bound, max_bound = bounds[i]
                translation[i] = (translation[i] - min_bound) / (max_bound - min_bound)

        # Extract the rotation component (upper-left 3x3 submatrix)
        rotation = tf_matrix[:3, :3]

        # Perform SVD on the rotation matrix
        U, _, Vt = np.linalg.svd(rotation)

        # Reconstruct the closest orthonormal rotation matrix
        rotation_normalized = U @ Vt

        # Ensure a right-handed coordinate system (determinant should be +1)
        if np.linalg.det(rotation_normalized) < 0:
            U[:, -1] *= -1
            rotation_normalized = U @ Vt

        # Construct the normalized 4x4 transformation matrix
        normalized_matrix = np.eye(4)
        normalized_matrix[:3, :3] = rotation_normalized
        if majority == 'column':
            normalized_matrix[:, 3] = translation
        elif majority == 'row':
            normalized_matrix[3, :] = translation

        return normalized_matrix
    
    def flatten_matrix(tf_matrix):
        """Flatten a 4x4 transformation matrix to a 1D array."""
        assert tf_matrix.shape == (4, 4), "Input matrix must be a 4x4 transformation matrix."
        if not isinstance(tf_matrix, np.ndarray):
            tf_matrix = np.array(tf_matrix)
        return tf_matrix.flatten()
    
    def unflatten_matrix(flat_matrix):
        """Unflatten a 1D array to a 4x4 transformation matrix. Matrix must be 12 or 16 elements.
        If 12 elements, the last row is assumed to be [0 0 0 1], and it is added to the matrix."""
        if isinstance(flat_matrix, dict):
            flat_matrix = np.array(list(flat_matrix.values()))
        if not isinstance(flat_matrix, np.ndarray):
            flat_matrix = np.array(flat_matrix)
        if len(flat_matrix) == 12:
            flat_matrix = np.concatenate((flat_matrix, [0, 0, 0, 1]))
        assert len(flat_matrix) == 16, "Input array must have 12 or 16 elements."
        return flat_matrix.reshape(4, 4)
    
    def batch_quaternion_to_matrix(quats:torch.tensor, positions:torch.tensor) -> torch.tensor:
        """Convert a batch of quaternions and positions to a batch of transformation matrices.
        Args:
            quats (torch.tensor): A tensor of shape (N, 4) representing N quaternions. Quaternions should be in the format (w, x, y, z).
            positions (torch.tensor): A tensor of shape (N, 3) representing N positions. Positions should be in the format (x, y, z).
        Returns:
            torch.tensor: A tensor of shape (N, 4, 4) representing N transformation matrices.
        """
        assert quats.shape[1] == 4, "Quaternions must be of shape (N, 4)."
        assert positions.shape[1] == 3, "Positions must be of shape (N, 3)."
        assert quats.shape[0] == positions.shape[0], "Quaternions and positions must have the same number of elements."
        
        # Normalize the quaternions
        quats = quats / torch.norm(quats, dim=1, keepdim=True)

        # Extract the components of the quaternion
        w = quats[:, 0]
        x = quats[:, 1]
        y = quats[:, 2]
        z = quats[:, 3]

        # Compute the rotation matrix
        rotation_matrix = torch.zeros((quats.shape[0], 3, 3), device=quats.device)
        rotation_matrix[:, 0, 0] = w * w + x * x - y * y - z * z
        rotation_matrix[:, 0, 1] = 2 * (x * y - w * z)
        rotation_matrix[:, 0, 2] = 2 * (x * z + w * y)
        rotation_matrix[:, 1, 0] = 2 * (x * y + w * z)
        rotation_matrix[:, 1, 1] = w * w - x * x + y * y - z * z
        rotation_matrix[:, 1, 2] = 2 * (y * z - w * x)
        rotation_matrix[:, 2, 0] = 2 * (x * z - w * y)
        rotation_matrix[:, 2, 1] = 2 * (y * z + w * x)
        rotation_matrix[:, 2, 2] = w * w - x * x - y * y + z * z

        # Create the transformation matrices
        transformation_matrices = torch.zeros((quats.shape[0], 4, 4), device=quats.device)
        transformation_matrices[:, :3, :3] = rotation_matrix
        transformation_matrices[:, :3, 3] = positions
        transformation_matrices[:, 3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=quats.device).repeat(quats.shape[0], 1)

        torch.cuda.empty_cache()

        return transformation_matrices
    
    def batch_normalize_quaternion(quat:torch.Tensor):
        """Normalize a quaternion."""
        if isinstance(quat, torch.Tensor):
            assert quat.shape == (4,), "Quaternion must be of shape (4,)."
            return quat / torch.norm(quat)
        elif isinstance(quat, np.ndarray):
            assert quat.shape == (4,), "Quaternion must be of shape (4,)."
            return quat / np.linalg.norm(quat)
    
    def batch_random_quaternions(batch_size:int, device=device) -> torch.Tensor:
        """Generate a batch of random quaternions (w,x,y,z). Quaternions are normalized to unit length.
        Args:
            batch (int): The number of quaternions to generate.
        Returns:
            torch.Tensor: A tensor of shape (batch, 4) representing the quaternions."""
        assert isinstance(batch_size, int), "Batch size must be an int."
        
        # Generate random quaternions
        quats = torch.randn(batch_size, 4, device=device)
        quats = quats / torch.norm(quats, dim=1, keepdim=True)  # Normalize to unit length
        return quats

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
                 usd_context=None,
                 voxel_groups:list[VoxelGroup]|np.ndarray[VoxelGroup]=None, 
                 weights:list[float]|np.ndarray[float]=None):
        """Initialize a new instance of the class.
        Args:
            usd_context (omni.usd.UsdContext): The USD context.
            voxel_groups (torch.Tensor[VoxelGroup]): The voxel groups.
            weights (torch.Tensor[float]): The weights of the voxel groups.
        """
        self.usd_context = usd_context
        if usd_context is not None:
            self.stage = usd_context.get_stage()
        else:
            self.stage = None
        self.voxel_groups = voxel_groups if voxel_groups is not None else np.array([])
        self.weights = weights if weights is not None else np.array([])

    def to_json(self, file_path=None):
        """
        Convert the perception space to a JSON serializable format.
        Returns:
            dict: A dictionary representation of the perception space.
        """
        data = {
            "voxel_groups": [],
            "weights": []
        }
        for voxel_group in self.voxel_groups:
            data["voxel_groups"].append({
                "name": voxel_group.name,
                # "voxels": voxel_group.voxels,
                "voxel_centers": voxel_group.voxel_centers.tolist(),
                "voxel_sizes": voxel_group.voxel_sizes.tolist()
            })
        data["weights"] = self.weights.tolist()

        if file_path is not None:
            import json
            with open(file_path, 'w') as f:
                json.dump(data, f)

        return data
    
    @staticmethod
    def from_json(json_dict):
        """
        Load the perception space from a JSON file.
        Args:
            json_dict (dict): A dictionary representation of the perception space.
        """
        if not isinstance(json_dict, dict):
            raise ValueError("Invalid perception_space data")
        
        assert "voxel_groups" in json_dict, "Invalid perception_space data, no voxel_groups found"
        assert "weights" in json_dict, "Invalid perception_space data, no weights found"
        
        voxel_groups = []
        for voxel_group in json_dict["voxel_groups"]:
            voxel_groups.append(PerceptionSpace.VoxelGroup(
                name=voxel_group["name"],
                voxels=['']*len(voxel_group["voxel_centers"]), # Just an empty list of the right length since the paths/prims are not needed
                voxel_centers=torch.tensor(voxel_group["voxel_centers"]),
                voxel_sizes=torch.tensor(voxel_group["voxel_sizes"])
            ))
        weights = np.array(json_dict["weights"])
        perception_space = PerceptionSpace(voxel_groups=voxel_groups, weights=weights)
        assert isinstance(perception_space, PerceptionSpace), "Invalid perception_space data"
        return perception_space

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
        indices = np.where(self.voxel_groups == voxel_group_name)
        if len(indices) > 1:
            raise ValueError(f"Multiple voxel groups with the same name {voxel_group_name} found.")
        elif len(indices) == 1:
            self.weights[indices[0]] = weight
        else:
            raise ValueError(f"Voxel group {voxel_group_name} not found.")
        
    
    def batch_ray_voxel_intersections(self, 
                                      ray_origins:torch.Tensor, 
                                      ray_directions:torch.Tensor, 
                                      batch_size:int=15000,
                                      verbose:bool=False,
                                      eps:float=1e-8,
                                      ) -> torch.Tensor:
        """
        Check if the rays intersect with the voxels in the voxel groups.
        Args:
            ray_origins (torch.Tensor): The origins of the rays.
            ray_directions (torch.Tensor): The directions of the rays.
        Returns:
            torch.Tensor: A tensor of shape (N,) where N is the number of voxels. Each element is the number of rays that intersect with the voxel.
        """

        if verbose:
            start_time = time.time()
        
        torch.cuda.empty_cache()
        
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

            # Clamp to avoid division by zero
            rays_d = torch.where(-eps < rays_d, torch.tensor(-eps), rays_d)
            rays_d = torch.where(eps > rays_d, torch.tensor(eps), rays_d)

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

        print(f" Batch ray voxel intersection traversal took {time.time() - start_time:.2f} seconds for {num_rays} rays and {num_voxels} voxels.") if verbose else None
        print(f"  VOXEL HITS max: {torch.max(v_hits)}, min: {torch.min(v_hits)}, mean: {torch.mean(v_hits)}") if verbose else None

        torch.cuda.empty_cache()

        return v_hits  # Shape: (N,), hits for each voxel
    
    def plot_me(self, fig=None, show=True, mode='centers'):
        """
        Plot the perception space.
        Args:
            fig (matplotlib.figure.Figure): The figure to plot on.
            title (str): The title of the plot.
            show (bool): Whether to show the plot.
            mode (str): The mode of the plot. Can be 'centers' or 'boxes'.
        """
        # Get voxel data
        centers = self.get_voxel_centers().cpu().numpy()
        weights = self.get_voxel_weights()
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        weights = weights.cpu().numpy()
        
        if mode == 'centers':
            fig = fig or go.Figure()
            fig.add_trace(go.Scatter3d(
                x=centers[:,0], y=centers[:,1], z=centers[:,2],
                mode='markers',
                marker=dict(size=5, color=weights, colorscale='Viridis', opacity=0.25),
                name='Perception Space'
            ))
            if show: fig.show()
            return fig

        # BOXES mode: build Mesh3d per voxel
        mins = self.get_voxel_mins().cpu().numpy()
        maxs = self.get_voxel_maxs().cpu().numpy()

        # Define static face‐index arrays for a cube
        i_faces = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j_faces = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k_faces = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

        mesh_traces = []
        for idx in range(len(centers)):
            x0,y0,z0 = mins[idx]
            x1,y1,z1 = maxs[idx]
            # Eight corners
            x_verts = [x0, x0, x1, x1, x0, x0, x1, x1]
            y_verts = [y0, y1, y1, y0, y0, y1, y1, y0]
            z_verts = [z0, z0, z0, z0, z1, z1, z1, z1]
            # Per‐face intensity (replicate weight per triangle)
            face_intensity = [weights[idx]] * len(i_faces)

            mesh_traces.append(go.Mesh3d(
                x=x_verts, y=y_verts, z=z_verts,
                i=i_faces, j=j_faces, k=k_faces,
                intensity=face_intensity, colorscale='Viridis',
                opacity=0.25, showscale=False,
                name='Perception Space'
            ))

        # Add traces to figure
        if fig is None:
            fig = go.Figure(data=mesh_traces)
        else:
            for mesh in mesh_traces:
                fig.add_trace(mesh)

        fig.update_layout(
            title='Perception Space',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        if show: fig.show()
        return fig

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
        self.body = body # The body of the sensor is saved relative to the xform
        if isinstance(focal_point, (list, tuple)):
            self.focal_point = np.array([[1, 0, 0, focal_point[0]],
                                         [0, 1, 0, focal_point[1]],
                                         [0, 0, 1, focal_point[2]],
                                         [0,0,0,1]])
        else:
            self.focal_point = focal_point

        self.ap_constants = ap_constants

    def __eq__(self, other) -> bool:
        """Check if two Sensor3D objects are equal based on just their subclass type and their name."""
        if not isinstance(other, self.__class__):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name and self.h_fov == self.h_fov and self.h_res == self.h_res and self.v_fov == self.v_fov and self.v_res == self.v_res and self.max_range == self.max_range and self.min_range == self.min_range and self.cost == self.cost
    
    def __hash__(self):
        # Hash by class and name (customize as needed)
        return hash((self.__class__, self.name, self.h_fov, self.h_res, self.v_fov, self.v_res, self.max_range, self.min_range, self.cost))
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, h_fov={self.h_fov}, h_res={self.h_res}, v_fov={self.v_fov}, v_res={self.v_res}, max_range={self.max_range}, min_range={self.min_range}, cost={self.cost})"

    @staticmethod
    def from_json(json_dict):
        """
        Load the sensor from JSON.
        Args:
            json_dict (dict): A dictionary representation of the sensor.
        """
        if not isinstance(json_dict, dict):
            raise ValueError("Invalid sensor data")
        
        assert "name" in json_dict, "Invalid sensor data, no name found"
        assert "h_fov" in json_dict, "Invalid sensor data, no h_fov found"
        assert "h_res" in json_dict, "Invalid sensor data, no h_res found"
        assert "v_fov" in json_dict, "Invalid sensor data, no v_fov found"
        assert "v_res" in json_dict, "Invalid sensor data, no v_res found"
        assert "max_range" in json_dict, "Invalid sensor data, no max_range found"
        assert "min_range" in json_dict, "Invalid sensor data, no min_range found"
        assert "cost" in json_dict, "Invalid sensor data, no cost found"

        return Sensor3D(
            name=json_dict["name"],
            h_fov=json_dict["h_fov"],
            h_res=json_dict["h_res"],
            v_fov=json_dict["v_fov"],
            v_res=json_dict["v_res"],
            max_range=json_dict["max_range"],
            min_range=json_dict["min_range"],
            cost=json_dict["cost"]
        )
    
    def to_json(self):
        """
        Convert the sensor to a JSON serializable format.
        Returns:
            dict: A dictionary representation of the sensor.
        """
        data = {
            "name": self.name,
            "h_fov": self.h_fov,
            "h_res": self.h_res,
            "v_fov": self.v_fov,
            "v_res": self.v_res,
            "max_range": self.max_range,
            "min_range": self.min_range,
            "cost": self.cost
        }
        return data

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
                 tf_sensor1:tuple[tuple[float, float, float], tuple[float, float, float, float]],
                 tf_sensor2:tuple[tuple[float, float, float], tuple[float, float, float, float]],
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

    @staticmethod
    def from_json(json_dict):
        """
        Load the sensor from JSON.
        Args:
            json_dict (dict): A dictionary representation of the sensor.
        """
        if not isinstance(json_dict, dict):
            raise ValueError("Invalid sensor data")
        
        assert "name" in json_dict, "Invalid sensor data, no name found"
        assert "sensor1" in json_dict, "Invalid sensor data, no sensor1 found"
        assert "sensor2" in json_dict, "Invalid sensor data, no sensor2 found"
        assert "tf_sensor1" in json_dict, "Invalid sensor data, no tf_sensor1 found"
        assert "tf_sensor2" in json_dict, "Invalid sensor data, no tf_sensor2 found"
        assert "cost" in json_dict, "Invalid sensor data, no cost found"

        return StereoCamera3D(
            name=json_dict["name"],
            sensor1=MonoCamera3D.from_json(json_dict["sensor1"]),
            sensor2=MonoCamera3D.from_json(json_dict["sensor2"]),
            tf_sensor1=json_dict["tf_sensor1"],
            tf_sensor2=json_dict["tf_sensor2"],
            cost=json_dict["cost"]
        )
    
    def to_json(self):
        """
        Convert the sensor to a JSON serializable format.
        Returns:
            dict: A dictionary representation of the sensor.
        """
        data = {
            "name": self.name,
            "sensor1": self.sensor1.to_json(),
            "sensor2": self.sensor2.to_json(),
            "tf_sensor1": self.tf_1,
            "tf_sensor2": self.tf_2,
            "cost": self.cost
        }
        return data


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
                 tf:tuple[tuple[float], tuple[float]],
                 usd_context=None,
                 name:str|None=None,
                 body:UsdGeom.Mesh=None,
                 ):
        """Initialize a new instance of the class.
        Args:
            sensor (Sensor3D): The sensor object.
            path (str): The path to the sensor in the USD stage.
            tf (tuple[tuple[float], tuple[float]]): The transformation of the sensor as a tuple of translation and rotation.
                The translation is a tuple of 3 floats (x, y, z) and the rotation is a tuple of 4 floats (w, x, y, z).
            usd_context (omni.usd.UsdContext): The USD context (if known).
            name (str|None): The name of the sensor instance. If None, use the sensor's name.
        """

        # assert isinstance(sensor, Sensor3D), f"Sensor must be of type Sensor3D, not {type(sensor)}"
        assert isinstance(path, str), "Path must be a string"
        assert isinstance(tf, (tuple, np.ndarray, list)) and len(tf) == 2, "Transformation must be a tuple, list, or np.ndarray of length 2 (translation and rotation)"
        assert isinstance(tf[0], (tuple, np.ndarray, list)) and len(tf[0]) == 3, "TF Translation must be a list or tuple of length 3 (x,y,z)"
        assert isinstance(tf[1], (tuple, np.ndarray, list)) and len(tf[1]) == 4, "TF Rotation must be a list or tuple of length 4 (w, x, y, z)"
        # assert isinstance(usd_context, (omni.usd.UsdContext, None)), "USD context must be of type omni.usd.UsdContext or None"
        assert isinstance(name, str) or name is None, "Name must be a string or None"

        self.name = name
        self.sensor = sensor

        self.translation, self.quat_rotation = tf

        self.usd_context = usd_context
        if self.usd_context is not None:
            self.stage = usd_context.get_stage()
        else:
            self.stage = None

        self.path = str(path)

        self.ray_casters = []
        self.body = body

        # If a usd_context is provided, create the sensor body and ray casters in isaac sim
        if ISAAC_SIM_MODE and self.usd_context is not None and self.body is not None:
            self.ray_casters = self.create_ray_casters()
            self.body = self.create_sensor_body(sensor.body)


    def replace_sensor(self, sensor:Sensor3D):
        """Replace the sensor in the instance with a new sensor. This will also replace the ray casters."""
        self.sensor = sensor
        # self.ray_casters = self.create_ray_casters() # Don't replace the ray casters because they should be the same
        # self.body = self.create_sensor_body(sensor.body) # Don't replace the body because isaac sim needs it
        

    def create_sensor_body(self, body:UsdGeom.Mesh):
        """Create the sensor body in the USD stage. Returns the created sensor body."""
        assert ISAAC_SIM_MODE, "This function is only available in Isaac Sim mode"
        assert body is not None, "Body must be a UsdGeom.Mesh"
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

    def set_omniverse_quaterion(self, quat:tuple[float, float, float, float]):
        assert isinstance(quat, (tuple, np.ndarray, list)) and len(quat) == 4, "Quaternion must be a tuple, list, or np.ndarray of length 4 (w, x, y, z)"
        assert ISAAC_SIM_MODE, "This function is only available in Isaac Sim mode"
        import isaacsim.core.utils.xforms as xforms_utils
        # Update the quaternion of the sensor in the stage
        prim = self.stage.GetPrimAtPath(self.path)

        xforms_utils.reset_and_set_xform_ops(
            prim=prim, 
            translation=Gf.Vec3d(*self.translation), 
            orientation=Gf.Quatd(*quat)
        )
        
        #Update self.tf's quat_rotation
        self.quat_rotation = quat

    def create_ray_casters(self, disable=False):
        """Check if the ray casters have been created in the stage. If not, create them. Sets self.ray_casters to the created ray casters. Returns the created ray casters in a list."""
        assert ISAAC_SIM_MODE, "This function is only available in Isaac Sim mode"
        import omni.kit.commands
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
        assert USD_MODE, "This function is only available in USD mode"
        assert ISAAC_SIM_MODE, "This function is only available in Isaac Sim mode"
        assert self.stage is not None, "Stage is not set"
        return self.stage.GetPrimAtPath(self.path)
    
    def get_world_transform(self) -> Gf.Matrix4d:
        """Get the world transform of a prim
        Returns:
            Gf.Matrix4d: The world transform of the prim
        """
        assert USD_MODE, "This function is only available in USD mode"
        assert ISAAC_SIM_MODE, "This function is only available in Isaac Sim mode"
        assert self.stage is not None, "Stage is not set"
        xform = UsdGeom.Xformable(self.get_prim())
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Extract position and rotation
        # position = Gf.Vec3d(world_transform.ExtractTranslation())
        # rotation = world_transform.ExtractRotationMatrix()
        
        return world_transform
    
    def get_tfs(self) -> list[tuple[float, float, float], tuple[float, float, float, float]]:
        """Get the translation and rotation of the sensor in world coordinates"""
        tfs = []
        if isinstance(self.sensor, StereoCamera3D):
            # Calculate the transforms robot -> sensor1 and robot -> sensor2
            for tf in [self.sensor.tf_1, self.sensor.tf_2]:
                mat_parent_to_sensor = tf_utils.tf_matrix_from_pose(translation=tf[0], orientation=tf[1]) #tf from parent to sensor
                mat_robot_to_parent = tf_utils.tf_matrix_from_pose(translation=self.translation, orientation=self.quat_rotation) #tf from robot to parent
                mat_robot_to_sensor = mat_parent_to_sensor @ mat_robot_to_parent #tf from robot to sensor
                pos, rot = tf_utils.pose_from_tf_matrix(mat_robot_to_sensor) #tf from robot to sensor
                tfs.append((pos, rot))
        else:
            # Just return the transform of the sensor inside a list
            tfs.append((self.translation, self.quat_rotation))
        return tfs

    def get_rays(self, verbose:bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (ray_origins, ray_directions) for this sensor instance, vectorized with torch."""

        start_time = time.time()
        torch.cuda.empty_cache()

        sensors = [self.sensor] if not isinstance(self.sensor, StereoCamera3D) else [self.sensor.sensor1, self.sensor.sensor2]
        tfs = self.get_tfs()

        ray_origins_list = []
        ray_directions_list = []

        for i, sensor in enumerate(sensors):
            # # Get the ray information from the ray caster object
            # hfov = ray_caster.GetAttribute("horizontalFov").Get() # degrees
            # vfov = ray_caster.GetAttribute("verticalFov").Get() # degrees
            # hres = int(hfov / ray_caster.GetAttribute("horizontalResolution").Get()) # number of rays, horizontal
            # vres = int(vfov / ray_caster.GetAttribute("verticalResolution").Get()) # Number of rays, vertical

            hfov = sensor.h_fov
            vfov = sensor.v_fov
            hres = int(sensor.h_fov / sensor.h_res) # number of rays, horizontal
            vres = int(sensor.v_fov / sensor.v_res) # number of rays, vertical

            # Check if there are zero norms in quat
            if np.linalg.norm(tfs[i][1]) == 0:
                print('\033[91m' + f"Warning: Quaternion used to generate rays for {sensor.name} has zero norm. Skipping sensor in instance {self.name}.\nQUAT: {tfs[i][1]}" + '\033[0m')
                continue

            rotation = R.from_quat(tfs[i][1]).as_matrix() # Convert quaternion to rotation matrix
            
            position = torch.tensor(tfs[i][0], dtype=torch.float32, device=device) # Simple x,y,z
            rotation = torch.tensor(rotation, dtype=torch.float32, device=device) # Rotation matrix

            # Generate grid of angles
            h_angles = torch.linspace(-hfov/2, hfov/2, hres, device=device) * (np.pi/180)
            v_angles = torch.linspace(-vfov/2, vfov/2, vres, device=device) * (np.pi/180)
            v_grid, h_grid = torch.meshgrid(v_angles, h_angles, indexing='ij')
            v_flat = v_grid.flatten()
            h_flat = h_grid.flatten()

            # Spherical to Cartesian (vectorized)
            x = -torch.cos(v_flat) * torch.cos(h_flat)   # negate X so +X is forward
            y =  torch.cos(v_flat) * torch.sin(h_flat)
            z =  torch.sin(v_flat)
            dirs = torch.stack([x, y, z], dim=1)
            dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)

            # Rotate directions
            rotated_dirs = torch.matmul(dirs, rotation.T)

            # Repeat origins
            origins = position.expand_as(rotated_dirs)

            ray_origins_list.append(origins)
            ray_directions_list.append(rotated_dirs)

        # Concatenate all rays from all ray_casters
        ray_origins = torch.cat(ray_origins_list, dim=0).to(device)
        ray_directions = torch.cat(ray_directions_list, dim=0).to(device)

        torch.cuda.empty_cache()

        print(f"Rays for {self.name} calculated in {time.time() - start_time:.3f} sec.") if verbose else None
        print(f"  RAY ORIGINS max: {torch.max(ray_origins)}, min: {torch.min(ray_origins)}, mean: {torch.mean(ray_origins)}") if verbose else None
        print(f"  RAY DIRECTIONS max: {torch.max(ray_directions)}, min: {torch.min(ray_directions)}, mean: {torch.mean(ray_directions)}") if verbose else None
        print(f"    H_ANGLES max: {torch.max(h_angles)}, min: {torch.min(h_angles)}, mean: {torch.mean(h_angles)}") if verbose else None
        print(f"    V_ANGLES max: {torch.max(v_angles)}, min: {torch.min(v_angles)}, mean: {torch.mean(v_angles)}") if verbose else None
        return ray_origins, ray_directions
    
    def plot_rays(self, ray_origins, ray_directions, ray_length=1.0, fig=None, show=True):
        """Plot the rays in 3D using plotly"""

        # Convert to numpy arrays
        ray_origins = ray_origins.cpu().numpy()
        ray_directions = ray_directions.cpu().numpy()

        # Create a 3D scatter plot
        if fig is None:
            fig = go.Figure()

        # Add rays
        for i in range(ray_origins.shape[0]):
            fig.add_trace(go.Scatter3d(
                x=[ray_origins[i, 0], ray_origins[i, 0] + ray_directions[i, 0]*ray_length],
                y=[ray_origins[i, 1], ray_origins[i, 1] + ray_directions[i, 1]*ray_length],
                z=[ray_origins[i, 2], ray_origins[i, 2] + ray_directions[i, 2]*ray_length],
                mode='lines',
                line=dict(color='violet', width=2),
                hoverinfo='none',
                showlegend=False
            ))

        # Add the sensor
        self.plot_me(fig)

        # Set the layout
        fig.update_layout(
            title='Rays',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=800,
            height=800,
        )

        if show:
            fig.show()

        return fig


    def plot_me(self, fig, vec_length=0.2):
        translation = self.translation
        quat = self.quat_rotation  # (w,x,y,z)
        
        scalar_last_quat = (quat[1], quat[2], quat[3], quat[0])  # (x,y,z,w)
        # Get direction vector
        rot_mat = R.from_quat(scalar_last_quat).as_matrix()
        x_forward = np.array([1, 0, 0])  # Local forward direction in sensor's local space
        y_forward = np.array([0, 1, 0])  # Local left direction in sensor's local space
        z_forward = np.array([0, 0, 1])  # Local up direction in sensor's local space
        x_direction = rot_mat @ x_forward  # Transform to world space
        y_direction = rot_mat @ y_forward  # Transform to world space
        z_direction = rot_mat @ z_forward  # Transform to world space
        # Normalize the direction vector
        x_direction = x_direction / np.linalg.norm(x_direction) * vec_length
        y_direction = y_direction / np.linalg.norm(y_direction) * vec_length
        z_direction = z_direction / np.linalg.norm(z_direction) * vec_length
        
        color = random_color(self.name)

        # Add the first point as a marker
        fig.add_trace(go.Scatter3d(
            x=[float(translation[0])],  # Only the first point
            y=[float(translation[1])],
            z=[float(translation[2])],
            mode='markers',  # Display as markers
            marker=dict(size=3, color=color),  # Marker size and color
            name=self.name,  # Legend label
            legendgroup=self.name,  # Group in the legend
            text=[
                f"Translation:<br>  (x={translation[0]:.2f},<br>  y={translation[1]:.2f},<br>  z={translation[2]:.2f})",  # Hover text for the first point
                f"Quaternion:<br>  (qw={self.quat_rotation[0]:.2f},"
                f"<br>  qx={self.quat_rotation[1]:.2f},"
                f"<br>  qy={self.quat_rotation[2]:.2f},"
                f"<br>  qz={self.quat_rotation[3]:.2f})"  # Hover text for the second point
            ],
            hoverinfo='text'  # Use custom hover text
        ))

        # Add a X+ line between the two points
        fig.add_trace(go.Scatter3d(
            x=[float(translation[0]), float(translation[0]) + float(x_direction[0])],  # Both points
            y=[float(translation[1]), float(translation[1]) + float(x_direction[1])],
            z=[float(translation[2]), float(translation[2]) + float(x_direction[2])],
            mode='lines',  # Display as a line
            line=dict(color='red', width=2),  # Line color and width
            showlegend=False,  # Hide legend for the line itself
            legendgroup=self.name,  # Group in the legend
        ))

        # Add a Y+ line between the two points
        fig.add_trace(go.Scatter3d(
            x=[float(translation[0]), float(translation[0]) + float(y_direction[0])],  # Both points
            y=[float(translation[1]), float(translation[1]) + float(y_direction[1])],
            z=[float(translation[2]), float(translation[2]) + float(y_direction[2])],
            mode='lines',  # Display as a line
            line=dict(color='green', width=2),  # Line color and width
            showlegend=False,  # Hide legend for the line itself
            legendgroup=self.name,  # Group in the legend
        ))

        # Add a Z+ line between the two points
        fig.add_trace(go.Scatter3d(
            x=[float(translation[0]), float(translation[0]) + float(z_direction[0])],  # Both points
            y=[float(translation[1]), float(translation[1]) + float(z_direction[1])],
            z=[float(translation[2]), float(translation[2]) + float(z_direction[2])],
            mode='lines',  # Display as a line
            line=dict(color='blue', width=2),  # Line color and width
            showlegend=False,  # Hide legend for the line itself
            legendgroup=self.name,  # Group in the legend
        ))
    
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
                 usd_context=None,
                 body:list[UsdGeom.Mesh]=None,
                 path:str=None,
                 sensor_pose_constraint:np.ndarray[float]=None,
                 sensors:list[Sensor3D_Instance]=[]):
        """
        Initialize a bot representation with a given shape, sensor coverage requirements, and optional color and sensor pose constraints.
        Args:
            name (str): The name of the bot.
            usd_context (omni.usd.UsdContext): The USD context (if there is one).
            body (list[UsdGeom.Mesh]): The body of the bot.
            path (str): The path to the bot in the USD stage.
            sensor_pose_constraint (np.ndarray): The pose constraint for the sensors as an array [[x-bounds], [y-bounds], [z-bounds]]. ex. [[0,1], [0,1], [0,1]]
            sensors (list[Sensor3D_Instance]): A list of sensor instances attached to the bot.
        """
        self.name = name
        self.path = path
        self.body = body
        self.sensor_pose_constraint = sensor_pose_constraint
        self.sensors = sensors

        self.usd_context = usd_context
        if self.usd_context is not None:
            self.stage = usd_context.get_stage()
        else:
            self.stage = None

        self.perception_entropy = 0.0
        self.perception_coverage_percentage = 0.0
        
        # TODO Remove self.body from any of the sensor_coverage_requirement meshes

    def __deepcopy__(self, memo):
        # Helper to shallow-copy lists/dicts containing USD objects
        def safe_copy(obj):
            # Always shallow-copy USD objects
            if isinstance(obj, (Sdf.Path, Usd.Prim, Usd.Stage)):
                return obj
            # If it's a list, shallow-copy any USD objects inside
            elif isinstance(obj, list):
                return [safe_copy(item) for item in obj]
            # If it's a dict, shallow-copy any USD objects inside
            elif isinstance(obj, dict):
                return {safe_copy(k): safe_copy(v) for k, v in obj.items()}
            # If it's a tuple, shallow-copy any USD objects inside
            elif isinstance(obj, tuple):
                return tuple(safe_copy(item) for item in obj)
            # Otherwise, use deepcopy
            else:
                try:
                    return copy.deepcopy(obj, memo)
                except Exception:
                    return obj  # fallback: shallow copy if deepcopy fails
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, safe_copy(v))
        return result
    
    def to_json(self, file_path:str=None):
        """Serialize the bot to JSON, and return a dict. If a file path is provided, save to that file."""
        import json
        bot_dict = {
            "name": self.name,
            "path": str(self.path),  # Convert Path to string
            "body": None,  # [self.body], # TODO: support mesh JSON serialization?
            "sensor_pose_constraint": None,  # [self.sensor_pose_constraint], # TODO: support mesh JSON serialization?
            "sensor_instances": [
                {
                    "name": sensor.name,
                    "path": str(sensor.path),  # Convert Path to string
                    "translation": sensor.translation.tolist() if isinstance(sensor.translation, np.ndarray) else sensor.translation,
                    "rotation": sensor.quat_rotation.tolist() if isinstance(sensor.quat_rotation, np.ndarray) else sensor.quat_rotation,
                    "sensor": sensor.sensor.to_json()  if not isinstance(sensor.sensor, StereoCamera3D) else None,
                    "sensor1": sensor.sensor.sensor1.to_json() if isinstance(sensor.sensor, StereoCamera3D) else None,
                    "sensor2": sensor.sensor.sensor2.to_json() if isinstance(sensor.sensor, StereoCamera3D) else None,
                }
                for sensor in self.sensors
            ],
        }
        if file_path is not None:
            with open(file_path, 'w') as f:
                json.dump(bot_dict, f)
        return bot_dict
    
    @staticmethod
    def from_json(json_dict:dict):
        """Deserialize from a dict to create a Bot3D object. The dict should be in the same format as the one returned by to_json()."""
        name = json_dict["name"]
        path = json_dict["path"]
        body = None, # TODO if you want to plot the body in the dash app, make this work.
        sensor_pose_constraint = None # TODO if you want to plot the body in the dash app, make this work.
        usd_context=None
        sensors = []
        for sensor in json_dict["sensor_instances"]:
            sensor_name = sensor["name"]
            sensor_path = sensor["path"]
            translation = sensor["translation"]
            rotation = sensor["rotation"]

            # Create a Sensor3D_Instance for each sensor
            if sensor["sensor"] is not None:
                sensor = Sensor3D.from_json(sensor["sensor"])
            elif sensor["sensor1"] is not None and sensor["sensor2"] is not None:
                sensor = StereoCamera3D.from_json(sensor["sensor1"], sensor["sensor2"])
            else:
                sensor = None
            sensor_instance = Sensor3D_Instance(sensor=sensor, 
                                                path=sensor_path, 
                                                tf=(translation, rotation))
            sensors.append(sensor_instance)
        return Bot3D(name=name, sensor_pose_constraint=sensor_pose_constraint, usd_context=usd_context, path=path, body=body, sensors=sensors)

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
    
    def get_unique_sensor_options(self):
        """Returns a list of all the unique sensors in the bot. These can be used as discreet sensor options in an optimization.
        This list is determined by the object type, and the sensor name.
        For example, if the bot has a Lidar3D sensor and a MonoCamera3D sensor, the list will contain two options: Lidar3D and MonoCamera3D.
        If the bot has two Lidar3D sensors with the same name, the list will only contain one option: a Lidar3D with the name."""
        
        return set([s.sensor for s in self.sensors])

    def clear_sensors(self):
        """Clears all the sensors from the bot."""
        self.sensors = []
        self.ray_casters = []
    
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
        
    def add_sensors_batch_quat(self, sensors:list, positions:torch.tensor, quaternions:torch.tensor) -> int:
        """
        Adds a batch of 3D sensors to the list of sensors. Only adds a sensor if it is not None.
        Parameters:
            sensors (list[Sensor3D]): The list of sensors to be added. lenght N
            positions (torch.tensor): The positions of the sensors. shape (N, 3)
            quaternions (torch.tensor): The quaternions of the sensors. shape (N, 4)
        Returns:
            int: The number of sensors successfully added.
        """
        assert len(sensors) == positions.shape[0] == positions.shape[0], "Sensors, positions, and quaternions must have the same length."
        assert all([isinstance(sensor, (Sensor3D, type(None))) for sensor in sensors]), "All sensors must be of type Sensor3D."
        assert positions.shape[1] == 3, "Positions must be of shape (N, 3)."
        assert quaternions.shape[1] == 4, "Quaternions must be of shape (N, 4)."
        assert positions.device == quaternions.device, "Positions and quaternions must be on the same device."

        # Concat the positions and quaternions to create the transformation tuples
        tfs = []
        for i in range(len(sensors)):
            translation = (positions[i, 0].item(), positions[i, 1].item(), positions[i, 2].item())
            rotation = (quaternions[i, 0].item(), quaternions[i, 1].item(), quaternions[i, 2].item(), quaternions[i, 3].item())
            tfs.append((translation, rotation))
        
        for i, sensor in enumerate(sensors):
            if sensor is not None:
                sensor_instance = Sensor3D_Instance(sensor=sensor, 
                                                    tf=tfs[i],
                                                    name=sensor.name,
                                                    path=f"{self.path}/{sensor.name}/{sensor.name}_{i}",
                                                    usd_context=self.usd_context)
                self.sensors.append(sensor_instance)
        
        
    def get_prim(self):
        """Get the USD prim for the bot"""
        return self.stage.GetPrimAtPath(self.path)

    def get_world_transform(self) -> Tuple[Gf.Vec3d, Gf.Rotation]:
        """Get the world transform of a prim"""
        xform = UsdGeom.Xformable(self.get_prim())
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        return world_transform
    
    def calculate_perception_entropy(self, perception_space:PerceptionSpace, verbose:bool=False) -> float:
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
                return torch.sum(sensor_measurements, dim=1)
        
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
            print(f"Calculating fused uncertainty for {uncertainties.shape[0]} voxels.") if verbose else None
            print(f"  uncertainties.shape: {uncertainties.shape}, should be ({uncertainties.shape[0]}, {uncertainties.shape[1]})")  if verbose else None
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
            print(f"Calculating AP for N={N} voxels and S={S} sensors.") if verbose else None
            print(f"  measurements.shape: {measurements.shape}, should be ({N}, {S})") if verbose else None
            print(f"  a.shape: {a.shape}, should be ({S},)") if verbose else None
            print(f"  b.shape: {b.shape}, should be ({S},)") if verbose else None

            bs_t = b.repeat(measurements.shape[0], 1).to(device) # a is (S,), so we need to repeat it N times and transpose it to (N, S)
            as_t = a.repeat(measurements.shape[0], 1).to(device) # b is (S,), so we need to repeat it N times and transpose it to (N, S)

            print(f"  as_t.shape: {as_t.shape}, should be ({N}, {S})") if verbose else None
            print(f"  bs_t.shape: {bs_t.shape}, should be ({N}, {S})") if verbose else None
        
            # ap = a * ln_m + b
            ap = as_t * torch.log(measurements) + bs_t

            # Transpose and clamp AP to valid range
            ap = torch.clamp(ap.T, min=0.001, max=0.999)
            print(f"  ap.shape: {ap.shape}, should be ({N}, {S})") if verbose else None
            
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
            print(f"Calculating entropy for {uncertainties.shape[0]} voxels.") if verbose else None
            print(f"  uncertainties.shape: {uncertainties.shape}, should be ({uncertainties.shape[0]},)") if verbose else None

            ln2pi_p_1 = torch.tensor(1 + 2 * np.log(np.pi)).to(device)
            entropy = 2 * torch.log(uncertainties) + ln2pi_p_1.repeat(uncertainties.shape[0], 1).T.to(device) # repeat for each voxel

            print(f"  entropy.shape: {entropy.shape}, should be ({uncertainties.shape[0]},)") if verbose else None

            # Reshape the entropy tensor to (N,) where N is the number of voxels
            entropy = entropy.view(-1)

            # Normalize the entropy to be between 0 and 1
            # entropy = (entropy - torch.min(entropy)) / (torch.max(entropy) - torch.min(entropy))
            
            return entropy
            
        ###################################################################################################
        
        # Create one tensor per sensor type (R, N, S) where R is the number of rays, 
        # N is the number of voxels, and S is the number of sensors.
        start_time = time.time()

        sensor_ms = {}
        for sensor_inst in self.sensors:
            o,d = sensor_inst.get_rays()

            name = sensor_inst.sensor.name

            # This is a tensor of shape (R, N) where R is the number of rays and N is the number of voxels. 
            # Each element is True if the ray intersects with the voxel, False otherwise.
            sensor_m:torch.Tensor = perception_space.batch_ray_voxel_intersections(o, d, verbose=False)
            print(f"sensor_m.shape: {sensor_m.shape}, should be (N,)")  if verbose else None
            print(f"sensor_m min: {sensor_m.min()}, sensor_m max: {sensor_m.max()}, sensor_m mean:{sensor_m.mean()}")  if verbose else None

            # Add the sensor measurements to the tensor for the sensor type
            if name not in sensor_ms or sensor_ms[name].numel() == 0:
                sensor_ms[name] = sensor_m.unsqueeze(1)
            else:
                sensor_ms[name] = torch.cat((sensor_ms[name], sensor_m.unsqueeze(1)), dim=1)
        
        if sensor_ms == {}:
            print("No sensors found in the bot. Returning 0.0 for perception entropy.")
            return 0.0, 0.0

        # Apply early fusion to combine measurements of the same sensor per voxel
        early_fusion_ms = []
        ap_as = torch.Tensor([])
        ap_bs = torch.Tensor([])
        for name, sensor_m_tensor in sensor_ms.items():
            sensors = self.get_sensors_by_name(name)
            if verbose:
                if len(sensors) > 1:
                    print(f" Multiple ({len(sensors)}) sensors '{name}'s found, using AP constants of the first one!")
                elif len(sensors) == 0:
                    print(f" No sensors '{name}'s found, using default AP constants!")
                else:
                    print(f" Found {len(sensors)} sensor '{name}'s, using AP constants of the first one!")
            sensor = sensors[0].sensor
            print(f"  a: {sensor.ap_constants['a']}, b: {sensor.ap_constants['b']}")  if verbose else None
            m = _apply_early_fusion(sensor_m_tensor).to(device)
            print(f"  m.shape: {m.shape}, should be (N,)")  if verbose else None

            early_fusion_ms.append(m.unsqueeze(1))  # shape (N, 1)
            ap_as = torch.cat((ap_as, torch.Tensor([sensor.ap_constants['a']])), dim=0)
            ap_bs = torch.cat((ap_bs, torch.Tensor([sensor.ap_constants['b']])), dim=0)
            # This is a tensor of shape (N, S) where N is the number of voxels and S is the number of sensors.
            # Each element is the number of rays that intersect with the voxel for that sensor type.
        
        early_fusion_ms = torch.cat(early_fusion_ms, dim=1)  # shape (N, S_types)
        print(f"early_fusion_ms.shape: {early_fusion_ms.shape}, should be (N, S_types)")  if verbose else None

        # Calculate the percent of the voxels that are covered by any sensor
        covered_voxels = torch.sum(early_fusion_ms > 0, dim=1) # shape (N,)
        num_voxels = early_fusion_ms.shape[0]
        self.perception_coverage_percentage = torch.sum(covered_voxels > 0).item() / num_voxels
        
        # Calculate the sensor AP for each voxel, where AP = a ln(m) + b
        # This is a tensor of shape (N, S) where N is the number of voxels, and S is the number of sensors.
        aps = _calc_aps(early_fusion_ms, ap_as, ap_bs)

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

        self.perception_entropy = entropy.item()

        return self.perception_entropy, self.perception_coverage_percentage
    
    def calculate_cost(self):
        """
        Calculate the cost of the bot based on the sensors. This is a simple sum of the sensor costs.
        Args:
            None

        Returns:
            float: The cost of the bot. Lower cost indicates better coverage.
        """
        cost = 0.0
        for sensor in self.sensors:
            cost += sensor.sensor.cost
        return cost

    def get_design_validity(self, aabb_sensor_collision=True, aabb_sensor_constraints=True, verbose=False) -> float:
        """ Returns the validity of the constraint mesh. A valid design meets the following criteria:
        1. All sensor meshes are within the sensor pose constraints mesh.
        2. No sensor meshes are in collision with each other.
        3. NOTE: the third constraint (No sensors in collision with the bot body) is implied by constraint 1, because we assume that the bot body has been subtracted from the sensor pose constraints mesh.
        
        Args:
            aabb_sensor_collision (bool): If True, check for AABB collision between sensors.
            aabb_sensor_constraints (bool): If True, check for AABB collision between sensors and sensor pose constraints.
            verbose (bool): If True, print debug information.
        Returns:
            float: from 0.0 to 1.0, where 0.0 is a completely invalid design and 1.0 is a completely valid design.
        """
        
        print("Checking design validity...") if verbose else None
        print("WARNING: Design validity check is not yet implemented! Returns True.") if verbose else None

        #TODO Implement this the rest of the way. Much of this code was generated by OpenAI's GPT 4.1 Preview as inspo.
        return True

        def get_mesh_aabb_torch(mesh, device="cpu"):
            # Get points (vertices) as numpy, then convert to torch
            points = np.array(mesh.GetPointsAttr().Get())
            # Get world transform as 4x4 numpy, convert to torch
            xform = UsdGeom.Xformable(mesh)
            world_tf = np.array(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
            points_world = (world_tf @ points_h.T).T[:, :3]
            points_world = torch.tensor(points_world, device=device, dtype=torch.float32)
            # AABB
            min_xyz = torch.min(points_world, dim=0).values
            max_xyz = torch.max(points_world, dim=0).values
            return min_xyz, max_xyz

        def sensors_not_in_collision_torch_aabb(self, device="cuda"):
            # Get AABBs for all sensor bodies
            aabbs = []
            for sensor in self.sensors:
                if sensor.body is not None:
                    aabbs.append(get_mesh_aabb_torch(sensor.body, device=device))
            if len(aabbs) < 2:
                return True  # No collision possible

            mins = torch.stack([a[0] for a in aabbs])  # (S, 3)
            maxs = torch.stack([a[1] for a in aabbs])  # (S, 3)
            S = mins.shape[0]

            # Compare all pairs (i, j) with i < j
            for i in range(S):
                for j in range(i+1, S):
                    # If all axes overlap, collision
                    overlap = torch.all(mins[i] <= maxs[j]) and torch.all(maxs[i] >= mins[j])
                    if overlap:
                        return False
            return True
        
        def all_sensors_within_constraint_torch_aabb(self, device="cuda"):
            # Assume single constraint mesh for simplicity
            constraint_mesh = self.sensor_pose_constraint[0] if isinstance(self.sensor_pose_constraint, list) else self.sensor_pose_constraint
            c_min, c_max = get_mesh_aabb_torch(constraint_mesh, device=device)
            for sensor in self.sensors:
                if sensor.body is None:
                    continue
                s_min, s_max = get_mesh_aabb_torch(sensor.body, device=device)
                # Check if sensor's AABB is fully inside constraint's AABB
                if not (torch.all(c_min <= s_min) and torch.all(c_max >= s_max)):
                    return False
            return True
        
        # Check if all sensors are within the sensor pose constraints
        if aabb_sensor_constraints:
            if verbose:
                print("Using AABB to check if all sensors are within the sensor pose constraints...")
            sensor_constraints_validity = all_sensors_within_constraint_torch_aabb(device=device)
        else:
            if verbose:
                print("Using mesh to check if all sensors are within the sensor pose constraints...")
            raise NotImplementedError("Mesh-based sensor pose constraint check is not yet implemented.")
        
        # Check if all sensors are in collision with each other
        if aabb_sensor_collision:
            if verbose:
                print("Using AABB to check for sensor collision...")
            sensors_collision_validity = sensors_not_in_collision_torch_aabb(device=device)
        else:
            if verbose:
                print("Using mesh to check for sensor collision...")
            raise NotImplementedError("Mesh-based sensor collision check is not yet implemented.")

        # Combine the two validity checks
        if verbose:
            print(f"Sensor pose constraints validity: {sensor_constraints_validity}")
            print(f"Sensor collision validity: {sensors_collision_validity}")

        validity = sensor_constraints_validity and sensors_collision_validity
        return validity

    def plot_bot_3d(
            self, 
            perception_space:PerceptionSpace=None, 
            show=True, 
            save_path:str=None,
            **kwargs):
        """Plot the bot in 3D using plotly.
        
        Plot contains:
            - the bot body as a mesh (if it exists)
            - each sensor in self.sensors as a point and a direction. Each sensor is colored by its type.
            - if perception_space is not None, the perception space is shown as a point cloud (a point for each voxel).

        Args:
            perception_space (PerceptionSpace): The perception space to plot. If None, the perception space is not plotted.
            show (bool): If True, show the plot. If False, return the figure.
            save_path (str): The path to save the plot. If None, the plot is not saved.
        Returns:
            fig (plotly.graph_objects.Figure): The plotly figure object.
        """
        
        def add_perception_space(fig):
            """ Add the perception space as a point cloud of all the voxel centers."""
            if perception_space is not None:
                # Get the voxel centers
                voxel_centers = perception_space.get_voxel_centers()
                # Get the weights of the voxels
                weights = perception_space.get_voxel_weights()
                # Normalize the weights to be between 0 and 1
                weights = (weights - torch.min(weights)) / (torch.max(weights) - torch.min(weights))
                # Convert to numpy arrayslen
                voxel_centers = voxel_centers.cpu().numpy()
                weights = weights.cpu().numpy()
                
                fig.add_trace(go.Scatter3d(
                    x=voxel_centers[:, 0],
                    y=voxel_centers[:, 1],
                    z=voxel_centers[:, 2],
                    mode='markers',
                    marker=dict(size=5, color=weights, colorscale='Viridis', opacity=0.25),
                    name='Perception Space'
                ))

        height = 500 if 'height' not in kwargs else kwargs['height']
        width = 500 if 'width' not in kwargs else kwargs['width']
        title = f"{self.name} Original" if 'title' not in kwargs else kwargs['title']

        # Create a plotly figure
        fig = go.Figure()

        # Add a dot at the origin
        fig.add_trace(go.Scatter3d(
            x=[0.0],  # X-coordinate
            y=[0.0],  # Y-coordinate
            z=[0.0],  # Z-coordinate
            mode='markers',  # Display as markers
            marker=dict(size=5, color='black'),  # Marker size and color
            name='ORIGIN'  # Legend label
        ))

        # Add the bot body to the plot
        # TODO: Add the bot body to the plot

        # Add the sensor pose constraints to the plot
        if self.sensor_pose_constraint is not None:
            print("TODO Adding sensor pose constraints to the plot.")

        # Add the sensors to the plot
        for sensor_i in self.sensors:
            sensor_i.plot_me(fig)

        # Add the perception space
        add_perception_space(fig)

        # Adjust the layout of the plot
        fig.update_layout(
            height=height,
            width=width,
            title=title,
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                camera=dict(
                    eye=dict(x=1.25, y=1.25, z=1.25),  # Adjust the camera position
                    center=dict(x=0, y=0, z=0),  # Center the camera on the origin
                    up=dict(x=0, y=0, z=1)  # Ensure the Z-axis is up
                ),
                aspectmode='data'  # Maintain the aspect ratio
            )
        )

        # Show or save the plot
        if show:
            fig.show()
        if save_path is not None:
            fig.write_image(save_path)

        return fig

def random_color(name):
    """Optional: Create a random color from a string, or define your own mapping"""
    if name is None:
        return "#000000"
    hash_digest = hashlib.md5(name.encode()).hexdigest()
    return f"#{hash_digest[:6]}"