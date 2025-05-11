import time

from typing import Type

import PIL
import PIL.ImageColor

import os
import copy
import random
import numpy as np
import math

from typing import List, Dict, Tuple, Optional, Union, Sequence
import hashlib

import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_MODE = True
except (ImportError, ModuleNotFoundError):
    print("Matplotlib not found; Matplotlib-specific features will not work.")
    MATPLOTLIB_MODE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_MODE = True
except (ImportError, ModuleNotFoundError):
    print("Plotly not found; Plotly-specific features will not work.")
    PLOTLY_MODE = False

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
    
    def tf_matrix_from_pose(translation: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        """
        Build a 4x4 homogeneous transform from a translation and a scalar-first quaternion.
        
        Args:
            translation: (3,) array [tx, ty, tz]
            quat_wxyz:    (4,) array [w, x, y, z]
        
        Returns:
            T: (4,4) array so that
            [R t] 
            [0 1]
            where R is the 3x3 rotation matrix and t is the 3x1 translation.
        """
        # Convert to SciPy’s [x, y, z, w] convention
        q_xyzw = np.roll(quat_wxyz, -1)
        
        # Build rotation matrix
        R_mat = R.from_quat(q_xyzw).as_matrix()
        
        # Assemble into 4x4
        T = np.eye(4, dtype=float)
        T[:3, :3] = R_mat
        T[:3,  3] = translation
        return T

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
        for i, voxel_group in enumerate(self.voxel_groups):
            data["voxel_groups"].append({
                "name": voxel_group.name,
                # "voxels": voxel_group.voxels,
                "voxel_centers": voxel_group.voxel_centers.tolist(),
                "voxel_sizes": voxel_group.voxel_sizes.tolist()
            })
            data["weights"].append(self.weights[i])

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
        
    def chunk_ray_voxel_intersections(self,
                                        ray_origins:torch.Tensor,
                                        ray_directions:torch.Tensor,
                                        rays_per_chunk:int=20000,
                                        voxels_per_chunk:int=2000,
                                        eps:float=1e-8,
                                        verbose:bool=False,
                                        ) -> torch.Tensor:
        """
        Count ray-vs-voxel intersections in chunks to limit peak memory.

        Args:
            origins:    Ray origins, shape (R,3).
            directions: Ray direction vectors, shape (R,3).
            max_rays_per_chunk: Maximum number of rays to process at once.
            max_voxels_per_chunk: Maximum number of voxels to process at once.
            eps: Small epsilon to avoid division-by-zero.

        Returns:
            counts: Tensor of shape (V,) giving number of rays intersecting each voxel.
        """

        if verbose:
            start_time = time.time()
        
        torch.cuda.empty_cache()

        # Precompute box mins/maxs once
        box_mins = self.get_voxel_mins().to(device)  # (V,3)
        box_maxs = self.get_voxel_maxs().to(device)  # (V,3)

        R = ray_origins.shape[0]
        V = box_mins.shape[0]
        counts = torch.zeros(V, dtype=torch.long, device=device)

        # Loop over ray‐chunks
        for i in range(0, R, rays_per_chunk):
            o_chunk = ray_origins[i : i + rays_per_chunk]      # (r_chunk,3)
            d_chunk = ray_directions[i : i + rays_per_chunk]   # (r_chunk,3)

            # Prepare for broadcasting
            O = o_chunk.unsqueeze(1)       # (r_chunk,1,3)
            D = d_chunk.unsqueeze(1)       # (r_chunk,1,3)
            D_safe = torch.where(D.abs() < eps, torch.full_like(D, eps), D)

            # Loop over voxel‐chunks
            for j in range(0, V, voxels_per_chunk):
                Bmin = box_mins[j : j + voxels_per_chunk].unsqueeze(0)  # (1,v_chunk,3)
                Bmax = box_maxs[j : j + voxels_per_chunk].unsqueeze(0)  # (1,v_chunk,3)

                # slab intersection distances
                t1 = (Bmin - O) / D_safe    # (r_chunk, v_chunk, 3)
                t2 = (Bmax - O) / D_safe    # (r_chunk, v_chunk, 3)

                t_enter = torch.max(torch.min(t1, t2), dim=2).values  # (r_chunk, v_chunk)
                t_exit  = torch.min(torch.max(t1, t2), dim=2).values  # (r_chunk, v_chunk)

                hits = (t_exit >= t_enter) & (t_exit >= 0)            # bool mask
                counts[j : j + voxels_per_chunk] += hits.sum(dim=0).to(torch.long)

        print(f" Batch ray voxel intersection traversal took {time.time() - start_time:.2f} seconds for {R} rays and {V} voxels.") if verbose else None
        print(f"  VOXEL HITS max: {torch.max(counts)}, min: {torch.min(counts)}") if verbose else None

        return counts
    
    def chunk_occluded_ray_voxel_intersections(
        self,
        ray_origins: torch.Tensor,        # (R,3)
        ray_directions: torch.Tensor,     # (R,3)
        body_aabbs: Optional[Sequence[Tuple[
            Tuple[float,float],           # (xmin, xmax)
            Tuple[float,float],           # (ymin, ymax)
            Tuple[float,float]            # (zmin, zmax)
        ]]] = None,
        rays_per_chunk: int = 20000,
        voxels_per_chunk: int = 2000,
        eps: float = 1e-8,
        min_distance: float = 0.0,
        max_distance: float = float('inf'),
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Count ray-vs-voxel intersections in chunks, with optional occlusion and distance windowing.

        Args:
        ray_origins:    (R,3) tensor of ray start points.
        ray_directions: (R,3) tensor of ray directions.
        body_aabbs:     optional list of occluder AABBs as
                        ((xmin,xmax),(ymin,ymax),(zmin,zmax)).
                        If None or empty, occlusion is skipped.
        rays_per_chunk, voxels_per_chunk: chunk sizes to limit GPU memory.
        eps:             tiny value to avoid div-zero.
        min_distance:    ignore hits closer than this.
        max_distance:    ignore hits farther than this.
        verbose:         print timing/stats if True.

        Returns:
        counts: (V,) long tensor, number of valid, un-occluded rays hitting each voxel.
        """
        device = ray_origins.device
        if verbose:
            t0 = time.time()

        torch.cuda.empty_cache()

        # Voxel bounds
        box_mins = self.get_voxel_mins().to(device)   # (V,3)
        box_maxs = self.get_voxel_maxs().to(device)   # (V,3)
        V = box_mins.size(0)

        # Occluder bounds (if any)
        if body_aabbs:
            oc_min_list, oc_max_list = [], []
            for (xmin, xmax), (ymin, ymax), (zmin, zmax) in body_aabbs:
                oc_min_list.append([xmin, ymin, zmin])
                oc_max_list.append([xmax, ymax, zmax])
            oc_mins = torch.tensor(oc_min_list, dtype=ray_origins.dtype, device=device)  # (O,3)
            oc_maxs = torch.tensor(oc_max_list, dtype=ray_origins.dtype, device=device)  # (O,3)
            has_occluders = True
        else:
            has_occluders = False

        R = ray_origins.size(0)
        counts = torch.zeros(V, dtype=torch.long, device=device)

        # Main chunked loop
        for i in range(0, R, rays_per_chunk):
            o_chunk = ray_origins[i : i + rays_per_chunk]    # (r,3)
            d_chunk = ray_directions[i : i + rays_per_chunk] # (r,3)

            O = o_chunk.unsqueeze(1)    # (r,1,3)
            D = d_chunk.unsqueeze(1)    # (r,1,3)
            D_safe = torch.where(D.abs() < eps,
                                torch.full_like(D, eps),
                                D)

            # 1) Occlusion: either compute first‐hit or set to ∞
            if has_occluders:
                Oc_min = oc_mins.unsqueeze(0)  # (1,O,3)
                Oc_max = oc_maxs.unsqueeze(0)  # (1,O,3)
                t1_oc = (Oc_min - O) / D_safe   # (r,O,3)
                t2_oc = (Oc_max - O) / D_safe   # (r,O,3)
                t_ent_oc = torch.max(torch.min(t1_oc, t2_oc), dim=2).values  # (r,O)
                t_exi_oc = torch.min(torch.max(t1_oc, t2_oc), dim=2).values  # (r,O)
                hit_oc = (t_exi_oc >= t_ent_oc) & (t_exi_oc >= 0)             # (r,O)
                inf = torch.full_like(t_ent_oc, float('inf'))
                t_ent_oc = torch.where(hit_oc, t_ent_oc, inf)                # (r,O)
                t_first_oc, _ = t_ent_oc.min(dim=1)                          # (r,)
            else:
                # no occluders → nothing blocks any ray
                t_first_oc = torch.full((o_chunk.size(0),), float('inf'),
                                        device=device, dtype=o_chunk.dtype)

            # 2) Voxel chunks
            for j in range(0, V, voxels_per_chunk):
                Bmin = box_mins[j : j + voxels_per_chunk].unsqueeze(0)  # (1,v,3)
                Bmax = box_maxs[j : j + voxels_per_chunk].unsqueeze(0)  # (1,v,3)

                t1 = (Bmin - O) / D_safe     # (r,v,3)
                t2 = (Bmax - O) / D_safe     # (r,v,3)
                t_ent = torch.max(torch.min(t1, t2), dim=2).values    # (r,v)
                t_exi = torch.min(torch.max(t1, t2), dim=2).values    # (r,v)

                # basic slab‐hit test
                hits = (t_exi >= t_ent)

                # distance window [min_distance, max_distance]
                hits &= (t_exi  >= min_distance)
                hits &= (t_ent <= max_distance)

                # occlusion cull
                oc_block = t_ent > t_first_oc.unsqueeze(1)  # (r,v)
                hits &= ~oc_block

                counts[j : j + voxels_per_chunk] += hits.sum(dim=0).to(torch.long)

        if verbose:
            print(f"Done in {time.time()-t0:.2f}s for {R} rays × {V} voxels; "
                f"hits max={counts.max().item()}, min={counts.min().item()}")

        return counts

    
    def plot_me(
            self, 
            fig=None, 
            show=True, 
            colors='weights', 
            entropy_scale=[0.0, 4.5, 17.0],
            bot=None, 
            mode='centers', 
            voxels_per_chunk=None, 
            rays_per_chunk=None,
            **kwargs
            ):
        """
        Plot the perception space.
        Args:
            fig (matplotlib.figure.Figure): The figure to plot on.
            title (str): The title of the plot.
            show (bool): Whether to show the plot.
            mode (str): The mode of the plot. Can be 'centers' or 'boxes'.
        """

        if fig is None:
            fig = fig or go.Figure()
            fig.update_layout(
                title='Perception Space',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                margin=dict(l=0, r=0, b=0, t=0)
            )
    
        # Get voxel data
        centers = self.get_voxel_centers().cpu().numpy()
        weights = self.get_voxel_weights().cpu().numpy()
        names = []
        for i, voxel_group in enumerate(self.voxel_groups):
            names += [voxel_group.name] * len(voxel_group.voxels)
        text = [f"{n}<br>Weight:{w:.2f}" for n,w in zip(names,weights)]

        entropies = None
        if colors == 'entropy':
            if bot is None:
                raise ValueError("bot must not be None if colors is 'ap'")
            assert isinstance(bot, Bot3D), "bot must be a Bot3D if colors is 'entropy'"
            assert voxels_per_chunk is not None, "voxels_per_chunk must passed if colors is 'entropy'"
            assert rays_per_chunk is not None, "rays_per_chunk must passed if colors is 'entropy'"
            # Calculate the per-voxel entropy
            entropies = bot.calculate_perception_entropies(
                perception_space=self, 
                voxels_per_chunk=voxels_per_chunk, 
                rays_per_chunk=rays_per_chunk
                ).cpu().numpy()
            text = [f"{n}<br>Weight:{w:.2f}<br>Entropy:{e:.4f}" for n,w,e in zip(names,weights,entropies)]

            # LOGARITHMIC COLORBAR
            # Assume color is a numpy array of scalar values
            max_entropy = np.max(entropies)
            min_entropy = np.min(entropies)
            mean_entropy = np.mean(entropies)
            print(f"Entropy color scale:  {entropy_scale}")
            print(f"Entropy actual scale: {[min_entropy, mean_entropy, max_entropy]}")
            if max_entropy > entropy_scale[2]:
                print(f" WARNING: max entropy {max_entropy} is HIGHER than the provided colorscale max {entropy_scale[2]}.")
            if min_entropy < entropy_scale[0]:
                print(f" WARNING: min entropy {min_entropy} is LOWER than the provided colorscale min {entropy_scale[0]}.")
            color = np.log10(np.clip(entropies, 1e-8, entropy_scale[2]))
            log_ticks = np.linspace(0, np.log10(entropy_scale[2]), num=5)
            tickvals = log_ticks
            ticktext = [f"{10**val:.3f}" for val in log_ticks]
            colorbar=dict(
                title="Entropy",
                tickvals=tickvals,
                ticktext=ticktext,
            )
            cmid = np.log10(entropy_scale[1])

        elif colors == 'weights':
            # Use weights as color
            color = weights
            colorbar=dict(
                title='Weight',
                tickvals=np.linspace(np.min(weights), np.max(weights), num=5),
                ticktext=[f"{w:.2f}" for w in np.linspace(np.min(weights), np.max(weights), num=5)],
            )
        elif colors == 'names':
            # Use names as color
            color = names
            colorbar=None
        else:
            raise ValueError("Invalid color mode. Use 'm', 'weights' or 'names'.")
        
        if mode == 'centers':

            fig.add_trace(go.Scatter3d(
                x=centers[:,0], 
                y=centers[:,1], 
                z=centers[:,2],
                mode='markers',
                marker=dict(
                    size=5, 
                    color=color, 
                    colorscale='Portland' if colors != 'weights' else 'Portland_r',
                    cmid=cmid if colors == 'entropy' else None,
                    opacity=0.25,
                    showscale=True if colors != 'names' else False,
                    colorbar=colorbar,
                    ),
                name='Perception Space',
                showlegend=False if colors != 'names' else True,
                text=text,
                hovertemplate='(%{x:.2f}, %{y:.2f}, %{z:.2f})<br>%{text}<extra></extra>',
            ))

        elif mode == 'boxes':
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
                    intensity=face_intensity, colorscale='Portland',
                    opacity=0.25, showscale=False,
                    name='Perception Space'
                ))

            # Add traces to figure
            for mesh in mesh_traces:
                fig.add_trace(mesh)

        else:
            raise ValueError("Invalid plot mode for perception_space. Use 'centers' or 'boxes'.")
        if show: fig.show()
        return fig, entropies

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
        """
        self.h_fov = h_fov
        self.h_res = h_res
        self.v_fov = v_fov
        self.v_res = v_res
        self.max_range = max_range
        self.min_range = min_range
        self.cost = cost
        self.name = name
        self.body = body # Not implemented

        self.ap_constants = ap_constants

    def __eq__(self, other) -> bool:
        """Check if two Sensor3D objects are equal based on just their subclass type and their name."""
        if not isinstance(other, self.__class__):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name and self.h_fov == self.h_fov and self.h_res == self.h_res and self.v_fov == self.v_fov and self.v_res == self.v_res and self.max_range == self.max_range and self.min_range == self.min_range and self.cost == self.cost
    
    def __hash__(self):
        # Hash by class and name (customize as needed)
        return hash((self.__class__, self.name, self.h_fov, self.h_res, self.v_fov, self.v_res))
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, h_fov={self.h_fov}, h_res={self.h_res}, v_fov={self.v_fov}, v_res={self.v_res}, max_range={self.max_range}, min_range={self.min_range}, cost={self.cost})"

    @staticmethod
    def from_json(json_dict):
        """
        Load the sensor from JSON.
        Args:
            json_dict (dict): A dictionary representation of the sensor.
        """
        if json_dict == "None" or json_dict is None:
            return None
        
        if not isinstance(json_dict, dict):
            raise ValueError("Invalid sensor data")
        
        assert "type" in json_dict, "Invalid sensor data, no type found"
        assert "name" in json_dict, "Invalid sensor data, no name found"

        if json_dict["type"] == "StereoCamera3D":
            return StereoCamera3D.from_json(json_dict)
        else:
            assert "h_fov" in json_dict, "Invalid sensor data, no h_fov found"
            assert "h_res" in json_dict, "Invalid sensor data, no h_res found"
            assert "v_fov" in json_dict, "Invalid sensor data, no v_fov found"
            assert "v_res" in json_dict, "Invalid sensor data, no v_res found"
            assert "max_range" in json_dict, "Invalid sensor data, no max_range found"
            assert "min_range" in json_dict, "Invalid sensor data, no min_range found"
            assert "cost" in json_dict, "Invalid sensor data, no cost found"
            assert "ap_constants" in json_dict, "Invalid sensor data, no ap_constants found"
            h_fov = json_dict["h_fov"]
            h_res = json_dict["h_res"]
            v_fov = json_dict["v_fov"]
            v_res = json_dict["v_res"]
            max_range = json_dict["max_range"]
            min_range = json_dict["min_range"]
            cost = json_dict["cost"]
            name = json_dict["name"]
            ap_constants = json_dict["ap_constants"]
            body = None # TODO: This should be the body of the sensor
            
            # Create a new instance of the class of type json_dict["type"]
            sensor_class = globals()[json_dict["type"]]
            sensor = sensor_class(name=name, h_fov=h_fov, h_res=h_res, v_fov=v_fov, v_res=v_res, max_range=max_range, min_range=min_range, cost=cost, body=body, ap_constants=ap_constants)
            return sensor
            

    
    def to_json(self):
        """
        Convert the sensor to a JSON serializable format.
        Returns:
            dict: A dictionary representation of the sensor.
        """
        data = {
            "type": self.__class__.__name__,
            "name": self.name,
            "h_fov": self.h_fov,
            "h_res": self.h_res,
            "v_fov": self.v_fov,
            "v_res": self.v_res,
            "max_range": self.max_range,
            "min_range": self.min_range,
            "cost": self.cost,
            "ap_constants": self.ap_constants
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
    
    def plot_ap_h_u_m(self, show=True, epsilon=1e-3, verbose=True) -> plt.Figure:
        """
        Plot the Average Precision (AP), sigma_i, and Perception Entropy H vs. m 
        using the sensor's AP constants.
        Args:
            show (bool): Whether to show the plot.
            epsilon (float): A small value to avoid division by zero.
            max_m (float): The maximum value of m for the plot.
        Returns:
            fig (matplotlib.figure.Figure): The figure object.
        """

        from mpl_toolkits.axes_grid1 import host_subplot
        import mpl_toolkits.axisartist as AA

        # Range of m values, assume that all rays can land on one voxel

        h_rays = int(self.h_fov/self.h_res)+1
        v_rays = int(self.v_fov/self.v_res)+1
        rays = h_rays * v_rays

        if isinstance(self, StereoCamera3D):
            rays = rays * 2

        if verbose:
            print(f"Sensor: {self.name}\n  FOV: {self.h_fov}x{self.v_fov}\n  Resolution: {h_rays}x{v_rays}\n  Rays: {rays}")
            print(f"  AP Constants: a={self.ap_constants['a']}, b={self.ap_constants['b']}")
        
        m_values = np.linspace(0.01, (h_rays*v_rays), 500)
        ln_m = np.log(m_values)

        # Calculate unclamped AP
        ap_raw = self.ap_constants["a"] * ln_m + self.ap_constants["b"]

        # Clamp AP
        ap_clamped = np.clip(ap_raw, epsilon, 1 - epsilon)

        # Calculate sigma_i
        sigma_i = 1 / ap_clamped - 1

        # Calculate Perception Entropy
        perception_entropy = 2 * np.log(sigma_i) + 1 + np.log(2 * np.pi)

        # Create host subplot for 3-axis plot
        fig = plt.figure(figsize=(8, 6))
        host = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)

        # Create parasite axes
        par1 = host.twinx()
        par2 = host.twinx()

        # Offset the third axis
        offset = 60
        new_fixed_axis = par2.get_grid_helper().new_fixed_axis
        par2.axis["right"] = new_fixed_axis(loc="right", axes=par2, offset=(offset, 0))
        par2.axis["right"].toggle(all=True)

        # Ensure par1 uses the middle right axis
        par1.axis["right"].toggle(all=True)

        # Plot AP
        p1, = host.plot(m_values, ap_clamped, color="#004666", label="Clamped AP")
        host.set_ylabel("Clamped AP")
        host.set_xlabel("m")
        host.axis["left"].label.set_color(p1.get_color())

        # Plot sigma_i
        p2, = par1.plot(m_values, sigma_i, color="orange", label=r"$\sigma_i$")
        par1.set_ylabel(r"$\sigma_i$")
        par1.axis["right"].label.set_color(p2.get_color())

        # Plot perception entropy
        p3, = par2.plot(m_values, perception_entropy, color="green", linestyle="--", label=r"$H_i(S_i|m_i,q)$")
        par2.set_ylabel(r"$H_i(S_i|m_i,q)$")
        par2.axis["right"].label.set_color(p3.get_color())

        # Title and legend
        subtitle = r"AP, $\sigma_i$, and $H_i$ vs. m"
        plt.title(f"{self.name}\n{subtitle}")
        lines = [p1, p2, p3]
        labels = [l.get_label() for l in lines]
        host.legend(lines, labels, loc='upper right')

        plt.show() if show else None
        return fig


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
                 body:UsdGeom.Mesh, # Not implemented
                 ap_constants = {
                        "a": 0.152,  # coefficient from Ma et. al. 2021
                        "b": 0.659   # coefficient from Ma et. al. 2021
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
        """
        super().__init__(name, h_fov, h_res, v_fov, v_res, max_range, min_range, cost, body, ap_constants=ap_constants)



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
                 ap_constants:dict = {
                        "a": 0.055,  # coefficient from the paper for camera
                        "b": 0.155   # coefficient from the paper for camera
                    },
                 h_fov:float=None, # If you get this, the sensor is being reconstructed from a json file
                 v_fov:float=None, # If you get this, the sensor is being reconstructed from a json file
                 max_range:float=100.0, # If you get this, the sensor is being reconstructed from a json file
                 min_range:float=0.0, # If you get this, the sensor is being reconstructed from a json file
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
        """
        self.name = name
        self.body = body
        self.cost = cost
        if h_fov is None or v_fov is None:
            assert focal_length is not None, "Focal length must be provided if h_fov and v_fov are not provided"
            assert h_aperture is not None, "Horizontal aperture must be provided if h_fov and v_fov are not provided"
            assert v_aperture is not None, "Vertical aperture must be provided if h_fov and v_fov are not provided"
            assert aspect_ratio is not None, "Aspect ratio must be provided if h_fov and v_fov are not provided"
            assert h_res is not None, "Horizontal resolution must be provided if h_fov and v_fov are not provided"
            assert v_res is not None, "Vertical resolution must be provided if h_fov and v_fov are not provided"
            self.h_aperture = h_aperture
            self.v_aperture = v_aperture
            self.aspect_ratio = aspect_ratio

            self.h_fov = np.rad2deg(2 * np.arctan(h_aperture / (2 * focal_length)))
            self.v_fov = np.rad2deg(2 * np.arctan(v_aperture / (2 * focal_length)))

            self.h_res = self.h_fov/h_res # number of degrees between pixels. It is the way it is for isaac sim ray casting, don't ask me why
            self.v_res = self.v_fov/v_res # number of degrees between pixels. It is the way it is for isaac sim ray casting, don't ask me why

            self.max_range = max_range
            self.min_range = min_range

            self.ap_constants = ap_constants
        else:
            assert h_fov is not None, "Horizontal field of view must be provided"
            assert v_fov is not None, "Vertical field of view must be provided"
            assert max_range is not None, "Maximum range must be provided"
            assert min_range is not None, "Minimum range must be provided"
            assert h_res is not None, "Horizontal resolution must be provided"
            assert v_res is not None, "Vertical resolution must be provided"
            assert cost is not None, "Cost must be provided"

            super().__init__(name, h_fov, h_res, v_fov, v_res, max_range, min_range, cost, body, ap_constants=ap_constants)
        

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
        # self.base_line = np.linalg.norm(tf_sensor1[0] - tf_sensor2[0])
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
        
        assert "name" in json_dict, "Invalid sensor data, no name found"
        assert "sensor1" in json_dict, "Invalid sensor data, no sensor1 found"
        assert "sensor2" in json_dict, "Invalid sensor data, no sensor2 found"
        assert "pos_sensor1" in json_dict, "Invalid sensor data, no pos_sensor1 found"
        assert "pos_sensor2" in json_dict, "Invalid sensor data, no pos_sensor2 found"
        assert "rot_sensor1" in json_dict, "Invalid sensor data, no rot_sensor1 found"
        assert "rot_sensor2" in json_dict, "Invalid sensor data, no rot_sensor2 found"
        assert "cost" in json_dict, "Invalid sensor data, no cost found"
        # assert "ap_constants" in json_dict, "Invalid sensor data, no ap_constants found"

        return StereoCamera3D(
            name=json_dict["name"],
            sensor1=MonoCamera3D.from_json(json_dict["sensor1"]),
            sensor2=MonoCamera3D.from_json(json_dict["sensor2"]),
            tf_sensor1=(json_dict["pos_sensor1"], json_dict["rot_sensor1"]),
            tf_sensor2=(json_dict["pos_sensor2"], json_dict["rot_sensor2"]),
            cost=json_dict["cost"],
            # ap_constants=json_dict["ap_constants"]
        )
    
    def to_json(self):
        """
        Convert the sensor to a JSON serializable format.
        Returns:
            dict: A dictionary representation of the sensor.
        """
        data = {
            "type": self.__class__.__name__,
            "name": self.name,
            "sensor1": self.sensor1.to_json(),
            "sensor2": self.sensor2.to_json(),
            "pos_sensor1": list(self.tf_1[0]), #(x,y,z)
            "rot_sensor1": list(self.tf_1[1]), #(qw,qx,qy,qz)
            "pos_sensor2": list(self.tf_2[0]), #(x,y,z)
            "rot_sensor2": list(self.tf_2[1]), #(qw,qx,qy,qz)
            "cost": self.cost,
            "ap_constants": self.ap_constants
        }
        return data


class Sensor3D_Instance:
    def __init__(self,
                 sensor:Sensor3D,
                 path:str,
                 tf:tuple[tuple[float], tuple[float]],
                 usd_context=None,
                 name:str|None=None,
                 body:UsdGeom.Mesh=None, # Not implemented
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

        # If a usd_context is provided, create the sensor body and ray casters in isaac sim
        # if ISAAC_SIM_MODE and self.usd_context is not None and self.body is not None:
        #     self.ray_casters = self.create_ray_casters()
        #     self.body = self.create_sensor_body(sensor.body)

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
    
    def get_world_tfs(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Get the translation and rotation of the sensor in world coordinates,
        using SciPy instead of tf_utils.
        Returns:
            list[tuple[np.ndarray, np.ndarray]]: A list of tuples containing the translation a
            nd rotation of the sensor in world coordinates. ((x,y,z), (w,x,y,z))
        """
        tfs = []

        if isinstance(self.sensor, StereoCamera3D):
            # robot to parent
            T_rp = TF.tf_matrix_from_pose(self.translation, self.quat_rotation)

            for pos_ps, q_ps_sf in [self.sensor.tf_1, self.sensor.tf_2]:
                # parent to sensor
                T_ps = TF.tf_matrix_from_pose(pos_ps, q_ps_sf)

                # robot to sensor
                T_rs = T_rp @ T_ps

                # extract translation and rotation
                pos_rs = T_rs[:3, 3]
                R_rs   = T_rs[:3, :3]
                quat_rs_xyzw = R.from_matrix(R_rs).as_quat()
                quat_rs_wxyz = np.roll(quat_rs_xyzw, 1) # I think there is a bug here? quaternions seem to be wrong sometimes. See HACK below
                
                # tfs.append((pos_rs, quat_rs_wxyz))
                # HACK: Uses the parent rotation quaternion all the time to fix stereo cameras
                tfs.append((pos_rs, self.quat_rotation))
        else:
            # no parent frame; sensor is directly on robot
            tfs.append((np.array(self.translation, dtype=float),
                        np.array(self.quat_rotation, dtype=float)))

        return tfs

    def get_rays(
            self, 
            verbose:bool=False
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (ray_origins, ray_directions) for this sensor instance, vectorized with torch.
        Args:
            verbose (bool): If True, print debug information.
            
        Returns:
            Tuple[torch.Tensor]: A tuple of two tensors: (ray_origins, ray_directions)
                ray_origins: The origins of the rays, shape (N, 3)
                ray_directions: The directions of the rays, shape (N, 3)"""

        start_time = time.time()
        torch.cuda.empty_cache()

        sensors = [self.sensor] if not isinstance(self.sensor, StereoCamera3D) else [self.sensor.sensor1, self.sensor.sensor2]
        tfs = self.get_world_tfs()

        print(f"Calculating rays for {type(self)} {self.name}, with {len(tfs)} sets of rays...") if verbose else None

        ray_origins_list = []
        ray_directions_list = []

        for i, sensor in enumerate(sensors):

            if verbose:
                print(f" Sensor {i}: {sensor.name}, is a ({type(sensor)})")
                print(f"  POS:  {tfs[i][0]} (x,y,z)")
                print(f"  QUAT: {tfs[i][1]} (w,x,y,z) - NOTE: HACK")

            hfov = sensor.h_fov
            vfov = sensor.v_fov
            hres = int(sensor.h_fov / sensor.h_res) # number of rays, horizontal
            vres = int(sensor.v_fov / sensor.v_res) # number of rays, vertical

            # Check if there are zero norms in quat
            if np.linalg.norm(tfs[i][1]) == 0:
                print('\033[91m' + f"Warning: Quaternion used to generate rays for {sensor.name} has zero norm. Skipping sensor in instance {self.name}.\nQUAT: {tfs[i][1]}" + '\033[0m')
                continue
            
            quat_scalar_last = (tfs[i][1][1], tfs[i][1][2], tfs[i][1][3], tfs[i][1][0])  # (x,y,z,w)
            rotation = R.from_quat(quat_scalar_last).as_matrix() # Convert quaternion to rotation matrix

            if verbose:
                print(f"  R:    {rotation[0]}")
                print(f"        {rotation[1]}")
                print(f"        {rotation[2]}")
            
            position = torch.tensor(tfs[i][0], dtype=torch.float32, device=device) # Simple x,y,z
            rotation = torch.tensor(rotation, dtype=torch.float32, device=device) # Rotation matrix

            # Generate grid of angles
            h_angles = torch.linspace(-hfov/2, hfov/2, hres, device=device) * (np.pi/180)
            v_angles = torch.linspace(-vfov/2, vfov/2, vres, device=device) * (np.pi/180)
            v_grid, h_grid = torch.meshgrid(v_angles, h_angles, indexing='ij')
            v_flat = v_grid.flatten()
            h_flat = h_grid.flatten()

            if verbose:
                print(f"  H_ANGLES: shape={h_angles.shape}, min={torch.min(h_angles)}, max={torch.max(h_angles)}")
                print(f"  V_ANGLES: shape={v_angles.shape}, min={torch.min(v_angles)}, max={torch.max(v_angles)}")
                print(f"  H_GRID: shape={h_grid.shape}, min={torch.min(h_grid)}, max={torch.max(h_grid)}")
                print(f"  V_GRID: shape={v_grid.shape}, min={torch.min(v_grid)}, max={torch.max(v_grid)}")

            # Spherical to Cartesian (vectorized)
            x = torch.cos(v_flat) * torch.cos(h_flat)
            y = torch.cos(v_flat) * torch.sin(h_flat)
            z = torch.sin(v_flat)
            
            if verbose:
                print(f"  +X: {x}")
                print(f"  +Y: {y}")
                print(f"  +Z: {z}")

            # Stack and normalize
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
        print(f"  RAY TENSOR shape: {ray_origins.shape}") if verbose else None
        print(f"  RAY ORIGINS max: {torch.max(ray_origins)}, min: {torch.min(ray_origins)}, mean: {torch.mean(ray_origins)}") if verbose else None
        print(f"  RAY DIRECTIONS max: {torch.max(ray_directions)}, min: {torch.min(ray_directions)}, mean: {torch.mean(ray_directions)}") if verbose else None
        return ray_origins, ray_directions
    
    def get_ray_distances(
            self,
            ray_origins:torch.Tensor, 
            ray_directions:torch.Tensor,
            occlusion_aabs:Optional[Sequence[Tuple[
                    Tuple[float,float],           # (xmin, xmax)
                    Tuple[float,float],           # (ymin, ymax)
                    Tuple[float,float]            # (zmin, zmax)
                ]]], 
            max_range:float,
            min_range:float=0.0
            ) -> torch.Tensor:
        """
        For each ray, get the distance to the nearest occlusion AABB in occlusion_aabs.
        If a ray never hits any occluder in front of it, distance = max_range.

        Args:
            ray_origins:     (R,3) tensor of ray start points.
            ray_directions:  (R,3) tensor of ray direction vectors.
            occlusion_aabs:  optional list of occluder AABBs as
                            [((xmin,xmax),(ymin,ymax),(zmin,zmax)), …].
            max_range:       the maximum sensor range.

        Returns:
            distances:       (R,) tensor of hit distances (clamped to [0, max_range]).
        """
        assert ray_origins.shape[0] == ray_directions.shape[0], "Ray origins and directions must have the same number of rays"
        assert isinstance(occlusion_aabs, (list, tuple)) or occlusion_aabs is None, "Occlusion aabs must be a list, tuple, or None"

        device = ray_origins.device
        R = ray_origins.size(0)

        # If no occluders provided, every ray travels full range
        if not occlusion_aabs:
            return torch.full((R,), max_range,
                            device=device,
                            dtype=ray_origins.dtype)

        # Build occluder min/max tensors
        oc_min_list = []
        oc_max_list = []
        for (xmin, xmax), (ymin, ymax), (zmin, zmax) in occlusion_aabs:
            oc_min_list.append([xmin, ymin, zmin])
            oc_max_list.append([xmax, ymax, zmax])
        oc_mins = torch.tensor(oc_min_list,
                            dtype=ray_origins.dtype,
                            device=device)  # (O,3)
        oc_maxs = torch.tensor(oc_max_list,
                            dtype=ray_origins.dtype,
                            device=device)  # (O,3)

        # Prepare for batched slab‐tests
        O = ray_origins.unsqueeze(1)     # (R,1,3)
        D = ray_directions.unsqueeze(1)  # (R,1,3)
        eps = 1e-8
        D_safe = torch.where(D.abs() < eps,
                            torch.full_like(D, eps),
                            D)

        Oc_min = oc_mins.unsqueeze(0)    # (1,O,3)
        Oc_max = oc_maxs.unsqueeze(0)    # (1,O,3)

        # Ray–occluder intersections
        t1 = (Oc_min - O) / D_safe        # (R,O,3)
        t2 = (Oc_max - O) / D_safe        # (R,O,3)
        t_near = torch.min(t1, t2)        # (R,O,3)
        t_far  = torch.max(t1, t2)        # (R,O,3)

        t_enter = torch.max(t_near, dim=2).values  # (R,O)
        t_exit  = torch.min(t_far,  dim=2).values  # (R,O)

        # Valid hits: exit ≥ enter and exit ≥ 0
        hit_mask = (t_exit >= t_enter) & (t_exit >= 0)

        # Replace non‐hits with +inf so they won’t be chosen
        inf = torch.full_like(t_enter, float('inf'))
        t_enter_valid = torch.where(hit_mask, t_enter, inf)  # (R,O)

        # Nearest occlusion‐entry per ray
        t_first, _ = t_enter_valid.min(dim=1)  # (R,)

        # Clamp to [0, max_range]
        distances = torch.clamp(t_first, min=min_range, max=max_range)

        return distances

    def plot_rays(
            self, 
            ray_origins, 
            ray_directions, 
            ray_length=1.0,
            occlusion_aabs=None,
            max_rays=100, 
            fig=None, 
            show=True,
            legendgroup=None,
            ):
        """Plot the rays in 3D using plotly"""

        ray_distances = self.get_ray_distances(ray_origins, ray_directions, occlusion_aabs, ray_length)
        
        # Convert to numpy arrays
        ray_origins = ray_origins.cpu().numpy()
        ray_directions = ray_directions.cpu().numpy()
        ray_distances = ray_distances.cpu().numpy()

        # Create a 3D scatter plot
        alone_on_fig = False
        if fig is None:
            fig = go.Figure()
            alone_on_fig = True

        # Add rays
        for i in range(0, ray_origins.shape[0], int(ray_origins.shape[0]/max_rays)+1):
            fig.add_trace(go.Scatter3d(
                x=[ray_origins[i, 0], ray_origins[i, 0] + ray_directions[i, 0]*ray_distances[i]],
                y=[ray_origins[i, 1], ray_origins[i, 1] + ray_directions[i, 1]*ray_distances[i]],
                z=[ray_origins[i, 2], ray_origins[i, 2] + ray_directions[i, 2]*ray_distances[i]],
                mode='lines',
                line=dict(color=random_color(self.name), width=2),
                hoverinfo='none',
                showlegend=False,
                legendgroup=legendgroup,
            ))

        # Set the layout
        if alone_on_fig:
            fig.update_layout(
                title=f'{self.name} Rays',
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


    def plot_sensor(
            self, 
            fig, 
            group_mode="type", 
            idx=None,
            plot_rays=False, 
            ray_length=1.0,
            occlusion_aabs=None, 
            max_rays=100, 
            vec_length=0.2,
            ):
        """Plot the sensor in 3D using plotly
        Args:
            fig (plotly.graph_objects.Figure): The figure to plot on.
            group_mode (str): The mode to group the sensor by. Can be "instance", "type", or "idx".
            vec_length (float): The length of the vectors to plot.
        """

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

        if group_mode == "type":
            assert idx is not None, "idx must be provided when group_mode is 'type'"
            legendgroup = self.sensor.name
            legendgrouptitle_text=self.sensor.name
            name = f"Sensor {idx}"
        if group_mode == "instance":
            legendgroup = "Sensors"
            legendgrouptitle_text = "Sensors"
            name = f"Sensor {idx}: {self.name}" if idx is not None else self.name
        if group_mode == "idx":
            legendgroup = None
            legendgrouptitle_text=None
            name = f"Sensor {idx}: {self.name}" if idx is not None else self.name
        
        color = random_color(self.name)

        # Add the first point as a marker
        fig.add_trace(go.Scatter3d(
            x=[float(translation[0])],  # Only the first point
            y=[float(translation[1])],
            z=[float(translation[2])],
            mode='markers',  # Display as markers
            marker=dict(size=3, color=color),  # Marker size and color
            name=name,  # Legend label
            legendgroup=legendgroup,  # Group in the legend
            legendgrouptitle_text=legendgrouptitle_text,  # Group title
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
            legendgroup=legendgroup,  # Group in the legend
        ))

        # Add a Y+ line between the two points
        fig.add_trace(go.Scatter3d(
            x=[float(translation[0]), float(translation[0]) + float(y_direction[0])],  # Both points
            y=[float(translation[1]), float(translation[1]) + float(y_direction[1])],
            z=[float(translation[2]), float(translation[2]) + float(y_direction[2])],
            mode='lines',  # Display as a line
            line=dict(color='green', width=2),  # Line color and width
            showlegend=False,  # Hide legend for the line itself
            legendgroup=legendgroup,  # Group in the legend
        ))

        # Add a Z+ line between the two points
        fig.add_trace(go.Scatter3d(
            x=[float(translation[0]), float(translation[0]) + float(z_direction[0])],  # Both points
            y=[float(translation[1]), float(translation[1]) + float(z_direction[1])],
            z=[float(translation[2]), float(translation[2]) + float(z_direction[2])],
            mode='lines',  # Display as a line
            line=dict(color='blue', width=2),  # Line color and width
            showlegend=False,  # Hide legend for the line itself
            legendgroup=legendgroup,  # Group in the legend
        ))

        if plot_rays:
            # Plot the rays
            ray_origins, ray_directions = self.get_rays()
            self.plot_rays(
                ray_origins, 
                ray_directions, 
                occlusion_aabs=occlusion_aabs, 
                max_rays=max_rays,
                ray_length=ray_length,
                fig=fig, 
                show=False,
                legendgroup=legendgroup)

        return fig
    
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
                 body:Optional[Sequence[Tuple[    # AABBs for the bot body
                    Tuple[float,float],           # (xmin, xmax)
                    Tuple[float,float],           # (ymin, ymax)
                    Tuple[float,float]            # (zmin, zmax)
                ]]] = None,
                 path:str=None,
                 sensor_pose_constraint:np.ndarray[float]=None,
                 sensors:list[Sensor3D_Instance]=[]):
        """
        Initialize a bot representation with a given shape, sensor coverage requirements, and optional color and sensor pose constraints.
        Args:
            name (str): The name of the bot.
            usd_context (omni.usd.UsdContext): The USD context (if there is one).
            body (list[UsdGeom.Mesh]): List of aabbs for the bot body. If None, no AABBs are used. [((x0, x1), (y0, y1), (z0, z1)),...]
            path (str): The path to the bot in the USD stage.
            sensor_pose_constraint (np.ndarray): The pose constraint for the sensors as an array. Only one constraint AABB is supported. ((x0, x1), (y0, y1), (z0, z1))
            sensors (list[Sensor3D_Instance]): A list of sensor instances attached to the bot.
        """
        self.name = name
        self.path = path
        self.sensor_pose_constraint = sensor_pose_constraint
        self.sensors = sensors
        self.body = body

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
            "body": self.body,
            "sensor_pose_constraint": self.sensor_pose_constraint.tolist() if isinstance(self.sensor_pose_constraint, np.ndarray) else self.sensor_pose_constraint,
            "sensor_instances": [
                {
                    "name": sensor.name,
                    "path": str(sensor.path),  # Convert Path to string
                    "translation": sensor.translation.tolist() if isinstance(sensor.translation, np.ndarray) else sensor.translation,
                    "rotation": sensor.quat_rotation.tolist() if isinstance(sensor.quat_rotation, np.ndarray) else sensor.quat_rotation,
                    "sensor": sensor.sensor.to_json(),
                }
                for sensor in self.sensors
            ],
        }
        if file_path is not None:
            with open(file_path, 'w') as f:
                json.dump(bot_dict, f)
        return bot_dict
    
    @staticmethod
    def from_json(json_dict:dict) -> 'Bot3D':
        """Deserialize from a dict to create a Bot3D object. The dict should be in the same format as the one returned by to_json()."""
        name = json_dict["name"]
        path = json_dict["path"]
        body = json_dict["body"]
        sensor_pose_constraint = json_dict["sensor_pose_constraint"]
        usd_context=None
        sensors = []
        for i, sensor in enumerate(json_dict["sensor_instances"]):
            sensor_name = sensor["name"]
            sensor_path = sensor["path"]
            translation = sensor["translation"]
            rotation = sensor["rotation"]

            # Create a Sensor3D_Instance for each sensor
            if sensor["sensor"] is not None:
                sensor = Sensor3D.from_json(sensor["sensor"])
            elif sensor["sensor1"] is not None and sensor["sensor2"] is not None:
                sensor = StereoCamera3D.from_json(sensor["sensor"])
            else:
                sensor = None
            sensor_instance = Sensor3D_Instance(name=f"s{i}={sensor_name}",
                                                sensor=sensor, 
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
    
    def calculate_perception_entropies(self, perception_space:PerceptionSpace, verbose:bool=False, rays_per_chunk:int=20000, voxels_per_chunk:int=2000) -> tuple:
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
            print(f"Calculating fused uncertainty for {uncertainties.shape[0]} voxels and {uncertainties.shape[1]} sensors.") if verbose else None
            print(f"  uncertainties.shape: {uncertainties.shape}, should be ({uncertainties.shape[0]}, {uncertainties.shape[1]})")  if verbose else None
            # Calculate the fused uncertainty using the formula
            # σ_fused = sqrt(1 / Σ(1/σ_i²))
            fused_uncertainty = torch.sqrt(1 / torch.sum(1 / (uncertainties ** 2), dim=1))

            print(f"  fused_uncertainty.shape: {fused_uncertainty.shape}, should be (N={uncertainties.shape[0]},)") if verbose else None
            print(f"  fused_uncertainty min: {fused_uncertainty.min()}, fused_uncertainty max: {fused_uncertainty.max()}, fused_uncertainty mean:{fused_uncertainty.float().mean()}") if verbose else None
            
            return fused_uncertainty
        
        def _calc_aps(measurements:torch.Tensor, a:torch.Tensor, b:torch.Tensor, eps=1e-3) -> torch.Tensor:
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
            print(f"  ap.shape: {ap.shape}, should be ({N}, {S})") if verbose else None
            print(f"  ap (unclamped) min: {ap.min()}, ap max: {ap.max()}, ap mean:{ap.float().mean()}") if verbose else None

            # clamp AP to valid range, avaoiding log(0)
            ap = torch.clamp(ap, min=eps, max=1.0-eps)
            print(f"  ap  (clamped)  min: {ap.min()}, ap max: {ap.max()}, ap mean:{ap.float().mean()}") if verbose else None
            
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

            print(f"Calculating uncertainty for {aps.shape[0]} voxels and {aps.shape[1]} sensors.") if verbose else None
            
            uncertainties = (1 / aps) - 1

            print(f"  uncertainties.shape: {uncertainties.shape}, should be {aps.shape}") if verbose else None
            print(f"  uncertainties min: {uncertainties.min()}, uncertainties max: {uncertainties.max()}, uncertainties mean:{uncertainties.float().mean()}") if verbose else None
            
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
            # H(S|m,q) = 2ln(σ + 1) + ln(2pi)
            print(f"Calculating entropy for {uncertainties.shape[0]} voxels.") if verbose else None
            print(f"  uncertainties.shape: {uncertainties.shape}, should be ({uncertainties.shape[0]},)") if verbose else None

            ln2pi = (torch.log(torch.tensor(2*np.pi))).repeat(uncertainties.shape[0]).to(device) # repeat for each voxel
            print(f"  '1+ln(2pi)'.shape: {ln2pi.shape}, should be ({uncertainties.shape[0]},)") if verbose else None
            print(f"  '1+ln(2pi)' min: {ln2pi.min()}, max: {ln2pi.max()}, mean:{ln2pi.float().mean()}") if verbose else None
            entropy = 2 * torch.log(uncertainties + 1) + ln2pi

            print(f"  entropy.shape: {entropy.shape}, should be ({uncertainties.shape[0]},)") if verbose else None
            print(f"  entropy min: {entropy.min()}, max: {entropy.max()}, mean:{entropy.float().mean()}") if verbose else None

            # Reshape the entropy tensor to (N,) where N is the number of voxels
            entropy = entropy.view(-1)
            
            return entropy
            
        ###################################################################################################
        
        # Create one tensor per sensor type (R, N, S) where R is the number of rays, 
        # N is the number of voxels, and S is the number of sensors.
        start_time = time.time()

        print(f"Calculating PE for {self.name}, with {len(self.sensors)} sensors.") if verbose else None

        # Get the robot body aabbs once TODO
        # body_mins = torch.zeros((len(self.sensors), 3)).to(device)
        # body_maxs = self.

        sensor_ms = {}
        for sensor_inst in self.sensors:

            print(f" Sensor: {sensor_inst.name}") if verbose else None

            o,d = sensor_inst.get_rays()

            name = sensor_inst.sensor.name

            # This is a tensor of shape (R, N) where R is the number of rays and N is the number of voxels. 
            # Each element is True if the ray intersects with the voxel, False otherwise.
            sensor_m:torch.Tensor = perception_space.chunk_ray_voxel_intersections(o, d, rays_per_chunk=rays_per_chunk, voxels_per_chunk=voxels_per_chunk, verbose=False)
            # sensor_m:torch.Tensor = perception_space.chunk_occluded_ray_voxel_intersections(o,d,body_mins, body_maxs) #TODO occluded measurements
            print(f"  sensor_m.shape: {sensor_m.shape}, should be (N,)")  if verbose else None
            print(f"  sensor_m min: {sensor_m.min()}, sensor_m max: {sensor_m.max()}, sensor_m mean:{sensor_m.float().mean()}")  if verbose else None

            # Add the sensor measurements to the tensor for the sensor type
            if name not in sensor_ms or sensor_ms[name].numel() == 0:
                sensor_ms[name] = sensor_m.unsqueeze(1)
            else:
                sensor_ms[name] = torch.cat((sensor_ms[name], sensor_m.unsqueeze(1)), dim=1)
        
        if sensor_ms == {}:
            print("No sensors found in the bot!! Returning max perception entropy, and no coverage.")
            return np.inf, 0.0

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
            print(f"  m.shape: {m.shape}, should be (N,) = num hits per voxel")  if verbose else None

            early_fusion_ms.append(m.unsqueeze(1))  # shape (N, 1)
            ap_as = torch.cat((ap_as, torch.Tensor([sensor.ap_constants['a']])), dim=0)
            ap_bs = torch.cat((ap_bs, torch.Tensor([sensor.ap_constants['b']])), dim=0)
            # This is a tensor of shape (N, S) where N is the number of voxels and S is the number of sensors.
            # Each element is the number of rays that intersect with the voxel for that sensor type.
        
        early_fusion_ms = torch.cat(early_fusion_ms, dim=1)  # shape (N, S_types)
        print(f"Early Fusion for {len(sensor_ms)} sensors...") if verbose else None
        print(f" early_fusion_ms.shape: {early_fusion_ms.shape}, should be (N, S_types) = num hits per voxel")  if verbose else None
        print(f" early_fusion_ms min: {early_fusion_ms.min()}, early_fusion_ms max: {early_fusion_ms.max()}, early_fusion_ms mean:{early_fusion_ms.float().mean()}")  if verbose else None

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
        # This is still a tensor of shape (N) where N is the number of voxels.
        entropies = _calc_entropies(late_fusion_Us)

        return entropies #, early_fusion_ms, aps, us, late_fusion_Us
    
    def calculate_perception_entropy(self, perception_space:PerceptionSpace, verbose:bool=False, rays_per_chunk:int=20000, voxels_per_chunk:int=2000) -> float:
        """
        Calculate the perception entropy of the bot based on the sensors and the perception space.
        Args:
            perception_space (PerceptionSpace): The perception space to calculate the entropy for.

        Returns:
            float: The perception entropy of the bot. Lower entropy indicates better coverage.
        """

        entropies = self.calculate_perception_entropies(
            perception_space, 
            rays_per_chunk=rays_per_chunk, 
            voxels_per_chunk=voxels_per_chunk, 
            verbose=verbose
            )

        # Calculate the weighted average entropy for the bot
        weights = perception_space.get_voxel_weights()
        sum_weights = torch.sum(weights, dim=0)
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
            perception_space_colors='weights',
            show_sensor_pose_constraints:bool=True,
            show_body:bool=True,
            show_sensor_rays:bool=None,
            ray_length:float=1.0,
            max_rays:int=100,
            show=True, 
            show_origin:bool=False,
            plot_margin_dict=dict(l=5, r=5, b=5, t=30),
            save_path:str=None,
            rays_per_chunk:int=10000,
            voxels_per_chunk:int=1000,
            group_sensors_by='type',
            **kwargs):
        """Plot the bot in 3D using plotly.
        
        Plot contains:
            - the bot body as a mesh (if it exists)
            - each sensor in self.sensors as a point and a direction. Each sensor is colored by its type.
            - if perception_space is not None, the perception space is shown as a point cloud (a point for each voxel).

        Args:
            perception_space (PerceptionSpace): The perception space to plot. If None, the perception space is not plotted.
            show_sensor_pose_constraints (bool): If True, show the sensor pose constraints as a mesh.
            show_body (bool): If True, show the bot body as a mesh.
            show_sensor_rays (bool): If None, no rays are shown. If True, show all rays from all sensors.
                    If an int, show the rays of that sensor; if a list, show the rays of all sensors in the list.
            ray_length (float): The length of the rays to show. Only used if show_sensor_rays is not None.
            max_rays (int): The maximum number of rays to show PER SENSOR. Only used if show_sensor_rays is not None.
            show (bool): If True, show the plot. If False, save the plot to save_path.
            save_path (str): The path to save the plot. If None, do not save the plot.
            rays_per_chunk (int): The number of rays to use per chunk. Only used if show_sensor_rays is not None.
            voxels_per_chunk (int): The number of voxels to use per chunk. Only used if show_sensor_rays is not None.
            **kwargs: Additional arguments to pass to the plot
        Returns:
            fig (plotly.graph_objects.Figure): The plotly figure object.
        """

        height = 500 if 'height' not in kwargs else kwargs['height']
        width = 500 if 'width' not in kwargs else kwargs['width']
        title = f"{self.name} Original" if 'title' not in kwargs else kwargs['title']

        # Create a plotly figure
        fig = go.Figure()

        # Add a dot at the bot origin
        fig.add_trace(go.Scatter3d(
            x=[0.0],  # X-coordinate
            y=[0.0],  # Y-coordinate
            z=[0.0],  # Z-coordinate
            mode='markers',  # Display as markers
            marker=dict(size=3, color='black'),  # Marker size and color
            name='ORIGIN'  # Legend label
        )) if show_origin else None

        # Add the sensor pose constraints to the plot
        if show_sensor_pose_constraints and self.sensor_pose_constraint is not None:
            mesh_data = box_mesh_data(
                self.sensor_pose_constraint,
                opacity=0.2,
                color='green',
                name='Sensor Constraint',
                legendgrouptitle_text="Problem Definition",
                legendgroup="Problem Definition",
                showlegend=True,
                hoverinfo='skip'
            )
            fig.add_trace(mesh_data)

        # Add the robot body to the plot
        if show_body and self.body is not None and None not in self.body:
            showlegend = True
            for i, aabb in enumerate(self.body):
                mesh_data = box_mesh_data(
                        extents=aabb, 
                        color="orange", 
                        opacity=0.2, 
                        name=f"Robot Body",
                        legendgroup="Problem Definition",
                        showlegend=showlegend,
                        hoverinfo='skip'
                    )
                fig.add_trace(mesh_data)
                showlegend = False  # Only show legend for the first box

        # Add the sensors to the plot
        if show_sensor_rays is None:
            show_sensor_rays = []
        if isinstance(show_sensor_rays, bool) and show_sensor_rays:
            show_sensor_rays = [i for i,s in enumerate(self.sensors)]
        if isinstance(show_sensor_rays, int):
            show_sensor_rays = [show_sensor_rays]

        for i, sensor_i in enumerate(self.sensors):
            if i in show_sensor_rays:
                sensor_i.plot_sensor(
                    fig, 
                    plot_rays=True, 
                    idx=i,
                    ray_length=ray_length, 
                    occlusion_aabs=self.body, 
                    max_rays=max_rays, 
                    group_mode=group_sensors_by,
                    )
            else:
                sensor_i.plot_sensor(
                    fig,
                    group_mode=group_sensors_by,
                    idx=i,
                    )

        # Add the perception space
        entropies = None
        if perception_space is not None:
            if perception_space_colors == 'entropy':
                assert rays_per_chunk is not None, "If using 'entropy' as colors, rays_per_chunk must be set."
                assert voxels_per_chunk is not None, "If using 'entropy' as colors, voxels_per_chunk must be set."
                _, entropies = perception_space.plot_me(
                    fig, show=False, bot=self, colors=perception_space_colors, mode='centers', 
                    voxels_per_chunk=voxels_per_chunk, rays_per_chunk=rays_per_chunk
                    )
            else:
                _, entropies = perception_space.plot_me(
                    fig, show=False, bot=self, colors=perception_space_colors, mode='centers',
                    )

        # Adjust the layout of the plot
        fig.update_layout(
            height=height,
            width=width,
            title=title,
            template='ggplot2',
            # margin=plot_margin_dict,
            scene=dict(
                xaxis=dict(
                    title='X',
                    zerolinecolor="black",
                    ),
                yaxis=dict(
                    title='Y',
                    zerolinecolor="black",
                    ),
                zaxis=dict(
                    title='Z',
                    zerolinecolor="black",
                    ),
                camera=dict(
                    eye=dict(x=1.25, y=1.25, z=1.25),  # Adjust the camera position
                    center=dict(x=0, y=0, z=0),  # Center the camera on the origin
                    up=dict(x=0, y=0, z=1)  # Ensure the Z-axis is up
                ),
                aspectmode='data',  # Maintain the aspect ratio
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0,
                xanchor="right",
                x=1
            ),
        )

        # Show or save the plot
        if show:
            fig.show()
        if save_path is not None:
            fig.write_image(save_path)

        return fig, entropies

def box_mesh_data(extents:tuple[tuple,tuple,tuple], opacity:float=0.25, **kwargs):
    """Create a box mesh data for a box with given extents.
    Args:
        extents (tuple[tuple,tuple,tuple]): The extents of the box in the form ((x0,x1),(y0,y1),(z0,z1)).
        **kwargs: Additional arguments to pass to the mesh data. Used directly to create the go.Mesh3D obj.
    Returns:
        mesh_data: The mesh data for the box. Can me used to create a mesh in plotly. For example, add to an existing fig:
            mesh_data = box_mesh_data(((0,1),(0,1),(0,1)))
            fig.add_trace(mesh_data)
    """
    x0, x1 = extents[0]
    y0, y1 = extents[1]
    z0, z1 = extents[2]

    # Eight corners
    x_verts = [x0, x0, x1, x1, x0, x0, x1, x1]
    y_verts = [y0, y1, y1, y0, y0, y1, y1, y0]
    z_verts = [z0, z0, z0, z0, z1, z1, z1, z1]
    
    # Define static face‐index arrays for a cube
    i_faces = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j_faces = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k_faces = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    return go.Mesh3d(
        x=x_verts, y=y_verts, z=z_verts,
        i=i_faces, j=j_faces, k=k_faces,
        opacity=opacity, showscale=False,
        **kwargs
    )

def random_color(unique_str:str|None) -> str:
    """Create a random color from a string, or define your own mapping"""
    if unique_str is None:
        return "#000000"
    hash_digest = hashlib.md5(unique_str.encode()).hexdigest()
    return f"#{hash_digest[:6]}"