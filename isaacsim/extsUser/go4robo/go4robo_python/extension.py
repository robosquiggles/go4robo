import gc

import omni.ext
import omni.ui as ui
import omni.kit.commands
import omni.physx as _physx
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.gui.components.element_wrappers import ScrollingWindow
from isaacsim.gui.components.menu import MenuItemDescription
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from pxr import UsdGeom, Gf, Sdf, Usd, UsdPhysics, Vt, UsdShade, PhysxSchema
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.collisions as collisions_utils
from isaacsim.sensors.physx import _range_sensor

import asyncio # Used to run sample asynchronously to not block rendering thread

from collections import Counter
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union
import carb
import time

import torch

import re

from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE

import os, sys

from .bot_3d_rep import *
from .bot_3d_problem import *

sensor_types = [MonoCamera3D, Lidar3D, StereoCamera3D]

default_sensor_aps = {
    "Lidar3D": {
        "a": 0.152,
        "b": 0.659,
    },
    "MonoCamera3D": {
        "a": 0.055,
        "b": 0.155,
    },
    "StereoCamera3D": {
        "a": 0.055,
        "b": 0.155,
    }
}

class GO4RExtension(omni.ext.IExt):
    """Extension that calculates perception entropy for cameras and LiDARs in Isaac Sim"""
    
    def on_startup(self, ext_id):
        """Initialize the extension"""

        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()

        self.stage = omni.usd.get_context().get_stage()                      # Used to access Geometry
        self.timeline = omni.timeline.get_timeline_interface()               # Used to interact with simulation
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface() # Used to interact with the LIDAR

        self._window = ScrollingWindow(title=EXTENSION_TITLE, width=600, height=700)
        # self._window.set_visibility_changed_fn(self._on_window)

        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.register_action(
            ext_id,
            f"CreateUIExtension:{EXTENSION_TITLE}",
            self._menu_callback,
            description=f"Add {EXTENSION_TITLE} Extension to UI toolbar",
        )
        self._menu_items = [
            MenuItemDescription(name=EXTENSION_TITLE, onclick_action=(ext_id, f"CreateUIExtension:{EXTENSION_TITLE}"))
        ]

        add_menu_items(self._menu_items, EXTENSION_TITLE)

        self.ui_elements = []

        self.selected_export_path = None

        # A place to store the robots
        self.robots: list[Bot3D] = []

        # These just help handle the stage selection and structure
        self.previous_selection = []
        
        # Perception Space Voxels
        self.voxel_size = 0.1
        # self.weighted_voxels = {} # {0: ([...], 1.0)} = Group 0, with [...] voxels and weight 1.0

        self.percep_entr_results_data = {} # {"robot_name": {"Entropy By Voxel Group": {group_id: entropy, ...},
                                           #                 "Entropy By Sensor Type": {sensor_name: entropy, ...},
                                           #                 "Total Entropy":           total_entropy}}

        # Logging
        self.log_messages = []
        self.max_log_messages = 100  # Default value
        
        # Add a property to track the selected perception area mesh
        self.perception_space:PerceptionSpace = PerceptionSpace(self._usd_context)
        self.perception_mesh = None
        self.perception_mesh_path = None
        
        self._build_ui()
        self._window.visible = True

        # Events
        events = self._usd_context.get_stage_event_stream()
        self._stage_event_sub = events.create_subscription_to_pop(self._on_stage_event)

    def on_shutdown(self):
        """Shutdown the extension"""

        # Cleanup UI
        self._cleanup_ui()

        # Remove menu items
        remove_menu_items(self._menu_items)

        # Unregister action
        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.unregister_action(self.ext_id, f"CreateUIExtension:{EXTENSION_TITLE}")

        # Cleanup stage event subscription
        if self._stage_event_sub:
            self._stage_event_sub = None

        # Cleanup window
        if self._window:
            self._window.visible = False
            self._window = None

        # Force garbage collection
        gc.collect()

    def _build_ui(self):
        """Build the UI for the extension"""

        # Build the UI
        try:
            with self._window.frame:
                with ui.VStack(spacing=5):

                    # Detected sensors section
                    with ui.CollapsableFrame("Robot & Sensors", height=0):
                        with ui.VStack(spacing=5, height=0):
                            with ui.HStack(spacing=5):
                                self.selected_robot_label = ui.Label("(Select one or more robot from the stage)", style = {"color": ui.color("#FF0000")})
                                self.refresh_sensors_btn = ui.Button("Refresh Robots & Sensors", clicked_fn=self._refresh_sensor_list, height=36, width=0)
                                self.disable_ui_element(self.refresh_sensors_btn, text_color=ui.color("#FF0000"))
                            self.sensor_list = ui.ScrollingFrame(height=300)
                            with ui.CollapsableFrame("Data Export", height=0, collapsed=True):
                                with ui.VStack(spacing=5, height=0):
                                    with ui.HStack(spacing=5):
                                        ui.Label("Save Location:", width=120)
                                        self.export_path_field = ui.StringField()
                                        self.browse_path_btn = ui.Button("Browse...", width=80, clicked_fn=self._browse_export_path)
                                    with ui.HStack(spacing=5, height=0):
                                        ui.Label("Export to Tabular:", width=120)
                                        # Add file path selection UI
                                        self.export_robot_btn = ui.Button("Export Robot", clicked_fn=self._export_robot_data, height=36)
                                        self.disable_ui_element(self.export_robot_btn)
                                        self.export_sensors_btn = ui.Button("Export Sensors", clicked_fn=self._export_sensor_data, height=36)
                                        self.disable_ui_element(self.export_sensors_btn)

                    ui.Spacer(height=10)
                    
                    with ui.CollapsableFrame("Perception Space", height=0):
                        with ui.VStack(spacing=5, height=0):

                            # Add a label with instructions
                            with ui.CollapsableFrame("Instructions", height=0, collapsed=True):
                                with ui.VStack(spacing=5, height=0):
                                    ui.Label("The perception Space is a collection of voxels that represent the environment.\nVoxels can be grouped and weighted according to the importance of being able to see in that volume.")
                                    ui.Label("Select a mesh and then click 'Voxelize' to create a voxel group.")
                                    ui.Label("Move voxels from one group to another by dragging them in the stage (See 'GO4R_PerceptionSpace).")
                                    ui.Label("Adjust the importance/weight of each group with the textbox.")

                            # Add sampling step size UI
                            with ui.CollapsableFrame("Voxelization", height=0):
                                with ui.VStack(spacing=5, height=0):
                                    with ui.HStack(spacing=5):
                                        ui.Label("Sampling Density:", width=120)
                                        self.voxel_size_field = ui.FloatField(width=60)
                                        self.voxel_size_field.model.set_value(self.voxel_size)
                                        ui.Label("meters between sample points", width=0)
                                        
                                        def _update_step_size(value:float):
                                            # Extract the actual float value from the model object
                                            float_value = value.get_value_as_float()
                                            self.voxel_size = float_value
                                            self._log_message(f"Sampling voxel size set to {self.voxel_size} meters")
                                        
                                        self.voxel_size_field.model.add_value_changed_fn(_update_step_size)
                                        
                                    # Add mesh selection UI
                                    with ui.HStack(spacing=5):
                                        ui.Label("Perception Mesh:", width=120)
                                        self.perception_mesh_label = ui.Label("(Not selected)", style={"color": ui.color("#FF0000")})
                                        self.select_mesh_btn = ui.Button("Voxelize", width=80, height=36, clicked_fn=self._on_voxelize_button_clicked)
                                        self.voxelize_progress_bar = ui.ProgressBar(width=120, height=36, val=0.0)

                            # Voxel Groups Section
                            with ui.CollapsableFrame("Voxel Groups", height=0):
                                with ui.VStack(spacing=5, height=0):

                                    # Header row
                                    with ui.HStack(spacing=5):
                                        ui.Label("Group", width=120)
                                        ui.Label("Weight", width=60)
                                        ui.Label("Voxel Count", width=120)
                                        ui.Spacer(width=120)  # Align with first column
                                        self.add_group_btn = ui.Button("+ New Group", clicked_fn=self._add_voxel_group, height=24, style={"color": ui.color("#00FF00")})

                                    # Container for dynamically added voxel groups
                                    self.voxel_groups_container = ui.VStack(spacing=5, height=0)
                    
                    ui.Spacer(height=10)
                    
                    # Results section
                    with ui.CollapsableFrame("Metric: Perception Entropy", height=0, collapsed=False):
                        with ui.VStack(spacing=5):
                            # Buttons for operations
                            with ui.HStack(spacing=5, height=0):
                                self.analyze_btn = ui.Button("Analyze Perception Entropy", clicked_fn=self._batch_calc_perception_entropies, width=120, height=36)
                                # Initially disable the button since no mesh is selected
                                self.disable_ui_element(self.analyze_btn, text_color=ui.color("#FF0000"))
                                self.analysis_progress_bar = ui.ProgressBar(height=36, val=0.0)
                            # self.reset_btn = ui.Button("Reset", clicked_fn=self._reset_settings, height=36)
                            self.results_list = ui.ScrollingFrame(height=250)
                            # Initialize with empty results
                            with self.results_list:
                                ui.Label("Run analysis to see results")

                    ui.Spacer(height=20)
                    
                    with ui.CollapsableFrame("Optimization", height=0, collapsed=False):
                        with ui.VStack(spacing=5):
                            # Buttons for operations
                            with ui.HStack(spacing=5, height=0):
                                self.optimize_btn = ui.Button("Optimize", clicked_fn=self._optimize_robot, height=36)
                                self.optimize_btn.set_style({"color": ui.color("#FF0000")})
                                self.optimize_btn.set_tooltip("Reset all settings to default values")
                            # self.analysis_progress_bar = ui.ProgressBar(height=36, val=0.0)
                            # Initialize with empty results
                            with self.results_list:
                                ui.Label("Run analysis to see results")
                    
                    ui.Spacer(height=20)
                    
                    # Log section for detailed information
                    with ui.CollapsableFrame("Log", collaped=True, height=0):
                        with ui.VStack(spacing=5):
                            with ui.HStack(spacing=5, height=0):
                                ui.Label("Max Messages:", width=100, height=18)
                                self.max_messages_field = ui.IntField(width=60, height=18)
                                self.max_messages_field.model.set_value(self.max_log_messages)
                                self.max_messages_field.model.add_value_changed_fn(self._update_max_log_messages)
                                self.clear_log_btn = ui.Button("Clear Log", width=80, clicked_fn=self._clear_log, height=18)

                            self.log_field = ui.StringField(
                                read_only=True,  # Make it read-only so users can't edit the log
                                multiline=True,  # Enable multi-line text
                                height=250,
                                style={
                                    "font_size": 14,
                                    "border_width": 0,  # No border for the field itself
                                }
                            )
        
        except Exception as e:
            print(f"Error building UI: {str(e)}")
            raise e
        
    def enable_ui_element(self, ui_element, text_color=None):
        """Enable a button and restore its normal style"""
        ui_element.enabled = True
        if text_color is not None:
            ui_element.set_style({"color": text_color})
        else:
            ui_element.set_style({})  # Reset to default style

    def disable_ui_element(self, ui_element, text_color=None):
        """Disable a button and apply disabled style"""
        ui_element.enabled = False
        ui_element.set_style({
            "background_color": ui.color("#555555"),
            "color": ui.color("#AAAAAA") if text_color is None else text_color,
            "opacity": 0.7
        })


    def _cleanup_ui(self): # TODO build the cleanup function
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from isaacsim.gui.components.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.ui_elements:
            if hasattr(ui_elem, 'cleanup'):
                ui_elem.cleanup()

    def _menu_callback(self):
        self._window.visible = not self._window.visible

    def _on_stage_event(self, event):

        selection = self._usd_context.get_selection().get_selected_prim_paths()

        # Only process if selection has actually changed
        if set(selection) == set(self.previous_selection):
            return  # Skip if selection hasn't changed

        # Store current selection for future comparison
        self.previous_selection = selection.copy()

        self.selected_prims = []

        if not selection:
            self.selected_robot_label.text = "(Select one or more robot from the stage)"
            self.selected_robot_label.style = {"color": ui.color("#FF0000")}
            self.disable_ui_element(self.refresh_sensors_btn, text_color=ui.color("#FF0000"))

            self.perception_mesh_label.text = "(Select one mesh from the stage)"
            self.perception_mesh_label.style = {"color": ui.color("#FF0000")}
            self.disable_ui_element(self.refresh_sensors_btn, text_color=ui.color("#FF0000"))
            return
        
        for robot_path in selection:
            bot_prim = get_current_stage().GetPrimAtPath(robot_path)
            self.selected_prims.append(bot_prim)

        self.selected_robot_label.text = f"Selected: {', '.join([prim_utils.get_prim_path(bot).split('/')[-1] for bot in self.selected_prims])}"
        self.enable_ui_element(self.refresh_sensors_btn, text_color=ui.color("#00FF00"))
        self.selected_robot_label.style = {"color": ui.color("#00FF00")}

        if len(self.selected_prims) > 1:
            self.perception_mesh_label.text = f"Selected: {', '.join([prim_utils.get_prim_path(bot).split('/')[-1] for bot in self.selected_prims])}"
            self.disable_ui_element(self.select_mesh_btn, text_color=ui.color("#FF0000"))
            self.disable_ui_element(self.analyze_btn, text_color=ui.color("#FF0000"))
            self.perception_mesh_label.style = {"color": ui.color("#FF0000")}
        elif len(self.selected_prims) == 1 and self.selected_prims[0].IsA(UsdGeom.Mesh):
            # One mesh is selected. Use it as your perception mesh??
            self.perception_mesh = self.selected_prims[0]
            self.perception_mesh_label.text = f"Selected: {self.selected_prims[0].GetName()}"
            self.enable_ui_element(self.select_mesh_btn, text_color=ui.color("#00FF00"))
            self.enable_ui_element(self.analyze_btn, text_color=ui.color("#00FF00"))
            self.perception_mesh_label.style = {"color": ui.color("#00FF00")}
        else:
            self.perception_mesh_label.text = "(Select one mesh from the stage)"
            self.perception_mesh_label.style = {"color": ui.color("#FF0000")}
            self.disable_ui_element(self.select_mesh_btn, text_color=ui.color("#FF0000"))
            self.disable_ui_element(self.analyze_btn, text_color=ui.color("#FF0000"))

        # If /World/GO4R_PerceptionVolume exists, update the voxel groups found there
        if get_current_stage().GetPrimAtPath("/World/GO4R_PerceptionVolume"):
            self._update_voxel_groups_from_stage()

    def _zero_gravity(self):
        """Disable gravity for all active prims in the stage"""
        if not self.stage:
            self.stage = get_current_stage()
        # Iterate over all prims in the stage
        for prim in self.stage.Traverse():
            # Check if the prim is active and not a prototype
            if prim.IsActive() and not prim.IsInstance():
                # Apply RigidBodyAPI and PhysxRigidBodyAPI
                UsdPhysics.RigidBodyAPI.Apply(prim)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

                # Access the PhysxRigidBodyAPI
                physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)

                # Disable gravity
                physx_api.CreateDisableGravityAttr(True)

                # Optionally, increase damping to prevent drifting
                physx_api.GetLinearDampingAttr().Set(1000.0)
                physx_api.GetAngularDampingAttr().Set(1000.0)

    def _gravity(self):
        for prim in self.stage.Traverse():
            # Check if the prim is active and not a prototype
            if prim.IsActive() and not prim.IsInstance():
                # Access the PhysxRigidBodyAPI
                physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)

                # Enable gravity
                if physx_api.GetDisableGravityAttr().IsValid():
                    physx_api.GetDisableGravityAttr().Set(False)

                # Reset damping values
                if physx_api.GetLinearDampingAttr().IsValid():
                    physx_api.GetLinearDampingAttr().Set(0.0)
                if physx_api.GetAngularDampingAttr().IsValid():
                    physx_api.GetAngularDampingAttr().Set(0.0)

    def _update_voxel_groups_from_stage(self):
        """Update voxel groups based on XForm hierarchy under GO4R_PerceptionVolume"""
        stage = get_current_stage()
        parent_path = "/World/GO4R_PerceptionVolume"
        parent_prim = stage.GetPrimAtPath(parent_path)
        
        if not parent_prim:
            self._log_message("Error: Perception volume not found in stage")
            return
            
        # Track all meshes found
        all_found_voxels = set()
              
        # Find all XForm children and assign group IDs
        xform_groups = [child for child in parent_prim.GetChildren() 
                        if child.IsA(UsdGeom.Xform) and not child.IsA(UsdGeom.Mesh)]
        
        # Clear the previously found voxel groups
        self.perception_space = PerceptionSpace(self._usd_context)
        
        # Process XForm groups first
        for xform in xform_groups:
            custom_data = xform.GetCustomData()
            weight = custom_data.get("perception_weight")
            if weight is None:
                xform.SetCustomDataByKey("perception_weight", 1.0)
                weight = 1.0
            xform_path = str(xform.GetPath())
            # The group id is the last part of the path
            group_id = xform_path.split("/")[-1]
            
            # Find all mesh children (voxels) under this XForm
            voxel_meshes = []
            voxel_sizes = []
            voxel_paths = []
            voxel_centers = []

            for child in xform.GetChildren():
                if child.IsA(UsdGeom.Mesh):
                    voxel_meshes.append(child)
                    all_found_voxels.add(str(child.GetPath()))

                    # Get the voxel size
                    boundable = UsdGeom.Boundable(child.GetPrim())
                    bbox = boundable.ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
                    box = bbox.ComputeAlignedBox()
                    min_point = box.GetMin()
                    max_point = box.GetMax()
                    size = [max_point[i] - min_point[i] for i in range(3)]
                    voxel_sizes.append(size)

                    # Get the center of the voxel
                    center = [(max_point[i] + min_point[i]) / 2 for i in range(3)]
                    voxel_centers.append(center)

                    # Get the path of the voxel
                    path = str(child.GetPath())
                    voxel_paths.append(path)
                
            vg = PerceptionSpace.VoxelGroup(group_id,
                                            voxel_paths, 
                                            torch.Tensor(voxel_centers), 
                                            torch.Tensor(voxel_sizes))
            self.perception_space.add_voxel_group(vg, weight)
        
        # Group ungrouped meshes to show the warning in the UI
        ungrouped_voxels = []
        for child in parent_prim.GetChildren():
            if child.IsA(UsdGeom.Mesh) and str(child.GetPath()) not in all_found_voxels:
                ungrouped_voxels.append(child)
                all_found_voxels.add(str(child.GetPath()))
        
        if ungrouped_voxels:
            # Create a new group for ungrouped voxels
            group_id = "UNGROUPED"
            voxel_paths = [str(mesh.GetPath()) for mesh in ungrouped_voxels]
            voxel_sizes = [mesh.GetExtentAttr().Get()[1] for mesh in ungrouped_voxels]
                
            vg = PerceptionSpace.VoxelGroup(group_id, voxel_paths, voxel_sizes)
            self.perception_space.add_voxel_group(vg, 0.0)
        
        
        # Update the UI
        self._update_voxel_groups_ui()

    def _update_max_log_messages(self, value):
        """Update the maximum number of log messages to keep"""
        self.max_log_messages = max(1, int(value))
        
    def _clear_log(self):
        """Clear all log messages"""
        self.log_messages = []
        self._update_log_display()
        
    def _log_message(self, message: str):
        """Add a message to the log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Add to message list
        self.log_messages.append(formatted_message)
        
        # Trim to max messages
        if len(self.log_messages) > self.max_log_messages:
            self.log_messages = self.log_messages[-self.max_log_messages:]
        
        # Update the log display with the new messages - this line is missing
        self._update_log_display()

    def _update_log_display(self):
        """Update the log display with current messages"""
        # Join the log messages with newlines and update the StringField
        # Ensure UI update happens on the main thread if called from async context
        async def update_ui():
            self.log_field.model.set_value("\n".join(self.log_messages))
        asyncio.ensure_future(update_ui())

    def _update_voxel_groups_ui(self):
        """Update the UI to show the current voxel groups"""
        # Clear existing UI
        self.voxel_groups_container.clear()
        
        with self.voxel_groups_container:
            for i, voxel_group in enumerate(self.perception_space.voxel_groups):
                group_name = voxel_group.name
                voxels = voxel_group.voxels
                weight = self.perception_space.weights[i]
                    
                with ui.HStack(spacing=5):

                    if group_name != "UNGROUPED":
                        ui.Label(group_name, width=120)
                    else:
                        # Special case for ungrouped voxels
                        ui.Label("UNGROUPED", width=120, style={"color": ui.color("#FF0000")})
                    
                    # Weight slider
                    weight_drag = ui.FloatDrag(width=60)
                    weight_drag.model.set_value(weight)
                    
                    def make_weight_changed_fn(g_id):
                        def on_weight_changed(value):
                            # Update the weight for this group
                            self.perception_space.set_voxel_group_weight(g_id, value.get_value_as_float())

                            # Update the custom weight data for the XForm
                            xform_path = f"/World/GO4R_PerceptionVolume/{g_id}"
                            xform_prim = get_current_stage().GetPrimAtPath(xform_path)
                            if xform_prim:
                                xform_prim.SetCustomDataByKey("perception_weight", value.get_value_as_float())
                            else:
                                self._log_message(f"Error: XForm {xform_path} not found in stage")

                            self._log_message(f"Updated {group_name} weight to {value.get_value_as_float()}")
                        return on_weight_changed
                    
                    if group_name != "UNGROUPED":
                        weight_drag.model.add_value_changed_fn(make_weight_changed_fn(group_name))
                    else:
                        # Disable the weight slider for ungrouped voxels
                        self.disable_ui_element(weight_drag, text_color=ui.color("#FF0000"))
                    
                    # Voxel count
                    
                    if group_name != "UNGROUPED":
                        ui.Label(f"{len(voxels)}", width=120)
                    else:
                        ui.Label(f"{len(voxels)}", width=120, style={"color": ui.color("#FF0000")})
        
        # Finally, if there are voxel groups, and robots to analyze, enable the analyze perception button
        names = list(self.perception_space.get_group_names())
        if len(names) > 0 and "UNGROUPED" not in names and len(self.robots) > 0:
            self.enable_ui_element(self.analyze_btn, text_color=ui.color("#00FF00"))

    def _add_voxel_group(self):
        """Add a new XForm group for voxels"""
        # Create a new XForm under the perception volume
        stage = get_current_stage()
        parent_path = "/World/GO4R_PerceptionVolume"
        
        # Create a new XForm voxel group
        xform_path = f"{parent_path}/VoxelGroup"
        id = 0
        while stage.GetPrimAtPath(xform_path):
            # Increment the group ID until we find a free one
            xform_path = f"{parent_path}/VoxelGroup_{id}"
            id += 1
        
        # Create the new XForm
        xform_prim = stage.DefinePrim(xform_path, "Xform")
        xform_prim.SetCustomDataByKey("perception_weight", 1.0)
        
        # Update groups from stage - this will pick up the new XForm
        self._update_voxel_groups_from_stage()
        
        self._log_message(f"Added new voxel group at {xform_path}")

    def _browse_export_path(self):
        """Open a file dialog to select export location"""
        import omni.kit.widget.filebrowser as fb
        
        # Create reference to persist the window
        self.file_browser_window = ui.Window("Select Export Location", width=800, height=0)
        
        def on_file_picked():
            self.export_path_field.model.set_value(self.selected_export_path.path)
            type_str = "Directory" if self.selected_export_path._is_folder else "File"
            if type_str == "Directory":
                self.export_path_field.set_style({"color": ui.color("#ff0000")})
                self.enable_ui_element(create_file_button, text_color=ui.color("#00FF00"))
            elif type_str == "File":
                self.export_path_field.style = {"color": ui.color("#00FF00")}
            else:
                self.export_path_field.style = {"color": ui.color("#00eeff")}
                
            # Enable export buttons when path is selected
            self.enable_ui_element(self.export_robot_btn, text_color=ui.color("#00FF00"))
            self.enable_ui_element(self.export_sensors_btn, text_color=ui.color("#00FF00"))
                
            self._log_message(f"Export path set to: {self.selected_export_path.path}")

        def on_browser_selection_changed(n, paths):
            if paths and len(paths) > 0:
                self.selected_export_path = paths[0]
                # Check if it's a directory
                type_str = "Directory" if paths[0]._is_folder else "File"
                if type_str == "Directory":
                    file_path_field.model.set_value(f"{self.selected_export_path.path}")
                    file_path_field.set_style({"color": ui.color("#00FF00")})
                    self.enable_ui_element(create_file_button, text_color=ui.color("#00FF00"))
                elif type_str == "File":
                    file_path = paths[0].path
                    file_name = file_path.split('/')[-1] if '/' in file_path else file_path
                    file_name_base = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                    extension = file_name.rsplit('.', 1)[1] if '.' in file_name else ""
                    
                    # Set the path field to the directory
                    directory = file_path[:-len(file_name)] if file_name in file_path else file_path
                    file_path_field.model.set_value(directory)
                    
                    # Update file name field with the selected file's name
                    file_name_field.model.set_value(file_name_base)
                    
                    # Update extension selection if applicable
                    if extension == "csv":
                        export_type_model.model.set_value(0)
                    elif extension == "xlsx":
                        export_type_model.model.set_value(1)  
                    elif extension == "npy":
                        export_type_model.model.set_value(2)
                    file_path_field.set_style({"color": ui.color("#00FF00")})

                    # Initially disable the create file button
                    self.disable_ui_element(create_file_button)
                on_file_picked()
            else:
                file_path_field.model.set_value("(No selection)")

        def _create_export_file():
            """Create an export file at the selected path"""
            path = self.selected_export_path.path if self.selected_export_path else None
            if len(path) == 0:
                self._log_message("Error: No export path selected")
                return
            
            type_str = "Directory" if self.selected_export_path._is_folder else "File"
            
            # If it's a directory, update the file name field with current path
            if type_str != "Directory":
                self._log_message("Error: Selected path is not a directory")
                return

            file_path = self.selected_export_path.path + "/" + file_name_field.model.get_value_as_string() + ['.csv','.xlsx','.npy'][export_type_model.model.get_value_as_int()]
            # Try to create the file or verify it can be written to
            try:
                # Just try opening the file to see if we can write to it
                with open(file_path, 'w') as f:
                    # Just create the file, we'll write data to it later
                    pass
                
                self._log_message(f"Successfully created file: {file_path}")
                
                # Update UI to show the selected path
                self.file_browser_window.visible = False
            except Exception as e:
                self._log_message(f"Error creating file: {str(e)}")
                return
            
            self.selected_export_path = fb.Path(file_path)
            self.export_path_label.text = f"Save to: {self.selected_export_path.path}"
            self.export_path_label.style = {"color": ui.color("#00FF00")}
            
            # Enable export buttons when path is selected
            self.enable_ui_element(self.export_robot_btn, text_color=ui.color("#00FF00"))
            self.enable_ui_element(self.export_sensors_btn, text_color=ui.color("#00FF00"))
            
            self._log_message(f"Export file created at: {file_path}")
            on_file_picked()
        
        with self.file_browser_window.frame:
            with ui.VStack(spacing=5):
                # Create the file browser widget
                browser = fb.FileBrowserWidget(
                    "Select File Location",
                    allow_multi_selection=False,
                    show_grid_view=True,
                    filter_fn=lambda item: item is not None and (item._is_folder or 
                                        # item.path.endswith(".csv") or 
                                        # item.path.endswith(".xlsx") or
                                        item.path.endswith(".npy"))
                    
                )
                
                # Use a file system model for local files
                model = fb.FileSystemModel("Local", "/home")
                browser.add_model_as_subtree(model)
                browser._selection_changed_fn = on_browser_selection_changed

                # Add text field to show selectionf
                with ui.HStack(spacing=5, height=30):
                    ui.Label("File Path:", width=0, height=30)
                    file_path_field = ui.StringField(height=30)
                    file_path_field.model.set_value("(No selection)")
                    self.disable_ui_element(file_path_field, text_color=ui.color("#FF0000"))
                    ui.Label("/", width=0, height=30)
                    file_name_field = ui.StringField(height=30, width=100)
                    self.enable_ui_element(file_name_field, text_color=ui.color("#00FF00"))
                    file_name_field.model.set_value("sensor_data")
                    with ui.HStack(spacing=5, height=30):
                        # Add the file extension selector with radio buttons
                        with ui.HStack(spacing=0, height=30):
                            export_type_model = ui.RadioCollection()
                            self.disable_ui_element(ui.RadioButton(radio_collection=export_type_model, text=".csv", width=45, height=0, value=0))
                            self.disable_ui_element(ui.RadioButton(radio_collection=export_type_model, text=".xlsx", width=45, height=0, value=1))
                            self.enable_ui_element(ui.RadioButton(radio_collection=export_type_model, text=".npy", width=45, height=0, value=2), text_color=ui.color("#00FF00"))

                            export_type_model.model.set_value(2)
                            
                            def on_extension_type_changed(ext_type):
                                file_name_path = file_name_field.model.get_value_as_string()
                                if file_name_path:
                                    extension = [".csv", ".xlsx", ".npy"][ext_type.as_int]
                                    # Strip any existing extension and add the new one
                                    base_name = file_name_path.rsplit('.', 1)[0] if '.' in file_name_path.split('/')[-1] else file_name_path
                                    new_path = f"{base_name}{extension}"
                                    file_name_field.model.set_value(new_path)
                            
                            export_type_model.model.add_value_changed_fn(on_extension_type_changed)
                    create_file_button = ui.Button("Create File", clicked_fn=_create_export_file, height=30, width=0, ooltip="Create the file at the location. Select a valid directory to enable this button.")
                    self.disable_ui_element(create_file_button)
        
        # Make the window visible
        self.file_browser_window.visible = True

    def _export_robot_data(self):
        """Export robot data to the selected file path"""
        if not self.selected_export_path:
            self._log_message("Error: No export path selected")
            return
        
        self._log_message(f"Exporting robot data to: {self.selected_export_path.path}")
        # Implement your robot data export logic here

    def _export_sensor_data(self):
        """Export sensor data to the selected file path"""
        if not self.selected_export_path:
            self._log_message("Error: No export path selected")
            return
        
        self._log_message(f"Exporting sensor data to: {self.selected_export_path.path}")
        # Implement your sensor data export logic here
    
    def _reset_settings(self):
        """Reset all settings to default values"""
        
        # Reset perception mesh settings
        self.perception_mesh = None
        self.perception_mesh_path = None
        self.perception_mesh_label.text = "(Not selected)"
        
        # Disable analyze button when mesh is reset
        self.disable_ui_element(self.analyze_btn, text_color=ui.color("#FF0000"))
        
        # Reset sampling step size
        self.voxel_size = 5.0
        self.voxel_size_field.model.set_value(self.voxel_size)
        
        self._log_message("Settings reset to default values")
    
    def _refresh_sensor_list(self):
        """Refresh the list of detected sensors without analysis"""

        def _remove_trailing_digits(name):
            """Remove trailing underscore+digits from a name, if they are present, using re"""
            return re.sub(r'(_\d+)?$', '', name)

        def _find_camera(prim:Usd.Prim) -> Sensor3D_Instance:
            """Find cameras that are descendants of the selected robot"""

            # self._log_message(f"DEBUG: Checking for CAMERA prim {prim.GetName()} of type {prim.GetTypeName()}")

            if prim.IsA(UsdGeom.Camera):
                # Skip editor cameras if a specific robot is selected
                name = prim.GetName()
                
                # Load the camera information into a MonoCamera3D
                cam_prim = UsdGeom.Camera(prim)

                # Get the aspect ratio
                aspect_ratio = self._get_prim_attribute(prim, "aspectRatio", None)
                if aspect_ratio is None:
                    v_aperture = cam_prim.GetVerticalApertureAttr().Get()
                    h_aperture = cam_prim.GetHorizontalApertureAttr().Get()
                    if v_aperture is not None and h_aperture is not None:
                        aspect_ratio = h_aperture / v_aperture
                if aspect_ratio is None:
                    self._log_message(f"Warning: Could not find aspect ratio for camera {prim.GetName()}, defaulting to 4:3")
                    aspect_ratio = 4.0/3.0

                # Get resolution more robustly - try different attribute names or patterns
                resolution = None
                # Try standard resolution attribute
                if prim.HasAttribute("resolution"):
                    resolution = prim.GetAttribute("resolution").Get()
                # Try common alternatives
                elif prim.HasAttribute("horizontalImageSize") and prim.HasAttribute("verticalImageSize"):
                    h_size = prim.GetAttribute("horizontalImageSize").Get()
                    v_size = prim.GetAttribute("verticalImageSize").Get()
                    if h_size is not None and v_size is not None:
                        resolution = (h_size, v_size)
                # Try isaac-specific attributes
                elif prim.HasAttribute("sensorWidth") and prim.HasAttribute("sensorHeight"):
                    resolution = (prim.GetAttribute("sensorWidth").Get(), 
                                prim.GetAttribute("sensorHeight").Get())
                
                # If resolution still not found, try to infer from other parameters
                if resolution is None:
                    # Log that we couldn't find resolution directly
                    self._log_message(f"Warning: Could not find resolution for camera {prim.GetName()}, defaulting to 720p")
                    # Default to HD resolution as fallback
                    resolution = (720, 720*aspect_ratio)

                try:
                    cam3d = MonoCamera3D(name=_remove_trailing_digits(name),
                                        focal_length=cam_prim.GetFocalLengthAttr().Get(),
                                        h_aperture=cam_prim.GetHorizontalApertureAttr().Get(),
                                        v_aperture=cam_prim.GetVerticalApertureAttr().Get(),
                                        aspect_ratio=aspect_ratio,
                                        h_res=resolution[0] if resolution else None,
                                        v_res=resolution[1] if resolution else None,
                                        body=_find_sensor_body(prim),
                                        cost=1.0,
                                        focal_point=(0, 0, 0)
                                        )
                    cam3d_instance = Sensor3D_Instance(cam3d, 
                                                       path=prim.GetPath(), 
                                                       name=_remove_trailing_digits(name), 
                                                       tf=self._get_robot_to_sensor_transform(prim, robot_prim),
                                                       usd_context=self._usd_context)
                    # self._log_message(f"Found camera: {cam3d_instance.name} with HFOV: {cam3d_instance.sensor.h_fov:.2f}°")
                except Exception as e:
                    self._log_message(f"Error extracting camera properties for {name}: {str(e)}")
                    raise e
                    
                return cam3d_instance
            
            else:
                return None

        def _find_lidar(prim:Usd.Prim) -> Sensor3D_Instance:
            """Find LiDARs that are descendants of the selected robot"""

            # self._log_message(f"DEBUG: Checking for LiDAR prim {prim.GetName()} of type {prim.GetTypeName()}")

            type_name = str(prim.GetTypeName()).lower()
            lidar_instance = None
            
            if "lidar" in type_name or "range" in type_name:
                name = prim.GetName()
                if "go4r_raycaster" in name.lower():
                    # Skip if this is a raycaster created by the extension
                    return None
                if name.lower() == "lidar":
                    # Get the name from the parent
                    name = prim.GetParent().GetName()
                # Direct LiDAR prim - extract properties
                try:
                    lidar = Lidar3D(name=_remove_trailing_digits(name),
                                    h_fov=self._get_prim_attribute(prim, "horizontalFov"),
                                    v_fov=self._get_prim_attribute(prim, "verticalFov"),
                                    h_res=self._get_prim_attribute(prim, "horizontalResolution"),
                                    v_res=self._get_prim_attribute(prim, "verticalResolution"),
                                    max_range=self._get_prim_attribute(prim, "maxRange"),
                                    min_range=self._get_prim_attribute(prim, "minRange"),
                                    body=_find_sensor_body(prim),
                                    cost=1.0,
                                    )
                    lidar_instance = Sensor3D_Instance(lidar, 
                                                       path=prim.GetPath(), 
                                                       name=_remove_trailing_digits(name), 
                                                       tf=self._get_robot_to_sensor_transform(prim, robot_prim),
                                                       usd_context=self._usd_context)
                    # self._log_message(f"Found LiDAR: {lidar_instance.name} with HFOV: {lidar_instance.sensor.h_fov:.2f}°")
                except Exception as e:
                    self._log_message(f"Error extracting LiDAR properties for {name}: {str(e)}")
            
            return lidar_instance
        

        def _find_sensor_body(prim:Usd.Prim) -> Usd.Prim:
            """Find the body of the sensor, which is the first mesh encountered in the local tree ancestry of the given prim"""
            def _recurse_find_sensor_body(prim):
                
                # First check at the most likely place
                prim_path = prim.GetPath()
                parent_path = prim_path.GetParentPath()
                likely_body_path = str(parent_path) + "GO4R_BODY"
                likely_body = get_current_stage().GetPrimAtPath(likely_body_path)
                if likely_body and likely_body.IsA(UsdGeom.Mesh):
                    return likely_body

                # Traverse the parent tree, looking for a mesh.
                # If no mesh in the children of the parent, go up a level and check again.
                # Do this recursively until we find meshes.

                if prim.IsA(UsdGeom.Mesh):
                    return prim
                else:
                    # Check if this is a mesh
                    for child in prim.GetChildren():
                        if child.IsA(UsdGeom.Mesh):
                            return child
                    # If not, check the parent
                    parent = prim.GetParent()
                    if parent:
                        return _recurse_find_sensor_body(parent)
                
                # If no mesh found, return None
                return None

            
            body = _recurse_find_sensor_body(prim)
            if body is None:
                self._log_message(f"Warning: No body found for sensor {prim.GetPath()}")
            return body



        def _assign_sensors_to_robot(prim, bot, processed_camera_paths=None):
            """Search a level for sensors and add them to the specified robot"""
            # Check if this node contains a sensor, regardless of whether it's a leaf
            # This allows finding sensors at any level of the hierarchy
            
            # Check for LiDAR first
            lidar = _find_lidar(prim)
            if lidar is not None:
                self._log_message(f"Adding LiDAR: {lidar.name} to robot {bot.name}")
                bot.sensors.append(lidar)
                # Don't return, continue searching in case there are other sensors

            # Check for camera
            camera = _find_camera(prim)
            if camera is not None:
                self._log_message(f"Adding camera: {camera.name} to robot {bot.name}")
                
                # Handle stereo camera pairing
                found_stereo_pair = False
                for role in ["left", "right"]:
                    if (role in camera.name.lower() or 
                        (prim.HasAttribute("stereoRole") and 
                         prim.GetAttribute("stereoRole").Get() and 
                         prim.GetAttribute("stereoRole").Get().lower() == role)):
                        this_cam_role = role
                        this_cam_name = camera.name
                        this_cam_path_str = prim.GetPath().pathString
                        other_cam_role = "left" if role == "right" else "right"
                        
                        # Case-insensitive replacement for the other camera name
                        pattern = re.compile(re.escape(role), re.IGNORECASE)
                        other_cam_name = pattern.sub(other_cam_role, camera.name)
                        other_cam_path_str = pattern.sub(other_cam_role, str(this_cam_path_str))

                        # Check if the other camera is already in the sensors list
                        for i, sensor_instance in enumerate(bot.sensors):
                            if isinstance(sensor_instance.sensor, MonoCamera3D):
                                # Case-insensitive comparison
                                if sensor_instance.name.lower() == other_cam_name.lower():
                                    # Found the other camera, create a stereo camera
                                    self._log_message(f"Pairing stereo cameras: {this_cam_name} and {other_cam_name} in robot {bot.name}")
                                    self._log_message(f"  Path 1: {this_cam_path_str}")
                                    self._log_message(f"  Path 2: {sensor_instance.path.pathString}")
                                    # Find the common parent of the two cameras based on the paths
                                    common_parent_path = os.path.commonpath([this_cam_path_str, sensor_instance.path.pathString])
                                    # self._log_message(f"  Common: {common_parent_path}")
                                    common_parent_prim = get_current_stage().GetPrimAtPath(common_parent_path)
                                    if not common_parent_prim:
                                        self._log_message(f"Error: Could not find common parent prim at path {common_parent_path}")
                                        common_parent_name=this_cam_name.replace(this_cam_role, "")
                                    else:
                                        common_parent_name = common_parent_prim.GetName()
                                    
                                    # Remove any training '_XX' suffix (where XX is two+ digits) from the common parent name
                                    if common_parent_name.endswith("_XX"):
                                        common_parent_name = common_parent_name[:-3]

                                    stereo_cam = StereoCamera3D(
                                        name=_remove_trailing_digits(common_parent_name),
                                        sensor1=sensor_instance.sensor if this_cam_role == "left" else camera.sensor,
                                        sensor2=sensor_instance.sensor if this_cam_role == "right" else camera.sensor,
                                        tf_sensor1=sensor_instance.tf if this_cam_role == "left" else camera.tf,
                                        tf_sensor2=sensor_instance.tf if this_cam_role == "right" else camera.tf,
                                        cost=sensor_instance.sensor.cost + camera.sensor.cost,
                                        body=prim
                                    )
                                    stereo_instance = Sensor3D_Instance(stereo_cam, 
                                                                        path=common_parent_path, 
                                                                        name=_remove_trailing_digits(common_parent_name), 
                                                                        tf=camera.tf,
                                                                        usd_context=self._usd_context)
                                    # Move the ray casters from the MonoCamera3D instance to the StereoCamera3D instance
                                    stereo_instance.ray_casters = sensor_instance.ray_casters + camera.ray_casters
                                    bot.sensors[i] = stereo_instance  # Replace the mono camera with stereo
                                    found_stereo_pair = True
                                    break
                        break  # Found a role, no need to check other roles
                        
                # Add camera only if it wasn't part of a stereo pair
                if not found_stereo_pair:
                    self._log_message(f"Adding stereo camera: {camera.name} to robot {bot.name}")
                    bot.sensors.append(camera)
            
            # Always continue recursively even if sensors were found at this level
            for child in prim.GetChildren():
                _assign_sensors_to_robot(child, bot, processed_camera_paths)

        self._log_message("Refreshing robots & sensors list...")

        stage = get_current_stage()

        self.robots = []
        
        total_sensors = dict.fromkeys(sensor_types, 0)
        for bot_prim in self.selected_prims:
            bot = Bot3D(bot_prim.GetName(), path=bot_prim.GetPath(), usd_context=self._usd_context)
            self.robots.append(bot)
            # Clear existing sensors before searching again
            bot.sensors = []
            
            # Find robot prim
            robot_prim = stage.GetPrimAtPath(bot.path)
            if not robot_prim:
                self._log_message(f"Error: Could not find prim at path {bot.path}")
                continue
                
            # Search for sensors in this robot (with a new empty processed_camera_paths set)
            _assign_sensors_to_robot(robot_prim, bot)
            
            found_sensors = {}
            for type in sensor_types:
                found_sensors[type.__name__] = bot.get_sensors_by_type(type)
                total_sensors[type] += len(found_sensors[type.__name__])
                self._log_message(f"Found {len(found_sensors[type.__name__])} {type.__name__} sensors for robot {bot.name}")
            
        self._log_message(f"Total sensors found: " + ', '.join([f"{total_sensors[type]} {type.__name__}(s)" for type in sensor_types]))
        
        # Update the UI
        self._update_sensor_list_ui()
        self._update_voxel_groups_ui()


    def _display_sensor_instance_properties(self, sensor_instance:Sensor3D_Instance):
        for attr, value in sensor_instance.__dict__.items():
            if "sensor" in attr:
                with ui.CollapsableFrame(attr, height=0, collapsed=True):
                    with ui.VStack(spacing=2):
                        self._display_sensor_instance_properties(value)
            elif "tf" in attr:
                with ui.CollapsableFrame(attr, height=0, collapsed=True):
                    with ui.VStack(spacing=2):
                        ui.Label(f"position: {value[0]}")
                        ui.Label(f"rotation: {value[1]}")
            elif "ap_constants" in attr:
                continue # Skip average precision attributes
            elif "ray_casters" in attr:
                with ui.CollapsableFrame(attr, height=0, collapsed=True):
                    with ui.VStack(spacing=2):
                        for i, rc in enumerate(value):
                            ui.Label(f"rc{i+1} at {prim_utils.get_prim_path(rc)}")
            else:
                ui.Label(f"{attr}: {value}")

    
    def _update_sensor_list_ui(self):
        """Update the sensor list UI with the detected sensors for all robots"""
        # Clear the current sensor list
        self.sensor_list.clear()
        
        with self.sensor_list:
            with ui.VStack(spacing=5):
                if not self.robots:
                    ui.Label("No robots selected")
                else:
                    # For each robot, create a collapsible frame
                    for robot in self.robots:
                            
                        # Create collapsible frame for this robot with blue border
                        with ui.CollapsableFrame(
                            f"Robot: {robot.name}", 
                            height=0, 
                            style={"border_width": 2, "border_color": ui.color("#0059ff")}, 
                            collapsed=False
                        ):
                            with ui.VStack(spacing=5):
                                for sensor_type in sensor_types:
                                    sensors = robot.get_sensors_by_type(sensor_type)
                                    if sensors:
                                        with ui.CollapsableFrame(f"{sensor_type.__name__}s: {len(sensors)}", height=0, style={"border_color": ui.color("#00c3ff")}, collapsed=False):
                                            with ui.VStack(spacing=5):
                                                for idx, sensor_instance in enumerate(sensors):
                                                    # Set default average precision if not set
                                                    if not hasattr(sensor_instance.sensor, 'ap_constants') or sensor_instance.sensor.ap_constants is None:
                                                        ap_constants = {'a': default_sensor_aps[sensor_instance.sensor.__class__.__name__]["a"],
                                                                        'b': default_sensor_aps[sensor_instance.sensor.__class__.__name__]["b"]}
                                                        sensor_instance.sensor.ap_constants = ap_constants
                                                        
                                                    with ui.CollapsableFrame(f"{idx+1}. {sensor_instance.name}", height=0, style={"border_color": ui.color("#FFFFFF")}, collapsed=True):
                                                        with ui.VStack(spacing=2):
                                                            # Add average precision input directly at the top level
                                                            with ui.HStack(spacing=5):
                                                                ui.Label("Average Precision:   a:", width=120)
                                                                a_field = ui.FloatField(width=80)
                                                                a_field.model.set_value(sensor_instance.sensor.ap_constants['a'])
                                                                
                                                                def on_a_val_changed(new_value, sensor=sensor_instance):
                                                                    a = max(0.0, min(1.0, new_value.get_value_as_float()))
                                                                    sensor.sensor.ap_constants['a'] = a
                                                                    self._log_message(f"Set {sensor.name} average precision to {a:.2f}")
                                                                
                                                                a_field.model.add_value_changed_fn(on_a_val_changed)

                                                                ui.Label("   b:", width=0)
                                                                b_field = ui.FloatField(width=80)
                                                                b_field.model.set_value(sensor_instance.sensor.ap_constants['b'])

                                                                def on_b_val_changed(new_value, sensor=sensor_instance):
                                                                    b = max(0.0, min(1.0, new_value.get_value_as_float()))
                                                                    sensor.sensor.ap_constants['b'] = b
                                                                    self._log_message(f"Set {sensor.name} average precision to {b:.2f}")
                                                                
                                                                b_field.model.add_value_changed_fn(on_b_val_changed)
                                                            
                                                            # Show other properties in collapsible section
                                                            with ui.CollapsableFrame("Properties", height=0, collapsed=True):
                                                                with ui.VStack(spacing=2):
                                                                    self._display_sensor_instance_properties(sensor_instance)

    def _batch_calc_perception_entropies(self):
        """Batch calculate perception entropies for all robots and sensors"""
        self._log_message("Batch calculating perception entropies...")
        
        results = {}
        for robot in self.robots:
            # Call the perception entropy calculation for each robot
            results[robot.name] = robot.calculate_perception_entropy(self.perception_space)

        print("Batch calculation results:")
        print(results)

        # Update the UI with the results
        self._update_percep_entr_results_ui()
        
        # Re-enable the analyze button
        self.enable_ui_element(self.analyze_btn, text_color=ui.color("#00FF00"))

        self._log_message("Batch calculation complete.")


    def _calc_perception_entropies(self, disable_raycasters=True):
        """Main function to analyze all sensors on all the robots.
        
        Must have a perception mesh voxelized, and at least one robot analyzed"""
        self._log_message("Disabling gravity...")
        self._zero_gravity()

        self._log_message("Starting perception entropy analysis...")
        self.analysis_progress_bar.model.set_value(0.0) # Reset progress bar

        # Check if we have robots to analyze, throw an error if not
        if not self.robots:
            self._log_message("Error: No robots selected for analysis.")
            self.analysis_progress_bar.model.set_value(0.0)
            return
        
        vg_names = list(self.perception_space.get_group_names())
        # Check if we have voxel groups, throw an error if not
        if len(vg_names) == 0:
            self._log_message("Error: No voxel groups found. Please create a perception mesh first.")
            self.analysis_progress_bar.model.set_value(0.0)
            return
        
        # Check if there are ungrouped voxels, warn the user
        if "UNGROUPED" in vg_names:
            self._log_message("Warning: There are ungrouped voxels that will not be considered!")

        percep_entr_results_data = {}
        percep_entr_results_data.update({robot.name: {"total": 0.0} for robot in self.robots})

        # Calculate total number of sensors for progress tracking
        total_sensors_to_process = sum(len(robot.sensors) for robot in self.robots)
        if total_sensors_to_process == 0:
            self._log_message("Warning: No sensors found on selected robots.")
            self.analysis_progress_bar.model.set_value(1.0) # Indicate completion (of nothing)
            return
            
        self.processed_sensors_count = 0

        async def update_progress():
            self.processed_sensors_count += 1
            progress = self.processed_sensors_count / total_sensors_to_process
            self.analysis_progress_bar.model.set_value(progress)
            # Force UI update
            await omni.kit.app.get_app().next_update_async()

        if disable_raycasters:
            for robot in self.robots:
                for sensor in robot.sensors:
                    if isinstance(sensor.sensor, Sensor3D) and hasattr(sensor, 'ray_casters') and sensor.ray_casters:
                        try:
                            omni.kit.commands.execute('ToggleActivePrims',
                                    prim_paths=[prim_utils.get_prim_path(rc) for rc in sensor.ray_casters if rc], # Check if rc is valid
                                    active=False,
                                    stage_or_context=self.stage)
                            self._log_message(f"Disabled raycaster(s) for sensor {sensor.name}")
                        except Exception as e:
                             self._log_message(f"Warning: Could not disable raycaster for {sensor.name}: {e}")


        async def run_all_analyses():
            tasks = []
            for robot in self.robots:
                # Pass the update_progress async function as a callback
                tasks.append(self._calc_perception_entropy(robot, update_progress_callback=update_progress, disable_raycasters=disable_raycasters))
            
            # Wait for all robot analyses to complete
            await asyncio.gather(*tasks)
            
            # Ensure progress bar reaches 100% at the end
            self.analysis_progress_bar.model.set_value(1.0)
            self._log_message("Analysis complete")
            # Update the results UI (consider making this async too if needed)
            self._update_percep_entr_results_ui()

        # Run the analyses asynchronously
        asyncio.ensure_future(run_all_analyses())

    async def _calc_perception_entropy(self, robot:Bot3D, update_progress_callback=None, disable_raycasters=True):
        """ Caluclate the perception entropy for a single robot."""

        self._log_message(f"Calculating perception entropy for robot {robot.name}...")

        # Add the robot to the results data dict. Results will be tracked at different levels and the UI updated at the end
        self.percep_entr_results_data.update({robot.name: {"Total Entropy": 0.0, 
                                                           "Entropy By Sensor Type": {}, 
                                                           "Entropy By Voxel Group": {}}})
        
        sensor_voxel_m = dict.fromkeys([s.name for s in robot.sensors], None)

        for sensor_instance in robot.sensors:
            # Pass the progress bar update callback to the measurement function
            sensor_voxel_m[sensor_instance.name] = await self._get_measurements_raycast(sensor_instance, disable_raycaster=disable_raycasters)
            # Call the progress update callback after measurements for this sensor are done
            if update_progress_callback:
                await update_progress_callback()
        
        vg_m_early_fusion_per_type = {}
        # Get the measurements for each sensor type on each voxel and apply early fusion
        unique_sensor_names = set([sensor.name for sensor in robot.sensors])
        for name in unique_sensor_names:
            measurements_for_sensor = {name: sensor_voxel_m[name] for name in unique_sensor_names if name in sensor_voxel_m and sensor_voxel_m[name] is not None}
            if measurements_for_sensor: # Check if dict is not empty
                vg_m_early_fusion_per_type.update({name: self._apply_early_fusion(measurements_for_sensor)})
        
        sum_normalized_voxel_entropy = 0.0
        for voxel_group in list(self.perception_space.get_voxel_group_names()): # TODO Parallelize this as much as possible
            if voxel_group == "UNGROUPED":
                self._log_message(f"Warning: Skipping {voxel_group} voxels")
                continue
            vg_measurements = vg_m_early_fusion_per_type[sensor_instance.sensor.name][voxel_group]
            
            vg_sensor_voxel_ap = dict.fromkeys(unique_sensor_names, None)
            vg_sensor_voxel_ap_clipped = dict.fromkeys(unique_sensor_names, None)
            vg_sensor_voxel_uncertainty = dict.fromkeys(unique_sensor_names, None)
            # For each unique sensor, calculate the average precision (AP) and uncertainty (σ) for each voxel
            for sensor_name in unique_sensor_names:
                sensor_instances = robot.get_sensors_by_name(sensor_name)
                if len(sensor_instances) == 0:
                    self._log_message(f"Warning: No sensors found with name {sensor_name}")
                    continue
                elif len(sensor_instances) > 1:
                    self._log_message(f"Warning: Multiple sensors found with name {sensor_name}, using the first one")
                sensor_instance = sensor_instances[0]

                # Calculate the sensor AP for each voxel, where AP = a ln(m) + b
                a = sensor_instance.sensor.ap_constants['a']
                b = sensor_instance.sensor.ap_constants['b']
                vg_sensor_voxel_ap[sensor_instance.sensor.name] = a * np.log(vg_measurements) + b

                # Clip the AP values to avoid log(0) or log(1)
                vg_sensor_voxel_ap_clipped[sensor_instance.sensor.name] = np.clip(vg_sensor_voxel_ap[sensor_instance.sensor.name], 0.0001, 0.9999)

                # Calculate the sensor uncertainty for each voxel, where σ = 1/AP - 1
                vg_sensor_voxel_uncertainty[sensor_instance.sensor.name] = 1.0 / vg_sensor_voxel_ap_clipped[sensor_instance.sensor.name] - 1.0

                # Calulate the total entropy for the sensor type
                vg_sensor_total_entropy = self._calc_total_entropy(vg_sensor_voxel_uncertainty[sensor_instance.sensor.name])
                self.percep_entr_results_data[robot.name]["Entropy By Sensor Type"].update( {sensor_name: np.sum(vg_sensor_total_entropy)} )
        
            # Apply per-voxel late fusion to get the total entropy where σ_fused = sqrt(1 / Σ(1/σ_i²))
            vg_voxel_uncertainty_late_fusion = self._apply_late_fusion(vg_sensor_voxel_uncertainty)
            
            # Calculate the per-voxel entropy H(S|m,q) = 2ln(σ) + 1 + ln(2pi)
            vg_voxel_entropy = self._calc_total_entropy(vg_voxel_uncertainty_late_fusion)
            self.percep_entr_results_data[robot.name]["Entropy By Voxel Group"].update( {voxel_group: np.sum(vg_voxel_entropy)} )

            # Calculate the total perception entropy, which is simply the weighted average of the voxel entropies
            vg_normalized_weight = self.weighted_voxels[voxel_group][1]/sum([w for vg, (v,w) in self.weighted_voxels.items() if vg != "UNGROUPED"])
            vg_normalized_voxel_entropy = vg_normalized_weight * vg_voxel_entropy
            sum_normalized_voxel_entropy += np.sum(vg_normalized_voxel_entropy)
        
        total_perception_entropy = sum_normalized_voxel_entropy / sum([len(v) for vg, (v,w) in self.weighted_voxels.items() if vg != "UNGROUPED"])
        self.percep_entr_results_data[robot.name]["Total Entropy"] = total_perception_entropy
        self._log_message(f"Total perception entropy for robot {robot.name}: {total_perception_entropy:.4f}")
        return total_perception_entropy

    def _calc_total_entropy(self, uncertainties:np.ndarray, normalize=True) -> float:
        """Calculate the total entropy from the uncertainties
        H(S|m,q) = 2ln(σ) + 1 + ln(2pi)"""
        entropy = 2 * np.log(uncertainties) + 1 + np.log(2*np.pi)
        if normalize:
            # Normalize the entropy to be between 0 and 1
            entropy = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))
        return entropy

        
    def _update_percep_entr_results_ui(self):
        """Update the results UI with entropy results for each robot"""
        # Clear existing results
        self.results_list.clear()
        
        with self.results_list:
            with ui.VStack(spacing=5):
                if not self.robots:
                    ui.Label("No robots selected")
                elif not self.percep_entr_results_data:
                    ui.Label("Run analysis to see results")
                else:
                    # For each robot, create a collapsible frame with its results
                    for robot_name, robot_results in self.percep_entr_results_data.items():
                        
                        with ui.CollapsableFrame(
                            f"Robot: {robot_name}", 
                            height=0,
                            style={"border_width": 2, "border_color": ui.color("#0059ff")},
                            collapsed=False
                        ):
                            with ui.VStack(spacing=5):
                                
                                for result_type, result_value in robot_results.items():
                                    if isinstance(result_value, dict):
                                        with ui.CollapsableFrame(
                                            result_type, 
                                            height=0,
                                            style={"border_color": ui.color("#00c3ff")},
                                            collapsed=False
                                        ):
                                            with ui.VStack(spacing=2):
                                                for key, entropy in result_value.items():
                                                    ui.Label(f"{key}: {entropy:.4f}")
                                    else:
                                        ui.Label(f"{result_type}: {result_value:.4f}")


    def _get_prim_attribute(self, prim, attr_name, default_value=None):
        """Get the value of a prim attribute or return a default value"""
        if not prim:
            self._log_message(f"Error: No prim provided for attribute extraction. Returning default value {default_value}.")
            return default_value
        if not attr_name:
            self._log_message(f"No such attribute '{attr_name}'. Returning default value {default_value}.")
            return default_value
        
        try:
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid():
                return attr.Get()
        except Exception as e:
            self._log_message(f"Error getting attribute {attr_name}: {str(e)}")
        
        return default_value
    
    def _get_world_transform(self, prim) -> Tuple[Gf.Vec3d, Gf.Rotation]:
        """Get the world transform (position and rotation) of a prim"""
        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Extract position and rotation
        position = Gf.Vec3d(world_transform.ExtractTranslation())
        rotation = world_transform.ExtractRotationMatrix()
        
        return position, rotation
    
    def _get_robot_to_sensor_transform(self, robot_prim, sensor_prim) -> Tuple[Gf.Vec3d, Gf.Rotation]:
        """Get the transform from the robot to the sensor"""
        xform = UsdGeom.Xformable(sensor_prim)
        sensor_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Get the world transform of the robot
        robot_position, robot_rotation = self._get_world_transform(robot_prim)
        
        # Calculate the transform from the robot to the sensor
        sensor_position = Gf.Vec3d(sensor_transform.ExtractTranslation())
        sensor_rotation = sensor_transform.ExtractRotationMatrix()
        
        # Calculate the relative transform from robot to sensor
        relative_position = sensor_position - robot_position
        relative_rotation = sensor_rotation * robot_rotation.GetInverse()
        
        return relative_position, relative_rotation
    
    def _on_voxelize_button_clicked(self):
        """Wrapper for the async function"""
        async def _initialize_and_voxelize():

            # Wait for stage to stabilize
            await omni.kit.app.get_app().next_update_async()    

            self.timeline.play()
            for _ in range(3):  # Wait for a few frames to ensure physics is ready
                await omni.kit.app.get_app().next_update_async()
            self.timeline.pause()

            voxels = await self._voxelize_perception_mesh()
            if voxels:
                self._update_voxel_groups_ui()  # Update UI to show voxels in first group
            return voxels
        
        asyncio.ensure_future(_initialize_and_voxelize())

    async def _voxelize_perception_mesh(self):
        """Select a mesh to use as the target perception area"""
        self.voxelize_progress_bar.model.set_value(0.0)  # Reset progress bar
        # Get current selection
        selection = self._usd_context.get_selection().get_selected_prim_paths()
        
        if not selection:
            self._log_message("No mesh selected. Please select a mesh prim in the stage.")
            self.perception_mesh_label.text = "(Not selected)"
            self.perception_mesh = None
            self.perception_mesh_path = None
            return
        
        # Use the Xform object at the top level called "PerceptionVolume" as the parent for all the voxels. If it doesn't exist, create it.
        stage = get_current_stage()
        xform_path = "/World/GO4R_PerceptionVolume"
        xform_prim = stage.GetPrimAtPath(xform_path)
        if not xform_prim:
            xform_prim = UsdGeom.Xform.Define(stage, xform_path)

        # Use the first selected item
        mesh_path = selection[0]
        stage = get_current_stage()
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        
        if not mesh_prim:
            self._log_message(f"Error: Could not find prim at path {mesh_path}")
            return
        
        # Check if it's a mesh or has a bounding box we can use
        if not (mesh_prim.IsA(UsdGeom.Mesh) or mesh_prim.IsA(UsdGeom.Boundable)):
            self._log_message(f"Selected prim {mesh_path} is not a mesh or boundable object")
            return
        
        # Try to apply the collision API to the selected mesh
        if not UsdPhysics.CollisionAPI.CanApply(mesh_prim):
            self._log_message(f"Error: Cannot apply CollisionAPI to {mesh_path}")
            return
        
        # Create an XFrom with the name of the mesh under the perception volume to add the voxels to
        mesh_name = mesh_prim.GetName()
        voxel_grp_xform_path = f"{xform_path}/{mesh_name}"
        i=0
        while stage.GetPrimAtPath(voxel_grp_xform_path):
            voxel_grp_xform_path = f"{xform_path}/{mesh_name}_{i}"
            i += 1
        voxel_grp_xform_prim = UsdGeom.Xform.Define(stage, voxel_grp_xform_path)
        
        voxels = await self.voxelize_mesh(mesh_prim, self.voxel_size, parent_path=voxel_grp_xform_path)
        self.voxelize_progress_bar.model.set_value(1.0)  # Set progress bar to complete

        if voxels:
            self._log_message(f"Created {len(voxels)} voxel meshes inside {mesh_path}")

            omni.kit.commands.execute('ToggleActivePrims',
                stage_or_context=self.stage,
                prim_paths=[mesh_path],
                active=False)

        else:
            self._log_message(f"No voxels created for {mesh_path}")
        
        self._log_message(f"Selected mesh '{mesh_path}' as perception area")

        return voxels

    async def voxelize_mesh(self, mesh_prim, voxel_size, parent_path):
        """
        Split a mesh into voxels of specified size and optionally create primitives for each voxel
        
        Args:
            mesh_prim (Usd.Prim): The source mesh to voxelize
            voxel_size (float): Size of each voxel in meters
            create_primitives (bool): If True, creates actual mesh primitives for each voxel
            parent_path (str): Path where voxel meshes will be created (defaults to mesh's parent)
            
        Returns:
            List of voxel mesh primitives if create_primitives=True, otherwise list of voxel bounds
        """
        
        # Get the mesh geometry
        mesh_geom = UsdGeom.Mesh(mesh_prim)
        mesh_name = mesh_prim.GetName()
        
        # Get the points and face indices
        points = mesh_geom.GetPointsAttr().Get()
        face_vertex_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
        
        if not points or not face_vertex_counts or not face_vertex_indices:
            self._log_message(f"Missing mesh data for {mesh_prim.GetPath()}")
            return []
        
        # Get the mesh transform
        xform = UsdGeom.Xformable(mesh_prim)
        
        # Get the mesh bounding box
        bounds = self._get_mesh_bounds(mesh_prim)
        if not bounds:
            self._log_message("Error: Could not determine mesh bounds.")
            return []
        
        min_point, max_point = bounds
        
        # Calculate grid dimensions
        grid_size_x = int((max_point[0] - min_point[0]) / voxel_size)
        grid_size_y = int((max_point[1] - min_point[1]) / voxel_size)
        grid_size_z = int((max_point[2] - min_point[2]) / voxel_size)
        
        self._log_message(f"Creating voxel grid: {grid_size_x}x{grid_size_y}x{grid_size_z} " + 
                        f"({grid_size_x * grid_size_y * grid_size_z} potential voxels)")

        # Create an array to store all voxel centers for later use
        voxel_centers = []

        # Target the prim mesh for ray casting
        self.target_prims_collision(prim_utils.get_prim_path(mesh_prim))
        await self._ensure_physics_updated(pause=False) # Don't pause the simulation

        # Iterate through the grid to find voxels intersected by triangles
        total_voxels = grid_size_x * grid_size_y * grid_size_z
        processed_voxels = 0

        for i in range(grid_size_x):
            for j in range(grid_size_y):
                for k in range(grid_size_z):
                    # Calculate voxel center
                    center_x = min_point[0] + (i + 0.5) * voxel_size
                    center_y = min_point[1] + (j + 0.5) * voxel_size
                    center_z = min_point[2] + (k + 0.5) * voxel_size
                    voxel_center = carb.Float3(center_x, center_y, center_z)
                    voxel_extent = carb.Float3(voxel_size, voxel_size, voxel_size)
                    
                    # Check if the voxel center is inside the mesh
                    # overlap = self._does_box_overlap_prim(voxel_center, voxel_extent, mesh_prim.GetPath()) #This only generates vozels at the edges of the mesh! Use _is_pont_in_mesh instead
                    center_inside = self._is_point_in_mesh(voxel_center, mesh_prim)
                    if center_inside == True:
                        voxel_centers.append(((i,j,k),voxel_center))
                    processed_voxels += 1
                    self.voxelize_progress_bar.model.set_value(processed_voxels / total_voxels /2) # Update progress bar up to 50%

        # Pause the simulation
        self.timeline.pause()
        
        created_voxels = []
        stage = get_current_stage()

        self.voxelize_progress_bar.model.set_value(0.5)

        min_x = - voxel_size / 2
        min_y = - voxel_size / 2
        min_z = - voxel_size / 2
        max_x = min_x + voxel_size
        max_y = min_y + voxel_size
        max_z = min_z + voxel_size
        
        # Define voxel as a cube
        voxel_points = [
            Gf.Vec3f(min_x, min_y, min_z),
            Gf.Vec3f(max_x, min_y, min_z),
            Gf.Vec3f(max_x, max_y, min_z),
            Gf.Vec3f(min_x, max_y, min_z),
            Gf.Vec3f(min_x, min_y, max_z),
            Gf.Vec3f(max_x, min_y, max_z),
            Gf.Vec3f(max_x, max_y, max_z),
            Gf.Vec3f(min_x, max_y, max_z)
        ]

        # Define the faces (6 faces, each with 4 vertices)
        face_vertex_counts = Vt.IntArray([4, 4, 4, 4, 4, 4])
        face_vertex_indices = Vt.IntArray([
            0, 1, 2, 3,  # bottom
            4, 5, 6, 7,  # top
            0, 1, 5, 4,  # front
            1, 2, 6, 5,  # right
            2, 3, 7, 6,  # back
            3, 0, 4, 7   # left
        ])
        
        # Create a mesh for each occupied voxel
        for (i,j,k), p in voxel_centers:
            
            # Create voxel mesh using USD API directly
            voxel_path = f"{parent_path}/{mesh_name}_voxel_{i}_{j}_{k}"
            mesh_def = UsdGeom.Mesh.Define(stage, voxel_path)
            mesh_def.CreatePointsAttr().Set(voxel_points)
            mesh_def.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_def.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)
            
            # Set the voxel's transform at its center
            xform = UsdGeom.Xformable(mesh_def)
            xform.AddTranslateOp().Set(Gf.Vec3d(p[0], p[1], p[2]))
            
            # Define or get the transparent material
            mat_path = Sdf.Path(f"/World/GO4R_PerceptionVolume/Looks/{mesh_name}_material")
            mat_prim = stage.GetPrimAtPath(mat_path)
            if not mat_prim:
                mat_prim = UsdShade.Material.Define(stage, mat_path)
                shader_path = mat_path.AppendPath("Shader")
                shader = UsdShade.Shader.Define(stage, shader_path)
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.8)) # Bluish tint
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.1) # Set transparency
                mat_prim.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

            # Bind the material to the voxel mesh
            UsdShade.MaterialBindingAPI(mesh_def.GetPrim()).Bind(UsdShade.Material(mat_prim))

            created_voxels.append(mesh_def.GetPrim())
            self.voxelize_progress_bar.model.set_value(0.5 + processed_voxels / total_voxels / 2) # Update progress bar to 100%
        
        return created_voxels

    def _get_mesh_bounds(self, mesh_prim):
        """Get the bounding box of a mesh prim"""
        try:
            if mesh_prim.IsA(UsdGeom.Boundable):
                # Get the bounding box in world space
                bound = UsdGeom.Boundable(mesh_prim).ComputeWorldBound(
                    Usd.TimeCode.Default(), UsdGeom.Tokens.default_
                )
                box = bound.ComputeAlignedBox()
                min_point = box.GetMin()
                max_point = box.GetMax()
                # self._log_message(f"Mesh bounds: {min_point} to {max_point}")
                return (min_point, max_point)
        except Exception as e:
            self._log_message(f"Error getting bounds of mesh: {str(e)}")
        
        return None
    
    def _does_box_overlap_prim(self, origin:carb.Float3, extent:carb.Float3, prim_path) -> bool:
        """Check if a box overlaps with a given prim path"""

        # Target the prim mash
        self.target_prims_collision(prim_path)

        rotation = carb.Float4(1.0, 0.0, 0.0, 0.0) # No rotation

        half_extent = carb.Float3(
            max(0.001, extent[0] / 2.0),
            max(0.001, extent[1] / 2.0), 
            max(0.001, extent[2] / 2.0)
        )

        prim_found = False

        def report_overlap(overlap):
            nonlocal prim_found
            if overlap.collision == prim_path:
                prim_found = True
            return True

        try:
            omni.physx.get_physx_scene_query_interface().overlap_box(half_extent, origin, rotation, report_overlap, False)
        
        except Exception as e:
            self._log_message(f"Error in overlap_box: {str(e)}")
        
        # if prim_found:
        #     self._log_message(f"Box overlaps with prim {prim_path}")
        # else:
        #     self._log_message(f"No overlap found with prim {prim_path}")
    
        return prim_found
    
    def _single_ray_cast_to_mesh(self, 
                                origin:Tuple[float, float, float]=(0.0,0.0,0.0), 
                                direction:Tuple[float, float, float]=(1.0,0.0,0.0), 
                                max_dist: float=100.0,
                                prim_path: str = None):
        """Projects a raycast in the given direction and checks for intersection with the target prim.
        If the ray hits the target prim, it returns the path to the geometry that was hit and the hit distance.
        If there is no target prim, it checks for intersection with anything with collision enabled in the entire scene annd returns the paths to the geometry that was hit and the hit distances.

        See https://docs.omniverse.nvidia.com/kit/docs/omni_physics/105.1/extensions/runtime/source/omni.physx/docs/index.html#raycast

        Args:
            position (np.array): origin's position for ray cast
            orientation (np.array): origin's orientation for ray cast
            offset (np.array): offset for ray cast
            max_dist (float, optional): maximum distance to test for collisions in stage units. Defaults to 100.0.
            prim_path (str, optional): path to the target prim. If None, will check for intersection with anything with collision enabled in the entire scene. Defaults to None.

        Returns:
            typing.Tuple[typing.Union[None, str], float]: path to geometry that was hit and hit distance, returns None, 10000 if no hit occurred
        """
        prim_path = str(prim_path)
        ray_hits = []

        def report_raycast(hit):
            nonlocal ray_hits
            if prim_path is None:
                # Add the hit to the list of hits
                ray_hits.append((hit.collision, hit.distance))
            elif hit.collision == prim_path:
                # If the hit is the target prim, add it to the list of hits
                ray_hits = (hit.collision, hit.distance)
                return True
            else:
                self._log_message(f"Hit something other than the target prim {prim_path}: hit {hit.collision} at distance {hit.distance}")
                # If the hit is not the target prim, ignore it
                # ray_hits.append((hit.collision, hit.distance))
            return False
            

        omni.physx.get_physx_scene_query_interface().raycast_all(origin, direction, max_dist, report_raycast, True)
        if ray_hits == []:
            # No hit found
            return None, max_dist
        return ray_hits
        

    def _is_point_in_mesh(self, point:Tuple[float,float,float], mesh_prim:Usd.Prim) -> bool:
        """
        Check if a point is inside a mesh using ray casting.
        This is accurate and works well even for sparse meshes.
        """

        # Quick bounds check first to avoid unnecessary calculations
        bounds = self._get_mesh_bounds(mesh_prim)
        if not bounds:
            self._log_message(f"Error: Could not determine bounds for mesh {mesh_prim.GetPath()}")
            return False

        min_point, max_point = bounds

        # Check if point is outside bounding box
        if (point[0] < min_point[0] or point[0] > max_point[0] or
            point[1] < min_point[1] or point[1] > max_point[1] or
            point[2] < min_point[2] or point[2] > max_point[2]):
            return False

        # Directions for ray casting
        directions = [
            (1.0, 0.0, 0.0),  # +X
            (-1.0, 0.0, 0.0), # -X
            (0.0, 1.0, 0.0),  # +Y
            (0.0, -1.0, 0.0), # -Y
            (0.0, 0.0, 1.0),  # +Z
            (0.0, 0.0, -1.0)  # -Z
        ]

        # Set up counters for inside/outside voting
        inside_votes = 0
        outside_votes = 0

        # Use twice the hypotenuse to determine the ray length
        ray_length = math.sqrt(
            (max_point[0] - min_point[0]) ** 2 +
            (max_point[1] - min_point[1]) ** 2 +
            (max_point[2] - min_point[2]) ** 2
        )

        for direction in directions:
            # Use ray_cast 
            hit= self._single_ray_cast_to_mesh(
                    point,              # Origin point as 3D vector np.array
                    direction,          # Direction as a 3D vector np.array
                    ray_length,         # Ray length
                    mesh_prim.GetPath() # Prim path to check for intersection
                )
            if hit[0] is None:
                return False

        # If we reach here, the point is inside the mesh
        return True

    def _apply_early_fusion(self, sensor_measurements: dict[str,dict[str,Tuple[np.ndarray[int],float]]]) -> dict[str,Tuple[List[int],float]]:
        """Apply early fusion strategy to combine measurements of the same sensor per voxel.
        This is just a sum of the measurements on the voxel.
        
        Parameters
        ----------
        measurements : dict[str,dict[str,Tuple[np.ndarray[int],float]]]
            Dictionary of measurements for each sensor type, where each value is a tuple of the measurements and the weight.
            The structure is {sensor_name: {group: ([measurements], weight)}}

        Returns
        -------
        dict[str,Tuple[np.ndarray[int],float]]
            Dictionary of combined measurements for each sensor type, where each value is a tuple of the combined measurements and the weight.
            The structure is {group: ([measurements], weight)}
        """

        # Initialize a dictionary to hold the combined measurements
        combined_measurements = {}

        # Iterate through each sensor type and its measurements
        for sensor_name, measurements in sensor_measurements.items(): #TODO ERROR HERE
            for group, (measurements_list, weight) in measurements.items():
                if group not in combined_measurements:
                    combined_measurements[group] = measurements_list

                # Assume that the measurements are in the same order for each sensor type
                combined_measurements[group] += measurements_list
        
        return combined_measurements
    
    def _apply_late_fusion(self, uncertainties:dict[str, np.ndarray[float]]) -> float:
        """Apply late fusion strategy to combine uncertainties (σ's)from different sensor types per voxel
        This is based on the formula from the "Perception Entropy..."
        
        σ_fused = sqrt(1 / Σ(1/σ_i²))

        Parameters
        ----------
        uncertainties : dict[str, list[float]]
            Dictionary of uncertainties (σ_i's) for each sensor type, where each value is a list of uncertainties.
            The structure is {group: [uncertainties]}
        
        Returns
        -------
        float
            The combined uncertainty (σ_fused) for the voxel.
        """
        if not uncertainties:
            return 0.0
            
        uncertainties_array = np.array(list(uncertainties.values())) # this should be a 2D array of uncertainties, where each row is a sensor and each column is a voxel
        # Calculate the sum of the inverse squares of the uncertainties
        sum_inverse_squares = np.sum(1 / (uncertainties_array ** 2), axis=0)
        # Calculate the fused uncertainty
        sig_fused = np.sqrt(1 / sum_inverse_squares)
        # Normalize the uncertainty
        # sig_fused = sig_fused / np.max(sig_fused) # Normalize to be between 0 and 1
        # Normalize the uncertainty to be between 0 and 1
        # sig_fused = np.clip(sig_fused, 0.0, 1.0)
        # self._log_message(f"Fused uncertainty: {sig_fused}")

        return sig_fused
            
    
    # def _on_window(self, visible):
    #     if self._window.visible:
    #         # Subscribe to Stage and Timeline Events

    #         self._usd_context = omni.usd.get_context()
    #         events = self._usd_context.get_stage_event_stream()
    #         self._stage_event_sub = events.create_subscription_to_pop(self._on_stage_event)
    #         stream = self._timeline.get_timeline_event_stream()
    #         self._timeline_event_sub = stream.create_subscription_to_pop(self._on_timeline_event)

    #         # self._build_ui()
    #     else:
    #         self._usd_context = None
    #         self._stage_event_sub = None
    #         self._timeline_event_sub = None
    #         self._cleanup_ui()


    def on_shutdown(self):
        """Clean up when the extension is unloaded"""
        remove_menu_items(self._menu_items, EXTENSION_TITLE)
    
        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.deregister_action(self.ext_id, f"CreateUIExtension:{EXTENSION_TITLE}")
        
        if self._window:
            self._window = None
        self._cleanup_ui()
        gc.collect()

    def target_prims_collision(self, prim_paths:str|Sdf.Path|List[Sdf.Path|str], disable_others:bool=True):
        """Set the target prim collisions on for ray cast / lidar sensing, set all the other prims to be non-collidable"""
        stage = get_current_stage()
        if not isinstance(prim_paths, list): # Means that there is only one prim path
            prim_paths = [prim_paths]
        else:
            prim_paths = [str(p) for p in prim_paths] # this creates a copy

        num_prims = len(prim_paths)
        
        # Search through all the prims in the stage. 
        # If the prim is in the list, set it to be collidable. 
        # If not, set it to be non-collidable (only if disable_others=True).
        for p in stage.Traverse():
            if p.IsA(UsdGeom.Mesh):
                collision_api = UsdPhysics.CollisionAPI(p)
                if not collision_api:
                    # Apply the CollisionAPI to the prim
                    collision_api = UsdPhysics.CollisionAPI.Apply(p)
                # Get the collision enabled attribute
                collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
                
                if p.GetPath() not in prim_paths:
                    if disable_others:
                        # Set the prim to be non-collidable
                        # self._log_message(f"Setting {p.GetPath()} as non-collidable")
                        collision_enabled_attr.Set(False)
                    # else:
                    #     self._log_message(f"NOT setting {p.GetPath()} as non-collidable")
                else:
                    # Set the target prim to be collidable
                    collision_enabled_attr.Set(True)
                    # Remove the prim from the list of target prims
                    prim_paths.remove(p.GetPath())
        
        if len(prim_paths) != 0:
            self._log_message(f"Warning: {len(prim_paths)} out of {num_prims} not found in the stage when setting collision targets!!")
        else:
            self._log_message(f"Set {num_prims} prims as target for ray cast / lidar sensing")
        

    def untarget_prims_collision(self, prim_paths:str|Sdf.Path|List[Sdf.Path|str]):
        """Set the target prim collisions off for ray cast / lidar sensing"""
        stage = get_current_stage()
        processed_count = 0
        original_count = len(prim_paths)
        for path_str in prim_paths:
            p = stage.GetPrimAtPath(path_str)
            # Check if prim exists and is a mesh before attempting to modify
            if p and p.IsA(UsdGeom.Mesh):
                collision_api = UsdPhysics.CollisionAPI(p)
                if not collision_api:
                    # This prim should already have had CollisionAPI applied by target_prims_collision
                    # If not, log a warning, but applying it here might be slow.
                    # Consider ensuring it's applied robustly in target_prims_collision or voxel creation.
                    self._log_message(f"Warning: CollisionAPI missing on {path_str} during untargeting.")
                    collision_api = UsdPhysics.CollisionAPI.Apply(p) # Apply if missing, might impact perf

                if collision_api:
                    collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
                    if collision_enabled_attr: # Check if attribute exists
                         collision_enabled_attr.Set(False)
                         processed_count += 1
                    else:
                         self._log_message(f"Warning: CollisionEnabled attribute missing on {path_str}.")
            # else:
            #     # DEBUG: Log if a path didn't correspond to a valid mesh prim
            #     self._log_message(f"Info: Path {path_str} not found or not a mesh during untargeting.")

        # # DEBUG: Log discrepancies if needed
        # if processed_count != original_count:
        #     self._log_message(f"Info: Untargeted {processed_count} out of {original_count} requested prims.")


    async def _ensure_physics_updated(self, pause=True, steps=1):
        """Ensures the physics scene is updated by stepping the simulation if paused, or waiting frames if playing.
        Call with `await self._ensure_physics_updated()` to ensure the physics scene is updated.

        Args:
            pause (bool, optional): Whether to pause the simulation after stepping. Defaults to True.
            steps (int, optional): The number of physics steps/frames to advance. Defaults to 1.
        """
        if not hasattr(self, 'timeline') or self.timeline is None:
            self._log_message("Warning: Timeline interface not found, attempting to re-acquire.")
            try:
                self.timeline = omni.timeline.get_timeline_interface()
            except Exception as e:
                self._log_message(f"Error: Failed to acquire timeline interface: {e}")
                # Cannot proceed without timeline, maybe raise an error or return
                raise AttributeError("Timeline interface could not be acquired.")
        
        was_playing = self.timeline.is_playing()
        if not was_playing:
            # If not playing, play before stepping
            self.timeline.play()
            # Wait for play command to take effect
            await omni.kit.app.get_app().next_update_async()

        # Advance the simulation by the specified number of steps
        for _ in range(steps):
            await omni.kit.app.get_app().next_update_async()

        if pause and not was_playing:
            # If it wasn't playing originally, pause it again
            self.timeline.pause()
        elif not pause and was_playing:
            # If it was playing originally and pause is False, ensure it continues playing
            # (This might be redundant if play() keeps it playing, but ensures state)
            self.timeline.play()
    
    async def _get_points_raycast(self, sensor_instance:Sensor3D_Instance, mesh_prim_path:Sdf.Path) -> List[Tuple[Gf.Vec3d, str, float]]:
        """Get the number of points from a raycast that land on the given prim"""

        points = []

        for i, ray_caster in enumerate(sensor_instance.ray_casters, start=1):
            i = i if len(sensor_instance.ray_casters) != 1 else ""
            print(f"i = {i}, and the sensor is a {sensor_instance.__getattribute__(f'sensor{i}').__class__}")
            self.target_prims_collision(mesh_prim_path)
            
            self.timeline.play()                                             # Play the simulation to get the ray data
            await omni.kit.app.get_app().next_update_async()                 # wait one frame for data
            self.timeline.pause()                                            # Pause the simulation to populate the LIDAR's depth buffers
            pathstr = prim_utils.get_prim_path(ray_caster)

            #Get the linear depth data from the lidar. If the linear depth of a point != the maximum range, it means that the point hit something
            lin_depth = self.lidarInterface.get_linear_depth_data(pathstr)
            for index in filter(lambda x: lin_depth[x] != self.lidarInterface.get_max_range(), range(len(lin_depth))):
                # Get the point in world space
                point = ray_caster.get_point(index)
                for obj, obj_data in self.target_objects.items():
                    weight = obj_data["weight"]
                    points.append((point, obj, weight))
            print(f"Num points hitting {mesh_prim_path}: {len(points)}")
            self._log_message(f"Num points hitting {mesh_prim_path}: {len(points)}")

        return points
    
    async def _get_measurements_raycast(self, sensor_instance:Sensor3D_Instance, disable_raycaster=False) -> dict[str,Tuple[np.ndarray[int],float]]:
        """Get the number of points from a raycaster that pass through each of the voxels in the given list
        The voxels and voxel groups will be stored in the same order as the weighted_voxels dictionary."""
        start_time = time.time()
        measurements = {}
        for gp_name, (voxels, weight) in self.weighted_voxels.items():
            if gp_name == "UNGROUPED":
                continue
            measurements[gp_name] = (np.zeros(len(voxels), dtype=int), weight)

        # Build a mapping from voxel center to index for fast lookup
        voxel_centers = {}
        voxel_idx_map = {}
        for gp_name, (voxels, weight) in self.weighted_voxels.items():
            if gp_name == "UNGROUPED":
                continue
            for idx, voxel in enumerate(voxels):
                center = voxel.GetAttribute("xformOp:translate").Get()
                center_tuple = tuple(center)
                voxel_centers[center_tuple] = (gp_name, idx)
                voxel_idx_map[center_tuple] = (gp_name, idx)

        # Compute grid bounds and voxel size
        all_centers = np.array(list(voxel_centers.keys()))
        grid_min = np.min(all_centers, axis=0)
        grid_max = np.max(all_centers, axis=0)
        # Assume uniform voxel size from your settings
        voxel_size = self.voxel_size if isinstance(self.voxel_size, float) else float(self.voxel_size[0])

        # For each ray caster (sensor), get rays and process
        ray_origins, ray_directions = sensor_instance.get_rays()
        # Get ray origins and directions for this sensor

        for ray_origin, ray_direction in zip(ray_origins, ray_directions):
            # 1. Ray-mesh intersection for occlusion
            hit_prim, hit_distance = self._single_ray_cast_to_mesh(
                origin=carb.Float3(float(ray_origin[0]), float(ray_origin[1]), float(ray_origin[2])), 
                direction=carb.Float3(float(ray_direction[0]), float(ray_direction[1]), float(ray_direction[2])), 
                max_dist=sensor_instance.sensor.max_range, 
                prim_path=None
            )

            # 2. DDA traversal up to hit_distance
            for voxel_idx in self.dda_ray_voxel_traversal(
                ray_origin, ray_direction, max_distance=hit_distance
            ):
                # Map voxel_idx (i,j,k) to world center
                center = (
                    grid_min[0] + (voxel_idx[0] + 0.5) * voxel_size,
                    grid_min[1] + (voxel_idx[1] + 0.5) * voxel_size,
                    grid_min[2] + (voxel_idx[2] + 0.5) * voxel_size,
                )
                # Find which group/index this voxel belongs to
                if center in voxel_idx_map:
                    gp_name, idx = voxel_idx_map[center]
                    measurements[gp_name][0][idx] += 1
        
        print(f"Measurements for {sensor_instance.name} took {time.time() - start_time:.2f} sec")
        self._log_message(f"Measurements for {sensor_instance.name} took {time.time() - start_time:.2f} sec")
        return measurements


    def _optimize_robot():
        """Optimize the robot's sensor poses"""
        raise NotImplementedError("Robot optimization is not implemented yet.")