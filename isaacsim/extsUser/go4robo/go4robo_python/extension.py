import gc

import omni.ext
import omni.ui as ui
import omni.kit.commands
import omni.physx as _physx
from omni.isaac.core.utils.stage import get_current_stage
from isaacsim.gui.components.element_wrappers import ScrollingWindow
from isaacsim.gui.components.menu import MenuItemDescription
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from pxr import UsdGeom, Gf, Sdf, Usd, UsdPhysics, Vt
import omni.isaac.core.utils.prims as prim_utils
import isaacsim.core.utils.collisions as collisions_utils
from isaacsim.sensors.physx import _range_sensor

import asyncio # Used to run sample asynchronously to not block rendering thread

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union
import carb
import time

from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE

import os, sys

bot_3d_rep_module_path = sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

print(f"Looking for bot_3d_rep in {bot_3d_rep_module_path}")
from bot_3d_rep import *

sensor_types = [MonoCamera3D, Lidar3D, StereoCamera3D]

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

        self.selected_export_path=None

        # Events
        events = self._usd_context.get_stage_event_stream()
        self._stage_event_sub = events.create_subscription_to_pop(self._on_stage_event)

        # A place to store the robots
        self.robots = []

        # These just help handle the stage selection and structure
        self.previous_selection = []
        
        # Voxel size for object representation (in meters)
        self.voxel_size = 0.1
        
        # Target object types with their typical dimensions (length, width, height in meters)
        self.target_objects = {
            "car": {"dimensions": [4.5, 1.8, 1.5], "weight": 1.0},
            "pedestrian": {"dimensions": [0.6, 0.6, 1.7], "weight": 1.0},
            "cyclist": {"dimensions": [1.8, 0.6, 1.7], "weight": 1.0}, 
            "truck": {"dimensions": [8.0, 2.5, 3.0], "weight": 1.0},
            "cone": {"dimensions": [0.3, 0.3, 0.5], "weight": 1.0}
        }

        self.log_messages = []
        self.max_log_messages = 100  # Default value
        
        # Add a property to track the selected perception area mesh
        self.perception_mesh = None
        self.perception_mesh_path = None
        
        self._build_ui()
        self._window.visible = True

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
                    
                    with ui.CollapsableFrame("Perception Entropy", height=0):
                        with ui.VStack(spacing=5, height=0):

                            # Add sampling step size UI
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
                                self.select_mesh_btn = ui.Button("Voxelize", width=80, height=36, clicked_fn=self._voxelize_perception_mesh)
                            
                            # Show bounds of selected mesh if available
                            self.mesh_bounds_label = ui.Label("Mesh Bounds: Not selected")
                        
                            with ui.CollapsableFrame("Object Types", height=0):
                                with ui.VStack(spacing=5, height=0):
                                    for obj_type in self.target_objects:
                                        with ui.HStack(spacing=5):
                                            ui.Label(f"{obj_type.capitalize()}", width=100)
                                            ui.Label("Weight:", width=50)
                                            weight_field = ui.FloatField(width=50)
                                            weight_field.model.set_value(self.target_objects[obj_type]["weight"])
                                            weight_field.model.add_value_changed_fn(
                                                lambda w, obj=obj_type: self._update_object_weight(obj, w)
                                            )
                    
                            # Buttons for operations
                            with ui.HStack(spacing=5, height=0):
                                self.analyze_btn = ui.Button("Analyze Perception", clicked_fn=self._calc_perception_entropy, height=36)
                                # Initially disable the button since no mesh is selected
                                self.disable_ui_element(self.analyze_btn, text_color=ui.color("#FF0000"))
                                self.reset_btn = ui.Button("Reset", clicked_fn=self._reset_settings, height=36)
                    
                    ui.Spacer(height=10)
                    
                    # Results section
                    with ui.CollapsableFrame("Results", height=0, collapsed=True):
                        with ui.VStack(spacing=5):
                            self.results_list = ui.ScrollingFrame(height=250)
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
        
        # Log the selected robots
        self._log_message(self.selected_robot_label.text)

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
        self.log_field.model.set_value("\n".join(self.log_messages))

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
        # Reset object weights
        for obj_type in self.target_objects:
            self.target_objects[obj_type]["weight"] = 1.0
        
        # Reset perception mesh settings
        self.perception_mesh = None
        self.perception_mesh_path = None
        self.perception_mesh_label.text = "(Not selected)"
        self.mesh_bounds_label.text = "Mesh Bounds: Not selected"
        
        # Disable analyze button when mesh is reset
        self.disable_ui_element(self.analyze_btn, text_color=ui.color("#FF0000"))
        
        # Reset sampling step size
        self.voxel_size = 5.0
        self.voxel_size_field.model.set_value(self.voxel_size)
        
        self._log_message("Settings reset to default values")
    
    def _refresh_sensor_list(self):
        """Refresh the list of detected sensors without analysis"""

        def _find_camera(prim:Usd.Prim) -> Sensor3D_Instance:
            """Find cameras that are descendants of the selected robot"""

            # self._log_message(f"DEBUG: Checking for CAMERA prim {prim.GetName()} of type {prim.GetTypeName()}")

            if prim.IsA(UsdGeom.Camera):
                # Skip editor cameras if a specific robot is selected
                name = prim.GetName()
                
                # Load the camera information into a MonoCamera3D
                cam_prim = UsdGeom.Camera(prim)

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
                    self._log_message(f"Warning: Could not find resolution for camera {prim.GetName()}")
                    # Default to HD resolution as fallback
                    resolution = (1280, 720)

                try:
                    cam3d = MonoCamera3D(name=name,
                                        focal_length=cam_prim.GetFocalLengthAttr().Get(),
                                        h_aperture=cam_prim.GetHorizontalApertureAttr().Get(),
                                        v_aperture=cam_prim.GetVerticalApertureAttr().Get(),
                                        aspect_ratio=self._get_prim_attribute(prim, "aspectRatio", None),
                                        h_res=resolution[0] if resolution else None,
                                        v_res=resolution[1] if resolution else None,
                                        body=prim,
                                        cost=1.0,
                                        focal_point=(0, 0, 0)
                                        )
                    cam3d_instance = Sensor3D_Instance(cam3d, path=prim.GetPath(), name=name, tf=self._get_robot_to_sensor_transform(prim, robot_prim))
                    cam3d_instance.create_ray_casters(get_current_stage())
                    self._log_message(f"Found camera: {cam3d_instance.name} with HFOV: {cam3d_instance.sensor.h_fov:.2f}°")
                except Exception as e:
                    self._log_message(f"Error extracting camera properties for {name}: {str(e)}")
                    raise e
                    
                return cam3d_instance
            
            else:
                return None

        def _find_lidar(prim:Usd.Prim) -> Dict:
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
                    lidar = Lidar3D(name=name,
                                    h_fov=self._get_prim_attribute(prim, "horizontalFov"),
                                    v_fov=self._get_prim_attribute(prim, "verticalFov"),
                                    h_res=self._get_prim_attribute(prim, "horizontalResolution"),
                                    v_res=self._get_prim_attribute(prim, "verticalResolution"),
                                    max_range=self._get_prim_attribute(prim, "maxRange"),
                                    min_range=self._get_prim_attribute(prim, "minRange"),
                                    body=prim,
                                    cost=1.0,
                                    )
                    lidar_instance = Sensor3D_Instance(lidar, path=prim.GetPath(), name=name, tf=self._get_robot_to_sensor_transform(prim, robot_prim))
                    lidar_instance.create_ray_casters(get_current_stage())
                    self._log_message(f"Found LiDAR: {lidar_instance.name} with HFOV: {lidar_instance.sensor.h_fov:.2f}°")
                except Exception as e:
                    self._log_message(f"Error extracting LiDAR properties for {name}: {str(e)}")
            
            return lidar_instance


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
                self._log_message(f"Found camera at path: {prim.GetPath()}")
                
                # Handle stereo camera pairing
                found_stereo_pair = False
                for role in ["left", "right"]:
                    if (role in camera.name.lower() or 
                        (prim.HasAttribute("stereoRole") and prim.GetAttribute("stereoRole").Get() == role)):
                        this_cam_role = role
                        this_cam_name = camera.name
                        this_cam_path_str = prim.GetPath().pathString
                        other_cam_role = "left" if role == "right" else "right"
                        other_cam_name = camera.name.replace(role, other_cam_role)
                        other_cam_path_str = str(this_cam_path_str).replace(role, other_cam_role)

                        # Check if the other camera is already in the sensors list
                        for i, sensor_instance in enumerate(bot.sensors):
                            if isinstance(sensor_instance.sensor, MonoCamera3D):
                                if sensor_instance.name == other_cam_name:
                                    # Found the other camera, create a stereo camera
                                    self._log_message(f"Pairing stereo cameras: {this_cam_name} and {other_cam_name} in robot {bot.name}")
                                    self._log_message(f"  Path 1: {this_cam_path_str}")
                                    self._log_message(f"  Path 2: {sensor_instance.path.pathString}")
                                    # Find the common parent of the two cameras based on the paths
                                    common_parent_path = os.path.commonpath([this_cam_path_str, sensor_instance.path.pathString])
                                    self._log_message(f"  Common: {common_parent_path}")
                                    common_parent_prim = get_current_stage().GetPrimAtPath(common_parent_path)
                                    if not common_parent_prim:
                                        self._log_message(f"Error: Could not find common parent prim at path {common_parent_path}")
                                        common_parent_name=this_cam_name.replace(this_cam_role, "")
                                    else:
                                        common_parent_name = common_parent_prim.GetName()


                                    stereo_cam = StereoCamera3D(
                                        name=common_parent_name,
                                        sensor1=sensor_instance.sensor if this_cam_role == "left" else camera.sensor,
                                        sensor2=sensor_instance.sensor if this_cam_role == "right" else camera.sensor,
                                        tf_sensor1=sensor_instance.tf if this_cam_role == "left" else camera.tf,
                                        tf_sensor2=sensor_instance.tf if this_cam_role == "right" else camera.tf,
                                        cost=sensor_instance.sensor.cost + camera.sensor.cost,
                                        body=prim
                                    )
                                    stereo_instance = Sensor3D_Instance(stereo_cam, path= common_parent_path, 
                                                                    name=common_parent_name, 
                                                                    tf=camera.tf)
                                    bot.sensors[i] = stereo_instance  # Replace the mono camera with stereo
                                    found_stereo_pair = True
                                    break
                        break  # Found a role, no need to check other roles
                        
                # Add camera only if it wasn't part of a stereo pair
                if not found_stereo_pair:
                    self._log_message(f"Adding camera: {camera.name} to robot {bot.name}")
                    bot.sensors.append(camera)
            
            # Always continue recursively even if sensors were found at this level
            for child in prim.GetChildren():
                _assign_sensors_to_robot(child, bot, processed_camera_paths)

        self._log_message("Refreshing robots & sensors list...")

        stage = get_current_stage()

        self.robots = []
        
        total_sensors = dict.fromkeys(sensor_types, 0)
        for bot_prim in self.selected_prims:
            bot = Bot3D(bot_prim.GetName(), path=bot_prim.GetPath())
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
        
        # Update the sensor list UI
        self._update_sensor_list_ui()


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
                self._log_message(f"Average Precision Constants for {sensor_instance.name}: {value}")
                with ui.CollapsableFrame("Average Precision Parameters", height=0, collapsed=True):
                    with ui.VStack(spacing=2):
                        ui.Label("AP = a ln(m) + b, values depend on the detection algorithm.")
                        ui.Label("Default values are from \"Perception Entropy [...]\" Ma et al. 2021.")
                        for const, val in value.items():
                            with ui.HStack(spacing=5):
                                ui.Label(f"{const}:", width=120)
                                ap_field = ui.FloatField(width=80)
                                ap_field.model.set_value(val)
                                
                                # Update the sensor instance's average_precision constant when changed
                                def on_param_changed(new_value):
                                    sensor_instance.ap_constants[const] = max(0.0, min(1.0, new_value.get_value_as_float()))
                                    self._log_message(f"Set {sensor_instance.name} {attr} to {val:.2f}")
                                    
                                ap_field.model.add_value_changed_fn(on_param_changed)
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
                                                    # if not hasattr(sensor_instance, 'average_precision') or sensor_instance.average_precision is None:
                                                    #     sensor_instance.average_precision = 0.9
                                                        
                                                    with ui.CollapsableFrame(f"{idx+1}. {sensor_instance.name}", height=0, style={"border_color": ui.color("#FFFFFF")}, collapsed=True):
                                                        # with ui.VStack(spacing=2):
                                                        #     # Add average precision input directly at the top level
                                                        #     with ui.HStack(spacing=5):
                                                        #         ui.Label("Average Precision:", width=120)
                                                        #         ap_field = ui.FloatField(width=80)
                                                        #         ap_field.model.set_value(sensor_instance.average_precision)
                                                                
                                                        #         # Update the sensor instance's average_precision when changed
                                                        #         def on_ap_changed(new_value, sensor=sensor_instance):
                                                        #             ap = max(0.0, min(1.0, new_value.get_value_as_float()))
                                                        #             sensor.average_precision = ap
                                                        #             self._log_message(f"Set {sensor.name} average precision to {ap:.2f}")
                                                                
                                                        #         ap_field.model.add_value_changed_fn(on_ap_changed)
                                                            
                                                            # Show other properties in collapsible section
                                                            with ui.CollapsableFrame("Properties", height=0, collapsed=True):
                                                                with ui.VStack(spacing=2):
                                                                    self._display_sensor_instance_properties(sensor_instance)

    def _update_object_weight(self, obj_type: str, weight: float):
        """Update the weight of a specific object type"""
        self.target_objects[obj_type]["weight"] = weight
        self._log_message(f"Updated weight for {obj_type} to {weight}")

    def _calc_perception_entropy(self):
        """Main function to analyze all sensors on the robot"""
        self._log_message("Starting perception entropy analysis...")
        
        # Check if we have robots to analyze
        if not self.robots:
            self._log_message("Error: No robots selected for analysis.")
            return
        
        # Check if we have a perception mesh
        if not self.perception_mesh:
            self._log_message("Error: No perception mesh selected.")
            return
        else:
            UsdPhysics.CollisionAPI.Apply(self.perception_mesh)

        results_data = {"combined": {"total": 0.0, "cameras": 0.0, "lidars": 0.0}}

        for robot in self.robots:
            for sensor_instance in robot.sensors:
                points = asyncio.ensure_future(self._get_points_raycast(sensor_instance, self.perception_mesh.GetPrimPath())) 
            
                if not points:
                    self._log_message(f"Error: No points from {sensor_instance.name} hit; check perception mesh.")
                    continue
                
                self._log_message(f"{sensor_instance.name} hit {len(points)} points")

                #TODO: Add the perception entropy calculation here
                
        
        # Update the results UI
        self._update_results_ui(results_data)
        
        self._log_message("Analysis complete")
        
    def _update_results_ui(self, results_data=None):
        """Update the results UI with entropy results for each robot"""
        # Clear existing results
        self.results_list.clear()
        
        with self.results_list:
            with ui.VStack(spacing=5):
                if not self.robots:
                    ui.Label("No robots selected")
                elif not results_data:
                    ui.Label("Run analysis to see results")
                else:
                    # Add an overall results section
                    with ui.CollapsableFrame(
                        "Overall Results", 
                        height=0,
                        style={"border_width": 2, "border_color": ui.color("#00aa00")},
                        collapsed=False
                    ):
                        with ui.VStack(spacing=5):
                            # Display combined results if available
                            if "combined" in results_data:
                                combined = results_data["combined"]
                                ui.Label(f"Total Perception Entropy: {combined['total']:.4f}")
                                ui.Label(f"All Cameras: {combined['cameras']:.4f}")
                                ui.Label(f"All LiDARs: {combined['lidars']:.4f}")
                    
                    # For each robot, create a collapsible frame with its results
                    for robot_name, robot_results in results_data.items():
                        if robot_name == "combined":
                            continue  # Skip the combined results as they're already displayed
                        
                        with ui.CollapsableFrame(
                            f"Robot: {robot_name}", 
                            height=0,
                            style={"border_width": 2, "border_color": ui.color("#0059ff")},
                            collapsed=False
                        ):
                            with ui.VStack(spacing=5):
                                ui.Label(f"Total Entropy: {robot_results['total']:.4f}")
                                
                                # Camera results
                                with ui.CollapsableFrame(
                                    f"Cameras: {robot_results['cameras']:.4f}", 
                                    height=0,
                                    style={"border_color": ui.color("#00c3ff")},
                                    collapsed=False
                                ):
                                    with ui.VStack(spacing=2):
                                        for camera_name, camera_entropy in robot_results.get('camera_details', {}).items():
                                            ui.Label(f"{camera_name}: {camera_entropy:.4f}")
                                
                                # LiDAR results
                                with ui.CollapsableFrame(
                                    f"LiDARs: {robot_results['lidars']:.4f}", 
                                    height=0,
                                    style={"border_color": ui.color("#00c3ff")},
                                    collapsed=False
                                ):
                                    with ui.VStack(spacing=2):
                                        for lidar_name, lidar_entropy in robot_results.get('lidar_details', {}).items():
                                            ui.Label(f"{lidar_name}: {lidar_entropy:.4f}")

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
    
    def _generate_sample_points(self) -> List[Tuple[Gf.Vec3d, str, float]]:
        """Generate sample points in the perception space"""
        sample_points = []
        
        # Check if we have a valid mesh
        if not self.perception_mesh or not self.perception_mesh.IsValid():
            self._log_message("Error: No valid perception mesh selected. Please select a mesh first.")
            return []
        
        # For more stuff that is not a mesh, use bounding box sampling
        if not self.perception_mesh.IsA(UsdGeom.Mesh):
            self._log_message("Warning: Selected prim is not a mesh. Using bounding box sampling instead.")
            return self._generate_box_sample_points()
        
        # Get the mesh geometry
        mesh_geom = UsdGeom.Mesh(self.perception_mesh)
        
        # Get the points and face indices
        points = mesh_geom.GetPointsAttr().Get()
        face_vertex_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
        
        if not points or not face_vertex_counts or not face_vertex_indices:
            self._log_message(f"Missing mesh data for {self.perception_mesh.GetPath()}. Using bounding box sampling instead.")
            return self._generate_box_sample_points()
        
        # Get the mesh transform
        xform = UsdGeom.Xformable(self.perception_mesh)
        local_to_world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Convert mesh to world space
        world_points = [Gf.Vec3d(local_to_world.Transform(Gf.Vec3d(p))) for p in points]
        
        # Get the mesh bounding box for determining sampling density
        bounds = self._get_mesh_bounds(self.perception_mesh)
        if not bounds:
            self._log_message("Error: Could not determine mesh bounds.")
            return []
        
        min_point, max_point = bounds
        dx = max_point[0] - min_point[0]
        dy = max_point[1] - min_point[1]
        dz = max_point[2] - min_point[2]
        
        # Estimate total volume and target number of samples
        volume = dx * dy * dz
        # Adjust the density factor based on how fine-grained you want the sampling to be
        density_factor = 1.0 / (self.voxel_size ** 3)
        target_samples = int(volume * density_factor)
        
        self._log_message(f"Generating approximately {target_samples} sample points for mesh")
        
        # Calculate centroid of mesh to use as starting point
        centroid = Gf.Vec3d(0, 0, 0)
        for p in world_points:
            centroid += p
        if len(world_points) > 0:
            centroid /= len(world_points)
        
        # Use an octree-inspired approach for adaptive sampling
        # Start with the bounding box and recursively subdivide where needed
        sample_points.extend(self._adaptive_sample_mesh(min_point, max_point, target_samples))
        
        self._log_message(f"Generated {len(sample_points)} sample points inside mesh area")
        return sample_points

    def _generate_box_sample_points(self) -> List[Tuple[Gf.Vec3d, str, float]]:
        """Generate sample points using the bounding box approach (fallback method)"""
        sample_points = []
        
        bounds = self._get_mesh_bounds(self.perception_mesh)
        if not bounds:
            self._log_message("Error: Could not determine mesh bounds.")
            return []
        
        min_point, max_point = bounds
        
        # Generate sample points within mesh bounds
        step = self.voxel_size  # Use the configurable step size
        
        # Place samples for each object type
        for obj_type, obj_data in self.target_objects.items():
            weight = obj_data["weight"]
            
            # Sample the mesh volume with the specified step size
            for x in np.arange(min_point[0], max_point[0], step):
                for y in np.arange(min_point[1], max_point[1], step):
                    for z in np.arange(min_point[2], max_point[2], step):
                        point = Gf.Vec3d(x, y, z)
                        
                        # Check if point is inside the mesh
                        if self._is_point_in_mesh(point, self.perception_mesh):
                            sample_points.append((point, obj_type, weight))
        
        self._log_message(f"Generated {len(sample_points)} sample points inside bounding box")

        print("Testing ray casting")
        self._single_ray_cast_all_collisions((0.0,0.0,0.0), (1.0,0.0,0.0), 1000.0)

        return sample_points

    def _adaptive_sample_mesh(self, min_corner, max_corner, target_samples, depth=0, max_depth=5):
        """Recursively sample a region of space, focusing on areas inside the mesh"""
        sample_points = []
        
        # Stop recursion if we've reached max depth or the region is too small
        if depth >= max_depth:
            # At max depth, apply regular grid sampling in this small region
            step = self.voxel_size
            
            # Create a smaller step size for regions we're focusing on
            adjusted_step = max(step, (max_corner[0] - min_corner[0]) / 3.0)
            
            # Place samples for each object type
            for obj_type, obj_data in self.target_objects.items():
                weight = obj_data["weight"]
                
                # Sample this subregion using a grid
                for x in np.arange(min_corner[0], max_corner[0], adjusted_step):
                    for y in np.arange(min_corner[1], max_corner[1], adjusted_step):
                        for z in np.arange(min_corner[2], max_corner[2], adjusted_step):
                            point = Gf.Vec3d(x, y, z)
                            
                            # Check if point is inside the mesh
                            if self._is_point_in_mesh(point, self.perception_mesh):
                                sample_points.append((point, obj_type, weight))
            return sample_points
        
        # Calculate the center of the current region
        center = (min_corner + max_corner) * 0.5
        
        # Probe the center point to see if it's inside the mesh
        center_inside = self._is_point_in_mesh(center, self.perception_mesh)
        
        # Probe the corners to see if they're inside the mesh
        corners = [
            Gf.Vec3d(min_corner[0], min_corner[1], min_corner[2]),
            Gf.Vec3d(max_corner[0], min_corner[1], min_corner[2]),
            Gf.Vec3d(min_corner[0], max_corner[1], min_corner[2]),
            Gf.Vec3d(max_corner[0], max_corner[1], min_corner[2]),
            Gf.Vec3d(min_corner[0], min_corner[1], max_corner[2]),
            Gf.Vec3d(max_corner[0], min_corner[1], max_corner[2]),
            Gf.Vec3d(min_corner[0], max_corner[1], max_corner[2]),
            Gf.Vec3d(max_corner[0], max_corner[1], max_corner[2])
        ]
        corners_inside = [self._is_point_in_mesh(c, self.perception_mesh) for c in corners]
        
        # If all corners and center are outside, skip this region
        if not center_inside and not any(corners_inside):
            return []
        
        # If all corners and center are inside, we can be more efficient with sampling
        if center_inside and all(corners_inside):
            # This region is fully inside the mesh, sample it with a coarser grid
            adjusted_step = self.voxel_size * 2  # Coarser sampling for interior regions
            for obj_type, obj_data in self.target_objects.items():
                weight = obj_data["weight"]
                
                # Determine how many points to sample based on volume
                dx = max_corner[0] - min_corner[0]
                dy = max_corner[1] - min_corner[1]
                dz = max_corner[2] - min_corner[2]
                volume = dx * dy * dz
                
                # Generate a uniform sampling but with larger step size
                for x in np.arange(min_corner[0], max_corner[0], adjusted_step):
                    for y in np.arange(min_corner[1], max_corner[1], adjusted_step):
                        for z in np.arange(min_corner[2], max_corner[2], adjusted_step):
                            point = Gf.Vec3d(x, y, z)
                            sample_points.append((point, obj_type, weight))
            return sample_points
        
        # Otherwise, this region intersects the mesh boundary, subdivide it
        mid_x = (min_corner[0] + max_corner[0]) / 2
        mid_y = (min_corner[1] + max_corner[1]) / 2
        mid_z = (min_corner[2] + max_corner[2]) / 2
        
        # Generate 8 octants
        octants = [
            (Gf.Vec3d(min_corner[0], min_corner[1], min_corner[2]), Gf.Vec3d(mid_x, mid_y, mid_z)),
            (Gf.Vec3d(mid_x, min_corner[1], min_corner[2]), Gf.Vec3d(max_corner[0], mid_y, mid_z)),
            (Gf.Vec3d(min_corner[0], mid_y, min_corner[2]), Gf.Vec3d(mid_x, max_corner[1], mid_z)),
            (Gf.Vec3d(mid_x, mid_y, min_corner[2]), Gf.Vec3d(max_corner[0], max_corner[1], mid_z)),
            (Gf.Vec3d(min_corner[0], min_corner[1], mid_z), Gf.Vec3d(mid_x, mid_y, max_corner[2])),
            (Gf.Vec3d(mid_x, min_corner[1], mid_z), Gf.Vec3d(max_corner[0], mid_y, max_corner[2])),
            (Gf.Vec3d(min_corner[0], mid_y, mid_z), Gf.Vec3d(mid_x, max_corner[1], max_corner[2])),
            (Gf.Vec3d(mid_x, mid_y, mid_z), Gf.Vec3d(max_corner[0], max_corner[1], max_corner[2]))
        ]
        
        # Recursively sample each octant
        for min_pt, max_pt in octants:
            # Adjust target samples based on volume ratio
            octant_volume = (max_pt[0] - min_pt[0]) * (max_pt[1] - min_pt[1]) * (max_pt[2] - min_pt[2])
            total_volume = (max_corner[0] - min_corner[0]) * (max_corner[1] - min_corner[1]) * (max_corner[2] - min_corner[2])
            octant_target = max(1, int(target_samples * (octant_volume / total_volume)))
            
            # Recursively sample this octant
            sample_points.extend(self._adaptive_sample_mesh(min_pt, max_pt, octant_target, depth + 1, max_depth))
        
        return sample_points
    
    def voxelize_mesh(self, mesh_prim, voxel_size, parent_path):
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
        local_to_world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
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

        # Iterate through the grid to find voxels intersected by triangles
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
                    # if _is_point_in_mesh(voxel_center, mesh_prim):
                    overlap = asyncio.ensure_future(self._does_box_overlap_prim(voxel_center, voxel_extent, mesh_prim.GetPath()))
                    if overlap:
                        voxel_centers.append(((i,k,j),voxel_center))

        # Convert voxel centers to a Vec3fArray for later use
        self._log_message(f"Generated {len(voxel_centers)} voxel centers")
        
        created_voxels = []
        stage = get_current_stage()
        
        # Create a mesh for each occupied voxel
        for (i,j,k), p in voxel_centers:
            # world_p = Gf.Vec3d(local_to_world.Transform(Gf.Vec3d(p))):
            # Calculate voxel corners
            min_x = p[0] - voxel_size / 2
            min_y = p[1] - voxel_size / 2
            min_z = p[2] - voxel_size / 2
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
            
            # Create voxel mesh using USD API directly
            voxel_path = f"{parent_path}/{mesh_name}_voxel_{i}_{j}_{k}"
            mesh_def = UsdGeom.Mesh.Define(stage, voxel_path)
            mesh_def.CreatePointsAttr().Set(voxel_points)
            mesh_def.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_def.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)
            created_voxels.append(mesh_def.GetPrim())
        
        self._log_message(f"Created {len(created_voxels)} voxel meshes")
        return created_voxels

    def _voxelize_perception_mesh(self):
        """Select a mesh to use as the target perception area"""
        # Get current selection
        selection = self._usd_context.get_selection().get_selected_prim_paths()
        
        if not selection:
            self._log_message("No mesh selected. Please select a mesh prim in the stage.")
            self.perception_mesh_label.text = "(Not selected)"
            self.mesh_bounds_label.text = "Mesh Bounds: Not selected"
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
        
        # Get and display bounds
        bounds = self._get_mesh_bounds(mesh_prim)
        if bounds:
            min_point, max_point = bounds
            self.mesh_bounds_label.text = (
                f"Mesh Bounds: \n     X: [{min_point[0]:.2f}, {max_point[0]:.2f}], \n"
                f"     Y: [{min_point[1]:.2f}, {max_point[1]:.2f}], \n"
                f"     Z: [{min_point[2]:.2f}, {max_point[2]:.2f}]"
            )
            
            # Calculate the largest dimension of the mesh
            dimensions = [
                max_point[0] - min_point[0],
                max_point[1] - min_point[1],
                max_point[2] - min_point[2]
            ]

        voxels = self.voxelize_mesh(mesh_prim, self.voxel_size, parent_path=xform_path)

        if voxels:
            self._log_message(f"Created {len(voxels)} voxel meshes inside {mesh_path}")
        else:
            self._log_message(f"No voxels created for {mesh_path}")
        
        self._log_message(f"Selected mesh '{mesh_path}' as perception area")

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
                self._log_message(f"Mesh bounds: {min_point} to {max_point}")
                return (min_point, max_point)
        except Exception as e:
            self._log_message(f"Error getting bounds of mesh: {str(e)}")
        
        return None
    
    async def _does_box_overlap_prim(self, origin:carb.Float3, extent:carb.Float3, prim_path):
        """Check if a box overlaps with a given prim path"""

        # Target the prim mash
        self.target_prim_collision(prim_path)

        rotation = carb.Float4(1.0, 0.0, 0.0, 0.0) # No rotation

        half_extent = carb.Float3(
            max(0.001, extent[0] / 2.0),
            max(0.001, extent[1] / 2.0), 
            max(0.001, extent[2] / 2.0)
        )

        prim_found = False

        def report_overlap(overlap):
            print(f"Overlap: {overlap}")
            nonlocal prim_found
            if overlap.collision == prim_path:
                print(f" - Found prim: {overlap.collision}")
                # Now that we have found our prim, return False to abort further search.voxel_size
                return False
            return True

        try:
            self.timeline.play()
            await omni.kit.app.get_app().next_update_async()
            omni.physx.get_physx_scene_query_interface().overlap_box(half_extent, origin, rotation, report_overlap, False)
            self.timeline.stop()
        
        except Exception as e:
            self._log_message(f"Error in overlap_box: {str(e)}")
        
        if prim_found:
            self._log_message(f"Box overlaps with prim {prim_path}")
        # else:
        #     self._log_message(f"No overlap found with prim {prim_path}")
    
        return prim_found
    
    def _single_ray_cast_all_collisions(self, 
                                        origin:Tuple[float, float, float]=(0.0,0.0,0.0), 
                                        direction:Tuple[float, float, float]=(1.0,0.0,0.0), 
                                        max_dist: float = 100.0) -> dict:
        """Projects a raycast forward along x axis with specified offset

        See https://docs.omniverse.nvidia.com/kit/docs/omni_physics/105.1/extensions/runtime/source/omni.physx/docs/index.html#raycast

        Args:
            position (np.array): origin's position for ray cast
            orientation (np.array): origin's orientation for ray cast
            offset (np.array): offset for ray cast
            max_dist (float, optional): maximum distance to test for collisions in stage units. Defaults to 100.0.

        Returns:
            typing.Tuple[typing.Union[None, str], float]: path to geometry that was hit and hit distance, returns None, 10000 if no hit occurred
        """
        # print(f"Raycast origin: {origin}, direction: {ray_dir}, max_dist: {max_dist}")

        def report_raycast(hit):
            print(f"Hit: {hit}")
            return hit["hit"]

        hit = omni.physx.get_physx_scene_query_interface().raycast_all(origin, direction, max_dist, report_raycast)
        print(f"Raycast hit: {hit}")
        if hit:
            usdGeom = UsdGeom.Mesh.Get(get_current_stage(), hit["rigidBody"])
            distance = hit["distance"]
            print(f"Hit! {usdGeom} at {distance}mm")
            return hit
        return None
        

    def _is_point_in_mesh(self, point:Tuple[float,float,float], mesh_prim:Usd.Prim) -> bool:
        """
        Check if a point is inside a mesh using ray casting.
        This is accurate and works well even for sparse meshes.
        """
        self._log_message(f"Checking if point {point} is inside mesh {mesh_prim.GetPath()}")

        # Target the prim mash
        self.target_prim_collision(prim_utils.get_prim_path(mesh_prim))

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
            self._log_message(f"Point {point} is outside the bounding box of the mesh")
            return False

        # Use fixed directions as Gf.Vec3d instead of numpy arrays
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

        # Calculate a ray length that's guaranteed to exit the bounding box
        ray_length = max(
            max_point[0] - min_point[0],
            max_point[1] - min_point[1],
            max_point[2] - min_point[2]
        ) *10

        for direction in directions:
            # Use ray_cast with properly formatted arguments - using Gf.Vec3d directly
            hit_info = self._single_ray_cast_all_collisions(
                point,            # Origin point as 3D vector np.array
                direction,        # Direction as a 3D vector np.array
                ray_length        # Ray length
            )

            # Count ray intersections
            if hit_info is not None:
                # If we hit something, count as an inside vote
                self._log_message(f"Hit! On {hit_info['collision']} at {hit_info['distance']}mm")

                inside_votes += 1
            else:
                # If no hit, the ray didn't intersect, likely outside
                outside_votes += 1

        # Use majority voting for robustness
        inside = inside_votes > outside_votes
        self._log_message(f"Point {point} was determined to be {'inside' if inside else 'outside'} the mesh. Votes: {inside_votes} inside, {outside_votes} outside")
        return inside_votes > outside_votes

    def _apply_early_fusion(self, entropies: List[float]) -> float:
        """Apply early fusion strategy to combine entropies (average them)"""
        if not entropies:
            return 0.0
        return sum(entropies) / len(entropies)
    
    def _apply_late_fusion(self, entropies: List[float]) -> float:
        """Apply late fusion strategy to combine entropies from different sensor types
        
        Based on the formula: σ_fused = sqrt(1 / Σ(1/σ_i²))
        """
        if not entropies:
            return 0.0
            
        # Convert entropies back to standard deviations
        sigmas = []
        for entropy in entropies:
            # Reverse the entropy formula to get sigma
            # H = 2*ln(σ) + 1 + ln(2π)
            # σ = exp((H - 1 - ln(2π))/2)
            sigma = math.exp((entropy - 1 - math.log(2 * math.pi)) / 2)
            sigmas.append(sigma)
        
        # Calculate fused sigma using the formula from the paper
        sum_inverse_sigma_squared = sum(1 / (sigma ** 2) for sigma in sigmas)
        if sum_inverse_sigma_squared > 0:
            sigma_fused = math.sqrt(1 / sum_inverse_sigma_squared)
            
            # Convert back to entropy
            entropy_fused = 2 * math.log(sigma_fused) + 1 + math.log(2 * math.pi)
            return entropy_fused
        else:
            return 0.0
    
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

    def target_prim_collision(self, prim_path):
        """Set the target prim collisions on for ray cast / lidar sensing, set all the other prims to be non-collidable"""
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        
        if not prim:
            return
        if not prim.IsValid():
            return
        if not prim.IsA(UsdGeom.Mesh):
            return
        
        # Search through all the prims in the stage and set them to be non-collidable
        for p in stage.Traverse():
            if p.IsA(UsdGeom.Mesh):
                collision_api = UsdPhysics.CollisionAPI(p)
                if not collision_api:
                    # Apply the CollisionAPI to the prim
                    collision_api = UsdPhysics.CollisionAPI.Apply(p)
                # Get the collision enabled attribute
                collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
                
                if p != prim:
                    # Set the other prims to be non-collidable
                    collision_enabled_attr.Set(False)
                else:
                    # Set the target prim to be collidable
                    collision_enabled_attr.Set(True)
        
        # self._log_message(f"Set {prim.GetPath()} as collision target for ray casting")

    def untarget_prim_collision(self, prim_path):
        """Set the target prim collisions off for ray casting. Don't change the rest."""
        prim = get_current_stage().GetPrimAtPath(prim_path)

        if not prim:
            return
        if not prim.IsValid():
            return
        if not prim.IsA(UsdGeom.Mesh):
            return
        
        # Set the target prim to be non-collidable
        collision_api = UsdPhysics.CollisionAPI(prim)

        # Check if it has the API
        if collision_api:
            # Get the collision enabled attribute
            collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
            
            # Disable collisions
            collision_enabled_attr.Set(False)
            self._log_message(f"Unset {prim.GetPath()} as target for ray cast / lidar sensing")
        else:
            self._log_message(f"Prim {prim.GetPath()} does not have a CollisionAPI, skipping")

    
    async def _get_points_raycast(self, sensor_instance:Sensor3D_Instance, mesh_prim_path:Sdf.Path) -> List[Tuple[Gf.Vec3d, str, float]]:
        """Get the number of points from a raycast that land on the given prim"""

        points = []

        for i, ray_caster in enumerate(sensor_instance.ray_casters, start=1):
            i = i if len(sensor_instance.ray_casters) != 1 else ""
            print(f"i = {i}, and the sensor is a {sensor_instance.__getattribute__(f'sensor{i}').__class__}")
            self.target_prim_collision(mesh_prim_path)
            
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

        
