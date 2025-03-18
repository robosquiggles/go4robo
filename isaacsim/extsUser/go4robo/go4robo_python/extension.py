import gc

import omni.ext
import omni.ui as ui
import omni.kit.commands
import omni.physx as _physx
from omni.isaac.core.utils.stage import get_current_stage
from isaacsim.gui.components.element_wrappers import ScrollingWindow
from isaacsim.gui.components.menu import MenuItemDescription
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from pxr import UsdGeom, Gf, Sdf, Usd
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.sensor as sensor

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import carb
import time

from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE

import os, sys

bot_3d_rep_module_path = sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

print(f"Looking for bot_3d_rep in {bot_3d_rep_module_path}")
from bot_3d_rep import *

sensor_types = [MonoCamera3D, Lidar3D]

class GO4RExtension(omni.ext.IExt):
    """Extension that calculates perception entropy for cameras and LiDARs in Isaac Sim"""
    
    def on_startup(self, ext_id):
        """Initialize the extension"""

        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()

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
        self._usd_context = omni.usd.get_context()
        events = self._usd_context.get_stage_event_stream()
        self._stage_event_sub = events.create_subscription_to_pop(self._on_stage_event)

        # A place to store the robots
        self.selected_robots = []

        # These just help handle the stage selection and structure
        self.previous_selection = []
        self.processed_lidar_paths = set()
        
        # Constants for calculation based on the paper
        # TODO: Make this based on mesh/objects in the scene
        self.perception_space = {
            "x_range": [-80.0, 80.0],  # meters
            "y_range": [-40.0, 40.0],  # meters
            "z_range": [0.0, 5.0]      # meters
        }
        
        # Constants for camera AP calculation
        self.camera_ap_constants = {
            "a": 0.055,  # coefficient from the paper
            "b": 0.155   # coefficient from the paper
        }
        
        # Constants for LiDAR AP calculation
        self.lidar_ap_constants = {
            "a": 0.152,  # coefficient from the paper
            "b": 0.659   # coefficient from the paper
        }
        
        # Voxel size for object representation (in meters)
        # TODO: Make this adjustable in the UI
        self.voxel_size = 0.1
        
        # Target object types with their typical dimensions (length, width, height in meters)
        # TODO: Make this based on objects in the scene?
        self.target_objects = {
            "car": {"dimensions": [4.5, 1.8, 1.5], "weight": 1.0},
            "pedestrian": {"dimensions": [0.6, 0.6, 1.7], "weight": 1.0},
            "cyclist": {"dimensions": [1.8, 0.6, 1.7], "weight": 1.0}, 
            "truck": {"dimensions": [8.0, 2.5, 3.0], "weight": 1.0},
            "cone": {"dimensions": [0.3, 0.3, 0.5], "weight": 1.0}
        }

        self.log_messages = []
        self.max_log_messages = 100  # Default value
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
                                self.refresh_sensors_btn = ui.Button("Refresh Sensor List", clicked_fn=self._refresh_sensor_list, height=36, width=0)
                                self.disable_ui_element(self.refresh_sensors_btn, text_color=ui.color("#FF0000"))
                            self.sensor_list = ui.ScrollingFrame(
                                height=250,
                            )
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
                    
                    with ui.CollapsableFrame("Settings", height=0):
                        with ui.VStack(spacing=5, height=0):
                            ui.Label("Perception Space (meters)")
                            
                            with ui.HStack(spacing=5):
                                ui.Label("X Range:", width=70)
                                self.x_min_field = ui.FloatField(width=50)
                                self.x_min_field.model.set_value(self.perception_space["x_range"][0])
                                ui.Label("to", width=20)
                                self.x_max_field = ui.FloatField(width=50)
                                self.x_max_field.model.set_value(self.perception_space["x_range"][1])
                            
                            with ui.HStack(spacing=5):
                                ui.Label("Y Range:", width=70)
                                self.y_min_field = ui.FloatField(width=50)
                                self.y_min_field.model.set_value(self.perception_space["y_range"][0])
                                ui.Label("to", width=20)
                                self.y_max_field = ui.FloatField(width=50)
                                self.y_max_field.model.set_value(self.perception_space["y_range"][1])
                            
                            with ui.HStack(spacing=5):
                                ui.Label("Z Range:", width=70)
                                self.z_min_field = ui.FloatField(width=50)
                                self.z_min_field.model.set_value(self.perception_space["z_range"][0])
                                ui.Label("to", width=20)
                                self.z_max_field = ui.FloatField(width=50)
                                self.z_max_field.model.set_value(self.perception_space["z_range"][1])
                    
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
                    
                    ui.Spacer(height=10)
                    
                    # Buttons for operations
                    with ui.HStack(spacing=5, height=0):
                        self.analyze_btn = ui.Button("Analyze Robot Sensors", clicked_fn=self._analyze_sensors, height=36)
                        self.reset_btn = ui.Button("Reset", clicked_fn=self._reset_settings, height=36)
                    
                    ui.Spacer(height=10)
                    
                    # Results section
                    with ui.CollapsableFrame("Results", height=0):
                        with ui.VStack(spacing=5):
                            self.camera_label = ui.Label("Cameras: Not analyzed")
                            self.lidar_label = ui.Label("LiDARs: Not analyzed")
                            self.total_label = ui.Label("Total Perception Entropy: N/A")
                            
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

                            self.log_text = ui.ScrollingFrame(
                                height=250, 
                                style={
                                    "border_width": 1, 
                                    "border_color": 0xFF0000FF, 
                                    "border_radius": 3,
                                    "alignment": ui.Alignment.TOP,
                                    "margin": 4
                                },
                                scroll_to_bottom_on_change=True
                            )
                            with self.log_text:
                                self.log_label = ui.Label("")
        
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

        self.selected_robots = []

        if not selection:
            self.selected_robot_label.text = "(Select one or more robot from the stage)"
            self.selected_robot_label.style = {"color": ui.color("#FF0000")}
            self.disable_ui_element(self.refresh_sensors_btn, text_color=ui.color("#FF0000"))
            return
        
        for robot_path in selection:
            bot = Bot3D(path=robot_path,
                        name=robot_path.split('/')[-1]
            )
            self.selected_robots.append(bot)
        self.selected_robot_label.text = f"Selected robots: {', '.join([bot.name for bot in self.selected_robots])}"
        self.enable_ui_element(self.refresh_sensors_btn, text_color=ui.color("#00FF00"))
        self.selected_robot_label.style = {"color": ui.color("#00FF00")}
        
        # Log the selected robots
        self._log_message(self.selected_robot_label.text)

    def _update_max_log_messages(self, value):
        """Update the maximum number of log messages to keep"""
        self.max_log_messages = max(1, int(value))
        self._update_log_display()  # Refresh display with new limit
        
    def _clear_log(self):
        """Clear all log messages"""
        self.log_messages = []
        self.log_label.text = ""
        
    def _log_message(self, message: str):
        """Add a message to the log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Add to message list
        self.log_messages.append(formatted_message)
        
        # Trim to max messages
        if len(self.log_messages) > self.max_log_messages:
            self.log_messages = self.log_messages[-self.max_log_messages:]
        
        # Update display
        self._update_log_display()
        
    def _update_log_display(self):
        """Update the log display with current messages"""
        self.log_label.text = "\n".join(self.log_messages)
        
        # Auto-scroll to bottom - this is handled by ScrollingFrame automatically
        # when content changes, but we could add explicit scrolling if needed

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
        self.perception_space = {
            "x_range": [-80.0, 80.0],
            "y_range": [-40.0, 40.0],
            "z_range": [0.0, 5.0]
        }
        
        # Update UI fields
        self.x_min_field.model.set_value(self.perception_space["x_range"][0])
        self.x_max_field.model.set_value(self.perception_space["x_range"][1])
        self.y_min_field.model.set_value(self.perception_space["y_range"][0])
        self.y_max_field.model.set_value(self.perception_space["y_range"][1])
        self.z_min_field.model.set_value(self.perception_space["z_range"][0])
        self.z_max_field.model.set_value(self.perception_space["z_range"][1])
        
        # Reset object weights
        for obj_type in self.target_objects:
            self.target_objects[obj_type]["weight"] = 1.0
            
        self._log_message("Settings reset to default values")
    
    def _refresh_sensor_list(self):
        """Refresh the list of detected sensors without analysis"""
        self._log_message("Refreshing sensor list...")

        stage = get_current_stage()

        total_cameras = 0
        total_lidars = 0
        
        for bot in self.selected_robots:

            #Clear sensors
            bot.sensors = []
            
            # Find robot prim
            robot_prim = stage.GetPrimAtPath(bot.path)
            if not robot_prim:
                self._log_message(f"Error: Could not find prim at path {bot.path}")
                continue
                
            # Search for sensors in this robot
            self._assign_sensors_to(robot_prim, bot)
            
            found_sensors = {}
            for type in sensor_types:
                found_sensors[type.__name__] = bot.get_sensors_by_type(type)
                if found_sensors[type.__name__]:
                    self._log_message(f"Found {len(found_sensors[type.__name__])} {type.__name__} sensors for robot {bot.name}")

            bot_cameras = found_sensors[MonoCamera3D.__name__]
            bot_lidars = found_sensors[Lidar3D.__name__]
            self._log_message(f"Robot {bot.name} has {bot_lidars} lidars and {bot_cameras} cameras")
            
            total_cameras += len(bot_cameras)
            total_lidars += len(bot_lidars)
        
        self._log_message(f"Total: {total_cameras} cameras and {total_lidars} LiDARs")
        
        # Update the sensor list UI
        self._update_sensor_list_ui()


    def _assign_sensors_to(self, prim, bot):
        """Search a level for sensors and add them to the specified robot"""
        # Track processed camera paths to avoid duplicates
        processed_camera_paths = set()
        
        for child in prim.GetChildren():
            # First check if this is part of a stereo pair
            stereo_camera = self._find_stereo_camera(child)
            if stereo_camera is not None:
                bot.sensors.append(stereo_camera)
                # Add the paths of both cameras in the stereo pair to processed paths
                # to avoid adding them individually
                if hasattr(stereo_camera.sensor, 'camera1') and hasattr(stereo_camera.sensor.camera1, 'body'):
                    processed_camera_paths.add(str(stereo_camera.sensor.camera1.body.GetPath()))
                if hasattr(stereo_camera.sensor, 'camera2') and hasattr(stereo_camera.sensor.camera2, 'body'):
                    processed_camera_paths.add(str(stereo_camera.sensor.camera2.body.GetPath()))
            
            # Check for mono camera if not part of a processed stereo pair
            if str(child.GetPath()) not in processed_camera_paths:
                camera = self._find_camera(child)
                if camera is not None:
                    bot.sensors.append(camera)
            
            # Check for LiDAR
            lidar = self._find_lidar(child)
            if lidar is not None:
                bot.sensors.append(lidar)
            
            # Recursively search child prims
            self._assign_sensors_to(child, bot)

    
    def _update_sensor_list_ui(self):
        """Update the sensor list UI with the detected sensors for all robots"""
        # Clear the current sensor list
        self.sensor_list.clear()
        
        with self.sensor_list:
            with ui.VStack(spacing=5):
                if not self.selected_robots:
                    ui.Label("No robots selected")
                else:
                    # For each robot, create a collapsible frame
                    for robot in self.selected_robots:

                        robot_name = robot.name
                        robot_stereo_cameras = robot.get_sensors_by_type(StereoCamera3D)
                        robot_cameras = robot.get_sensors_by_type(MonoCamera3D)
                        robot_lidars = robot.get_sensors_by_type(Lidar3D)
                        
                        # Skip robots with no sensors
                        if not robot_cameras and not robot_lidars:
                            continue
                            
                        # Create collapsible frame for this robot with blue border
                        with ui.CollapsableFrame(
                            f"Robot: {robot_name}", 
                            height=0, 
                            style={"border_width": 2, "border_color": ui.color("#0088FF")}, 
                            collapsed=False
                        ):
                            with ui.VStack(spacing=5):
                                # Display stereo cameras for this robot
                                if robot_stereo_cameras:
                                    with ui.CollapsableFrame(f"Stereo Cameras: {len(robot_stereo_cameras)}", height=0, style={"border_color": ui.color("#00FF00")}, collapsed=False):
                                        with ui.VStack(spacing=5):
                                            for idx, stereo_camera in enumerate(robot_stereo_cameras):
                                                with ui.CollapsableFrame(f"{idx+1}. {stereo_camera.name}", height=0, style={"border_color": ui.color("#FFFFFF")}, collapsed=True):
                                                    with ui.VStack(spacing=2):
                                                        for property_name, property_value in stereo_camera.__dict__.items():
                                                            ui.Label(f"{property_name}: {property_value}")
                                                        
                                                        # Show additional properties if available
                                                        if 'properties' in stereo_camera.__dict__.keys():
                                                            with ui.CollapsableFrame("Additional Properties", height=0, collapsed=True):
                                                                with ui.VStack(spacing=2):
                                                                    for prop_name, prop_value in stereo_camera['properties'].items():
                                                                        ui.Label(f"{prop_name}: {prop_value}")

                                # Display mono cameras for this robot
                                if robot_cameras:
                                    with ui.CollapsableFrame(f"Cameras: {len(robot_cameras)}", height=0, style={"border_color": ui.color("#FF00FF")}, collapsed=False):
                                        with ui.VStack(spacing=5):
                                            for idx, camera in enumerate(robot_cameras):
                                                with ui.CollapsableFrame(f"{idx+1}. {camera.name}", height=0, style={"border_color": ui.color("#FFFFFF")}, collapsed=True):
                                                    with ui.VStack(spacing=2):
                                                        for property_name, property_value in camera.__dict__.items():
                                                            ui.Label(f"{property_name}: {property_value}")
                                                        
                                                        # Show additional properties if available
                                                        if 'properties' in camera.__dict__.keys():
                                                            with ui.CollapsableFrame("Additional Properties", height=0, collapsed=True):
                                                                with ui.VStack(spacing=2):
                                                                    for prop_name, prop_value in camera['properties'].items():
                                                                        ui.Label(f"{prop_name}: {prop_value}")
                                
                                # Display LiDARs for this robot
                                if robot_lidars:
                                    with ui.CollapsableFrame(f"LiDARs: {len(robot_lidars)}", height=0, style={"border_color": ui.color("#00FFFF")}, collapsed=False):
                                        with ui.VStack(spacing=5):
                                            for idx, lidar in enumerate(robot_lidars):
                                                with ui.CollapsableFrame(f"{idx+1}. {lidar.name}", height=0, style={"border_color": ui.color("#FFFFFF")}, collapsed=True):
                                                    with ui.VStack(spacing=2):
                                                        for property_name, property_value in lidar.__dict__.items():
                                                            ui.Label(f"{property_name}: {property_value}")
                                                            
                                                        # Show additional properties if available
                                                        if 'properties' in lidar.__dict__.keys():
                                                            with ui.CollapsableFrame("Additional Properties", height=0, collapsed=True):
                                                                with ui.VStack(spacing=2):
                                                                    for prop_name, prop_value in lidar['properties'].items():
                                                                        ui.Label(f"{prop_name}: {prop_value}")
    
    def _update_object_weight(self, obj_type: str, weight: float):
        """Update the weight of a specific object type"""
        self.target_objects[obj_type]["weight"] = weight
        self._log_message(f"Updated weight for {obj_type} to {weight}")

    def _analyze_sensors(self):
        """Main function to analyze all sensors on the robot"""
        self._log_message("Starting sensor analysis...")
        
        # Update perception space from UI fields
        self.perception_space["x_range"] = [self.x_min_field.model.get_value_as_float(), 
                                           self.x_max_field.model.get_value_as_float()]
        self.perception_space["y_range"] = [self.y_min_field.model.get_value_as_float(), 
                                           self.y_max_field.model.get_value_as_float()]
        self.perception_space["z_range"] = [self.z_min_field.model.get_value_as_float(), 
                                           self.z_max_field.model.get_value_as_float()]
        
        # Refresh sensor list
        self._refresh_sensor_list()
        
        # Use the detected sensors for analysis
        cameras = self.detected_cameras
        lidars = self.detected_lidars
        
        # Calculate perception entropy for cameras
        camera_entropy = self._calculate_camera_entropy(cameras)
        self.camera_label.text = f"Cameras: {camera_entropy:.4f}"
        
        # Calculate perception entropy for LiDARs
        lidar_entropy = self._calculate_lidar_entropy(lidars)
        self.lidar_label.text = f"LiDARs: {lidar_entropy:.4f}"
        
        # Calculate total perception entropy (late fusion of sensors)
        if cameras and lidars:
            # Using late fusion strategy from the paper
            total_entropy = self._apply_late_fusion([camera_entropy, lidar_entropy])
            self.total_label.text = f"Total Perception Entropy: {total_entropy:.4f}"
        elif cameras:
            self.total_label.text = f"Total Perception Entropy: {camera_entropy:.4f}"
        elif lidars:
            self.total_label.text = f"Total Perception Entropy: {lidar_entropy:.4f}"
        else:
            self.total_label.text = "Total Perception Entropy: N/A"
            self._log_message("No sensors found for analysis")
            
        self._log_message("Analysis complete")

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
    
    def _find_camera(self, prim) -> Sensor3D_Instance:
        """Find cameras that are descendants of the selected robot"""

        # self._log_message(f"DEBUG: Checking for CAMERA prim {prim.GetName()} of type {prim.GetTypeName()}")

        if prim.IsA(UsdGeom.Camera):
            # Skip editor cameras if a specific robot is selected
            name = prim.GetName()
            
            # Load the camera information into a MonoCamera3D
            cam_prim = UsdGeom.Camera(prim)

            resolution = self._get_prim_attribute(prim, "aspectRatio", None)

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
            cam3d_instance = Sensor3D_Instance(cam3d, path=prim.GetPath(), name=name, tf=self._get_world_transform(prim))
            
            self._log_message(f"Found camera: {cam3d_instance.name} with HFOV: {cam3d_instance.sensor.h_fov:.2f}°")
                
            return cam3d_instance
        
        else:
            return None
        
    def _find_stereo_camera(self, prim, left_prim=None) -> Sensor3D_Instance:
        """Find stereo camera pairs and combine them into a StereoCamera3D
        
        Args:
            prim (Usd.Prim): The prim to check for camera pairs
            left_prim (Usd.Prim, optional): A previously found left camera to pair with
            
        Returns:
            Sensor3D_Instance: A StereoCamera3D instance if a pair is found, None otherwise
        """
        # Skip non-camera prims
        if not prim.IsA(UsdGeom.Camera):
            return None
        
        # Get camera name
        name = prim.GetName()
        path = str(prim.GetPath())
        
        # Check if this is a right camera looking for its left pair
        if "_right" in name.lower() or "right_" in name.lower() or "right" == name.lower():
            # This is a right camera, look for corresponding left camera
            left_name = name.lower().replace("right", "left")
            parent_path = str(prim.GetParent().GetPath())
            possible_left_paths = [
                f"{parent_path}/{left_name}",
                f"{parent_path}/{left_name.capitalize()}",
                f"{parent_path}/{left_name.upper()}",
                f"{parent_path.replace('right', 'left')}/{left_name}",
                f"{parent_path.replace('right', 'left')}/{left_name.capitalize()}",
                f"{parent_path.replace('right', 'left')}/{left_name.upper()}"
            ]
            
            stage = get_current_stage()
            left_prim = None
            
            # Try to find the left camera
            for left_path in possible_left_paths:
                left_prim = stage.GetPrimAtPath(left_path)
                if left_prim and left_prim.IsA(UsdGeom.Camera):
                    self._log_message(f"Found left camera at {left_path} to pair with right camera {name}")
                    break
            
            if not left_prim:
                self._log_message(f"Could not find matching left camera for right camera {name}")
                return None
            
            # Create individual camera instances
            left_cam_instance = self._find_camera(left_prim)
            right_cam_instance = self._find_camera(prim)
            
            if not left_cam_instance or not right_cam_instance:
                return None
            
            # Calculate baseline (distance between cameras)
            left_pos, left_rot = left_cam_instance.tf
            right_pos, right_rot = right_cam_instance.tf
            
            # Create stereo camera
            stereo_name = name.replace("_right", "").replace("right_", "").replace("right", "")
            if not stereo_name:
                stereo_name = "stereo_camera"
            
            try:
                stereo_cam = StereoCamera3D(
                    name=stereo_name,
                    camera1=left_cam_instance.sensor,
                    camera2=right_cam_instance.sensor,
                    tf_camera1=left_cam_instance.tf,
                    tf_camera2=right_cam_instance.tf,
                    cost=left_cam_instance.sensor.cost + right_cam_instance.sensor.cost,
                    body=prim
                )
                
                # Use the left camera's position for the stereo camera
                stereo_instance = Sensor3D_Instance(
                    stereo_cam, 
                    path=left_prim.GetPath(),
                    tf=left_cam_instance.tf,
                    name=stereo_name
                )
                
                self._log_message(f"Created stereo camera {stereo_name} with baseline: {stereo_cam.base_line:.3f} units")
                
                return stereo_instance
                
            except Exception as e:
                self._log_message(f"Error creating stereo camera: {str(e)}")
                return None
        
        # Check if this is a left camera (for when we encounter left first)
        elif "_left" in name.lower() or "left_" in name.lower() or "left" == name.lower():
            # This is a left camera, look for corresponding right camera
            right_name = name.lower().replace("left", "right")
            parent_path = str(prim.GetParent().GetPath())
            possible_right_paths = [
                f"{parent_path}/{right_name}",
                f"{parent_path}/{right_name.capitalize()}",
                f"{parent_path}/{right_name.upper()}",
                f"{parent_path.replace('left', 'right')}/{right_name}",
                f"{parent_path.replace('left', 'right')}/{right_name.capitalize()}",
                f"{parent_path.replace('left', 'right')}/{right_name.upper()}"
            ]
            
            stage = get_current_stage()
            right_prim = None
            
            # Try to find the right camera
            for right_path in possible_right_paths:
                right_prim = stage.GetPrimAtPath(right_path)
                if right_prim and right_prim.IsA(UsdGeom.Camera):
                    # Call this function with the right camera as primary
                    return self._find_stereo_camera(right_prim, left_prim=prim)
            
            # No matching right camera found
            self._log_message(f"Could not find matching right camera for left camera {name}")
            return None
        
        # This is neither a left nor right camera
        return None

    def _find_lidar(self, prim:Usd.Prim) -> Dict:
        """Find LiDARs that are descendants of the selected robot"""

        # Skip if the prim has already been processed as a child
        if str(prim.GetPath()) in self.processed_lidar_paths:
            return None

        def is_lidar_prim(prim:Usd.Prim) -> str:
            """Check if a prim is a LiDAR"""
            # Check by type
            type_name = str(prim.GetTypeName()).lower()
            if "lidar" in type_name or "range" in type_name:
                self._log_message(f"Found LiDAR: '{prim.GetName()}' using type name")
                return "lidar by type"
            
            # Check by name - this is a fallback approach
            name = prim.GetName().lower()
            lidar_prim_names = ["lidar", "velodyne", "ouster", "hesai", "sick", "lms"]
            if any(lidar_name in name for lidar_name in lidar_prim_names):
                # If this is named lidar, likely there is a lidar under this prim
                self._log_message(f"Found LiDAR: '{prim.GetName()}' using prim name")
                return "lidar by name"
            
            return False
        
        def find_actual_lidar_child(parent_prim):
            """Search for the actual LiDAR component among children"""
            for child in parent_prim.GetChildren():
                child_type = str(child.GetTypeName()).lower()
                child_name = child.GetName().lower()
                
                # Look for a child that is actually a LiDAR type
                if "lidar" in child_type or "range" in child_type or "lidar" in child_name:
                    return child
                    
            # If no specific LiDAR child found, just return the first child
            children = list(parent_prim.GetChildren())
            if children:
                return children[0]
            return None
                
        lidar_data = None
        name = prim.GetName()
        is_lidar = is_lidar_prim(prim)
        
        if is_lidar == "lidar by type":
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
                lidar_data = Sensor3D_Instance(lidar, path=prim.GetPath(), name=name, tf=self._get_world_transform(prim))
            except Exception as e:
                self._log_message(f"Error extracting LiDAR properties for {name}: {str(e)}")
        
        elif is_lidar == "lidar by name":
            # This is a container/frame for a LiDAR - find the actual LiDAR component
            actual_lidar_prim = find_actual_lidar_child(prim)
            
            if actual_lidar_prim:
                
                self.processed_lidar_paths.add(str(actual_lidar_prim.GetPath())) # Add to the already explored prims

                # Use name from the parent frame but properties from the child
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
                    lidar_data = Sensor3D_Instance(lidar, path=prim.GetPath(), name=name, tf=self._get_world_transform(prim))
                except Exception as e:
                    self._log_message(f"Error extracting LiDAR properties for {name}: {str(e)}")
            else:
                # No suitable child found
                lidar_data = None
        else:
            lidar_data = None
        
        return lidar_data

    
    def _get_world_transform(self, prim) -> Tuple[Gf.Vec3d, Gf.Rotation]:
        """Get the world transform (position and rotation) of a prim"""
        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Extract position and rotation
        position = Gf.Vec3d(world_transform.ExtractTranslation())
        rotation = world_transform.ExtractRotationMatrix()
        
        return position, rotation
    
    def _calculate_camera_entropy(self, cameras: List[Dict]) -> float:
        """Calculate perception entropy for all cameras"""
        if not cameras:
            return 0.0
        
        # Generate sample points in the perception space
        sample_points = self._generate_sample_points()
        
        # For each camera, calculate entropy at each sample point
        camera_entropies = []
        
        for camera in cameras:
            entropy_sum = 0.0
            total_weight = 0.0
            
            for point, obj_type, weight in sample_points:
                # Calculate pixel count (sensor measurement) for this point
                pixel_count = self._calculate_camera_pixel_count(camera, point, obj_type)
                
                # Convert pixel count to AP
                ap = self._calculate_camera_ap(pixel_count)
                
                # Convert AP to standard deviation
                sigma = self._ap_to_sigma(ap)
                
                # Calculate entropy
                point_entropy = self._gaussian_entropy(sigma)
                entropy_sum += point_entropy * weight
                total_weight += weight
            
            if total_weight > 0:
                camera_entropies.append(entropy_sum / total_weight)
                self._log_message(f"Camera {camera['name']} entropy: {camera_entropies[-1]:.4f}")
        
        # Use early fusion strategy for multiple cameras (sum of measurements)
        # In practice, we'd need a more sophisticated fusion model
        if len(camera_entropies) > 1:
            combined_entropy = self._apply_early_fusion(camera_entropies)
            return combined_entropy
        elif camera_entropies:
            return camera_entropies[0]
        else:
            return 0.0
    
    def _calculate_lidar_entropy(self, lidars: List[Dict]) -> float:
        """Calculate perception entropy for all LiDARs"""
        if not lidars:
            return 0.0
        
        # Generate sample points in the perception space
        sample_points = self._generate_sample_points()
        
        # For each LiDAR, calculate entropy at each sample point
        lidar_entropies = []
        
        for lidar in lidars:
            entropy_sum = 0.0
            total_weight = 0.0
            
            for point, obj_type, weight in sample_points:
                # Calculate point count (sensor measurement) for this point
                point_count = self._calculate_lidar_point_count(lidar, point, obj_type)
                
                # Convert point count to AP
                ap = self._calculate_lidar_ap(point_count)
                
                # Convert AP to standard deviation
                sigma = self._ap_to_sigma(ap)
                
                # Calculate entropy
                point_entropy = self._gaussian_entropy(sigma)
                entropy_sum += point_entropy * weight
                total_weight += weight
            
            if total_weight > 0:
                lidar_entropies.append(entropy_sum / total_weight)
                self._log_message(f"LiDAR {lidar['name']} entropy: {lidar_entropies[-1]:.4f}")
        
        # Use early fusion strategy for multiple LiDARs (sum of measurements)
        if len(lidar_entropies) > 1:
            combined_entropy = self._apply_early_fusion(lidar_entropies)
            return combined_entropy
        elif lidar_entropies:
            return lidar_entropies[0]
        else:
            return 0.0
    
    def _generate_sample_points(self) -> List[Tuple[Gf.Vec3d, str, float]]:
        """Generate sample points in the perception space with associated object types and weights"""
        sample_points = []
        
        # Coarse sampling for efficiency; adjust step size as needed
        step_size = 5.0  # meters
        
        x_range = self.perception_space["x_range"]
        y_range = self.perception_space["y_range"]
        z_range = self.perception_space["z_range"]
        
        # Place samples for each object type
        for obj_type, obj_data in self.target_objects.items():
            weight = obj_data["weight"]
            
            for x in np.arange(x_range[0], x_range[1], step_size):
                for y in np.arange(y_range[0], y_range[1], step_size):
                    # For simplicity, we place objects on the ground (z=0)
                    # In a more complex simulation, we'd vary height as well
                    point = Gf.Vec3d(x, y, z_range[0])
                    sample_points.append((point, obj_type, weight))
        
        return sample_points
    
    def _calculate_camera_pixel_count(self, camera: Dict, point: Gf.Vec3d, obj_type: str) -> int:
        """Calculate the number of pixels an object at given point would occupy in the camera"""
        # Extract camera properties
        cam_pos, cam_rot_matrix = camera["transform"]
        hfov = camera["hfov"]
        resolution = camera["resolution"]
        
        # Get object dimensions
        obj_dimensions = self.target_objects[obj_type]["dimensions"]
        
        # Calculate vector from camera to point in world space
        direction_vector = point - cam_pos
        distance = direction_vector.GetLength()
        
        # Skip if too far
        if distance <= 0 or distance > 150:  # Maximum reasonable distance
            return 0
        
        # Transform the direction vector to camera space using the rotation matrix
        # Camera looks along +Z axis, with +Y up and +X to the right
        # We need to invert the rotation matrix to convert from world to camera space
        cam_rot_inverse = cam_rot_matrix.GetInverse()
        direction_camera_space = cam_rot_inverse * direction_vector
        
        # Check if the point is in front of the camera (positive Z in camera space)
        if direction_camera_space[2] <= 0:
            return 0  # Behind the camera
        
        # Calculate the horizontal and vertical angles in camera space
        horizontal_angle = math.atan2(direction_camera_space[0], direction_camera_space[2])
        vertical_angle = math.atan2(direction_camera_space[1], direction_camera_space[2])
        
        # Check if within camera FOV
        # Assuming the vertical FOV is derived from horizontal FOV and aspect ratio
        aspect_ratio = resolution[0] / resolution[1]
        vfov = hfov / aspect_ratio
        
        if abs(horizontal_angle) > hfov/2 or abs(vertical_angle) > vfov/2:
            return 0  # Outside field of view
        
        # Calculate object dimensions in camera view
        obj_width = obj_dimensions[1]   # Width of object
        obj_height = obj_dimensions[2]  # Height of object
        
        # Calculate angular width and height in radians
        angular_width = math.atan2(obj_width, distance)
        angular_height = math.atan2(obj_height, distance)
        
        # Calculate pixel width and height based on angular size
        pixel_width = (angular_width / hfov) * resolution[0]
        pixel_height = (angular_height / vfov) * resolution[1]
        
        # Calculate pixel count
        pixel_count = max(0, int(pixel_width * pixel_height))
        
        return pixel_count
    
    def _calculate_lidar_point_count(self, lidar: Dict, point: Gf.Vec3d, obj_type: str) -> int:
        """Calculate the number of LiDAR points that would hit an object at the given point"""
        # Extract LiDAR properties
        lidar_pos, lidar_rot_matrix = lidar["transform"]
        channels = lidar["channels"]
        fov_vertical = lidar["fov_vertical"]
        fov_horizontal = lidar["fov_horizontal"]
        max_range = lidar["max_range"]
        
        # Get object dimensions
        obj_dimensions = self.target_objects[obj_type]["dimensions"]
        
        # Calculate vector from LiDAR to point in world space
        direction_vector = point - lidar_pos
        distance = direction_vector.GetLength()
        
        # Skip if the point is too far
        if distance > max_range or distance <= 0:
            return 0
        
        # Transform the direction vector to LiDAR space using the rotation matrix
        # Assuming LiDAR looks along +X axis, with +Z up and +Y to the right
        # We need to invert the rotation matrix to convert from world to LiDAR space
        lidar_rot_inverse = lidar_rot_matrix.GetInverse()
        direction_lidar_space = lidar_rot_inverse * direction_vector
        
        # Normalize the vector in LiDAR space
        dir_norm = direction_lidar_space.GetNormalized()
        
        # Calculate angles in LiDAR space
        # Horizontal angle (in xy plane) - azimuth
        horizontal_angle = math.atan2(dir_norm[1], dir_norm[0])
        
        # Vertical angle (elevation from xy plane)
        vertical_angle = math.asin(dir_norm[2])
        
        # Check if point is within LiDAR's FOV
        if abs(horizontal_angle) > fov_horizontal/2 or abs(vertical_angle) > fov_vertical/2:
            return 0  # Outside field of view
        
        # Estimate point density based on angular resolution and channels
        # Horizontal resolution depends on scan rate and rotation speed
        # Simplification: assuming uniform angular resolution
        horizontal_resolution = 0.2 * math.degrees(fov_horizontal) # points per degree horizontally
        vertical_resolution = channels / math.degrees(fov_vertical) # points per degree vertically
        
        # Calculate angular dimensions of the object from the LiDAR's perspective
        obj_width = obj_dimensions[1]   # Width of object
        obj_height = obj_dimensions[2]  # Height of object
        
        # Calculate angular width and height in degrees
        angular_width_deg = math.degrees(math.atan2(obj_width, distance))
        angular_height_deg = math.degrees(math.atan2(obj_height, distance))
        
        # Calculate estimated point count based on angular size and resolution
        horizontal_points = angular_width_deg * horizontal_resolution
        vertical_points = angular_height_deg * vertical_resolution
        
        # Calculate total points, with distance attenuation factor
        # Points decrease with square of distance due to beam divergence
        attenuation_factor = 50.0 / (distance * distance) # Adjust the constant as needed
        attenuation_factor = min(1.0, max(0.1, attenuation_factor)) # Clamp to reasonable range
        
        point_count = max(1, int(horizontal_points * vertical_points * attenuation_factor))
        
        return point_count
    
    def _calculate_camera_ap(self, pixel_count: int) -> float:
        """Calculate Average Precision (AP) based on pixel count using the paper's formula"""
        if pixel_count <= 0:
            return 0.001  # Minimal AP for numerical stability
        
        # Using the formula from the paper: AP ≈ a * ln(m) + b
        a = self.camera_ap_constants["a"]
        b = self.camera_ap_constants["b"]
        
        ap = a * math.log(pixel_count) + b
        
        # Clamp AP to valid range
        ap = max(0.001, min(0.999, ap))
        
        return ap
    
    def _calculate_lidar_ap(self, point_count: int) -> float:
        """Calculate Average Precision (AP) based on point count using the paper's formula"""
        if point_count <= 0:
            return 0.001  # Minimal AP for numerical stability
        
        # Using the formula from the paper: AP ≈ a * ln(m) + b
        a = self.lidar_ap_constants["a"]
        b = self.lidar_ap_constants["b"]
        
        ap = a * math.log(point_count) + b
        
        # Clamp AP to valid range
        ap = max(0.001, min(0.999, ap))
        
        return ap
    
    def _ap_to_sigma(self, ap: float) -> float:
        """Convert Average Precision (AP) to standard deviation using the paper's formula"""
        # Using the formula from the paper: σ = 1/AP - 1
        sigma = (1 / ap) - 1
        return sigma
    
    def _gaussian_entropy(self, sigma: float) -> float:
        """Calculate the entropy of a 2D Gaussian distribution with given standard deviation"""
        # Using the formula from the paper: H(S|m, q) = 2*ln(σ) + 1 + ln(2π)
        entropy = 2 * math.log(sigma) + 1 + math.log(2 * math.pi)
        return entropy
    
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