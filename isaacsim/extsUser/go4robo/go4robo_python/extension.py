import gc

import omni.ext
import omni.ui as ui
import omni.kit.commands
import omni.physx as _physx
from omni.isaac.core.utils.stage import get_current_stage
from isaacsim.gui.components.element_wrappers import ScrollingWindow
from isaacsim.gui.components.menu import MenuItemDescription
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from omni.ui import color as cl
from pxr import UsdGeom, Gf, Sdf, Usd

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import carb
import time

from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE

class PerceptionEntropyExtension(omni.ext.IExt):
    """Extension that calculates perception entropy for cameras and LiDARs in Isaac Sim"""
    
    def on_startup(self, ext_id):
        """Initialize the extension"""

        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()

        self._window = ScrollingWindow(title=EXTENSION_TITLE, width=400, height=700)
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

        # Events
        self._usd_context = omni.usd.get_context()
        events = self._usd_context.get_stage_event_stream()
        self._stage_event_sub = events.create_subscription_to_pop(self._on_stage_event)

        # self._timeline = omni.timeline.get_timeline_interface()
        # stream = self._timeline.get_timeline_event_stream()
        # self._timeline_event_sub = stream.create_subscription_to_pop(self._on_timeline_event)

        # Store the robot
        self.robot = ""  # Initialize empty robot path
        self.selected_robot_path = "/"  # Initialize with default path

        # Store detected sensors
        self.detected_cameras = []
        self.detected_lidars = []
        
        # Constants for calculation based on the paper
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
        self.max_log_messages = 50  # Default value
        self._build_ui()
        self._window.visible = True
    

    def _build_ui(self):
        """Build the UI for the extension"""

        # Build the UI
        try:
            with self._window.frame:
                with ui.VStack(spacing=5):

                    # Detected sensors section
                    with ui.CollapsableFrame("Sensors", height=0):
                        with ui.VStack(spacing=5):
                            with ui.HStack(spacing=5):
                                ui.Label("Select the Robot from the Stage, and click:")
                                self.refresh_sensors_btn = ui.Button("Refresh Sensor List", clicked_fn=self._refresh_sensor_list, height=36)
                            self.selected_robot_label = ui.Label("(No robot selected)", width=0, style = {"color": cl("#FF0000")})
                            self.sensor_list = ui.ScrollingFrame(
                                height=250,
                                style={"border_width": 1, "border_color": 0xFF0000FF, "border_radius": 3}
                            )

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
                                self.max_messages_field = ui.IntField(width=60, height=18, )
                                self.max_messages_field.model.set_value(self.max_log_messages)
                                self.max_messages_field.model.add_value_changed_fn(self._update_max_log_messages)
                                self.clear_log_btn = ui.Button("Clear Log", width=80, clicked_fn=self._clear_log, height=18)

                            self.log_text = ui.ScrollingFrame(height=250)
                            with self.log_text:
                                self.log_label = ui.Label("")
        
        except Exception as e:
            print(f"Error building UI: {str(e)}")
            raise e


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
        if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
            selection = self._usd_context.get_selection().get_selected_prim_paths()

            if not selection:
                self._log_message("Error: No prim selected")
                self.selected_robot_label.text = "(No robot selected)"
                self.selected_robot_label.style = {"color": cl("#FF0000")}
                return
            if len(selection) > 1:
                self._log_message("Error: Multiple prims selected")
                self.selected_robot_label.text = "(Please only select one prim)"
                self.selected_robot_label.style = {"color": cl("#FFFF00")}
                return
            self.robot = selection[0]
            self.selected_robot_label.text = f"Selected: {self.robot}"
            self.selected_robot_label.style = {"color": cl("#00FFFF")}
        
        self._log_message(f"Found Robot: {len(self.robot)}")

    # def _on_timeline_event(self, event):
    #     # if event.type == int(omni.timeline.TimelineEventType.PLAY):
    #     #     if not self._physx_subscription:
    #     #         self._physx_subscription = self._physxIFace.subscribe_physics_step_events(self._on_physics_step)
    #     # elif event.type == int(omni.timeline.TimelineEventType.STOP):
    #     #     self._physx_subscription = None

    #     # self.ui_builder.on_timeline_event(event)
    #     pass

    def _refresh_robot(self, levels=2):
        """Find all potential robots in the scene and update the dropdown"""
        self._log_message("Selecting robot...")

        # Get the current stage
        stage = get_current_stage()
        if not stage:
            self._log_message("Error: No stage is loaded")
            return

        selection = self._usd_context.get_selection().get_selected_prim_paths()

        if not selection:
            self._log_message("Error: No prim selected")
            return
        if len(selection) > 1:
            self._log_message("Error: Multiple prims selected")
            return
        self.robot = selection[0]
        
        self._log_message(f"Found {len(self.robot)} potential robots")

    def _update_object_weight(self, obj_type: str, weight: float):
        """Update the weight of a specific object type"""
        self.target_objects[obj_type]["weight"] = weight
        self._log_message(f"Updated weight for {obj_type} to {weight}")

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
        
        # Find camera and LiDAR sensors
        self.detected_cameras = self._find_cameras(stage, self.robot)
        self.detected_lidars = self._find_lidars(stage, self.robot)
        
        self._log_message(f"Found {len(self.detected_cameras)} cameras and {len(self.detected_lidars)} LiDARs")
        
        # Update the sensor list UI
        self._update_sensor_list_ui()
    
    def _update_sensor_list_ui(self):
        """Update the sensor list UI with the detected sensors"""
        # Clear the current sensor list
        self.sensor_list.clear()
        
        # Create a new UI for the sensor list
        with self.sensor_list:
            with ui.VStack(spacing=5):
                if not self.detected_cameras and not self.detected_lidars:
                    ui.Label("No sensors detected")
                else:
                    # Display cameras
                    if self.detected_cameras:
                        with ui.CollapsableFrame(f"Cameras: {len(self.detected_cameras)}", height=0, style={"border_color": cl("#FF00FF")}, collapsed=False):
                            with ui.VStack(spacing=5):
                                for idx, camera in enumerate(self.detected_cameras):
                                    with ui.CollapsableFrame(f"{idx+1}. {camera['name']}", height=0, style={"border_color": cl("#FFFFFF")}, collapsed=True):
                                        with ui.VStack(spacing=2):
                                            pos, rot = camera["transform"]
                                            ui.Label(f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                                            ui.Label(f"HFOV: {math.degrees(camera['hfov']):.1f}°")
                                            ui.Label(f"Resolution: {camera['resolution'][0]}x{camera['resolution'][1]}")
                                            ui.Label(f"Focal Length: {camera['focal_length']:.2f} mm")
                                            
                                            # Show additional properties if available
                                            if 'properties' in camera:
                                                with ui.CollapsableFrame("Additional Properties", height=0, collapsed=True):
                                                    with ui.VStack(spacing=2):
                                                        for prop_name, prop_value in camera['properties'].items():
                                                            ui.Label(f"{prop_name}: {prop_value}")
                    
                    # Display LiDARs
                    if self.detected_lidars:
                        with ui.CollapsableFrame(f"LiDARs: {len(self.detected_lidars)}", height=0):
                            with ui.VStack(spacing=5):
                                for idx, lidar in enumerate(self.detected_lidars):
                                    with ui.CollapsableFrame(f"{idx+1}. {lidar['name']}", height=0, style={"border_color": cl("#FFFFFF")}, collapsed=True):
                                        with ui.VStack(spacing=2):
                                            pos, rot = lidar["transform"]
                                            ui.Label(f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                                            ui.Label(f"Channels: {lidar['channels']}")
                                            ui.Label(f"Vertical FOV: {math.degrees(lidar['fov_vertical']):.1f}°")
                                            ui.Label(f"Horizontal FOV: {math.degrees(lidar['fov_horizontal']):.1f}°")
                                            ui.Label(f"Max Range: {lidar['max_range']:.1f} m")
                                            if 'model' in lidar and lidar['model'] != "Unknown":
                                                ui.Label(f"Model: {lidar['model']}")
                                                
                                            # Show additional properties if available
                                            if 'properties' in lidar and lidar['properties']:
                                                with ui.CollapsableFrame("Additional Properties", height=0, collapsed=True):
                                                    with ui.VStack(spacing=2):
                                                        for prop_name, prop_value in lidar['properties'].items():
                                                            ui.Label(f"{prop_name}: {prop_value}")
    
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
    
    def _find_cameras(self, stage, robot_path) -> List[Dict]:
        """Find cameras that are descendants of the selected robot"""
        cameras = []
        
        # Get the root prim for filtering
        root_prim = stage.GetPrimAtPath(robot_path)
        if not root_prim:
            self._log_message(f"Error: Could not find prim at path {robot_path}")
            return cameras
        
        # Search for camera prims under the selected robot
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Camera):
                # Skip editor cameras if a specific robot is selected
                name = prim.GetName()
                if robot_path != "/" and any(editor_cam in name for editor_cam in ["OmniverseKit", "editorCamera"]):
                    continue
                
                # The rest of the camera detection code remains unchanged
                # [existing camera extraction code...]
                cam = UsdGeom.Camera(prim)
                
                # Get camera properties
                focal_length = cam.GetFocalLengthAttr().Get()
                h_aperture = cam.GetHorizontalApertureAttr().Get()
                
                # Calculate horizontal field of view
                hfov = 2 * math.atan(h_aperture / (2 * focal_length))
                
                # Try to get resolution from prim attributes or use default
                resolution = (1920, 1080)  # Default resolution
                
                # Try to get additional attributes if available
                try:
                    if prim.HasAttribute("resolution"):
                        res_attr = prim.GetAttribute("resolution")
                        if res_attr.IsValid():
                            res_value = res_attr.Get()
                            if isinstance(res_value, tuple) and len(res_value) == 2:
                                resolution = res_value
                except Exception as e:
                    self._log_message(f"Error getting resolution for camera {prim.GetName()}: {str(e)}")
                
                camera_data = {
                    "name": prim.GetName(),
                    "path": str(prim.GetPath()),
                    "transform": self._get_world_transform(prim),
                    "focal_length": focal_length,
                    "hfov": hfov,
                    "resolution": resolution,
                    "h_aperture": h_aperture,
                    "properties": {
                        "focal_length (mm)": focal_length,
                        "horizontal_aperture (mm)": h_aperture,
                        "horizontal_fov (degrees)": math.degrees(hfov),
                        "resolution": f"{resolution[0]}x{resolution[1]}"
                    }
                }
                
                cameras.append(camera_data)
                self._log_message(f"Found camera: {camera_data['name']} with HFOV: {math.degrees(camera_data['hfov']):.2f}°")
                
        return cameras

    def _find_lidars(self, stage, robot_path) -> List[Dict]:
        """Find LiDARs that are descendants of the selected robot"""
        lidars = []
        
        # Get the root prim for filtering
        root_prim = stage.GetPrimAtPath(robot_path)
        if not root_prim:
            self._log_message(f"Error: Could not find prim at path {robot_path}")
            return lidars
        
        # Look for LiDAR-type prims under the selected robot
        lidar_prim_names = ["lidar", "Lidar", "LiDAR", "LIDAR"]
        lidar_prim_types = ["LidarPrim", "SensorPrim", "RangeSensor"]
        
        for prim in Usd.PrimRange(root_prim):
            
            name = prim.GetName()
            prim_type = prim.GetTypeName()
            
            is_lidar = (
                any(lidar_name in name.lower() for lidar_name in lidar_prim_names) or
                str(prim_type) in lidar_prim_types
            )
            
            if is_lidar:

                properties = {}

                try:
                    fov_vertical = self._get_prim_attribute(prim, "verticalFov", math.radians(30.0)) # degrees
                    fov_horizontal = self._get_prim_attribute(prim, "horizontalFov", math.radians(360.0)) # degrees
                    res_vertical = self._get_prim_attribute(prim, "verticalResolution", 0.4) # degrees between rays
                    res_horizontal = self._get_prim_attribute(prim, "horizontalResolution", 4.0)  # degrees between rays
                    max_range = self._get_prim_attribute(prim, "maxRange", 100.0) # meters
                    min_range = self._get_prim_attribute(prim, "minRange", 0.0) # meters
                    channels = self._get_prim_attribute(prim, "channels", 64)
                    
                except Exception as e:
                    self._log_message(f"Error extracting LiDAR properties for {name}: {str(e)}")
                
                # Extract model info etc...
                # [existing model detection code]
                
                lidar_data = {
                    "name": name,
                    "path": str(prim.GetPath()),
                    "transform": self._get_world_transform(prim),
                    "channels": channels,
                    "fov_vertical": fov_vertical,
                    "fov_horizontal": fov_horizontal,
                    "max_range": max_range,
                    "model": name,
                    "properties": properties
                }
                
                lidars.append(lidar_data)
                self._log_message(f"Found LiDAR: {lidar_data['name']} ({name}, {channels} channels)")
        
        return lidars
    
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