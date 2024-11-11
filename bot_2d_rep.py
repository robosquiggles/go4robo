import shapely
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
import numpy as np
    
class FOV2D:
    def __init__(self, polygon: Polygon, focal_point: tuple[float] = (0, 0), color: str = 'blue', rotation: float = 0):
        self.polygon = polygon
        self.focal_point = focal_point
        self.color = color
        self.rotation = rotation
        self.translate(*self.focal_point)
        self.rotate(self.rotation)

    def plot_fov(self, whole_plot=False, show=False):
        x, y = self.polygon.exterior.xy
        plt.plot(x, y, color=self.color)
        plt.fill(x, y, alpha=0.5, color=self.color, edgecolor='none')
        plt.scatter(*self.focal_point, color=self.color)  # Add a dot at the focal point
        if whole_plot:
            plt.title('Field of View')
            plt.xlabel('Distance (m)')
            plt.ylabel('Distance (m)')
            plt.grid(True)
            plt.axis('equal')
        if show:
            plt.show()

    def translate(self, dx, dy):
        self.polygon = shapely.affinity.translate(self.polygon, xoff=dx, yoff=dy)
        return self

    def rotate(self, angle):
        self.polygon = shapely.affinity.rotate(self.polygon, angle, origin=self.focal_point, use_radians=False)
        return self

class FOV2D_Simple(FOV2D):
    def __init__(self, hfov: float, distance: float, color: str = 'blue', focal_point: tuple[float] = (0, 0), rotation: float = 0):
        half_angle = np.radians(hfov / 2)
        points = [
            (0, 0),  # origin
            (distance * np.cos(np.pi/2 - half_angle), distance * np.sin(np.pi/2 - half_angle)),  # left edge
            (distance * np.cos(np.pi/2 + half_angle), distance * np.sin(np.pi/2 + half_angle))  # right edge
        ]
        num_points = 100  # number of points to create the arc
        angles = np.linspace(-half_angle, half_angle, num_points)
        arc_points = [(distance * np.cos(angle), distance * np.sin(angle)) for angle in angles]
        fov_points = [points[0]] + arc_points + [points[0]]
        fov_polygon = Polygon(fov_points)
        super().__init__(fov_polygon, focal_point, color, rotation+90)
    

class SimpleBot2d:
    def __init__(self, shape:shapely.geometry.Polygon, sensor_coverage_requirement, bot_color:str="purple", sensor_pose_constraint=None):
        self.shape = shape
        self.color = bot_color
        self.sensors = []
        if type(sensor_pose_constraint) is not list:
            self.sensor_pose_constraint = [sensor_pose_constraint]
        else:
            self.sensor_pose_constraint = sensor_pose_constraint
        
        if type(sensor_coverage_requirement) is not list:
            self.sensor_coverage_requirement = [sensor_coverage_requirement]
        else:
            self.sensor_coverage_requirement = sensor_coverage_requirement

    def add_sensor_2d(self, sensor:FOV2D):
        self.sensors.append(sensor)

    def add_sensors_2d(self, sensors:list[FOV2D]):
        for sensor in sensors:
            self.sensors.append(sensor)

    def plot_bot(self, show_constraint=True, show_coverage_requirement=True, show_sensors=True):
        x, y = self.shape.exterior.xy
        plt.fill(x, y, color=self.color)
        plt.plot(x, y, color=self.color)
        
        if show_constraint and self.sensor_pose_constraint:
            for constraint in self.sensor_pose_constraint:
                cx, cy = constraint.exterior.xy
                plt.fill(cx, cy, color='green', alpha=0.25)
        
        if show_coverage_requirement and self.sensor_coverage_requirement:
            for requirement in self.sensor_coverage_requirement:
                rx, ry = requirement.exterior.xy
                plt.plot(rx, ry, color='black', linestyle='dotted')
        
        if show_sensors and self.sensors:
            for sensor in self.sensors:
                sensor.plot_fov(whole_plot=False)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


#################################
##  This part doesn't work yet ##
#################################

import os
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt

# Define a dictionary to map package names to their base directories
package_paths = {
    'jackal_description': './_datasets/robo_forms/jackal/jackal_description'
}

def resolve_package_uri(uri):
    if uri.startswith('package://'):
        parts = uri.split('/')
        package_name = parts[2]
        relative_path = '/'.join(parts[3:])
        if package_name in package_paths:
            return os.path.join(package_paths[package_name], relative_path)
    return uri

def plot_robot_outline_from_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    polygons = []
    
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            for geometry in visual.findall('geometry'):
                for mesh in geometry.findall('mesh'):
                    filename = mesh.get('filename')
                    if filename:
                        # Resolve package:// URI to actual file path
                        filename = resolve_package_uri(filename)
                        
                        # Assuming the mesh file is in STL format and contains 2D coordinates
                        # You might need to adjust this part based on the actual format and content of your mesh files
                        with open(filename, 'r') as f:
                            points = []
                            for line in f:
                                if line.startswith('vertex'):
                                    _, x, y, _ = line.split()
                                    points.append((float(x), float(y)))
                            if points:
                                polygons.append(Polygon(points))
    
    if polygons:
        multi_polygon = MultiPolygon(polygons)
        x, y = multi_polygon.exterior.xy
        plt.plot(x, y)
        plt.fill(x, y, alpha=0.5, fc='r', ec='black')
        plt.title('2D Top-Down View of Robot Outline')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    else:
        print("No valid polygons found in the URDF file.")

# Call the function with the URDF file path
# plot_robot_outline_from_urdf('./_datasets/robo_forms/jackal/jackal_description/urdf/jackal.urdf.xacro')