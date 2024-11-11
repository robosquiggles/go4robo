import shapely
from shapely.geometry import Polygon

from matplotlib.path import Path
from matplotlib.patches import PathPatch

import matplotlib.pyplot as plt
import numpy as np

def plot_polygon_with_holes(polygon, **kwargs):
    exterior_coords = list(polygon.exterior.coords)
    codes = [Path.MOVETO] + [Path.LINETO] * (len(exterior_coords) - 1) + [Path.CLOSEPOLY]
    vertices = exterior_coords + [exterior_coords[0]]
    
    if polygon.interiors is not None:
        for interior in polygon.interiors:
            interior_coords = list(interior.coords)
            codes += [Path.MOVETO] + [Path.LINETO] * (len(interior_coords) - 1) + [Path.CLOSEPOLY]
            vertices += interior_coords + [interior_coords[0]]
    
    path = Path(vertices, codes)
    patch = PathPatch(path, **kwargs)
    plt.gca().add_patch(patch)
    
class FOV2D:
    def __init__(self, fov_polygon: Polygon, cost:float, bounds_polygon:Polygon=None, focal_point:tuple[float]=(0, 0), color: str = 'blue', rotation: float = 0):
        """
        Initialize a new instance of the class.
        Args:
            polygon (Polygon): The polygon object to be represented.
            focal_point (tuple[float], optional): The focal point for the polygon. Defaults to (0, 0).
            color (str, optional): The color of the polygon. Defaults to 'blue'.
            rotation (float, optional): The initial rotation angle of the polygon in degrees. Defaults to 0.
            bound_polygon (Polygon): The polygon object representing the physical bounds of the sensor.
        """
        self.bounds = bounds_polygon
        self.fov = fov_polygon
        self.focal_point = focal_point
        self.color = color
        self.rotation = rotation
        self.translate(*self.focal_point)
        self.rotate(self.rotation)
        self.cost=cost

    def plot_fov(self, whole_plot=False, show=False) -> bool:
        """
        Plots the field of view (FOV) of the object.
        Parameters:
            whole_plot (bool): If True, adds title, labels, grid, and sets axis to equal. Default is False.
            show (bool): If True, displays the plot. Default is False.
        Returns:
            None
        """

        x, y = self.fov.exterior.xy
        plt.plot(x, y, color=self.color)
        plt.fill(x, y, alpha=0.5, color=self.color, edgecolor='none')
        if self.bounds is not None:
            bx, by = self.bounds.exterior.xy
            plt.plot(bx, by, color=self.color)
            plt.fill(bx, by, alpha=0.8, color=self.color, edgecolor='none')
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
        """Translates the FOV by the given dx, dy. Also returns the self (FOV2D object) for quick use."""
        self.fov = shapely.affinity.translate(self.fov, xoff=dx, yoff=dy)
        if self.bounds is not None:
            self.bounds = shapely.affinity.translate(self.bounds, xoff=dx, yoff=dy)
        return self

    def rotate(self, angle):
        """Rotates the FOV by the given angle (+ is ccw). Also returns the self (FOV2D object) for quick use."""
        center = self.focal_point
        self.fov = shapely.affinity.rotate(self.fov, angle, origin=center, use_radians=False)
        if self.bounds is not None:
            self.bounds = shapely.affinity.rotate(self.bounds, angle, origin=center, use_radians=False)
        return self
    
    def contained_in(self, fov:Polygon):
        """Returns whether or not the sensor is within the given polygon."""
        if self.bounds is not None:
            return fov.contains(shapely.geometry.Point(self.focal_point)) and fov.contains(self.bounds)
        else:
            return fov.contains(shapely.geometry.Point(self.focal_point))


class FOV2D_Simple(FOV2D):
    def __init__(self, hfov: float, distance: float, cost:float, color: str = 'blue', focal_point: tuple[float] = (0, 0), rotation: float = 0, bounds_polygon:Polygon=None):
        """
        Initializes the 2D representation of a robot's field of view (FOV).
        Args:
            hfov (float): The horizontal field of view in degrees.
            distance (float): The distance from the origin to the edge of the FOV.
            color (str, optional): The color of the FOV representation. Defaults to 'blue'.
            focal_point (tuple[float], optional): The focal point of the FOV. Defaults to (0, 0).
            rotation (float, optional): The rotation angle of the FOV in degrees. Defaults to 0.
            bound_polygon (Polygon): The polygon object representing the physical bounds of the sensor.
        Attributes:
            fov_polygon (Polygon): The polygon representing the FOV.
        """

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
        fov_polygon = shapely.affinity.rotate(Polygon(fov_points), 90, (0,0))
        super().__init__(fov_polygon=fov_polygon, cost=cost, focal_point=focal_point, color=color, rotation=rotation, bounds_polygon=bounds_polygon)
    

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
        fig, ax = plt.subplots()
        plot_polygon_with_holes(self.shape, facecolor=self.color, alpha=0.5, edgecolor=self.color)

        if show_constraint and self.sensor_pose_constraint:
            for constraint in self.sensor_pose_constraint:
                plot_polygon_with_holes(constraint, facecolor='green', alpha=0.25)
        
        if show_coverage_requirement and self.sensor_coverage_requirement:
            for requirement in self.sensor_coverage_requirement:
                plot_polygon_with_holes(requirement, facecolor='none', edgecolor='black', linestyle='dotted')
        
        if show_sensors and self.sensors:
            for sensor in self.sensors:
                sensor.plot_fov(whole_plot=False)

        ax.set_aspect('equal', adjustable='box')
        plt.show()
        return fig

    def is_valid(self, verbose=True):
        """
        Check if the current configuration of sensors is valid.
        This method performs two checks:
        1. Ensures that all sensors are within the defined sensor pose constraints.
        2. Ensures that no two sensors intersect with each other.
        Returns:
            bool: True if the configuration is valid, False otherwise.
        """

        valid = True

        # Check if all sensors are within the sensor pose constraint
        for sensor in self.sensors:
            if not any(constraint.contains(sensor.bounds) for constraint in self.sensor_pose_constraint):
                valid = False
                if verbose:
                    print("Bot Sensor Package is invalid because sensor is outside of physical constraints.")
                break

        # Check if sensors do not touch each other
        for i, sensor1 in enumerate(self.sensors):
            for j, sensor2 in enumerate(self.sensors):
                if i != j and sensor1.bounds.intersects(sensor2.bounds):
                    valid = False
                    if verbose:
                        print("Sensor Package is invalid because sensors intersect.")
                    break
            if not valid:
                break
        if valid:
            print("Bot Sensor Package is Valid!") 
        return valid
    
    def get_sensor_coverage(self):
        if not self.sensor_coverage_requirement:
            return 0.0

        total_coverage = shapely.geometry.Polygon()
        for sensor in self.sensors:
            total_coverage = total_coverage.union(sensor.fov)
        total_coverage = total_coverage.intersection(self.sensor_coverage_requirement[0])
        coverage_area = total_coverage.area
        requirement_area = self.sensor_coverage_requirement[0].area

        return (coverage_area / requirement_area)
    
    def get_pkg_cost(self):
        return sum(sensor.cost for sensor in self.sensors)


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