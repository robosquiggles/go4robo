import shapely
from shapely.geometry import Polygon

import pointpats

from matplotlib.path import Path
from matplotlib.patches import PathPatch

import matplotlib.pyplot as plt
import numpy as np
import random

def plot_polygon_with_holes(polygon, **kwargs):
    """
    Plots a polygon with holes using Matplotlib.
    Parameters:
        polygon (shapely.geometry.Polygon): The polygon to be plotted, which may contain holes.
        **kwargs: Additional keyword arguments to be passed to the PathPatch constructor.
    Returns:
        None
    """

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
    def __init__(self, fov_polygon: Polygon, cost:float, bounds_polygon:Polygon=None, focal_point:tuple[float]=(0, 0), color: str = 'purple', rotation: float = 0):
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
        self.focal_point = (0,0)
        self.color = color
        self.rotation = rotation
        self.translate(*focal_point)
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
        # plt.plot(x, y, color=self.color)
        plt.fill(x, y, alpha=0.5, color=self.color, edgecolor='none')
        if self.bounds is not None:
            bx, by = self.bounds.exterior.xy
            plt.plot(bx, by, color=self.color)
            plt.fill(bx, by, alpha=0.8, color=self.color, edgecolor='none')
        plt.scatter(*self.focal_point, color=self.color,marker='.')  # Add a dot at the focal point
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
        self.focal_point = (self.focal_point[0] + dx, self.focal_point[1] + dy)
        return self
    
    def set_translation(self, x, y):
        """Sets the absolute x and y of the focal point of the sensor."""
        dx = x - self.focal_point[0]
        dy = y - self.focal_point[1]
        self.translate(dx, dy)

    def rotate(self, angle):
        """Rotates the FOV by the given angle (+ is ccw). Also returns the self (FOV2D object) for quick use."""
        center = self.focal_point
        self.fov = shapely.affinity.rotate(self.fov, angle, origin=center, use_radians=False)
        if self.bounds is not None:
            self.bounds = shapely.affinity.rotate(self.bounds, angle, origin=center, use_radians=False)
        self.rotation = (self.rotation + angle) % 360
        return self
    
    def set_rotation(self, angle):
        """Sets the absolute rotation of the FOV to the given angle."""
        self.rotate(angle - self.rotation)
    
    def contained_in(self, fov:Polygon):
        """Returns whether or not the sensor is within the given polygon."""
        if self.bounds is not None:
            return fov.contains(shapely.geometry.Point(self.focal_point)) and fov.contains(self.bounds)
        else:
            return fov.contains(shapely.geometry.Point(self.focal_point))


class FOV2D_Simple(FOV2D):
    def __init__(self, hfov: float, distance: float, cost:float, color: str = 'purple', focal_point: tuple[float] = (0, 0), rotation: float = 0, bounds_polygon:Polygon=None):
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
        self.bounds_polygon_xy = list(bounds_polygon.exterior.coords)
        self.half_angle = np.radians(hfov / 2)
        self.dist = distance
        points = [
            (0, 0),  # origin
            (distance * np.cos(np.pi/2 - self.half_angle), distance * np.sin(np.pi/2 - self.half_angle)),  # left edge
            (distance * np.cos(np.pi/2 + self.half_angle), distance * np.sin(np.pi/2 + self.half_angle))  # right edge
        ]
        num_points = 100  # number of points to create the arc
        angles = np.linspace(-self.half_angle, self.half_angle, num_points)
        arc_points = [(distance * np.cos(angle), distance * np.sin(angle)) for angle in angles]
        fov_points = [points[0]] + arc_points + [points[0]]
        fov_polygon = shapely.affinity.rotate(Polygon(fov_points), 90, (0,0))
        super().__init__(fov_polygon=fov_polygon, cost=cost, focal_point=focal_point, color=color, rotation=rotation, bounds_polygon=bounds_polygon)

    def __eq__(self, other):
        if not isinstance(other, FOV2D_Simple):
            return False
        return (self.half_angle == other.half_angle and
                self.dist == other.dist and
                self.cost == other.cost and
                self.color == other.color and
                self.bounds_polygon_xy == other.bounds_polygon_xy)

    def __hash__(self):
        return hash((self.half_angle, self.dist, self.cost, self.color, self.bounds_polygon))
    

class SimpleBot2d:
    def __init__(self, shape:shapely.geometry.Polygon, sensor_coverage_requirement, bot_color:str="blue", sensor_pose_constraint=None):
        """
        Initialize a bot representation with a given shape, sensor coverage requirements, and optional color and sensor pose constraints.
        Args:
            shape (shapely.geometry.Polygon): The geometric shape representing the bot.
            sensor_coverage_requirement (list or shapely.geometry.Polygon): The required sensor coverage areas. If a single polygon is provided, it will be converted to a list.
            bot_color (str, optional): The color of the bot. Defaults to "blue".
            sensor_pose_constraint (list or optional): Constraints on the sensor poses. If a single constraint is provided, it will be converted to a list. Defaults to None.
        Attributes:
            shape (shapely.geometry.Polygon): The geometric shape representing the bot.
            color (str): The color of the bot.
            sensors (list): A list to store sensors associated with the bot.
            sensor_pose_constraint (list): Constraints on the sensor poses.
            sensor_coverage_requirement (list): The required sensor coverage areas with the bot's shape removed from each.
        """
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

        # Remove self.shape from any of the sensor_coverage_requirement shapes
        self.sensor_coverage_requirement = [
            req.difference(self.shape) for req in self.sensor_coverage_requirement
        ]

    def add_sensor_2d(self, sensor:FOV2D|None):
        """
        Adds a 2D sensor to the list of sensors. Only adds a sensor if it is not None.
        Parameters:
            sensor (FOV2D|None): The 2D sensor to be added (or None).
        Returns:
            bool: True if the sensor was added successfully, False otherwise.
        """
        if sensor is not None:
            self.sensors.append(sensor)
            return True
        return False

    def add_sensor_valid_pose(self, sensor:FOV2D, max_tries:int=25, verbose=False):
        """
        Adds a sensor to a valid location within the defined constraints.
        This method generates random points within the bounding box of the 
        sensor pose constraints and translates the sensor to these points. 
        It checks if the new sensor pose is valid and, if so, adds the sensor 
        to the list of sensors.
        Args:
            sensor (FOV2D): The sensor to be added, which will be translated 
                    to a valid location within the constraints.
        """
        for i in range(max_tries):
            x, y = pointpats.random.poisson(self.sensor_pose_constraint[0], size=1)
            rotation = np.degrees(np.arctan2(y, x)) - 90

            sensor.set_translation(x, y)
            sensor.set_rotation(rotation) #this isn't quite right but good enough
            
            if self.is_valid_sensor_pose(sensor):
                self.add_sensor_2d(sensor)
                break
            if i == max_tries and verbose:
                print(f"Did not find a valid sensor pose in {max_tries} tries. Quitting!")
        return sensor

    def remove_sensor_by_index(self, index):
        """Removes a sensor from the sensors list by its index."""
        del self.sensors[index]

    def clear_sensors(self):
        """Removes all sensors from the sensors list."""
        self.sensors = []

    def add_sensors_2d(self, sensors:list[FOV2D]):
        """
        Adds a list of 2D sensors to the current object.
        Args:
            sensors (list[FOV2D]): A list of FOV2D sensor objects to be added.
        """
        for sensor in sensors:
            self.add_sensor_2d(sensor)

    def plot_bot(self, show_constraint=True, show_coverage_requirement=True, show_sensors=True, title=None):
        """
        Plots the robot's shape, sensor constraints, coverage requirements, and sensors on a 2D plot.
        Parameters:
        -----------
        show_constraint : bool, optional
            If True, plots the sensor pose constraints (default is True).
        show_coverage_requirement : bool, optional
            If True, plots the sensor coverage requirements (default is True).
        show_sensors : bool, optional
            If True, plots the sensors' fields of view (default is True).
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.
        """

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
        if title is not None:
            plt.title(title)
        plt.show()
        return fig
    
    def is_valid_sensor_pose(self, sensor:FOV2D, verbose=False):
        """
        Verifies if the sensor's position is within the defined 
        sensor pose constraints and does not intersect with any existing sensors.

        Parameters:
        sensor (FOV2D): The sensor object whose pose needs to be validated.
        verbose (bool): If True, prints detailed information about why a sensor 
                        pose is invalid. Default is False.

        Returns:
        bool: True if the sensor pose is valid, False otherwise.
        """

        # Check if the sensor is within the sensor pose constraint
        if not any(constraint.contains(sensor.bounds) for constraint in self.sensor_pose_constraint):
            if verbose:
                print(f"A Sensor at {sensor.focal_point} is invalid because it is outside of physical constraints.")
            return False

        # Check if the sensor does not intersect with any existing sensors
        for existing_sensor in self.sensors:
            if sensor.bounds.intersects(existing_sensor.bounds):
                if verbose:
                    print(f"A Sensor at {sensor.focal_point} is invalid because it intersects with the sensor at {existing_sensor.focal_point}.")
                return False

        return True

    def is_valid_pkg(self, verbose=False):
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
            if verbose:
                print("Checking validity of", sensor, " in ", self.sensor_pose_constraint)
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
        if valid and verbose:
            print("Bot Sensor Package is Valid!") 
        return valid
    
    def get_sensor_coverage(self):
        """
        Calculate the coverage percentage of the sensors based on the required coverage area.
        This method computes the total area covered by all sensors and compares it to the required 
        coverage area. It returns the ratio of the covered area to the required area as a percentage.
        Returns:
            float: The coverage percentage of the sensors. Returns 0.0 if there is no coverage requirement.
        """

        if not self.sensor_coverage_requirement:
            return 0.0

        total_coverage = shapely.geometry.Polygon()
        for sensor in self.sensors:
            if sensor is not None:
                total_coverage = total_coverage.union(sensor.fov)
        total_coverage = total_coverage.intersection(self.sensor_coverage_requirement[0])
        coverage_area = total_coverage.area
        requirement_area = self.sensor_coverage_requirement[0].area

        return (coverage_area / requirement_area)
    
    def get_pkg_cost(self):
        return sum([sensor.cost for sensor in self.sensors if sensor is not None])
    
    def convert_to_1D():
        return None