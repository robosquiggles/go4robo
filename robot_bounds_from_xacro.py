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