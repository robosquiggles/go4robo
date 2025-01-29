import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
print("Setting PATH to script directory:", script_dir)
sys.path.append(script_dir)

import bpy

from bot_3d_rep import *


######### SENSORS ############
print("SENSORS")
sensor_mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.075)
sensor_mesh.transform(TF.translation_matrix(-.05,-.05,-.05))

sensor = FOV3D(h_fov=np.radians(90), 
            v_fov=np.radians(70), 
            distance=5, 
            cost=80.00, 
            name="toy_sensor", 
            color=(0,130,170), 
            body=Mesh(sensor_mesh), 
            focal_point=(0, 0, 0))

sensor.transform(TF.translation_matrix(.5,.5,0))
# sensor.transform(TF.translation_matrix(0,0,.5))
##############################

######### BLENDER ############
print("ADDING TO BLENDER")
# Clear all objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Create a new mesh

sensor_meshes = sensor.get_viz_meshes(viz_fov=False, show_now=False)
for mesh in sensor_meshes:
    mesh.blender_show()

##############################
