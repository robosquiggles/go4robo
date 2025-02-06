import os, sys
import shutil
import numpy as np
# import pybullet as p
# import pybullet_data
import subprocess
import threading
from tqdm import tqdm

def main():

    # Set environment variables and paths
    os_env = os.environ.copy()
    DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DIR = os.path.join(DATASETS_DIR, "raw")  # Where all URDFs are stored
    SENSOR_MESH_DIR = os.path.join(DATASETS_DIR, "sensors")  # Where sensor meshes will be copied
    ROBOTS_MESH_DIR = os.path.join(DATASETS_DIR, "robots")  # Where robot meshes will be copied
    ROBOTS_NPY = os.path.join(ROBOTS_MESH_DIR, "robots_data.npy")  # Sensor data output file
    SENSORS_NPY = os.path.join(SENSOR_MESH_DIR, "sensors_data.npy")  # Sensor data output file

    print(f"➡️ DATASETS_DIR: {DATASETS_DIR}")
    print(f"  ➡️ RAW_DIR: {RAW_DIR}")
    print(f"  ➡️ SENSOR_MESH_DIR: {SENSOR_MESH_DIR}")
    print(f"  ➡️ ROBOTS_MESH_DIR: {ROBOTS_MESH_DIR}")
    print(f"  ➡️ ROBOTS_NPY: {ROBOTS_NPY}")
    print(f"  ➡️ SENSORS_NPY: {SENSORS_NPY}")

    #Find ROS2
    try:
        ROS_DIR = os.path.join(os.sep, 'opt', 'ros', os_env['ROS_DISTRO'])
        ROS_SRC_DIR = os.path.join(ROS_DIR, 'src')
        subprocess.run(["ls", ROS_DIR], check=True)
        print(f"➡️ ROS_DIR: {ROS_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"⚠ ROS2 not found: {e}")
        user_input = input("Press enter to continue or 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Exiting...")
            sys.exit(1)

    def run_ros2_command(command):
        """Runs a ROS 2 CLI command with proper sourcing."""
        source_path = os.path.join(ROS_DIR, "setup.bash")
        source_command = f"source {source_path} && " + command
        result = subprocess.run(["bash", "-c", source_command], capture_output=True, text=True)
        return result

    def get_ros2_packages():
        """Returns a list of installed ROS 2 package names."""
        result = run_ros2_command("ros2 pkg list")
        return result.stdout.strip().split("\n")

    def get_package_prefix(package_name):
        """Returns the install path of a ROS 2 package."""
        result = run_ros2_command(f"ros2 pkg prefix {package_name}")
        return result.stdout.strip() if result.returncode == 0 else None

    def find_description_files(package_name, package_prefix):
        """Searches for .xacro and .urdf files within a package."""
        description_files = []
        if not package_prefix or not os.path.isdir(package_prefix):
            return description_files
        
        for root, _, files in os.walk(package_prefix):
            for file in files:
                if file.endswith(".xacro") or file.endswith(".urdf"):
                    description_files.append(os.path.join(root, file))
        
        return description_files

    # # Ensure output directories exist
    # os.makedirs(SENSOR_MESH_DIR, exist_ok=True)
    # os.makedirs(os.path.dirname(ROBOTS_NPY), exist_ok=True)

    # # Start PyBullet in headless mode
    # p.connect(p.DIRECT)
    # p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Ensure default assets are available

    # Function to get absolute transformation matrix
    def get_absolute_transform(robot_id, link_index):
        """Returns the absolute 4x4 transformation matrix of a given link."""
        pos, orn = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)[:2]
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = np.array(pos)
        
        return transform

    # Function to copy mesh files
    def copy_mesh(mesh_path):
        """Copies the mesh file to SENSOR_MESH_DIR and returns the new path."""
        if not mesh_path or mesh_path.startswith("package://"):
            return None  # Ignore ROS package paths

        abs_mesh_path = os.path.abspath(mesh_path)
        if not os.path.exists(abs_mesh_path):
            print(f"⚠ Warning: Mesh not found at {abs_mesh_path}")
            return None

        # Copy to sensor mesh directory
        dest_path = os.path.join(SENSOR_MESH_DIR, os.path.basename(mesh_path))
        shutil.copy2(abs_mesh_path, dest_path)
        return dest_path

    def has_robot_name(urdf_path):
        with open(urdf_path, "r") as f:
            for line in f:
                if "<robot" in line and "name=" in line:
                    return True
        return False

    def convert_xacro_to_urdf(xacro_path):
        """Converts a .urdf.xacro file to a .urdf file using ROS2 xacro."""
        urdf_path = xacro_path.replace(".xacro", "")
        try:
            print(f"⚠ Xacro encountered at {xacro_path} Attempting to Convert to .urdf")
            subprocess.run(
                ["ros2", "run", "xacro", "xacro", xacro_path, "-o", urdf_path],
                # ["xacro", xacro_path, "-o", urdf_path],
                check=True
            )
            return urdf_path
        except subprocess.CalledProcessError as e:
            print(f"⚠ Xacro conversion failed for {xacro_path}: {e}")
            return None
        
    print(f"🔍 Searching for installed robot and sensor descriptions")
    robots = []
    sensors = []

    ros2_packages = get_ros2_packages()
    descriptions = {}
    
    for package in tqdm(ros2_packages, desc="Scanning packages"):
        package_prefix = get_package_prefix(package)
        description_files = find_description_files(package, package_prefix)
        
        if description_files:
            descriptions[package] = description_files
    
    if descriptions:
        print("\nFound the following robot and sensor descriptions:")
        for package, files in descriptions.items():
            print(f"\nPackage: {package}")
            for file in files:
                print(f"  - {file}")
    else:
        print("No robot or sensor descriptions found.")

    print(f"🚀 Starting data extraction!!")

    print(f"✅ Finished processing {len(robots)} robots!")
    print(f"  📂 Robot data saved at {ROBOTS_NPY}")
    print(f"  📂 Robot meshes saved at {ROBOTS_MESH_DIR}")

    print(f"✅ Finished processing {len(sensors)} sensors!")
    print(f"  📂 Sensor data saved at {SENSORS_NPY}")
    print(f"  📂 Sensor meshes saved at {SENSOR_MESH_DIR}")


if __name__ == "__main__":
    main()