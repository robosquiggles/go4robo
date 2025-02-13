import os, sys
import shutil
import numpy as np
# import pybullet as p
# import pybullet_data
import subprocess
import threading
from tqdm import tqdm

TODO = f"❗❗ TODO ❗❗"

# Set environment variables and paths
os_env = os.environ.copy()
DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(DATASETS_DIR, "raw")  # Where all URDFs are stored
SENSOR_MESH_DIR = os.path.join(DATASETS_DIR, "sensors")  # Where sensor meshes will be copied
ROBOTS_MESH_DIR = os.path.join(DATASETS_DIR, "robots")  # Where robot meshes will be copied
ROBOTS_NPY = os.path.join(ROBOTS_MESH_DIR, "robots_data.npy")  # Sensor data output file
SENSORS_NPY = os.path.join(SENSOR_MESH_DIR, "sensors_data.npy")  # Sensor data output file

print(f"➡️ 📂 DATASETS_DIR: {DATASETS_DIR}"
      f"\n➡️ 📂 RAW_DIR: {RAW_DIR}
      f"\n➡️ 📂 SENSOR_MESH_DIR: {SENSOR_MESH_DIR}"
      f"\n➡️ 📂 ROBOTS_MESH_DIR: {ROBOTS_MESH_DIR}
      f"\n➡️ 📂 ROBOTS_NPY: {ROBOTS_NPY}"
      f"\n➡️ 📂 SENSORS_NPY: {SENSORS_NPY}")

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

def classify_description_file(file_path):
    """Classifies URDF/XACRO files as robot, sensor, or other based on content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
            if any(tag in content for tag in ["<robot", "base_link", "joint"]):
                return "robot"
            elif any(tag in content for tag in ["<sensor", "lidar", "camera", "imu"]):
                return "sensor"
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
    return "other"

def build_dependency_graph(package_prefix):
    """Builds a dependency graph of URDF/XACRO files."""
    dependencies = {}
    if not os.path.isdir(package_prefix):
        return dependencies
    
    for root, _, files in os.walk(package_prefix):
        for file in files:
            if file.endswith(".xacro") or file.endswith(".urdf"):
                file_path = os.path.join(root, file)
                dependencies[file_path] = set()
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().lower()
                        for line in content.split("\n"):
                            if ("<xacro:include" in line or "<include" in line) and "file=" in line:
                                try:
                                    included_file = line.split("file=")[1].split("\"")[1]
                                    dependencies[file_path].add(os.path.join(root, included_file))
                                except IndexError:
                                    print(f"Warning: Malformed include statement in {file_path}: {line.strip()}")
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
    return dependencies

def find_top_level_files(dependencies):
    """Identifies top-level URDF/XACRO files by finding files that are not included by others."""
    included_files = {file for includes in dependencies.values() for file in includes}
    return [file for file in dependencies if file not in included_files]

def find_description_files(package_name, package_prefix):
    """Searches for .xacro and .urdf files within a package, classifies them, and filters top-level files."""
    dependencies = build_dependency_graph(package_prefix)
    return find_top_level_files(dependencies)

def main():
    ros2_packages = get_ros2_packages()
    top_level_files = []
    
    for package in tqdm(ros2_packages, desc="Scanning packages"):
        package_prefix = get_package_prefix(package)
        description_files = find_description_files(package, package_prefix)
        top_level_files.extend(description_files)
    
    if top_level_files:
        print("\nFound the following top-level URDF/XACRO files:")
        for file in top_level_files:
            print(f"  - {file}")
    else:
        print("No relevant top-level URDF/XACRO files found.")




    ############################## Extract robot and sensor data ################################
    print(f"🚀 Starting data extraction!!")

    print(TODO)

    print(f"✅ Finished processing {len(top_level_files)} top-level URDFs!")
    print(f"  📂 Robot data saved at {ROBOTS_NPY}")
    print(f"  📂 Robot meshes saved at {ROBOTS_MESH_DIR}")

    print(f"✅ Finished processing {TODO} sensors!")
    print(f"  📂 Sensor data saved at {SENSORS_NPY}")
    print(f"  📂 Sensor meshes saved at {SENSOR_MESH_DIR}")


if __name__ == "__main__":
    main()
