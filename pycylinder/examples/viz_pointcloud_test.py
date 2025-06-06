import os
import sys
import numpy as np
import open3d as o3d

# Add the current directory to the path to allow importing local modules
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from the pycylinder package
from pycylinder.logger import get_logger

# Initialize logger
logger = get_logger()

# Minimal point cloud visualization test
# Generate a synthetic point cloud similar to your main script
np.random.seed(42)
points = np.random.uniform(-1, 1, size=(2000, 3))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

logger("[TEST] Visualizing ONLY the synthetic point cloud with draw_plotly...")
try:
    o3d.visualization.draw_plotly([pcd])
    logger("[RESULT] Point cloud visualization succeeded.")
except Exception as e:
    logger(f"[FAIL] Point cloud visualization crashed: {e}")
    raise  # Re-raise the exception to see the full traceback
