import open3d as o3d
import numpy as np

# Minimal point cloud visualization test
# Generate a synthetic point cloud similar to your main script
np.random.seed(42)
points = np.random.uniform(-1, 1, size=(2000, 3))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

print("[TEST] Visualizing ONLY the synthetic point cloud with draw_plotly...")
try:
    o3d.visualization.draw_plotly([pcd])
    print("[RESULT] Point cloud visualization succeeded.")
except Exception as e:
    print("[FAIL] Point cloud visualization crashed:", e)
