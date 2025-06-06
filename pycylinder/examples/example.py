"""
Example: Run cylinder detection on a synthetic cylinder point cloud.
"""
import os
import sys
import numpy as np
import open3d as o3d

# Add the current directory to the path to allow importing local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from the pycylinder package
from pycylinder.detector import CylinderDetector
from pycylinder.logger import CylinderLogger, set_logger
from pycylinder.synthetic import generate_cylinder_point_cloud
from pycylinder.pointcloud import PointCloud

def main():
    # Set up console logger for debug output
    logger = CylinderLogger(mode='console')
    set_logger(logger)
    
    import numpy as np
    np.random.seed(42)  # For reproducible synthetic data
    # Generate a synthetic cylinder
    center = np.array([0, 0, 0])
    axis = np.array([0, 0, 1])
    radius = 0.5
    height = 2.0
    pcd = generate_cylinder_point_cloud(center, axis, radius, height, n_points=2000, noise=0.01)
    logger(f"Generated {np.asarray(pcd.points).shape[0]} points.")
    logger("\n[GROUND TRUTH CYLINDER]")
    logger(f"  Center: {center}")
    logger(f"  Axis:   {axis}")
    logger(f"  Radius: {radius}")
    logger(f"  Height: {height}")

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
    logger(f"Estimated {len(pcd.normals)} normals.")
    # Wrap in pycylinder PointCloud
    pc = PointCloud(pcd)
    # Run detection with relaxed parameters
    # --- Tighter detection parameters for improved selectivity ---
    # --- Moderate detection parameters for debugging synthetic data ---
    # --- Very permissive parameters for debugging region growing ---
    detector = CylinderDetector(
        pc,
        directions_samples=40,
        distance_threshold=0.05,    # More permissive
        normal_threshold=0.8,       # More permissive
        min_component_points=5,     # Allow tiny clusters for debug
        circle_residual=0.02,
        min_inliers=15,
        min_length=0.02,
        max_radius=2.0,
        angle_thresh=0.1,
        center_thresh=0.05,
        radius_thresh=0.05
    )


    logger("Running detection...")
    cylinders = detector.detect()
    logger(f"Detected {len(cylinders)} cylinders.")
    # Compare detected cylinders to ground truth
    def axis_angle_deg(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    gt_center = center
    gt_axis = axis / np.linalg.norm(axis)
    gt_radius = radius

    logger("\n[DETECTED CYLINDERS]")
    min_err = float('inf')
    best_idx = -1
    best_errs = None
    for i, cyl in enumerate(cylinders):
        center_dist = np.linalg.norm(cyl.center - gt_center)
        axis_err = axis_angle_deg(cyl.axis, gt_axis)
        radius_err = abs(cyl.radius - gt_radius)
        combined_err = center_dist + axis_err/10.0 + radius_err  # simple combined metric
        logger(f"Cylinder {i}: center={cyl.center}, axis={cyl.axis}, radius={cyl.radius}, inliers={len(cyl.inliers)}")
        logger(f"    [Error] center_dist={center_dist:.4f}, axis_angle_deg={axis_err:.2f}, radius_err={radius_err:.4f}, combined_err={combined_err:.4f}")
        if combined_err < min_err:
            min_err = combined_err
            best_idx = i
            best_errs = (center_dist, axis_err, radius_err)
    if best_idx >= 0:
        logger(f"\n[NEAREST DETECTED CYLINDER: {best_idx}]")
        cyl = cylinders[best_idx]
        logger(f"  center={cyl.center}, axis={cyl.axis}, radius={cyl.radius}, inliers={len(cyl.inliers)}")
        logger(f"  Errors: center_dist={best_errs[0]:.4f}, axis_angle_deg={best_errs[1]:.2f}, radius_err={best_errs[2]:.4f}")
    else:
        logger("\n[No detected cylinders]")

    # --- Visualize ground truth and best detected cylinder as meshes ---
    def create_cylinder_mesh(center, axis, radius, height, color=[1, 0, 0]):
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=30, split=4)
        mesh.paint_uniform_color(color)
        axis = axis / np.linalg.norm(axis)
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, axis)
        c = np.dot(z_axis, axis)
        if np.linalg.norm(v) < 1e-6:
            R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
        mesh.rotate(R, center=(0, 0, 0))
        mesh.translate(center - axis * height / 2)
        return mesh

    import numpy as np

    def has_invalid_geometry(geometry):
        if hasattr(geometry, 'points'):
            pts = np.asarray(geometry.points)
            if not np.all(np.isfinite(pts)):
                return True
        if hasattr(geometry, 'vertices'):
            verts = np.asarray(geometry.vertices)
            if not np.all(np.isfinite(verts)):
                return True
        return False

    # Minimal visualization attempt: visualize ground truth and best detected cylinder
    try:
        from open3d.visualization import draw_plotly
        meshes = []
        # Visualize ground truth cylinder (green)
        gt_mesh = create_cylinder_mesh(center, axis, radius, height, color=[0, 1, 0])
        meshes.append(gt_mesh)
        # Visualize best detected cylinder (blue)
        if best_idx >= 0:
            best_cyl = cylinders[best_idx]
            # Use detected cylinder's own height estimate
            # Estimate length from inlier projections if available
            pts = np.asarray(pcd.points)[best_cyl.inliers] if hasattr(best_cyl, 'inliers') and len(best_cyl.inliers) > 0 else None
            if pts is not None:
                rel = pts - best_cyl.center
                proj = np.dot(rel, best_cyl.axis)
                cyl_height = proj.max() - proj.min()
                if cyl_height < 1e-3:
                    cyl_height = height  # fallback
            else:
                cyl_height = height
            best_mesh = create_cylinder_mesh(best_cyl.center, best_cyl.axis, best_cyl.radius, cyl_height, color=[0, 0, 1])
            meshes.append(best_mesh)
        # Visualize point cloud (gray)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        draw_plotly([pcd] + meshes, width=800, height=600)
        logger("Visualization succeeded.")
    except Exception as e:
        logger(f"Visualization failed: {e}")

    # [CLEANUP] Removed old mesh visualization logic based on filtered_cylinders.
    # Visualization is now handled above with ground truth and best detected cylinder meshes.

    logger("\n[DIAG] Minimal visualization attempts:")
    # 1. Visualize only the point cloud
    logger("[TEST] Visualizing ONLY the point cloud...")
    try:
        o3d.visualization.draw_plotly([pcd])
        logger("[RESULT] Point cloud visualization succeeded.")
    except Exception as e:
        logger(f"[FAIL] Point cloud visualization crashed: {e}")

    # 2. Visualize only the ground truth mesh (if available)
    logger("[TEST] Visualizing ONLY the ground truth cylinder mesh (no point cloud)...")
    try:
        gt_mesh = create_cylinder_mesh(center, axis, radius, height, color=[0, 1, 0])
        o3d.visualization.draw_plotly([gt_mesh])
        logger("[RESULT] Single mesh visualization succeeded.")
    except Exception as e:
        logger(f"[FAIL] Single mesh visualization crashed: {e}")

    # 3. Visualize point cloud + ground truth mesh
    logger("[TEST] Visualizing point cloud + ground truth mesh...")
    try:
        o3d.visualization.draw_plotly([pcd, gt_mesh])
        logger("[RESULT] Point cloud + mesh visualization succeeded.")
    except Exception as e:
        logger(f"[FAIL] Point cloud + mesh visualization crashed: {e}")

    # 4. Visualize all selected meshes + point cloud
    logger("[TEST] Visualizing all detected cylinders as meshes...")
    try:
        o3d.visualization.draw_plotly([pcd, *meshes])
        logger("[RESULT] Full visualization succeeded.")
    except Exception as e:
        logger(f"[FAIL] Full visualization crashed: {e}")
        logger("[INFO] Visualization skipped.")

if __name__ == "__main__":
    main()
