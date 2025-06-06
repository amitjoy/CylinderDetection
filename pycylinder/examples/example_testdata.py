import os
import sys
import numpy as np
import open3d as o3d

# Import from the pycylinder package
from pycylinder.detector import CylinderDetector, fibonacci_sphere
from pycylinder.logger import CylinderLogger, set_logger, LogLevel
from pycylinder.pointcloud import PointCloud

def main():
    # Set up logger with different levels for console and file
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', 'cylinder_debug.log')
    
    # Create logger with:
    # - Console: INFO level and above (avoids debug spam in console)
    # - File: DEBUG level and above (includes all messages)
    logger = CylinderLogger(
        mode='both',
        log_file=log_file,
        console_level=LogLevel.INFO,  # Only show INFO and above in console
        file_level=LogLevel.DEBUG,    # Show all messages in log file
        include_timestamp=True
    )
    set_logger(logger)
    
    # Log startup information
    logger.info(f"Starting cylinder detection with Open3D {o3d.__version__}")
    logger.debug(f"Debug logging enabled in {log_file}")
    
    # Path to test data OBJ file
    testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    obj_path = os.path.join(testdata_dir, 'Pipes.obj')
    if not os.path.exists(obj_path):
        logger(f"Test data OBJ file not found: {obj_path}")
        return

    # Load mesh and sample points
    logger(f"Loading mesh from {obj_path} ...")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    # Sample points from mesh (adjust number as needed)
    pcd = mesh.sample_points_poisson_disk(20000)
    pcd.estimate_normals()
    logger(f"Sampled {np.asarray(pcd.points).shape[0]} points from mesh.")

    # Create point cloud wrapper from the sampled points
    point_cloud = PointCloud(pcd)

    # Print point cloud statistics for debugging
    points = point_cloud.to_numpy()
    normals = point_cloud.normals_numpy()
    logger(f"Point cloud bounds: {np.min(points, axis=0)}, {np.max(points, axis=0)}")
    mean_spacing = np.mean(np.linalg.norm(points[1:] - points[:-1], axis=1))
    logger(f"Mean spacing between consecutive points: {mean_spacing}")
    logger(f"First 5 normals: {normals[:5] if normals is not None else 'No normals'}")

    # Enable debug logging through the logger configuration above

    # Automated threshold selection for region growing
    min_visualize_size = 100    # Find connected components with the chosen threshold
    AUTOMATIC_THRESHOLD_SELECTION = True
    if AUTOMATIC_THRESHOLD_SELECTION:
        logger("Automatically selecting distance threshold...")
        # Import utility functions for automatic threshold selection
        from pycylinder.utils import find_optimal_distance_threshold, compute_mean_spacing
        # Use the utility function to find optimal threshold and large regions
        chosen_threshold, large_regions = find_optimal_distance_threshold(
            points=pcd,
            min_large_regions=20,  # Target number of large regions
            min_region_size=100,    # Minimum points for a region to be considered 'large'
            logger=logger
        )
        
        # Visualize the segmentation with the chosen threshold
        region_colors = np.full((len(points), 3), 0.7)  # Gray for unassigned
        for i, region in enumerate(large_regions):
            color = np.random.rand(3)  # Random color for each region
            region_colors[region.indices] = color
        pcd.colors = o3d.utility.Vector3dVector(region_colors)
        o3d.visualization.draw([pcd])
    else:
        # Fallback to default threshold if automatic selection is disabled
        from .utils import compute_mean_spacing
        mean_spacing = compute_mean_spacing(pcd)
        chosen_threshold = mean_spacing * 1.0  # Fallback to mean spacing
        logger(f"Using default threshold: {chosen_threshold:.6f}")

    # Use the chosen threshold for cylinder detection
    detector = CylinderDetector(
        point_cloud=pcd,
        directions_samples=60,
        distance_threshold=chosen_threshold,
        normal_threshold=0.92,
        min_component_points=5,
        circle_residual=0.03,
        min_length=0.08,
        min_inliers=5,
        max_radius=2.0,
        angle_thresh=0.2,
        center_thresh=0.12,
        radius_thresh=0.08
    )
    cylinders = detector.detect()
    logger(f"Detected {len(cylinders)} cylinders.")

    # Overlay detected cylinders on input point cloud
    base_colors = np.full(points.shape, 0.7)  # gray
    if len(cylinders) > 0:
        for idx, cyl in enumerate(cylinders):
            # Color inliers for each cylinder
            color = np.random.rand(3)
            base_colors[cyl.inliers] = color
        pcd.colors = o3d.utility.Vector3dVector(base_colors)
        o3d.visualization.draw([pcd])
    else:
        logger("No cylinders to overlay.")

if __name__ == "__main__":
    main()
