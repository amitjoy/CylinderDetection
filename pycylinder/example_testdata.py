import open3d as o3d
import numpy as np
import os
from detector import CylinderDetector, CylinderLogger, fibonacci_sphere

def main():
    # Set up file logger for debug output
    CylinderDetector.LOGGER = CylinderLogger(mode='file', log_file='../../logs/cylinder_debug.log')
    logger = CylinderDetector.LOGGER or print
    # Path to test data OBJ file
    testdata_dir = os.path.join(os.path.dirname(__file__), 'examples/testdata')
    obj_path = os.path.join(testdata_dir, 'Pipes.obj')
    if not os.path.exists(obj_path):
        msg = f"Test data OBJ file not found: {obj_path}"
        if callable(logger):
            logger.log(msg)
        else:
            logger.log(msg)
        return

    # Load mesh and sample points
    msg = f"Loading mesh from {obj_path} ..."
    if callable(logger):
        logger.log(msg)
    else:
        logger.log(msg)
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    # Sample points from mesh (adjust number as needed)
    pcd = mesh.sample_points_poisson_disk(20000)
    pcd.estimate_normals()
    msg = f"Sampled {np.asarray(pcd.points).shape[0]} points from mesh."
    if callable(logger):
        logger.log(msg)
    else:
        logger.log(msg)

    # Convert Open3D point cloud to numpy arrays
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # Minimal PointCloud class for CylinderDetector
    class SimplePointCloud:
        def __init__(self, points, normals):
            self._points = points
            self._normals = normals
            class DummyO3dPcd:
                def __init__(self, normals):
                    self._normals = normals
                @property
                def normals(self):
                    return self._normals
            self.o3d_pcd = DummyO3dPcd(normals)
        def to_numpy(self):
            return self._points
        @property
        def normals(self):
            return self._normals
        def normals_numpy(self):
            return self._normals

    pc = SimplePointCloud(points, normals)

    # Print point cloud statistics for debugging
    msg = f"Point cloud bounds: {np.min(points, axis=0)}, {np.max(points, axis=0)}"
    if callable(logger):
        logger.log(msg)
    else:
        logger.log(msg)
    mean_spacing = np.mean(np.linalg.norm(points[1:] - points[:-1], axis=1))
    msg = f"Mean spacing between consecutive points: {mean_spacing}"
    if callable(logger):
        logger.log(msg)
    else:
        logger.log(msg)
    msg = f"First 5 normals: {normals[:5]}"
    if callable(logger):
        logger.log(msg)
    else:
        logger.log(msg)

    # Enable debug prints globally
    CylinderDetector.DEBUG_PRINTS = True

    # Automated threshold selection for region growing
    min_visualize_size = 100  # Only visualize regions with at least this many points
    directions = fibonacci_sphere(60)
    first_direction = directions[0]
    # Compute mean nearest neighbor distance
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_spacing = np.mean(distances[:, 1])
    logger.log(f"Mean nearest neighbor distance: {mean_spacing}")
    multipliers = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50]
    chosen_threshold = None
    for k in multipliers:
        dist_thresh = mean_spacing * k
        logger.log(f"\nTrying distance_threshold = {dist_thresh:.3f} (k={k})")
        detector = CylinderDetector(
            pc,
            directions_samples=60,
            distance_threshold=dist_thresh,
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
        regions = detector.find_connected_components(first_direction)
        region_sizes = [len(region.indices) for region in regions]
        logger.log(f"Found {len(regions)} regions.")
        logger.log(f"Region sizes: {region_sizes}")
        large_regions = [region for region in regions if len(region.indices) >= min_visualize_size]
        logger.log(f"Regions with >= {min_visualize_size} points: {len(large_regions)}")
        if len(large_regions) >= 20:
            chosen_threshold = dist_thresh
            logger.log(f"Selected distance_threshold = {dist_thresh:.3f} (k={k}) with {len(large_regions)} large regions.")
            # Visualize the chosen segmentation
            region_colors = np.full((len(points), 3), 0.7)
            for i, region in enumerate(large_regions):
                color = np.random.rand(3)
                region_colors[region.indices] = color
            pcd.colors = o3d.utility.Vector3dVector(region_colors)
            o3d.visualization.draw([pcd])
            break
    if chosen_threshold is None:
        logger.log("No threshold produced at least 20 large regions. Using the largest tried value.")
        chosen_threshold = mean_spacing * multipliers[-1]

    # Use the chosen threshold for cylinder detection
    detector = CylinderDetector(
        pc,
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
    logger.log(f"Detected {len(cylinders)} cylinders.")

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
        logger.log("No cylinders to overlay.")

    # After parameter sweep, run detection with the last threshold and log number of cylinders detected
    detector = CylinderDetector(
        pc,
        directions_samples=60,
        distance_threshold=distance_thresholds[-1],
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
    logger.log(f"Detected {len(cylinders)} cylinders.")

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
        logger.log("No cylinders to overlay.")

if __name__ == "__main__":
    main()
