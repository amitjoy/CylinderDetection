import open3d as o3d
import numpy as np
import os
from detector import CylinderDetector, CylinderLogger

def main():
    # Set up file logger for debug output
    from detector import CylinderLogger, CylinderDetector
    CylinderDetector.LOGGER = CylinderLogger(mode='file', log_file='../../logs/cylinder_debug.log')
    logger = CylinderDetector.LOGGER or print
    # Path to test data OBJ file
    testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')
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

    # Run cylinder detection
    detector = CylinderDetector(
        pc,
        directions_samples=60,
        distance_threshold=0.03,
        normal_threshold=0.92,
        min_component_points=20,
        circle_residual=0.03,
        min_length=0.08,
        min_inliers=20,
        max_radius=2.0,
        angle_thresh=0.2,
        center_thresh=0.12,
        radius_thresh=0.08
    )

    msg = "Detecting cylinders..."
    if callable(logger):
        logger.log(msg)
    else:
        logger.log(msg)
    cylinders = detector.detect()
    msg = f"Detected {len(cylinders)} cylinders."
    if callable(logger):
        logger.log(msg)
    else:
        logger.log(msg)

    # Optionally, visualize detected cylinders (as point clouds)
    # Visualize the original point cloud and color detected inliers
    inlier_colors = np.full(points.shape, 0.7)  # gray
    for idx, cyl in enumerate(cylinders):
        color = np.random.rand(3)
        inlier_colors[cyl.inliers] = color
    pcd.colors = o3d.utility.Vector3dVector(inlier_colors)
    o3d.visualization.draw([pcd])

if __name__ == "__main__":
    main()
