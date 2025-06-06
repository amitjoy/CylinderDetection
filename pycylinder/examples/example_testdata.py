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
    
    # Normalize the mesh to fit in unit cube centered at origin
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    mesh.translate(-mesh.get_center())
    
    mesh.compute_vertex_normals()
    # Sample points from mesh (adjust number as needed)
    pcd = mesh.sample_points_poisson_disk(5000)  # Reduced number of points for faster processing
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

    # Configure cylinder detection with optimized parameters
    logger("Configuring cylinder detection with optimized parameters...")
    
    # Calculate mean spacing for adaptive thresholding
    mean_spacing = np.mean(np.linalg.norm(points[1:] - points[:-1], axis=1))
    logger(f"Mean point spacing: {mean_spacing:.6f}")
    
    # Create and configure detector with improved parameters
    logger("Creating CylinderDetector...")
    detector = CylinderDetector(
        point_cloud=point_cloud,
        directions_samples=30,          # Balanced between speed and coverage
        distance_threshold=mean_spacing * 3.5,  # Slightly increased for better connectivity
        normal_threshold=0.75,          # More relaxed for better detection
        min_component_points=8,         # Further reduced to detect smaller cylinders
        circle_residual=0.07,           # More relaxed for better fitting
        min_length=0.06,                # Reduced to detect shorter cylinders
        min_inliers=6,                  # Reduced to detect smaller cylinders
        max_radius=0.5,                 # Keep focus on smaller pipes
        angle_thresh=0.45,              # More tolerant for merging
        center_thresh=0.18,             # More relaxed for merging
        radius_thresh=0.25,             # More tolerant for merging
        max_mean_distance_to_surface=0.04,  # More lenient surface fitting
        debug_prints=False
    )
    
    # Run cylinder detection
    logger("Starting cylinder detection...")
    cylinders = detector.detect()
    logger(f"Detected {len(cylinders)} cylinders.")
    
    # Log details of detected cylinders
    for i, cylinder in enumerate(cylinders):
        logger(f"Cylinder {i+1}:")
        logger(f"  Center: {np.round(cylinder.center, 4)}")
        logger(f"  Axis: {np.round(cylinder.axis, 4)}")
        logger(f"  Radius: {cylinder.radius:.6f}")
        logger(f"  Length: {cylinder.length:.6f}")
        logger(f"  Points: {len(cylinder.inliers)}")

    # Create a visualizer with better visualization of cylinders
    def create_cylinder_mesh(cylinder, color=None):
        """Create a mesh for a cylinder with proper orientation and dimensions.
        
        Args:
            cylinder: Cylinder object with center, axis, radius, and length
            color: Optional RGB color (if None, a random color will be used)
            
        Returns:
            Tuple of (cylinder_mesh, axis_line_set, start_sphere, end_sphere)
        """
        if color is None:
            color = np.random.rand(3)
        
        # Get cylinder parameters
        radius = cylinder.radius
        height = cylinder.length
        center = cylinder.center
        axis = cylinder.axis / np.linalg.norm(cylinder.axis)  # Ensure unit vector
        
        # Calculate start and end points
        half_height = height / 2
        start_point = center - axis * half_height
        end_point = center + axis * half_height
        
        # Create a unit cylinder along Z-axis
        mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, 
            height=1.0,  # Unit height, we'll scale it
            resolution=32,  # More segments for smoother cylinder
            split=1
        )
        
        # Scale to desired height
        mesh.scale(height, center=(0, 0, 0))
        
        # Rotate to align with the cylinder axis
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, axis)
        
        # Handle the case where the axis is aligned with Z (avoid division by zero)
        if np.linalg.norm(rotation_axis) < 1e-6:
            if np.dot(z_axis, axis) < 0:  # 180 degree rotation needed
                rotation_axis = np.array([1, 0, 0])
                rotation_angle = np.pi
            else:  # No rotation needed
                rotation_axis = np.array([1, 0, 0])
                rotation_angle = 0
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, axis), -1.0, 1.0))
        
        # Create rotation matrix
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        mesh.rotate(rotation_matrix, center=(0, 0, 0))
        
        # Translate to the correct position
        mesh.translate(center)
        
        # Set color and enable smooth shading
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        
        # Create a line set for the axis
        axis_points = np.array([start_point, end_point])
        lines = [[0, 1]]
        colors = [[1, 0, 0]]  # Red color for axis
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(axis_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # Create spheres at the start and end points
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius*1.02, resolution=20)
        start_sphere.translate(start_point)
        start_sphere.paint_uniform_color(color)
        start_sphere.compute_vertex_normals()
        
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius*1.02, resolution=20)
        end_sphere.translate(end_point)
        end_sphere.paint_uniform_color(color)
        end_sphere.compute_vertex_normals()
        
        return mesh, line_set, start_sphere, end_sphere

    # Create visualization
    vis_geometries = []
    
    # Add the original point cloud (semi-transparent)
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(points)
    pcd_vis.colors = o3d.utility.Vector3dVector(np.ones_like(points) * 0.7)  # Light gray
    vis_geometries.append(pcd_vis)
    
    # Add each detected cylinder as a mesh
    for i, cylinder in enumerate(cylinders):
        try:
            # Generate a consistent color based on cylinder index
            color = np.array([
                (i * 0.618033988749895) % 1.0,  # Golden ratio for color distribution
                (i * 0.3819660112501051) % 1.0, # Complementary ratio
                (i * 0.2360679774997897) % 1.0  # Another ratio for blue channel
            ])
            
            # Create cylinder visualization components
            cylinder_mesh, axis_line, start_sphere, end_sphere = create_cylinder_mesh(cylinder, color=color)
            
            # Add all components to visualization
            vis_geometries.extend([cylinder_mesh, axis_line, start_sphere, end_sphere])
            
        except Exception as e:
            logger(f"Error visualizing cylinder {i}: {str(e)}")
    
    # Set up visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Cylinder Detection", width=1200, height=900)
    
    # Add all geometries to the visualizer
    for geometry in vis_geometries:
        vis.add_geometry(geometry)
    
    # Set up view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # Add lighting
    render_option = vis.get_render_option()
    render_option.light_on = True
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    render_option.point_size = 2.0
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
