"""
Utility functions for math, direction generation, projections, etc.
"""
import numpy as np
import numpy.linalg as la
from scipy.spatial import KDTree
from .logger import get_logger

def fibonacci_sphere(samples=100):
    """Generate evenly distributed directions on a sphere."""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)


def project_points_onto_plane(points, plane_point, plane_normal):
    """
    Project 3D points onto a plane defined by plane_point and plane_normal.
    Args:
        points: (N, 3) numpy array
        plane_point: (3,) point on the plane
        plane_normal: (3,) normal vector of the plane (will be normalized)
    Returns:
        (N, 3) numpy array of projected points
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    diff = points - plane_point
    dist = np.dot(diff, plane_normal)
    return points - np.outer(dist, plane_normal)


def pairwise_distances(a, b):
    """
    Compute pairwise Euclidean distances between two sets of points.
    Args:
        a: (N, 3)
        b: (M, 3)
    Returns:
        (N, M) array
    """
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def fit_cylinder_least_squares(points, direction_init=None):
    """
    Fit a cylinder to 3D points using least-squares.
    Args:
        points: (N, 3) numpy array
        direction_init: (3,) initial axis direction (optional)
    Returns:
        center: (3,) numpy array (point on axis)
        axis: (3,) numpy array (unit vector)
        radius: float
    """
    import numpy as np
    from scipy.optimize import minimize
    points = np.asarray(points)
    if direction_init is None:
        # Use PCA to estimate axis
        pts_mean = np.mean(points, axis=0)
        cov = np.cov(points - pts_mean, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)]
    else:
        axis = direction_init / np.linalg.norm(direction_init)
    def cost(params):
        # params: center (3), axis (2, spherical angles), radius (1)
        cx, cy, cz, theta, phi, r = params
        axis_vec = np.array([
            np.sin(theta)*np.cos(phi),
            np.sin(theta)*np.sin(phi),
            np.cos(theta)
        ])
        c = np.array([cx, cy, cz])
        d = points - c
        proj = d - np.outer(np.dot(d, axis_vec), axis_vec)
        dist = np.linalg.norm(proj, axis=1)
        return np.mean((dist - r)**2)
    # Initial guess
    c0 = np.mean(points, axis=0)
    # Convert axis to spherical angles
    theta0 = np.arccos(axis[2])
    phi0 = np.arctan2(axis[1], axis[0])
    r0 = np.mean(np.linalg.norm(points - c0, axis=1))
    x0 = np.array([*c0, theta0, phi0, r0])
    res = minimize(cost, x0, method='L-BFGS-B')
    cx, cy, cz, theta, phi, r = res.x
    axis_vec = np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])
    center = np.array([cx, cy, cz])
    axis = axis_vec / np.linalg.norm(axis_vec)
    radius = abs(r)
    return center, axis, radius


def compute_mean_spacing(points, k=10):
    """
    Compute the mean nearest neighbor distance for each point.
    
    Args:
        points: (N, 3) numpy array of 3D points, SimplePointCloud, or Open3D PointCloud
        k: Number of neighbors to consider (including the point itself)
        
    Returns:
        float: Mean distance to the k-th nearest neighbor
    """
    # Convert SimplePointCloud to numpy array
    if hasattr(points, '_points'):  # SimplePointCloud
        points = points._points
    # Convert Open3D PointCloud to numpy array
    elif hasattr(points, 'points'):  # Open3D PointCloud
        points = np.asarray(points.points)
    # Ensure points is a numpy array
    points = np.asarray(points)
    
    # Handle case where k+1 > number of points
    if len(points) <= 1:
        return 0.0
    k = min(k, len(points) - 1)
    if k < 1:
        return 0.0
    
    # Compute mean spacing
    tree = KDTree(points)
    distances, _ = tree.query(points, k=k+1)  # k+1 because point is its own neighbor
    return float(np.mean(distances[:, 1:]))  # Exclude the point itself


def find_optimal_distance_threshold(points, min_large_regions=20, min_region_size=100, logger=None):
    """
    Automatically find an optimal distance threshold for region growing.
    
    Args:
        points: (N, 3) numpy array of 3D points
        min_large_regions: Desired minimum number of large regions
        min_region_size: Minimum points for a region to be considered 'large'
        logger: Optional logger for debug output
        
    Returns:
        float: Optimal distance threshold
        list: List of large connected components
    """
    if logger is None:
        logger = get_logger()
    
    # Compute mean spacing
    mean_spacing = compute_mean_spacing(points)
    logger(f"Mean point spacing: {mean_spacing:.6f}")
    
    # Try different multipliers of the mean spacing
    multipliers = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    best_threshold = None
    best_large_regions = []
    best_num_large_regions = 0
    
    for multiplier in multipliers:
        threshold = mean_spacing * multiplier
        logger(f"\nTrying threshold: {threshold:.6f} (x{multiplier:.1f} of mean spacing)")
        
        # Find connected components with this threshold
        from .detector import CylinderDetector
        detector = CylinderDetector(
            points,
            distance_threshold=threshold,
            normal_threshold=0.92,  # Default value
            min_component_points=5   # Keep this small to get all potential regions
        )
        
        # Get all connected components
        components = detector.find_connected_components(None)  # Direction doesn't matter here
        large_regions = [c for c in components if len(c.indices) >= min_region_size]
        num_large = len(large_regions)
        
        logger(f"Found {len(components)} total regions, {num_large} large (≥{min_region_size} points)")
        
        # Update best threshold
        if num_large > best_num_large_regions:
            best_num_large_regions = num_large
            best_threshold = threshold
            best_large_regions = large_regions
            
        # Early exit if we found enough large regions
        if best_num_large_regions >= min_large_regions:
            break
    
    if best_threshold is None:
        logger("WARNING: No suitable threshold found. Using mean spacing as fallback.")
        best_threshold = mean_spacing
        best_large_regions = []
    
    logger(f"\nSelected threshold: {best_threshold:.6f} with {len(best_large_regions)} large regions")
    return best_threshold, best_large_regions
