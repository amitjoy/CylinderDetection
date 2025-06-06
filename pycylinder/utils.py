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


def fit_cylinder_least_squares(points, direction_init=None, max_radius=0.5, max_iterations=100, tolerance=1e-6):
    """
    Fit a cylinder to 3D points using RANSAC followed by least squares optimization.
    
    Args:
        points: Nx3 array of 3D points
        direction_init: Initial guess for the cylinder axis (unit vector)
        max_radius: Maximum allowed radius (for regularization)
        max_iterations: Maximum number of optimization iterations
        tolerance: Convergence tolerance
        
    Returns:
        center, axis, radius: Fitted cylinder parameters
    """
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.distance import cdist
    
    points = np.asarray(points, dtype=np.float64)
    n_points = len(points)
    
    if n_points < 10:  # Not enough points for reliable fitting
        return np.mean(points, axis=0), np.array([0, 0, 1.0]), 0.1
    
    # Center the points for better numerical stability
    center = np.mean(points, axis=0)
    centered = points - center
    
    # Initialize axis using PCA if not provided
    if direction_init is None:
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort eigenvalues and vectors in descending order
        sort_inds = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_inds]
        eigvecs = eigvecs[:, sort_inds]
        
        # Check if points are approximately planar (smallest eigenvalue is much smaller than others)
        if eigvals[2] < 0.1 * eigvals[1]:
            # Points are nearly planar, use the normal to the plane as initial axis
            axis = eigvecs[:, 2]
        else:
            # Use direction of maximum variance
            axis = eigvecs[:, 0]
    else:
        axis = np.asarray(direction_init, dtype=np.float64)
    
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    
    # Improved RANSAC for better inlier selection
    best_inliers = None
    best_radius = 0.1
    best_axis = axis
    best_center = center
    best_score = -np.inf
    
    n_samples = min(200, n_points // 2)  # Increased number of RANSAC samples
    inlier_threshold = 0.05  # 5cm threshold for inliers
    
    for _ in range(n_samples):
        # Sample 3 points with some minimum distance between them
        while True:
            sample_idx = np.random.choice(n_points, size=3, replace=False)
            sample = points[sample_idx]
            # Ensure points are not too close to each other
            if np.min(cdist(sample, sample) + np.eye(3) * 1e6) > 0.1:
                break
        
        try:
            # Get two vectors in the plane
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-10:  # Points are colinear
                continue
                
            normal = normal / normal_norm
            
            # Project points to plane
            proj = sample - np.outer(np.dot(sample, normal), normal)
            
            # Fit circle in 2D using a more stable circle fitting method
            # Using Taubin's method for better numerical stability
            x = proj[:, 0]
            y = proj[:, 1]
            z = x * x + y * y
            
            # Form the design matrix
            n = len(x)
            X = np.column_stack([x, y, np.ones(n)])
            Y = z.reshape(-1, 1)
            
            # Solve using SVD for better numerical stability
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            c = Vt.T @ np.diag(1.0 / S) @ U.T @ Y
            
            # Extract circle parameters
            xc = c[0] / 2
            yc = c[1] / 2
            radius = np.sqrt(c[2] + xc**2 + yc**2)
            
            # Skip if radius is too large or too small
            if radius > max_radius or radius < 0.01:
                continue
                
            # Find inliers using a more robust distance measure
            dists = np.abs(np.sqrt((proj[:, 0] - xc)**2 + (proj[:, 1] - yc)**2) - radius)
            inliers = dists < inlier_threshold
            inlier_count = np.sum(inliers)
            
            # Score based on number of inliers and fit quality
            if inlier_count < 5:  # Require minimum number of inliers
                continue
                
            # Calculate a score that balances number of inliers and fit quality
            score = inlier_count / (1.0 + np.median(dists[inliers]) if inlier_count > 0 else 1.0)
            
            if score > best_score:
                best_score = score
                best_inliers = inliers
                best_radius = radius
                best_axis = normal
                # Recalculate center using all inliers
                if inlier_count > 5:
                    best_center = np.mean(points[inliers], axis=0)
                else:
                    best_center = np.mean(points, axis=0)

        except Exception as e:
            continue
    
    # Use RANSAC inliers for final fitting if we found good ones
    if best_inliers is not None and np.sum(best_inliers) > 5:
        points = points[best_inliers]
        center = best_center
        centered = points - center
        axis = best_axis
    
    # Initial guess: [radius, axis_x, axis_y, center_x, center_y, center_z]
    # Ensure axis is unit vector by only parameterizing two components and solving for the third
    x0 = np.concatenate([
        [np.clip(best_radius, 0.01, max_radius)],  # radius
        best_axis[:2],  # Only store x,y components of axis (z is determined by unit length)
        best_center  # center x,y,z
    ])
    
    def residuals(x):
        # Extract parameters
        radius = np.clip(x[0], 0.01, max_radius)  # Prevent zero or negative radius
        
        # Reconstruct axis (enforce unit vector by only parameterizing two components)
        axis_xy = x[1:3]
        axis_z = np.sqrt(max(0, 1 - np.sum(axis_xy**2)))  # Ensure unit length
        axis = np.array([axis_xy[0], axis_xy[1], axis_z])
        axis = axis / (np.linalg.norm(axis) + 1e-10)  # Ensure unit vector
        
        center = x[3:6]
        
        # Vector from center to points
        vec_to_points = points - center
        
        # Project points onto plane perpendicular to axis
        proj_lengths = np.dot(vec_to_points, axis)
        proj_vectors = vec_to_points - np.outer(proj_lengths, axis)
        dists = np.linalg.norm(proj_vectors, axis=1) - radius
        
        # Add regularization to keep axis close to initial guess
        axis_reg = 0.05 * (x[1:3] - x0[1:3])
        
        # Add regularization for radius to prevent it from growing too large
        radius_reg = 0.01 * (radius - x0[0])
        
        # Combine residuals
        return np.concatenate([
            dists,
            axis_reg,
            [radius_reg]
        ])
    
    # Set bounds for optimization
    bounds = (
        [0.01, -1, -1, -np.inf, -np.inf, -np.inf],  # Lower bounds
        [max_radius, 1, 1, np.inf, np.inf, np.inf]   # Upper bounds
    )
    
    # Scale parameters for better numerical stability
    x0_scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Run optimization with multiple restarts for better convergence
    best_result = None
    best_cost = np.inf
    logger = get_logger()
    
    try:
        for _ in range(3):  # Try up to 3 different initializations
            try:
                # Add small random noise to initial guess for different starting points
                if best_result is not None:
                    x0_perturbed = x0 + 0.1 * (np.random.random(6) - 0.5) * x0_scale
                    x0_perturbed[0] = np.clip(x0_perturbed[0], 0.01, max_radius)
                    x0_perturbed[1:3] = np.clip(x0_perturbed[1:3], -1, 1)
                else:
                    x0_perturbed = x0
                
                # Run least squares optimization
                result = least_squares(
                    residuals,
                    x0_perturbed,
                    bounds=bounds,
                    max_nfev=max_iterations,
                    xtol=tolerance * 0.1,  # Tighter tolerance
                    ftol=tolerance * 0.1,  # Tighter tolerance
                    method='trf',
                    loss='soft_l1',  # More robust to outliers
                    f_scale=0.1,  # Scale for robust loss
                    x_scale=x0_scale,
                    verbose=0
                )
                
                # Check if this is the best result so far
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
                    
            except Exception as e:
                logger.debug(f"Optimization attempt failed: {str(e)}")
                continue
        
        # Extract optimized parameters from best result
        if best_result is not None and best_result.success:
            # Extract optimized parameters
            radius = np.clip(best_result.x[0], 0.01, max_radius)
            
            # Reconstruct axis (enforce unit vector)
            axis_xy = best_result.x[1:3]
            axis_z = np.sqrt(max(0, 1 - np.sum(axis_xy**2)))
            axis = np.array([axis_xy[0], axis_xy[1], axis_z])
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            
            center = best_result.x[3:6]
            
            # Ensure the axis points in the same general direction as the initial guess
            if np.dot(axis, best_axis) < 0:
                axis = -axis
                
            return center, axis, radius
    
    except Exception as e:
        logger.warning(f"Cylinder optimization failed: {str(e)}")
    
    # Fallback to RANSAC result if optimization fails
    if best_inliers is not None and np.sum(best_inliers) > 5:
        logger.debug("Falling back to RANSAC result")
        return best_center, best_axis, best_radius
    
    # Last resort: return a cylinder using PCA
    logger.debug("Falling back to PCA-based cylinder")
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    
    # Project points onto axis and compute radius
    projections = np.dot(centered, axis)
    proj_pts = centered - np.outer(projections, axis)
    radius = np.median(np.linalg.norm(proj_pts, axis=1))
    radius = np.clip(radius, 0.01, max_radius)
    
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
