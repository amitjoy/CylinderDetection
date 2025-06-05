"""
Utility functions for math, direction generation, projections, etc.
"""
import numpy as np

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
