"""
Synthetic point cloud generator for cylinders.
"""
import numpy as np
import open3d as o3d

def generate_cylinder_point_cloud(center, axis, radius, height, n_points=2000, noise=0.002):
    """
    Generate a synthetic cylinder point cloud.
    Args:
        center: (3,) center of the cylinder (at midpoint)
        axis: (3,) axis direction (will be normalized)
        radius: float
        height: float
        n_points: int, number of points
        noise: float, stddev of Gaussian noise
    Returns:
        open3d.geometry.PointCloud
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    # Find orthogonal vectors
    if np.allclose(axis, [1,0,0]):
        ortho1 = np.array([0,1,0])
    else:
        ortho1 = np.cross(axis, [1,0,0])
    ortho1 = ortho1 / np.linalg.norm(ortho1)
    ortho2 = np.cross(axis, ortho1)
    ortho2 = ortho2 / np.linalg.norm(ortho2)
    # Sample angles and heights
    angles = np.random.uniform(0, 2*np.pi, n_points)
    heights = np.random.uniform(-height/2, height/2, n_points)
    xs = center[0] + heights * axis[0] + radius * np.cos(angles) * ortho1[0] + radius * np.sin(angles) * ortho2[0]
    ys = center[1] + heights * axis[1] + radius * np.cos(angles) * ortho1[1] + radius * np.sin(angles) * ortho2[1]
    zs = center[2] + heights * axis[2] + radius * np.cos(angles) * ortho1[2] + radius * np.sin(angles) * ortho2[2]
    pts = np.stack([xs, ys, zs], axis=1)
    pts += np.random.normal(scale=noise, size=pts.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd
