"""
PointCloud class for loading, storing, and processing point cloud data using Open3D.
"""
import open3d as o3d
import numpy as np

class PointCloud:
    def __init__(self, o3d_pcd):
        """Initialize with an Open3D PointCloud object."""
        self.o3d_pcd = o3d_pcd

    @classmethod
    def from_file(cls, filename):
        """Load point cloud from file (PLY, XYZ, etc.)."""
        pcd = o3d.io.read_point_cloud(filename)
        return cls(pcd)

    def estimate_normals(self, radius=0.05, max_nn=30):
        """Estimate normals for the point cloud."""
        self.o3d_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )

    def to_numpy(self):
        """Return points as Nx3 numpy array."""
        return np.asarray(self.o3d_pcd.points)

    def normals_numpy(self):
        """Return normals as Nx3 numpy array."""
        return np.asarray(self.o3d_pcd.normals)
