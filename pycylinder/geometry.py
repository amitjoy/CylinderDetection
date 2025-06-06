"""
Geometry primitives: Cylinder, Circle, ConnectedComponent
"""
import numpy as np

class Cylinder:
    def __init__(self, center, axis, radius, inliers=None, points=None):
        """
        Initialize a cylinder with center, axis, radius, and optional inliers and points.
        
        Args:
            center: 3D point on the cylinder axis
            axis: 3D vector defining the cylinder axis direction (will be normalized)
            radius: Radius of the cylinder
            inliers: List of indices of points belonging to this cylinder
            points: 3D points belonging to this cylinder
        """
        self.center = np.asarray(center, dtype=np.float64)
        self.axis = np.asarray(axis, dtype=np.float64)
        self.axis = self.axis / np.linalg.norm(self.axis)  # Ensure unit vector
        self.radius = float(radius)
        self.inliers = inliers if inliers is not None else []
        
        # Initialize cache
        self._points = None
        self._length = None
        self._start_point = None
        self._end_point = None
        
        # If points are provided, use them to initialize
        if points is not None:
            self.points = np.asarray(points, dtype=np.float64)
        
    @property
    def points(self):
        """Get the points that define this cylinder.
        
        Returns:
            numpy.ndarray: Array of points, or None if not available.
        """
        if self._points is not None:
            return self._points
            
        # If no explicit points but we have inliers, try to use them
        if self.inliers is not None and len(self.inliers) > 0:
            inliers_array = np.asarray(self.inliers)
            if inliers_array.ndim == 2 and inliers_array.shape[1] == 3:
                return inliers_array
                
        # If we can't get points from inliers, return None
        return None
        
    @points.setter
    def points(self, points):
        """Set the points for this cylinder and update derived properties.
        
        Args:
            points: Nx3 array of 3D points, or None to clear the points.
        """
        self._points = np.asarray(points, dtype=np.float64) if points is not None else None
        self._update_derived_properties()
    
    def _update_derived_properties(self):
        """Update length and endpoints based on current points."""
        # Clear cached properties
        self._length = None
        self._start_point = None
        self._end_point = None
        
        if self._points is None or len(self._points) < 2:
            return
            
        try:
            # Project points onto the cylinder axis
            rel_points = self._points - self.center
            projections = np.dot(rel_points, self.axis)
            
            if len(projections) > 0:
                # Calculate length as distance between min and max projections
                min_proj = np.min(projections)
                max_proj = np.max(projections)
                self._length = float(max_proj - min_proj)
                
                # Update start and end points
                self._start_point = self.center + self.axis * min_proj
                self._end_point = self.center + self.axis * max_proj
                
                # Update center to be the midpoint
                self.center = (self._start_point + self._end_point) / 2
        except Exception as e:
            logger = get_logger()
            logger.warning(f"Error updating cylinder properties: {str(e)}")
    
    @property
    def length(self) -> float:
        """Get the length of the cylinder based on its points.
        
        Returns:
            float: The length of the cylinder, or 0 if not enough points are available.
        """
        if self._length is None:
            self._update_derived_properties()
        return self._length if self._length is not None else 0.0
        
    @property
    def start_point(self):
        """Get the start point of the cylinder along its axis.
        
        Returns:
            np.ndarray: 3D start point of the cylinder.
        """
        if self._start_point is None:
            self._update_derived_properties()
        return self._start_point if self._start_point is not None else self.center.copy()
        
    @property
    def end_point(self):
        """Get the end point of the cylinder along its axis.
        
        Returns:
            np.ndarray: 3D end point of the cylinder.
        """
        if self._end_point is None:
            self._update_derived_properties()
        return self._end_point if self._end_point is not None else self.center.copy()

class Circle:
    def __init__(self, center, radius, inliers=None):
        self.center = np.asarray(center)
        self.radius = float(radius)
        self.inliers = inliers if inliers is not None else []

class ConnectedComponent:
    def __init__(self, points, normals, direction):
        self.points = np.asarray(points)  # Array of points
        self.normals = np.asarray(normals) if normals is not None else None  # Optional array of normals
        self.direction = np.asarray(direction)  # Direction of the component
        
    @property
    def indices(self):
        """For backward compatibility, return indices if needed"""
        return np.arange(len(self.points))  # Return indices of points
