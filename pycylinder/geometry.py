"""
Geometry primitives: Cylinder, Circle, ConnectedComponent
"""
import numpy as np

class Cylinder:
    def __init__(self, center, axis, radius, inliers=None):
        self.center = np.asarray(center)
        self.axis = np.asarray(axis)
        self.radius = float(radius)
        self.inliers = inliers if inliers is not None else []
        self._points = None  # Will store points if they're set explicitly
        
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
    def points(self, value):
        """Set the points that define this cylinder.
        
        Args:
            value (numpy.ndarray): Array of points with shape (N, 3).
        """
        if value is not None:
            self._points = np.asarray(value)
        else:
            self._points = None
        
    @property
    def length(self) -> float:
        """Calculate the length of the cylinder based on its points.
        
        Returns:
            float: The length of the cylinder, or 0 if not enough points are available.
        """
        if not hasattr(self, '_length'):
            points = self.points
            if points is None or len(points) < 2:
                self._length = 0.0
            else:
                try:
                    # Project points onto the cylinder axis
                    projections = np.dot(points - self.center, self.axis)
                    # Calculate the distance between the min and max projections
                    self._length = float(np.max(projections) - np.min(projections))
                except Exception as e:
                    # If any error occurs during calculation, return 0
                    self._length = 0.0
        return self._length
        
    @property
    def start_point(self):
        """Get the start point of the cylinder along its axis."""
        if not hasattr(self, '_start_point'):
            if not hasattr(self, '_length') or self._length == 0:
                return self.center
            self._start_point = self.center - self.axis * (self._length / 2)
        return self._start_point
        
    @property
    def end_point(self):
        """Get the end point of the cylinder along its axis."""
        if not hasattr(self, '_end_point'):
            if not hasattr(self, '_length') or self._length == 0:
                return self.center
            self._end_point = self.center + self.axis * (self._length / 2)
        return self._end_point

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
