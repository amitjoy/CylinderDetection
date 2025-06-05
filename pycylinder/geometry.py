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

class Circle:
    def __init__(self, center, radius, inliers=None):
        self.center = np.asarray(center)
        self.radius = float(radius)
        self.inliers = inliers if inliers is not None else []

class ConnectedComponent:
    def __init__(self, indices, direction):
        self.indices = indices  # List of point indices
        self.direction = np.asarray(direction)
