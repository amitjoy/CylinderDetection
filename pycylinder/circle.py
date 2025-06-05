"""
Circle detection in 2D using RANSAC.
"""
import numpy as np

class Circle2DRansac:
    def __init__(self, min_samples=3, residual_threshold=0.01, max_trials=1000):
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials

    def fit(self, points):
        """
        Fit a circle to 2D points using RANSAC.
        Args:
            points: (N, 2) numpy array
        Returns:
            center: (2,) numpy array
            radius: float
            inliers: list of indices
        """
        best_inliers = []
        best_circle = (None, None)
        N = len(points)
        if N < self.min_samples:
            return None, None, []
        for _ in range(self.max_trials):
            idxs = np.random.choice(N, self.min_samples, replace=False)
            sample = points[idxs]
            # Fit circle to 3 points
            circle = self._fit_circle_3pts(sample)
            if circle is None:
                continue
            center, radius = circle
            dists = np.linalg.norm(points - center, axis=1)
            residuals = np.abs(dists - radius)
            inliers = np.where(residuals < self.residual_threshold)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_circle = (center, radius)
        if len(best_inliers) < self.min_samples:
            return None, None, []
        # Refit using all inliers
        center, radius = self._fit_circle_least_squares(points[best_inliers])
        return center, radius, best_inliers.tolist()

    def _fit_circle_3pts(self, pts):
        # pts: (3,2)
        A = pts[0]
        B = pts[1]
        C = pts[2]
        temp = B - A
        temp2 = C - A
        d = 2 * (temp[0]*temp2[1] - temp[1]*temp2[0])
        if np.abs(d) < 1e-8:
            return None
        a = np.dot(A, A)
        b = np.dot(B, B)
        c = np.dot(C, C)
        ux = ((a - b) * temp2[1] - (a - c) * temp[1]) / d
        uy = ((a - c) * temp[0] - (a - b) * temp2[0]) / d
        center = np.array([ux, uy]) + A
        radius = np.linalg.norm(center - A)
        return center, radius

    def _fit_circle_least_squares(self, pts):
        # Algebraic circle fit (Kasa method)
        x = pts[:,0]
        y = pts[:,1]
        A = np.column_stack((2*x, 2*y, np.ones(len(pts))))
        b = x**2 + y**2
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, c = sol
        center = np.array([xc, yc])
        radius = np.sqrt(c + xc**2 + yc**2)
        return center, radius
