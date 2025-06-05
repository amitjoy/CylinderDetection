"""
CylinderDetector: Main detection pipeline (stub)
"""
from geometry import Cylinder
from utils import fibonacci_sphere

class CylinderLogger:
    def __init__(self, mode='console', log_file=None):
        self.mode = mode
        self.log_file = log_file
        if self.mode == 'file' and self.log_file is not None:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w') as f:
                f.write('')  # Clear file at start
    def log(self, msg):
        if self.mode == 'console':
            print(msg)
        elif self.mode == 'file' and self.log_file is not None:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
        else:
            print(msg)

import os

class CylinderDetector:
    # Set CylinderDetector.DEBUG_PRINTS = True to enable all debug output globally.
    # Do not use per-instance debug toggling.
    DEBUG_PRINTS = False
    LOGGER = None
    def __init__(self, point_cloud, directions_samples=60, distance_threshold=0.02, normal_threshold=0.95, min_component_points=30, circle_residual=0.01, min_length=0.05, min_inliers=30, max_radius=2.0, angle_thresh=0.1, center_thresh=0.05, radius_thresh=0.05, debug_prints=None):
        # debug_prints argument is ignored; use CylinderDetector.DEBUG_PRINTS

        """point_cloud: PointCloud instance"""
        self.point_cloud = point_cloud
        self.directions_samples = directions_samples
        self.distance_threshold = distance_threshold
        self.normal_threshold = normal_threshold
        self.min_component_points = min_component_points
        self.circle_residual = circle_residual
        self.min_length = min_length
        self.min_inliers = min_inliers
        self.max_radius = max_radius
        self.angle_thresh = angle_thresh
        self.center_thresh = center_thresh
        self.radius_thresh = radius_thresh

    def find_connected_components(self, direction):
        """
        Segment the point cloud into connected components along a given direction.
        Args:
            direction: (3,) numpy array, main axis
        Returns:
            List of ConnectedComponent objects
        """
        import numpy as np
        from geometry import ConnectedComponent
        points = self.point_cloud.to_numpy()
        normals = self.point_cloud.normals_numpy() if len(self.point_cloud.o3d_pcd.normals) else None
        N = len(points)
        if CylinderDetector.DEBUG_PRINTS:
            logger = CylinderDetector.LOGGER or print
            msg = f"[DEBUG][find_connected_components] Direction: {direction}, N_points={N}, N_normals={len(normals) if normals is not None else 0}"
            if callable(logger):
                logger(msg)
            else:
                logger.log(msg)
            logger = CylinderDetector.LOGGER or print
            msg = f"[DEBUG][find_connected_components] distance_threshold={self.distance_threshold}, normal_threshold={self.normal_threshold}, min_component_points={self.min_component_points}"
            if callable(logger):
                logger(msg)
            else:
                logger.log(msg)
        for i in range(min(10, N)):
            pt_str = np.array2string(points[i], precision=3)
            if normals is not None:
                norm_str = np.array2string(normals[i], precision=3)
                if CylinderDetector.DEBUG_PRINTS:
                    logger = CylinderDetector.LOGGER or print
                    msg = f"  Point {i}: {pt_str}, Normal: {norm_str}"
                    if callable(logger):
                        logger(msg)
                    else:
                        logger.log(msg)
            else:
                if CylinderDetector.DEBUG_PRINTS:
                    logger = CylinderDetector.LOGGER or print
                    msg = f"  Point {i}: {pt_str}, Normal: None"
                    if callable(logger):
                        logger(msg)
                    else:
                        logger.log(msg)
            # Print number of neighbors for this point
            dists = np.linalg.norm(points - points[i], axis=1)
            neighbors = np.where((dists < self.distance_threshold))[0]
            if CylinderDetector.DEBUG_PRINTS:
                logger = CylinderDetector.LOGGER or print
                msg = f"    [DEBUG] Num neighbors within distance_threshold: {len(neighbors)}"
                if callable(logger):
                    logger(msg)
                else:
                    logger.log(msg)
        visited = np.zeros(N, dtype=bool)
        components = []
        for idx in range(N):
            if visited[idx]:
                continue
            # Start region growing
            if CylinderDetector.DEBUG_PRINTS:
                logger = CylinderDetector.LOGGER or print
                msg = f"[DEBUG][find_connected_components] Starting region from idx={idx}"
                if callable(logger):
                    logger(msg)
                else:
                    logger.log(msg)
            queue = [idx]
            current = []
            while queue:
                i = queue.pop()
                if visited[i]:
                    continue
                visited[i] = True
                current.append(i)
                # Find neighbors (brute force, can be optimized)
                dists = np.linalg.norm(points - points[i], axis=1)
                neighbors = np.where((dists < self.distance_threshold) & (~visited))[0]
                for n in neighbors:
                    if normals is not None:
                        dot = np.dot(normals[i], normals[n])
                        if dot < self.normal_threshold:
                            continue
                    queue.append(n)
            if CylinderDetector.DEBUG_PRINTS:
                logger = CylinderDetector.LOGGER or print
                msg = f"[DEBUG][find_connected_components] Finished region with size={len(current)}"
                if callable(logger):
                    logger(msg)
                else:
                    logger.log(msg)
            if len(current) >= self.min_component_points:
                if CylinderDetector.DEBUG_PRINTS:
                    logger = CylinderDetector.LOGGER or print
                    msg = f"[DEBUG][find_connected_components] --> Region accepted as component."
                    if callable(logger):
                        logger(msg)
                    else:
                        logger.log(msg)
                components.append(ConnectedComponent(current, direction))
            else:
                if CylinderDetector.DEBUG_PRINTS:
                    logger = CylinderDetector.LOGGER or print
                    msg = f"[DEBUG][find_connected_components] --> Region rejected (too small)."
                    if callable(logger):
                        logger(msg)
                    else:
                        logger.log(msg)
        return components

    def project_component(self, component):
        """
        Project the points of a component onto a plane perpendicular to its direction.
        Returns Nx2 numpy array (2D projected points)
        """
        import numpy as np
        from utils import project_points_onto_plane
        pts = self.point_cloud.to_numpy()[component.indices]
        plane_point = np.mean(pts, axis=0)
        plane_normal = component.direction
        projected = project_points_onto_plane(pts, plane_point, plane_normal)
        # Find 2 orthogonal axes on the plane
        u = np.cross(plane_normal, [1,0,0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(plane_normal, [0,1,0])
        u = u / np.linalg.norm(u)
        v = np.cross(plane_normal, u)
        v = v / np.linalg.norm(v)
        # Express projected points in (u,v) basis
        proj2d = np.stack([np.dot(p - plane_point, u) for p in projected])
        proj2d = np.column_stack((proj2d, [np.dot(p - plane_point, v) for p in projected]))
        return proj2d

    def detect(self):
        """
        Full detection pipeline: directions -> components -> project -> circle detect -> cylinder candidates
        Returns: list of Cylinder objects
        """
        import numpy as np
        from utils import fibonacci_sphere
        from geometry import Cylinder
        from circle import Circle2DRansac
        cylinders = []
        directions = fibonacci_sphere(self.directions_samples)
        if CylinderDetector.DEBUG_PRINTS:
            logger = CylinderDetector.LOGGER or print
            msg = f"[DEBUG] Sampled {len(directions)} directions for detection."
            if callable(logger):
                logger(msg)
            else:
                logger.log(msg)
        circle_detector = Circle2DRansac(residual_threshold=self.circle_residual)
        total_components = 0
        total_circles = 0
        total_valid_candidates = 0
        for direction_idx, direction in enumerate(directions):
            components = self.find_connected_components(direction)
            if CylinderDetector.DEBUG_PRINTS:
                logger = CylinderDetector.LOGGER or print
                msg = f"[DEBUG] Direction {direction_idx}: Found {len(components)} connected components."
                if callable(logger):
                    logger(msg)
                else:
                    logger.log(msg)
            total_components += len(components)
            for comp_idx, comp in enumerate(components):
                proj2d = self.project_component(comp)
                center2d, radius2d, inliers2d = circle_detector.fit(proj2d)
                if center2d is None or len(inliers2d) == 0:
                    continue
                total_circles += 1
                # Map inliers2d back to 3D indices
                indices_3d = [comp.indices[i] for i in inliers2d]
                pts3d = self.point_cloud.to_numpy()[indices_3d]
                # --- Advanced 3D cylinder fitting ---
                from utils import fit_cylinder_least_squares
                center3d, axis, radius = fit_cylinder_least_squares(pts3d, direction_init=comp.direction)
                candidate = Cylinder(center3d, axis, radius, inliers=indices_3d)
                if self.is_cylinder_valid(candidate):
                    total_valid_candidates += 1
                    cylinders.append(candidate)
        if CylinderDetector.DEBUG_PRINTS:
            logger = CylinderDetector.LOGGER or print
            msg1 = f"[DEBUG] Total connected components across all directions: {total_components}"
            msg2 = f"[DEBUG] Total 2D circles detected: {total_circles}"
            msg3 = f"[DEBUG] Total valid cylinder candidates before merging: {total_valid_candidates}"
            for msg in [msg1, msg2, msg3]:
                if callable(logger):
                    logger(msg)
                else:
                    logger.log(msg)
        # Post-processing: merge overlapping cylinders
        cylinders = self.merge_cylinders(cylinders)
        cylinders = self.merge_tiny_clusters(cylinders)
        if CylinderDetector.DEBUG_PRINTS:
            logger = CylinderDetector.LOGGER or print
            msg = f"[DEBUG] Total cylinders after merging: {len(cylinders)}"
            if callable(logger):
                logger(msg)
            else:
                logger.log(msg)
        if callable(logger):
            logger(msg)
        else:
            logger.log(msg)
        return cylinders

    def is_cylinder_valid(self, cylinder):
        """
        Validate cylinder geometry and support.
        """
        if len(cylinder.inliers) < self.min_inliers:
            if CylinderDetector.DEBUG_PRINTS:
                logger = CylinderDetector.LOGGER or print
            msg = f"[DEBUG] Cylinder rejected: too few inliers ({len(cylinder.inliers)} < {self.min_inliers})"
            if callable(logger):
                logger(msg)
            else:
                logger.log(msg)
            return False
        if not (0 < cylinder.radius < self.max_radius):
            if CylinderDetector.DEBUG_PRINTS:
                logger = CylinderDetector.LOGGER or print
            msg = f"[DEBUG] Cylinder rejected: invalid radius ({cylinder.radius})"
            if callable(logger):
                logger(msg)
            else:
                logger.log(msg)
            return False
        # Check length (project inliers onto axis)
        from numpy import dot
        pts = self.point_cloud.to_numpy()[cylinder.inliers]
        rel = pts - cylinder.center
        proj = dot(rel, cylinder.axis)
        length = proj.max() - proj.min()
        if length < self.min_length:
            if CylinderDetector.DEBUG_PRINTS:
                logger = CylinderDetector.LOGGER or print
            msg = f"[DEBUG] Cylinder rejected: too short (length {length} < {self.min_length})"
            if callable(logger):
                logger(msg)
            else:
                logger.log(msg)
            return False
        return True

    def merge_cylinders(self, cylinders):
        """
        Merge similar/overlapping cylinders (simple greedy clustering).
        """
        import numpy as np
        merged = []
        used = set()
        for i, cyl1 in enumerate(cylinders):
            if i in used:
                continue
            group = [cyl1]
            for j, cyl2 in enumerate(cylinders):
                if j <= i or j in used:
                    continue
                angle = np.arccos(np.clip(np.dot(cyl1.axis, cyl2.axis), -1, 1))
                center_dist = np.linalg.norm(cyl1.center - cyl2.center)
                radius_diff = abs(cyl1.radius - cyl2.radius)
                if angle < self.angle_thresh and center_dist < self.center_thresh and radius_diff < self.radius_thresh:
                    group.append(cyl2)
                    used.add(j)
            # Merge group by averaging
            if len(group) == 1:
                merged.append(group[0])
            else:
                centers = np.array([c.center for c in group])
                axes = np.array([c.axis for c in group])
                radii = np.array([c.radius for c in group])
                inliers = sum([c.inliers for c in group], [])
                merged.append(type(group[0])(centers.mean(axis=0), axes.mean(axis=0), radii.mean(), inliers))
            used.add(i)
        return merged

    def merge_tiny_clusters(self, cylinders, min_inliers_merge=10, axis_thresh=0.2, center_thresh=0.1, radius_thresh=0.1):
        """
        Merge tiny cylinders (with inlier count below min_inliers_merge) into their nearest similar larger neighbor.
        """
        import numpy as np
        if not cylinders:
            return []
        large = [c for c in cylinders if len(c.inliers) >= min_inliers_merge]
        tiny = [c for c in cylinders if len(c.inliers) < min_inliers_merge]
        merged = large[:]
        for tc in tiny:
            best = None
            best_score = float('inf')
            for lc in large:
                angle = np.arccos(np.clip(np.dot(tc.axis, lc.axis), -1, 1))
                center_dist = np.linalg.norm(tc.center - lc.center)
                radius_diff = abs(tc.radius - lc.radius)
                score = angle + center_dist + radius_diff
                if angle < axis_thresh and center_dist < center_thresh and radius_diff < radius_thresh and score < best_score:
                    best = lc
                    best_score = score
            if best is not None:
                # Merge inliers and re-average center, axis, radius
                all_inliers = best.inliers + tc.inliers
                centers = np.vstack([best.center, tc.center])
                axes = np.vstack([best.axis, tc.axis])
                radii = np.array([best.radius, tc.radius])
                merged.remove(best)
                merged.append(type(best)(centers.mean(axis=0), axes.mean(axis=0), radii.mean(), all_inliers))
            else:
                merged.append(tc)
        return merged
