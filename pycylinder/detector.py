"""
CylinderDetector: Main detection pipeline
"""
import numpy as np
# Use relative imports for package modules
from .geometry import Cylinder
from .utils import fibonacci_sphere, fit_cylinder_least_squares
from .logger import get_logger, LogLevel

class CylinderDetector:
    LOGGER = get_logger()  # Use the module-level logger
    
    def __init__(self, point_cloud, directions_samples=60, distance_threshold=0.02, normal_threshold=0.95, 
                 min_component_points=30, circle_residual=0.01, min_length=0.05, min_inliers=30, 
                 max_radius=2.0, angle_thresh=0.1, center_thresh=0.05, radius_thresh=0.05, 
                 max_mean_distance_to_surface=0.01, debug_prints=None):
        """
        Initialize the CylinderDetector.
        
        Args:
            point_cloud: PointCloud instance containing the 3D points and normals
            directions_samples: Number of directions to sample for initial detection
            distance_threshold: Maximum distance between points to be considered neighbors
            normal_threshold: Minimum dot product between normals for points to be considered similar
            min_component_points: Minimum number of points in a connected component
            circle_residual: Maximum allowed residual for circle fitting
            min_length: Minimum length of a valid cylinder
            min_inliers: Minimum number of inliers for a valid cylinder
            max_radius: Maximum allowed radius for a valid cylinder
            angle_thresh: Maximum angle (in radians) between axes for merging cylinders
            center_thresh: Maximum distance between centers for merging cylinders
            radius_thresh: Maximum radius difference for merging cylinders
            max_mean_distance_to_surface: Maximum allowed mean distance of points to cylinder surface
            debug_prints: Deprecated, kept for backward compatibility
        """
        # debug_prints is kept for backward compatibility but not used
        self._debug = debug_prints if debug_prints is not None else False
        self.logger = get_logger()  # Get the module logger

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
        self.max_mean_distance_to_surface = max_mean_distance_to_surface

        # Cylinder fitting parameters
        self.min_radius = 0.02  # Minimum cylinder radius in meters
        self.max_radius = 0.5   # Maximum cylinder radius in meters
        self.distance_threshold = 0.02  # Increased distance threshold for cylinder fitting (was 0.01)
        self.ransac_n = 20      # Number of points for RANSAC
        self.num_iterations = 1000  # Max number of iterations for RANSAC
        
        # Validation parameters - made more permissive
        self.min_inliers = 50    # Reduced minimum number of inliers required (was 100)
        self.min_cylinder_length = 0.05  # Reduced minimum length of a valid cylinder (was 0.1)
        self.max_mean_distance_to_surface = 0.08  # Increased max mean distance to surface (was 0.05)
        self.max_median_distance_to_surface = 0.1  # New parameter for median distance threshold
        self.mad_multiplier = 3.0  # Increased multiplier for MAD threshold (was 1.5)

    def find_connected_components(self, direction):
        """
        Segment the point cloud into connected components along a given direction.
        Args:
            direction: (3,) numpy array, main axis
        Returns:
            List of ConnectedComponent objects
        """
        import numpy as np
        from .geometry import ConnectedComponent
        
        # Use the correct methods to get points and normals
        points = self.point_cloud.to_numpy()
        normals = self.point_cloud.normals_numpy()  # This will be None if normals aren't available
        N = len(points)
        
        self.LOGGER.debug(f"[find_connected_components] Direction: {direction}, N_points={N}, N_normals={len(normals) if normals is not None else 0}")
        self.LOGGER.debug(f"[find_connected_components] distance_threshold={self.distance_threshold}, normal_threshold={self.normal_threshold}, min_component_points={self.min_component_points}")
        # Only log point details at debug level
        if self.LOGGER.isEnabledFor(LogLevel.DEBUG):
            for i in range(min(10, N)):
                pt_str = np.array2string(points[i], precision=3)
                if normals is not None:
                    norm_str = np.array2string(normals[i], precision=3)
                    self.LOGGER.debug(f"  Point {i}: {pt_str}, Normal: {norm_str}")
                else:
                    self.LOGGER.debug(f"  Point {i}: {pt_str}, Normal: None")
                # Log number of neighbors for this point
                dists = np.linalg.norm(points - points[i], axis=1)
                neighbors = np.where((dists < self.distance_threshold))[0]
                self.LOGGER.debug(f"    Num neighbors within distance_threshold: {len(neighbors)}")
        visited = np.zeros(N, dtype=bool)
        components = []
        self.LOGGER.debug(f"[find_connected_components] Beginning region growing over {N} points (potential regions).")
        # Throttling parameters
        MAX_REGION_LOGS = 10  # log details for first 10 regions
        REGION_LOG_EVERY = 1000  # then log every 1000th region
        region_count = 0
        accepted_count = 0
        rejected_count = 0
        region_sizes = []
        for idx in range(N):
            if visited[idx]:
                continue
            # Start region growing
            region_count += 1
            log_this_region = (
                region_count <= MAX_REGION_LOGS or
                region_count % REGION_LOG_EVERY == 0
            )
            if log_this_region:
                self.LOGGER.debug(f"[find_connected_components] Starting region from idx={idx} (region {region_count})")
            elif region_count == MAX_REGION_LOGS + 1:
                self.LOGGER.debug(f"[find_connected_components] ... (further region logs suppressed; will log every {REGION_LOG_EVERY}th region)")
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
            region_sizes.append(len(current))
            if len(current) >= self.min_component_points:
                component = ConnectedComponent(points[current], normals[current] if normals is not None else None, direction)
                components.append(component)
                if log_this_region:
                    self.LOGGER.debug("[find_connected_components] --> Region accepted as component.")
                accepted_count += 1
            else:
                if log_this_region:
                    self.LOGGER.debug("[find_connected_components] --> Region rejected (too small).")
                rejected_count += 1
        # Summary statistics
        if region_sizes:  # Only calculate stats if we have regions
            self.LOGGER.debug(
                f"[find_connected_components] Summary: {region_count} regions processed, "
                f"{accepted_count} accepted, {rejected_count} rejected. "
                f"Region sizes: min={min(region_sizes)}, "
                f"max={max(region_sizes)}, "
                f"mean={sum(region_sizes)/len(region_sizes):.2f}"
            )
        else:
            self.LOGGER.debug("[find_connected_components] No regions were processed")
        return components

    def project_component(self, component):
        """
        Project the points of a component onto a plane perpendicular to its direction.
        Returns Nx2 numpy array (2D projected points)
        """
        import numpy as np
        from .utils import project_points_onto_plane
        all_pts = self.point_cloud.to_numpy()
        pts = all_pts[component.indices]
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
        from .utils import fibonacci_sphere
        from .geometry import Cylinder
        from .circle import Circle2DRansac
        cylinders = []
        directions = fibonacci_sphere(self.directions_samples)
        circle_detector = Circle2DRansac(residual_threshold=self.circle_residual)
        total_components = 0
        total_circles = 0
        total_valid_candidates = 0
        for direction_idx, direction in enumerate(directions):
            components = self.find_connected_components(direction)
            self.logger.debug(f"[detect] Direction {direction_idx}: Found {len(components)} connected components.")
            total_components += len(components)
            for comp_idx, comp in enumerate(components):
                proj2d = self.project_component(comp)
                center2d, radius2d, inliers2d = circle_detector.fit(proj2d)
                if center2d is None or len(inliers2d) == 0:
                    continue
                total_circles += 1
                # Map inliers2d back to 3D indices
                indices_3d = [comp.indices[i] for i in inliers2d]
                all_pts = self.point_cloud.to_numpy()
                pts3d = all_pts[indices_3d]
                try:
                    # Fit cylinder to 3D points with improved fitting
                    center3d, axis, radius = fit_cylinder_least_squares(
                        pts3d, 
                        direction_init=comp.direction,
                        max_radius=self.max_radius,
                        max_iterations=100,
                        tolerance=1e-6
                    )
                    
                    # Create cylinder with points and inliers
                    candidate = Cylinder(
                        center=center3d,
                        axis=axis,
                        radius=radius,
                        inliers=indices_3d,
                        points=pts3d  # Pass the actual 3D points
                    )
                    
                    self.logger.debug(f"[detect] Fitted cylinder - "
                                    f"center: {np.round(center3d, 4)}, "
                                    f"axis: {np.round(axis, 4)}, "
                                    f"radius: {radius:.6f}, "
                                    f"points: {len(pts3d)}")
                    
                    if self.is_cylinder_valid(candidate):
                        total_valid_candidates += 1
                        cylinders.append(candidate)
                        self.logger.debug("[detect] Cylinder accepted")
                    else:
                        self.logger.debug("[detect] Cylinder rejected during validation")
                        
                except Exception as e:
                    self.logger.warning(f"[detect] Error fitting cylinder: {str(e)}")
                    continue
        self.logger.debug(f"[detect] Total valid cylinder candidates before merging: {total_valid_candidates}")
        # Post-processing: merge overlapping cylinders
        cylinders = self.merge_cylinders(cylinders)
        cylinders = self.merge_tiny_clusters(cylinders)
        self.logger.debug(f"[detect] Total cylinders after merging: {len(cylinders)}")
        return cylinders

    def is_cylinder_valid(self, cylinder):
        """Validate cylinder geometry and support.
        
        Args:
            cylinder: The cylinder to validate.
            
        Returns:
            bool: True if the cylinder is valid, False otherwise.
        """
        try:
            # Check if we have enough inliers
            if len(cylinder.inliers) < self.min_inliers:
                self.logger.debug(
                    f"Cylinder rejected: too few inliers ({len(cylinder.inliers)} < {self.min_inliers})"
                )
                return False
                
            # Check radius is within bounds
            if not (self.min_radius <= cylinder.radius <= self.max_radius):
                self.logger.debug(
                    f"Cylinder rejected: invalid radius ({cylinder.radius:.6f} not in "
                    f"[{self.min_radius:.2f}, {self.max_radius:.2f}])"
                )
                return False
                
            # Get the points for this cylinder and set them
            all_pts = self.point_cloud.to_numpy()
            pts = all_pts[cylinder.inliers]
            if len(pts) < 3:  # Need at least 3 points to define a cylinder
                self.logger.debug("Cylinder rejected: too few points after filtering")
                return False
                
            # Update cylinder with points (this will also update length and endpoints)
            cylinder.points = pts
            
            # Check length
            if cylinder.length < self.min_cylinder_length:
                self.logger.debug(
                    f"Cylinder rejected: too short (length {cylinder.length:.6f} < {self.min_cylinder_length:.6f})"
                )
                return False
                
            # Check if the cylinder has sufficient support (points close to the surface)
            rel = pts - cylinder.center
            proj = np.dot(rel, cylinder.axis)
            radial_vectors = rel - np.outer(proj, cylinder.axis)
            dist_to_surface = np.abs(np.linalg.norm(radial_vectors, axis=1) - cylinder.radius)
            
            # Calculate robust statistics for distance to surface
            median_dist = np.median(dist_to_surface)
            mean_dist = np.mean(dist_to_surface)
            mad = np.median(np.abs(dist_to_surface - median_dist))  # Median Absolute Deviation
            
            # Use a dynamic threshold based on the new parameters
            mad_threshold = self.mad_multiplier * mad
            
            # Log the fit quality for debugging
            self.logger.debug(
                f"Cylinder fit quality - median: {median_dist:.6f}, mean: {mean_dist:.6f}, "
                f"MAD: {mad:.6f} (threshold: {mad_threshold:.6f})"
            )
            
            # Check median distance to surface (more robust than mean)
            if median_dist > self.max_median_distance_to_surface:
                self.logger.debug(
                    f"Cylinder rejected: median distance too large ({median_dist:.6f} > "
                    f"{self.max_median_distance_to_surface:.6f})"
                )
                return False
                
            # Check mean distance to surface
            if mean_dist > self.max_mean_distance_to_surface:
                self.logger.debug(
                    f"Cylinder rejected: mean distance too large ({mean_dist:.6f} > "
                    f"{self.max_mean_distance_to_surface:.6f})"
                )
                return False
                
            # Check for axis stability
            if np.linalg.norm(cylinder.axis) < 1e-6:  # Check for zero vector
                self.logger.debug("Cylinder rejected: invalid axis (zero vector)")
                return False
                
            # Check that the cylinder has sufficient extent in 3D space
            if np.max(proj) - np.min(proj) < self.min_cylinder_length * 0.5:  # At least half the min length
                self.logger.debug("Cylinder rejected: insufficient extent along axis")
                return False
                
            # Check for degenerate cases (e.g., all points in a plane)
            if np.max(dist_to_surface) < 1e-6:
                self.logger.debug("Cylinder rejected: points lie on a plane (zero radius)")
                return False
                
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating cylinder: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

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
                
                # Combine points from all cylinders in the group
                all_points = []
                for c in group:
                    if hasattr(c, 'points') and c.points is not None:
                        all_points.append(c.points)
                
                # Create new cylinder with combined points
                new_cylinder = type(group[0]) (
                    center=centers.mean(axis=0),
                    axis=axes.mean(axis=0),
                    radius=radii.mean(),
                    inliers=inliers,
                    points=np.vstack(all_points) if all_points else None
                )
                merged.append(new_cylinder)
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
                
                # Combine points from both cylinders
                all_points = []
                for c in [best, tc]:
                    if hasattr(c, 'points') and c.points is not None:
                        all_points.append(c.points)
                
                # Create new cylinder with combined points
                new_cylinder = type(best)(
                    center=centers.mean(axis=0),
                    axis=axes.mean(axis=0),
                    radius=radii.mean(),
                    inliers=all_inliers,
                    points=np.vstack(all_points) if all_points else None
                )
                
                merged.remove(best)
                merged.append(new_cylinder)
            else:
                merged.append(tc)
        return merged
