"""
Geospatial Utilities for PathSnap Trace-Lite
Local metric projection, exact geometric point-to-segment computations,
heading differences, and spatial indexing.
"""
import math
import numpy as np
from shapely.geometry import LineString, box
from shapely.strtree import STRtree
from src.config import EARTH_RADIUS_METERS

class LocalProjector:
    """
    Equirectangular projection centered at (lon0, lat0).
    Maps (lon, lat) in degrees to metric coordinates (x, y) in meters.
    """
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = lon0
        self.lat0 = lat0
        self.lat0_rad = math.radians(lat0)
        self.cos_lat0 = math.cos(self.lat0_rad)
        self.R = EARTH_RADIUS_METERS

    def project(self, lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project degrees to meters."""
        x = self.R * np.radians(lons - self.lon0) * self.cos_lat0
        y = self.R * np.radians(lats - self.lat0)
        return x, y

    def unproject(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unproject meters back to degrees."""
        lons = np.degrees(x / (self.R * self.cos_lat0)) + self.lon0
        lats = np.degrees(y / self.R) + self.lat0
        return lons, lats


def point_to_segment_distance(
    px: float, py: float,
    x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of exact point-to-segment distance and projected position.
    
    Parameters:
        px, py: point coordinates (scalar)
        x1, y1: start coordinates of segments (array of shape M)
        x2, y2: end coordinates of segments (array of shape M)
        
    Returns:
        distances: array of shape M
        fractions: projection fraction t in [0, 1]
        proj_x: projected point x
        proj_y: projected point y
    """
    vx = x2 - x1
    vy = y2 - y1
    seg_len_sq = vx * vx + vy * vy
    
    # Vector from p1 to point
    ux = px - x1
    uy = py - y1
    
    # Avoid division by zero for zero-length segments
    zero_mask = seg_len_sq <= 1e-12
    seg_len_sq_safe = np.where(zero_mask, 1.0, seg_len_sq)
    
    # Projection fraction t = dot(u, v) / ||v||^2
    t = (ux * vx + uy * vy) / seg_len_sq_safe
    t = np.clip(t, 0.0, 1.0)
    t = np.where(zero_mask, 0.0, t)
    
    # Closest point on segment
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    
    # Distance from point to closest point
    dx = px - proj_x
    dy = py - proj_y
    distances = np.hypot(dx, dy)
    
    return distances, t, proj_x, proj_y


def calculate_heading(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Calculate compass heading in degrees [0, 360) where 0 is North, 90 is East.
    dx = Easting change, dy = Northing change.
    """
    heading = np.degrees(np.arctan2(dx, dy))
    heading = (heading + 360.0) % 360.0
    return heading


def heading_difference(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """
    Calculate minimal absolute angle difference in degrees [0, 180].
    """
    diff = np.abs(h1 - h2) % 360.0
    return np.minimum(diff, 360.0 - diff)


class SpatialIndex:
    """
    Spatial index over road segments using Shapely STRtree.
    """
    def __init__(self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.num_segments = len(x1)
        
        # Build LineStrings for STRtree
        # In shapely >= 2.0, from_ragged_array or list of LineStrings
        lines = [LineString([(x1[i], y1[i]), (x2[i], y2[i])]) for i in range(self.num_segments)]
        self.tree = STRtree(lines)
        
    def query_radius(self, px: float, py: float, radius: float) -> np.ndarray:
        """
        Query segment indices within bounding box of given radius.
        Returns array of segment (arc) indices.
        """
        search_box = box(px - radius, py - radius, px + radius, py + radius)
        indices = self.tree.query(search_box)
        return indices
