"""
Data Loader for PathSnap Trace-Lite
Parses .track, .nodes, .arcs, and .route files per trip.
Maintains compact NumPy representations and local metric projections.
"""
from pathlib import Path
from typing import Optional
import numpy as np

from src.config import TRAIN_DIR, TEST_DIR
from src.geo_utils import LocalProjector, SpatialIndex, calculate_heading

class TripData:
    """
    Encapsulates all data and spatial indexing for a single trip.
    """
    def __init__(self, trip_id: str, trip_dir: Path, has_route: bool = True):
        self.trip_id = trip_id
        self.trip_dir = Path(trip_dir)
        self.has_route = has_route
        
        self.track_lons: np.ndarray = np.array([])
        self.track_lats: np.ndarray = np.array([])
        self.track_times: np.ndarray = np.array([])
        
        self.nodes_lons: np.ndarray = np.array([])
        self.nodes_lats: np.ndarray = np.array([])
        
        self.arcs_from: np.ndarray = np.array([], dtype=np.int32)
        self.arcs_to: np.ndarray = np.array([], dtype=np.int32)
        
        self.reference_route: Optional[np.ndarray] = None
        
        self._load_data()
        self._setup_projection_and_index()

    def _load_data(self):
        # 1. Parse .track (lon, lat, timestamp)
        track_file = self.trip_dir / f"{self.trip_id}.track"
        track_arr = np.loadtxt(track_file, dtype=np.float64)
        if track_arr.ndim == 1:
            track_arr = track_arr.reshape(1, -1)
        self.track_lons = track_arr[:, 0]
        self.track_lats = track_arr[:, 1]
        self.track_times = track_arr[:, 2]

        # 2. Parse .nodes (lon, lat)
        nodes_file = self.trip_dir / f"{self.trip_id}.nodes"
        nodes_arr = np.loadtxt(nodes_file, dtype=np.float64)
        if nodes_arr.ndim == 1:
            nodes_arr = nodes_arr.reshape(1, -1)
        self.nodes_lons = nodes_arr[:, 0]
        self.nodes_lats = nodes_arr[:, 1]

        # 3. Parse .arcs (from_node, to_node)
        arcs_file = self.trip_dir / f"{self.trip_id}.arcs"
        arcs_arr = np.loadtxt(arcs_file, dtype=np.int32)
        if arcs_arr.ndim == 1:
            arcs_arr = arcs_arr.reshape(1, -1)
        self.arcs_from = arcs_arr[:, 0]
        self.arcs_to = arcs_arr[:, 1]

        # 4. Parse .route (optional)
        route_file = self.trip_dir / f"{self.trip_id}.route"
        if route_file.exists():
            route_arr = np.loadtxt(route_file, dtype=np.int32)
            if route_arr.ndim == 0:
                self.reference_route = np.array([int(route_arr)], dtype=np.int32)
            else:
                self.reference_route = route_arr
        else:
            self.reference_route = None

    def _setup_projection_and_index(self):
        # Center projection on trajectory centroid
        lon0 = float(np.mean(self.track_lons))
        lat0 = float(np.mean(self.track_lats))
        self.projector = LocalProjector(lon0, lat0)

        # Project track coordinates
        self.track_x, self.track_y = self.projector.project(self.track_lons, self.track_lats)

        # Project node coordinates
        self.nodes_x, self.nodes_y = self.projector.project(self.nodes_lons, self.nodes_lats)

        # Segment endpoints
        self.x1 = self.nodes_x[self.arcs_from]
        self.y1 = self.nodes_y[self.arcs_from]
        self.x2 = self.nodes_x[self.arcs_to]
        self.y2 = self.nodes_y[self.arcs_to]

        # Segment lengths and headings
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        self.arc_lengths = np.hypot(dx, dy)
        self.arc_headings = calculate_heading(dx, dy)

        # Build spatial index
        self.spatial_index = SpatialIndex(self.x1, self.y1, self.x2, self.y2)

        # Build forward adjacency list for network routing
        # mapping node -> list of (neighbor_node, arc_id, length)
        self.num_nodes = len(self.nodes_x)
        self.num_arcs = len(self.arcs_from)
        self.adj: list[list[tuple[int, int, float]]] = [[] for _ in range(self.num_nodes)]
        for arc_id in range(self.num_arcs):
            u = int(self.arcs_from[arc_id])
            v = int(self.arcs_to[arc_id])
            length = float(self.arc_lengths[arc_id])
            self.adj[u].append((v, arc_id, length))


def list_trips(partition: str = "train") -> list[str]:
    """List trip IDs in train or student_test partition."""
    p = TRAIN_DIR if partition == "train" else TEST_DIR
    trip_dirs = sorted([d.name for d in p.iterdir() if d.is_dir()])
    return trip_dirs


def load_trip(trip_id: str, partition: str = "train") -> TripData:
    """Load a specific trip by ID."""
    p = TRAIN_DIR if partition == "train" else TEST_DIR
    trip_dir = p / trip_id
    return TripData(trip_id, trip_dir)
