"""
Custom Log-Space Viterbi / HMM Route Decoder for PathSnap Trace-Lite
Combines supervised classifier emission probabilities with network-constrained
transition likelihoods and Dijkstra shortest-path route reconstruction.
"""
import heapq
import numpy as np

from src.data_loader import TripData
from src.candidate_gen import ObservationCandidates

class DijkstraRouter:
    """
    Directed road-network shortest-path router with bounded Dijkstra and caching.
    """
    def __init__(self, adj: list[list[tuple[int, int, float]]]):
        self.adj = adj
        self.cache: dict[tuple[int, int], tuple[float, list[int]]] = {}

    def get_shortest_path(self, start_node: int, end_node: int, max_dist: float = 400.0) -> tuple[float, list[int]]:
        """
        Find shortest directed path distance and intermediate arc sequence between start_node and end_node.
        Returns:
            distance: float (inf if unreachable within max_dist)
            arcs: list of intermediate arc IDs
        """
        if start_node == end_node:
            return 0.0, []
            
        key = (start_node, end_node)
        if key in self.cache:
            return self.cache[key]
            
        # Dijkstra with bounded priority queue
        # pq entries: (dist, current_node, path_arcs)
        pq = [(0.0, start_node, [])]
        best_dists = {start_node: 0.0}
        
        found_dist = float("inf")
        found_arcs = []
        
        while pq:
            d, u, path = heapq.heappop(pq)
            if d > best_dists.get(u, float("inf")):
                continue
            if d > max_dist:
                break
            if u == end_node:
                found_dist = d
                found_arcs = path
                break
                
            for v, arc_id, length in self.adj[u]:
                new_d = d + length
                if new_d < max_dist and new_d < best_dists.get(v, float("inf")):
                    best_dists[v] = new_d
                    heapq.heappush(pq, (new_d, v, path + [arc_id]))
                    
        # Cache result
        self.cache[key] = (found_dist, found_arcs)
        return found_dist, found_arcs


class ViterbiRouteDecoder:
    """
    Custom log-space HMM Viterbi Decoder.
    """
    def __init__(
        self,
        trip: TripData,
        beta: float = 15.0,
        max_speed: float = 35.0,
        disconn_penalty: float = -3.5,
        backward_tol_m: float = 15.0
    ):
        self.trip = trip
        self.beta = beta
        self.max_speed = max_speed  # 35 m/s ≈ 126 km/h
        self.disconn_penalty = disconn_penalty  # Calibrated transition penalty to avoid state entrapment
        self.backward_tol_m = backward_tol_m    # Physical backward distance tolerance in meters for stationary GPS jitter
        self.router = DijkstraRouter(trip.adj)

    def decode(
        self,
        candidates_list: list[ObservationCandidates],
        candidate_probabilities: list[np.ndarray]
    ) -> tuple[np.ndarray, list[int], dict]:
        """
        Run Viterbi decoding over candidate lattice.
        
        Parameters:
            candidates_list: List of ObservationCandidates of length T
            candidate_probabilities: List of 1D arrays of model probabilities p(y=1) per candidate
            
        Returns:
            aligned_pred_arcs: (T,) arc ID per observation
            reconstructed_route: full list of arc IDs including intermediate segments
            diagnostics: dictionary with decode statistics
        """
        T = len(candidates_list)
        if T == 0:
            return np.array([], dtype=np.int32), [], {"status": "empty"}

        # 1. Compute normalized log emissions
        log_emissions = []
        eps = 1e-6
        for t in range(T):
            c_probs = np.copy(candidate_probabilities[t])
            if len(c_probs) == 0:
                log_emissions.append(np.array([-50.0], dtype=np.float32))
                continue
            # Clamp and normalize
            c_probs = np.clip(c_probs, eps, 1.0)
            p_sum = np.sum(c_probs)
            norm_p = c_probs / p_sum if p_sum > 0 else np.full_like(c_probs, 1.0 / len(c_probs))
            log_emissions.append(np.log(np.clip(norm_p, eps, 1.0)))

        # 2. Viterbi DP Tables
        NEG_INF = -1e9
        V = [np.copy(log_emissions[0])]
        backpointers = [np.full(len(candidates_list[0].arc_ids), -1, dtype=np.int32)]
        
        restarts_count = 0
        total_transitions = 0
        
        for t in range(1, T):
            K_prev = len(candidates_list[t-1].arc_ids)
            K_curr = len(candidates_list[t].arc_ids)
            
            # GPS displacement
            dx = self.trip.track_x[t] - self.trip.track_x[t-1]
            dy = self.trip.track_y[t] - self.trip.track_y[t-1]
            d_gps = float(np.hypot(dx, dy))
            dt = max(1.0, float(self.trip.track_times[t] - self.trip.track_times[t-1]))
            
            curr_v = np.full(K_curr, NEG_INF, dtype=np.float32)
            curr_bp = np.full(K_curr, -1, dtype=np.int32)
            
            c_prev = candidates_list[t-1]
            c_curr = candidates_list[t]
            
            for j in range(K_curr):
                arc_curr = int(c_curr.arc_ids[j])
                frac_curr = float(c_curr.fractions[j])
                v_node = int(self.trip.arcs_from[arc_curr])
                len_curr = float(self.trip.arc_lengths[arc_curr])
                
                best_score = NEG_INF
                best_i = -1
                
                for i in range(K_prev):
                    if V[t-1][i] <= NEG_INF / 2:
                        continue
                        
                    arc_prev = int(c_prev.arc_ids[i])
                    frac_prev = float(c_prev.fractions[i])
                    u_node = int(self.trip.arcs_to[arc_prev])
                    len_prev = float(self.trip.arc_lengths[arc_prev])
                    
                    # Network distance calculation
                    if arc_prev == arc_curr:
                        delta_dist = (frac_curr - frac_prev) * len_prev
                        if delta_dist >= -self.backward_tol_m:
                            d_net = max(0.0, delta_dist)
                        else:
                            # Moving backwards beyond physical GPS jitter tolerance
                            d_net = 500.0
                    elif u_node == v_node:
                        d_net = (1.0 - frac_prev) * len_prev + frac_curr * len_curr
                    else:
                        d_path, _ = self.router.get_shortest_path(u_node, v_node, max_dist=400.0)
                        if d_path < float("inf"):
                            d_net = (1.0 - frac_prev) * len_prev + d_path + frac_curr * len_curr
                        else:
                            d_net = 600.0
                            
                    # Transition log-likelihood
                    if d_net >= 500.0:
                        log_trans = self.disconn_penalty
                    else:
                        speed = d_net / dt
                        speed_penalty = max(0.0, speed - self.max_speed) * 2.0
                        log_trans = - (abs(d_net - d_gps) / self.beta) - speed_penalty
                        
                    score = V[t-1][i] + log_trans
                    if score > best_score:
                        best_score = score
                        best_i = i
                        
                if best_score > NEG_INF / 2:
                    curr_v[j] = log_emissions[t][j] + best_score
                    curr_bp[j] = best_i
                    
            # Restart if all branches broke
            if np.max(curr_v) <= -150.0 or np.all(curr_bp == -1):
                restarts_count += 1
                curr_v = np.copy(log_emissions[t])
                curr_bp = np.zeros(K_curr, dtype=np.int32)
                
            V.append(curr_v)
            backpointers.append(curr_bp)
            total_transitions += 1

        # 3. Backtracking
        best_last_idx = int(np.argmax(V[-1]))
        chosen_indices = np.zeros(T, dtype=np.int32)
        chosen_indices[-1] = best_last_idx
        
        for t in range(T - 1, 0, -1):
            chosen_indices[t-1] = backpointers[t][chosen_indices[t]]
            if chosen_indices[t-1] == -1:
                chosen_indices[t-1] = 0

        # Aligned predicted arc per observation
        aligned_pred_arcs = np.zeros(T, dtype=np.int32)
        for t in range(T):
            idx = chosen_indices[t]
            if len(candidates_list[t].arc_ids) > idx:
                aligned_pred_arcs[t] = candidates_list[t].arc_ids[idx]
            else:
                aligned_pred_arcs[t] = -1

        # 4. Reconstruct intermediate arcs for feasible transitions
        reconstructed_route = []
        for t in range(T):
            a_curr = aligned_pred_arcs[t]
            if a_curr == -1:
                continue
            if not reconstructed_route or reconstructed_route[-1] != a_curr:
                if reconstructed_route:
                    a_prev = reconstructed_route[-1]
                    u = int(self.trip.arcs_to[a_prev])
                    v = int(self.trip.arcs_from[a_curr])
                    if u != v:
                        _, intermediate_arcs = self.router.get_shortest_path(u, v, max_dist=400.0)
                        for mid_arc in intermediate_arcs:
                            if not reconstructed_route or reconstructed_route[-1] != mid_arc:
                                reconstructed_route.append(mid_arc)
                reconstructed_route.append(a_curr)

        diagnostics = {
            "total_observations": T,
            "restarts_count": restarts_count,
            "restart_frequency": float(restarts_count / T) if T > 0 else 0.0,
            "reconstructed_arcs_count": len(reconstructed_route),
            "unique_arcs_predicted": len(np.unique(aligned_pred_arcs))
        }

        return aligned_pred_arcs, reconstructed_route, diagnostics
