"""
runtime/som/som_map.py — Phase 2.5 + 2.6: Live SOM Topology
============================================================

Phase 2.5 (done):
  decay_step()      — weight erosion per PULSE
  prune_check()     — hard-zero weights below threshold
  node_strength()   — scalar weight magnitude
  mark_activated()  — protect from this pulse's decay

Phase 2.6 (new):
  reorg_map()          — REORG_MAP: rebalance topology after pruning fills dead zones
  set_region_decay_rate() — DECAY_RATE_SET: per-region custom decay rates
  neighbour_coords()   — utility for NEIGHBOR_SYNC radius queries
"""
from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

VEC_DIM = 8


# ── SOM node ─────────────────────────────────────────────────────────────────

@dataclass
class SomNode:
    row: int
    col: int
    weights: List[float] = field(default_factory=lambda: [random.random() for _ in range(VEC_DIM)])
    activation: float = 0.0    # last Gaussian influence received
    hit_count: int = 0         # times this node was BMU
    # Phase 2.5 — decay fields
    decay_rate:       float = 0.001   # per-pulse weight erosion rate
    activated_this_pulse: bool = False  # set True by TRAIN, cleared by DECAY_STEP
    emotion_protected: bool = False     # set by DECAY_PROTECT, blocks decay


class LiveSomMap:
    """
    Thread-safe live SOM topology.

    Usage
    -----
    som = LiveSomMap(rows=16, cols=16)
    bmu_r, bmu_c = som.bmu(vec)
    som.train(vec, bmu_r, bmu_c)
    som.walk_gradient(agent_r, agent_c)  # → (new_r, new_c)
    leader_r, leader_c = som.elect(agents)
    """

    def __init__(self, rows: int = 16, cols: int = 16,
                 dims: int = VEC_DIM, lr: float = 0.5, sigma: float = 3.0):
        self.rows  = rows
        self.cols  = cols
        self.dims  = dims
        self.lr    = lr
        self.sigma = sigma
        self.epoch = 0

        self._lock = threading.RLock()

        self.nodes: List[List[SomNode]] = [
            [SomNode(r, c, [random.gauss(0.5, 0.15) for _ in range(dims)])
             for c in range(cols)]
            for r in range(rows)
        ]

    # ── Init ────────────────────────────────────────────────────────────────

    def init_random(self):
        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    self.nodes[r][c].weights = [random.gauss(0.5, 0.15) for _ in range(self.dims)]
                    self.nodes[r][c].activation = 0.0
                    self.nodes[r][c].hit_count  = 0

    def init_grid(self):
        """Distributed initialisation — prevents dead neurons."""
        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    xf = c / max(self.cols - 1, 1)
                    yf = r / max(self.rows - 1, 1)
                    self.nodes[r][c].weights = [
                        xf if d % 2 == 0 else yf
                        for d in range(self.dims)
                    ]
                    self.nodes[r][c].activation = 0.0
                    self.nodes[r][c].hit_count  = 0

    # ── BMU ─────────────────────────────────────────────────────────────────

    def bmu(self, vec: List[float]) -> Tuple[int, int]:
        """SOM_BMU — find best matching unit. Returns (row, col)."""
        if len(vec) < self.dims:
            vec = vec + [0.0] * (self.dims - len(vec))
        vec = vec[:self.dims]

        best_dist = float("inf")
        best_r, best_c = 0, 0

        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    d = _euclidean_sq(vec, self.nodes[r][c].weights)
                    if d < best_dist:
                        best_dist = d
                        best_r, best_c = r, c
        return best_r, best_c

    # ── Train ───────────────────────────────────────────────────────────────

    def train(self, vec: List[float], bmu_r: int, bmu_c: int,
              lr: Optional[float] = None, sigma: Optional[float] = None):
        """SOM_TRAIN — Kohonen update with Gaussian neighbourhood."""
        if len(vec) < self.dims:
            vec = vec + [0.0] * (self.dims - len(vec))
        vec = vec[:self.dims]

        lr    = lr    if lr    is not None else self.lr
        sigma = sigma if sigma is not None else self.sigma

        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    dr = r - bmu_r
                    dc = c - bmu_c
                    dist_sq = dr * dr + dc * dc
                    h = math.exp(-dist_sq / (2 * sigma * sigma))
                    if h < 1e-4:
                        continue
                    node = self.nodes[r][c]
                    node.weights = [
                        w + lr * h * (v - w)
                        for w, v in zip(node.weights, vec)
                    ]
                    if h > node.activation:
                        node.activation = h

            bmu_node = self.nodes[bmu_r][bmu_c]
            bmu_node.hit_count += 1
            bmu_node.activated_this_pulse = True
            self.epoch += 1

    # ── Walk ────────────────────────────────────────────────────────────────

    def walk_gradient(self, r: int, c: int) -> Tuple[int, int]:
        """SOM_WALK — move toward highest-activation neighbour."""
        best_val = -float("inf")
        best_r, best_c = r, c

        with self._lock:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        val = self.nodes[nr][nc].activation
                        if val > best_val:
                            best_val = val
                            best_r, best_c = nr, nc
        return best_r, best_c

    def walk_random(self, r: int, c: int) -> Tuple[int, int]:
        """Random walk — one step in a random cardinal direction."""
        options = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                options.append((nr, nc))
        return random.choice(options) if options else (r, c)

    # ── Elect ───────────────────────────────────────────────────────────────

    def elect(self, agent_positions: List[Tuple[int, int, int]]) -> int:
        """
        SOM_ELECT — democratic leader election.
        Agent with highest hit_count at its SOM position wins.
        Returns winning agent_id.
        """
        if not agent_positions:
            return 0

        best_score = -1
        leader_id  = agent_positions[0][0]

        with self._lock:
            for agent_id, r, c in agent_positions:
                score = self.nodes[r][c].hit_count
                if score > best_score:
                    best_score = score
                    leader_id  = agent_id

        return leader_id

    # ── LR / sigma decay ────────────────────────────────────────────────────

    def decay(self, lr_rate: float = 0.01, sigma_rate: float = 0.005):
        """LR_DECAY — reduce learning rate and neighbourhood."""
        self.lr    = max(0.001, self.lr    * (1.0 - lr_rate))
        self.sigma = max(0.5,   self.sigma * (1.0 - sigma_rate))

    # ── Sense ───────────────────────────────────────────────────────────────

    def sense(self, r: int, c: int) -> float:
        """SOM_SENSE — mean weight of node (r, c). Clamped to [-1, 1]."""
        with self._lock:
            w = self.nodes[r][c].weights
        avg = sum(w) / len(w) if w else 0.0
        return max(-1.0, min(1.0, avg))

    def node_dist(self, r1: int, c1: int, r2: int, c2: int) -> float:
        """Topological distance between two SOM nodes."""
        return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

    # ── Phase 2.5: Decay step ────────────────────────────────────────────────

    def decay_step(self,
                   protected_coords: Optional[List[Tuple[int,int]]] = None,
                   base_rate: float = 0.001) -> int:
        """
        DECAY_STEP opcode — fires on every PULSE.

        For every node NOT activated this pulse AND NOT emotion-protected:
            w(t+1) = w(t) * (1 - decay_rate)

        Returns number of nodes that were decayed.
        """
        protected_set = set(protected_coords) if protected_coords else set()
        decayed = 0

        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    node = self.nodes[r][c]
                    if node.activated_this_pulse or (r, c) in protected_set:
                        node.activated_this_pulse = False
                        continue
                    rate = node.decay_rate if node.decay_rate > 0 else base_rate
                    node.weights = [w * (1.0 - rate) for w in node.weights]
                    decayed += 1
        return decayed

    # ── Phase 2.5: Prune check ───────────────────────────────────────────────

    def prune_check(self, threshold: float = 0.01) -> int:
        """
        PRUNE_CHECK opcode — hard-zero weights below threshold.
        Returns count of pruned nodes.
        """
        pruned = 0
        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    node = self.nodes[r][c]
                    strength = math.sqrt(
                        sum(w * w for w in node.weights)
                    ) / math.sqrt(self.dims)
                    if 0 < strength < threshold:
                        node.weights = [0.0] * self.dims
                        node.activation = 0.0
                        pruned += 1
        return pruned

    def mark_activated(self, r: int, c: int) -> None:
        """Mark node (r, c) as activated this pulse — shields from DECAY_STEP."""
        with self._lock:
            self.nodes[r][c].activated_this_pulse = True

    def node_strength(self, r: int, c: int) -> float:
        """RMS weight magnitude at node (r, c). Range [0, ~1]."""
        with self._lock:
            w = self.nodes[r][c].weights
        return math.sqrt(sum(v * v for v in w) / max(len(w), 1))

    # ── Phase 2.6: REORG_MAP ─────────────────────────────────────────────────

    def reorg_map(self, dead_threshold: float = 1e-4,
                  spread_radius: int = 2) -> int:
        """
        REORG_MAP opcode — rebalance SOM topology after heavy pruning.

        Dead nodes (all weights ≈ 0) are re-seeded from the average of
        their living neighbours within `spread_radius`. This prevents
        permanent dead zones and allows the map to re-fill over subsequent
        TRAIN pulses.

        Returns number of nodes that were reseeded.
        """
        reseeded = 0

        with self._lock:
            # First pass: collect dead node coords
            dead_coords = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if node_is_dead(self.nodes[r][c], dead_threshold):
                        dead_coords.append((r, c))

            # Second pass: reseed each dead node from living neighbours
            for (r, c) in dead_coords:
                neighbour_weights: List[List[float]] = []
                for dr in range(-spread_radius, spread_radius + 1):
                    for dc in range(-spread_radius, spread_radius + 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            n = self.nodes[nr][nc]
                            if not node_is_dead(n, dead_threshold):
                                neighbour_weights.append(n.weights)

                if neighbour_weights:
                    # Average neighbours + small noise
                    avg = [
                        sum(nw[d] for nw in neighbour_weights)
                        / len(neighbour_weights)
                        for d in range(self.dims)
                    ]
                    noise_scale = 0.05
                    self.nodes[r][c].weights = [
                        max(0.0, w + random.gauss(0, noise_scale))
                        for w in avg
                    ]
                    self.nodes[r][c].activation = 0.0
                    self.nodes[r][c].hit_count  = 0
                    reseeded += 1
                else:
                    # No living neighbours — random reseed
                    self.nodes[r][c].weights = [
                        random.gauss(0.5, 0.15) for _ in range(self.dims)
                    ]
                    reseeded += 1

        return reseeded

    # ── Phase 2.6: DECAY_RATE_SET ────────────────────────────────────────────

    def set_region_decay_rate(self,
                              center_r: int, center_c: int,
                              radius: float,
                              rate: float) -> int:
        """
        DECAY_RATE_SET opcode — apply a custom decay rate to a circular
        region of the SOM.

        Low rate (e.g. 0.0001) = expertise zone — memories persist.
        High rate (e.g. 0.05)  = stale zone — memories fade fast.

        Returns number of nodes updated.
        """
        rate    = max(0.0, min(1.0, rate))
        updated = 0

        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    dist = math.sqrt(
                        (r - center_r) ** 2 + (c - center_c) ** 2
                    )
                    if dist <= radius:
                        self.nodes[r][c].decay_rate = rate
                        updated += 1

        return updated

    # ── Phase 2.6: Neighbour coords ──────────────────────────────────────────

    def neighbour_coords(self, r: int, c: int,
                         radius: float = 2.0) -> List[Tuple[int, int]]:
        """
        Return all (row, col) coords within topological radius of (r, c).
        Used by NEIGHBOR_SYNC to find who to share with.
        """
        coords = []
        ir = int(math.ceil(radius))
        for dr in range(-ir, ir + 1):
            for dc in range(-ir, ir + 1):
                nr, nc = r + dr, c + dc
                if (dr == 0 and dc == 0):
                    continue
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if math.sqrt(dr * dr + dc * dc) <= radius:
                    coords.append((nr, nc))
        return coords

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            total    = self.rows * self.cols
            dead     = sum(
                1 for r in range(self.rows)
                for c in range(self.cols)
                if node_is_dead(self.nodes[r][c])
            )
            mean_str = sum(
                self.node_strength(r, c)
                for r in range(self.rows)
                for c in range(self.cols)
            ) / total
        with self._lock:
            nodes = [
                {
                    "row": r, "col": c,
                    "activation": self.nodes[r][c].activation,
                    "hit_count":  self.nodes[r][c].hit_count,
                    "strength":   round(self.node_strength(r, c), 4),
                    "is_dead":    node_is_dead(self.nodes[r][c]),
                }
                for r in range(self.rows)
                for c in range(self.cols)
            ]
        return {
            "rows": self.rows, "cols": self.cols,
            "epoch": self.epoch,
            "lr": self.lr, "sigma": self.sigma,
            "dead_nodes": dead,
            "live_nodes": total - dead,
            "mean_node_strength": round(mean_str, 4),
            "nodes": nodes,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def node_is_dead(node: SomNode, threshold: float = 1e-4) -> bool:
    """True if all weights are effectively zero."""
    return all(abs(w) < threshold for w in node.weights)


def _euclidean_sq(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))
