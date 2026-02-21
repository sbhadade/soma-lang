"""
runtime/som/som_map.py — Phase 2 + 2.5: Live SOM topology
==========================================================

Thread-safe SOM map shared across all SOMA agents.

Phase 2  : BMU / TRAIN / WALK / ELECT — real Kohonen update, activation tracking
Phase 2.5: DECAY_STEP / PRUNE_CHECK — weight erosion + synaptic pruning

Paper: "A Path to AGI Part II: Liveliness"
  w(t+1) = w(t) * (1 - decay_rate)  if node NOT activated this pulse
  w(t+1) = w(t)                      if node WAS activated (protected by emotion)
"""
from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

VEC_DIM = 8          # matches soma_runtime.h SOMA_VEC_DIM


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

        self._lock = threading.RLock()   # RW simulation via reentrant lock

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
        """Distributed init — prevents dead neurons."""
        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    xf = c / max(self.cols - 1, 1)
                    yf = r / max(self.rows - 1, 1)
                    self.nodes[r][c].weights = [
                        xf if d % 2 == 0 else yf for d in range(self.dims)
                    ]
                    self.nodes[r][c].activation = 0.0
                    self.nodes[r][c].hit_count  = 0

    # ── BMU ─────────────────────────────────────────────────────────────────

    def bmu(self, vec: List[float]) -> Tuple[int, int]:
        """Find Best Matching Unit. Thread-safe read."""
        best_dist = float("inf")
        best_r, best_c = 0, 0
        # Pad / truncate vec to self.dims
        v = (list(vec) + [0.0] * self.dims)[:self.dims]
        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    d = _euclidean_sq(self.nodes[r][c].weights, v)
                    if d < best_dist:
                        best_dist = d
                        best_r, best_c = r, c
        return best_r, best_c

    # ── TRAIN ────────────────────────────────────────────────────────────────

    def train(self, vec: List[float], bmu_r: int, bmu_c: int,
              lr: Optional[float] = None, sigma: Optional[float] = None):
        """
        Kohonen update with Gaussian neighbourhood.
        Updates activation on every node touched.
        """
        lr    = lr    if lr    is not None else self.lr
        sigma = sigma if sigma is not None else self.sigma
        v = (list(vec) + [0.0] * self.dims)[:self.dims]

        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    dist_sq = (r - bmu_r) ** 2 + (c - bmu_c) ** 2
                    influence = math.exp(-dist_sq / (2.0 * sigma * sigma))
                    if influence < 1e-5:
                        continue
                    node = self.nodes[r][c]
                    node.weights = [
                        w + lr * influence * (vi - w)
                        for w, vi in zip(node.weights, v)
                    ]
                    if influence > node.activation:
                        node.activation = influence

            self.nodes[bmu_r][bmu_c].hit_count += 1
            self.nodes[bmu_r][bmu_c].activated_this_pulse = True  # Phase 2.5
            self.epoch += 1

    # ── WALK ────────────────────────────────────────────────────────────────

    def walk_gradient(self, r: int, c: int) -> Tuple[int, int]:
        """
        Move one step toward the highest-activation neighbour (8-directional).
        Returns new (row, col).
        """
        best_act = -1.0
        best_r, best_c = r, c

        with self._lock:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        act = self.nodes[nr][nc].activation
                        if act > best_act:
                            best_act = act
                            best_r, best_c = nr, nc
        return best_r, best_c

    def walk_random(self, r: int, c: int) -> Tuple[int, int]:
        """Random walk — fallback when map has no activation yet."""
        dr = random.choice([-1, 0, 1])
        dc = random.choice([-1, 0, 1])
        nr = max(0, min(self.rows - 1, r + dr))
        nc = max(0, min(self.cols - 1, c + dc))
        return nr, nc

    # ── ELECT ────────────────────────────────────────────────────────────────

    def elect(self, agent_positions: List[Tuple[int, int, int]]) -> int:
        """
        Democratic leader election.

        Parameters
        ----------
        agent_positions : list of (agent_id, som_r, som_c)

        Returns
        -------
        agent_id of the agent whose SOM node has the highest hit_count.
        Ties broken by lowest agent_id.
        """
        best_hits = -1
        leader_id = -1

        with self._lock:
            for aid, r, c in agent_positions:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    hits = self.nodes[r][c].hit_count
                    if hits > best_hits or (hits == best_hits and
                       (leader_id == -1 or aid < leader_id)):
                        best_hits = hits
                        leader_id = aid

        return leader_id if leader_id != -1 else (agent_positions[0][0] if agent_positions else 0)

    # ── Decay ────────────────────────────────────────────────────────────────

    def decay(self, lr_rate: float = 0.01, sigma_rate: float = 0.005):
        """Decay learning rate and neighbourhood sigma each epoch."""
        self.lr    = max(0.001, self.lr    * (1.0 - lr_rate))
        self.sigma = max(0.5,   self.sigma * (1.0 - sigma_rate))

    # ── Sense ────────────────────────────────────────────────────────────────

    def sense(self, r: int, c: int) -> float:
        """Return mean activation of node (r, c). Range [0, 1]."""
        with self._lock:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                return self.nodes[r][c].activation
        return 0.0

    def node_dist(self, r1: int, c1: int, r2: int, c2: int) -> float:
        return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

    # ── Phase 2.5: DECAY_STEP ────────────────────────────────────────────────

    def decay_step(self,
                   protected_coords: Optional[List[Tuple[int,int]]] = None,
                   base_rate: float = 0.001) -> int:
        """
        DECAY_STEP opcode — fires on every PULSE.

        For every node NOT activated this pulse AND NOT emotion-protected:
            w(t+1) = w(t) * (1 - decay_rate)

        Activated nodes and emotion-protected nodes are shielded.
        Returns number of nodes that were decayed.

        Parameters
        ----------
        protected_coords : list of (r, c) that are emotion-protected
                           (comes from EmotionRegistry).
        base_rate : default decay rate if node.decay_rate not overridden.
        """
        protected_set = set(protected_coords) if protected_coords else set()
        decayed = 0

        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    node = self.nodes[r][c]
                    # Skip: activated this pulse, or emotion-protected
                    if node.activated_this_pulse or (r, c) in protected_set:
                        node.activated_this_pulse = False   # reset for next pulse
                        continue
                    rate = node.decay_rate if node.decay_rate > 0 else base_rate
                    node.weights = [w * (1.0 - rate) for w in node.weights]
                    node.activation *= (1.0 - rate * 0.5)  # activation fades too
                    node.activated_this_pulse = False
                    decayed += 1

        return decayed

    # ── Phase 2.5: PRUNE_CHECK ───────────────────────────────────────────────

    def prune_check(self, threshold: float = 0.01) -> int:
        """
        PRUNE_CHECK opcode — zero out weights below `threshold` strength.

        Synaptic pruning: if a node's L2 norm falls below threshold,
        its weights are reset to zero. This frees the node for
        new learning.

        Returns number of nodes pruned.
        """
        pruned = 0
        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    node = self.nodes[r][c]
                    strength = math.sqrt(sum(w*w for w in node.weights))
                    if strength < threshold:
                        node.weights   = [0.0] * self.dims
                        node.activation = 0.0
                        pruned += 1
        return pruned

    def mark_activated(self, r: int, c: int) -> None:
        """
        Called by TRAIN to mark a node as activated this pulse.
        Prevents DECAY_STEP from eroding it.
        """
        with self._lock:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.nodes[r][c].activated_this_pulse = True

    def node_strength(self, r: int, c: int) -> float:
        """L2 norm of node weights. Used by memory consolidation ranking."""
        with self._lock:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                return math.sqrt(sum(w*w for w in self.nodes[r][c].weights))
        return 0.0

    # ── Snapshot ────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of the SOM state."""
        with self._lock:
            return {
                "rows": self.rows,
                "cols": self.cols,
                "epoch": self.epoch,
                "lr": self.lr,
                "sigma": self.sigma,
                "nodes": [
                    {
                        "row": r, "col": c,
                        "activation": self.nodes[r][c].activation,
                        "hit_count":  self.nodes[r][c].hit_count,
                    }
                    for r in range(self.rows)
                    for c in range(self.cols)
                ],
            }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _euclidean_sq(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))
