"""
runtime/som/terrain.py — Phase 3: SomTerrain
=============================================

The SomTerrain extends every SOM node with collective memory fields.
It is the map's own record of its history — accumulated across thousands
of agent lifetimes — and nobody programs it.  It emerges.

Geography that emerges:
  - Hot zones   : nodes where agents repeatedly felt positive valence
  - Cold zones  : nodes where agents felt danger or failure
  - Sacred places: high cultural_deposit — where dying agents left memories
  - Virgin territory: attractor_count ≈ 0, high exploration_reward — the frontier

Opcodes implemented here:
    TERRAIN_READ  0x66  — read collective state of current SOM node
    TERRAIN_MARK  0x67  — write emotional data into terrain at (r, c)

Paper: "A Path to AGI Part III: Curiosity", §4 The SomTerrain
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Per-node terrain data ─────────────────────────────────────────────────────

@dataclass
class TerrainNode:
    """
    Collective memory extension of a single SOM node.

    Fields set by accumulated agent experience — never by the programmer.
    """
    row: int
    col: int

    # Accumulated emotional history
    collective_valence:   float = 0.0   # rolling mean of all EMOT_TAGs here
    emotional_intensity:  float = 0.0   # peak intensity ever recorded
    attractor_count:      int   = 0     # how many agents visited
    cultural_deposit:     float = 0.0   # salience left by dying agents

    # Curiosity / exploration
    exploration_reward:   float = 1.0   # starts high; decays as agents visit
    last_visited_pulse:   int   = -1
    visit_streak:         int   = 0     # consecutive visits (expertise buildup)

    # Danger / caution signal
    danger_level:         float = 0.0   # accumulated negative valence

    def __post_init__(self) -> None:
        pass

    @property
    def is_virgin(self) -> bool:
        """True if almost no agent has been here — frontier territory."""
        return self.attractor_count < 3 and self.exploration_reward > 0.7

    @property
    def is_sacred(self) -> bool:
        """True if this node holds significant cultural memory."""
        return self.cultural_deposit > 0.5

    @property
    def is_hot_zone(self) -> bool:
        """True if this region has strongly positive collective valence."""
        return self.collective_valence > 0.3 and self.attractor_count >= 10

    @property
    def is_cold_zone(self) -> bool:
        """True if this region is collectively dangerous."""
        return self.danger_level > 0.3 and self.attractor_count >= 5

    def visit(self, pulse: int, valence: float, intensity: float) -> None:
        """
        Called when any agent fires EMOT_TAG at this coordinate.
        Updates collective emotional history.
        """
        n = max(self.attractor_count, 1)
        # Rolling mean with exponential decay
        decay = 0.95
        self.collective_valence  = (self.collective_valence * decay
                                    + valence * intensity * (1 - decay))
        self.emotional_intensity = max(self.emotional_intensity,
                                       intensity * 0.9)
        self.attractor_count     += 1
        self.last_visited_pulse  = pulse

        # Update danger level
        if valence < -0.2:
            self.danger_level = min(1.0, self.danger_level + abs(valence) * intensity * 0.1)
        else:
            self.danger_level = max(0.0, self.danger_level - 0.02)

        # Exploration reward decays as visits accumulate
        self.exploration_reward = math.exp(-self.attractor_count / 50.0)

    def deposit(self, salience: float) -> None:
        """
        Called when a dying agent deposits its memories here (SOUL_QUERY result).
        Marks this as a sacred place.
        """
        self.cultural_deposit = min(1.0, self.cultural_deposit + salience * 0.2)

    def decay_tick(self) -> None:
        """Slow cultural decay — sacred places fade across many generations."""
        self.cultural_deposit    = max(0.0, self.cultural_deposit - 0.0001)
        self.exploration_reward  = min(1.0, self.exploration_reward + 0.001)
        self.danger_level        = max(0.0, self.danger_level - 0.005)

    def to_dict(self) -> dict:
        return {
            "row":                self.row,
            "col":                self.col,
            "collective_valence": round(self.collective_valence, 4),
            "emotional_intensity":round(self.emotional_intensity, 4),
            "attractor_count":    self.attractor_count,
            "cultural_deposit":   round(self.cultural_deposit, 4),
            "exploration_reward": round(self.exploration_reward, 4),
            "danger_level":       round(self.danger_level, 4),
            "is_virgin":          self.is_virgin,
            "is_sacred":          self.is_sacred,
            "is_hot_zone":        self.is_hot_zone,
            "is_cold_zone":       self.is_cold_zone,
        }


# ── SomTerrain ────────────────────────────────────────────────────────────────

class SomTerrain:
    """
    Layer of collective memory overlaid on the SOM map.

    One TerrainNode per (row, col) coordinate, shared across all agents.
    Thread-safe.  Persists for the lifetime of the SOM map (even as agents
    are born, die, and are replaced by EVOLVE selection).

    Usage
    -----
        terrain = SomTerrain(rows=16, cols=16)

        # Agent fires EMOT_TAG at (3, 7) with valence +0.8
        terrain.mark(row=3, col=7, pulse=t,
                     valence=0.8, intensity=0.9)

        # Another agent reads terrain before navigating
        info = terrain.read(row=3, col=7)
        if info["is_hot_zone"]:
            ...  # navigate toward this attractor

        # Find most-curious destination (virgin + high exploration reward)
        r, c = terrain.most_curious_node()
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self._nodes: List[List[TerrainNode]] = [
            [TerrainNode(r, c) for c in range(cols)]
            for r in range(rows)
        ]
        self._pulse: int = 0
        self._lock  = threading.Lock()

    def _node(self, row: int, col: int) -> TerrainNode:
        r = max(0, min(self.rows - 1, row))
        c = max(0, min(self.cols - 1, col))
        return self._nodes[r][c]

    # ── TERRAIN_MARK opcode ───────────────────────────────────────────────────

    def mark(self, row: int, col: int,
             pulse: int,
             valence: float, intensity: float) -> TerrainNode:
        """
        TERRAIN_MARK — agent fires emotional tag at (row, col).
        Updates collective terrain memory.
        """
        with self._lock:
            node = self._node(row, col)
            node.visit(pulse, valence, intensity)
            return node

    def deposit_soul(self, row: int, col: int, salience: float) -> None:
        """
        Called when a dying agent deposits cultural memory at its last position.
        Creates or deepens a 'sacred place' in the terrain.
        """
        with self._lock:
            self._node(row, col).deposit(salience)

    # ── TERRAIN_READ opcode ───────────────────────────────────────────────────

    def read(self, row: int, col: int) -> dict:
        """
        TERRAIN_READ — return collective memory state at (row, col).
        Used by curious agents deciding where to explore next.
        """
        with self._lock:
            return self._node(row, col).to_dict()

    # ── Navigation helpers ────────────────────────────────────────────────────

    def most_curious_node(self) -> Tuple[int, int]:
        """
        Return the (row, col) with the highest exploration_reward.
        Curious agents navigate here when goal stalls.
        """
        with self._lock:
            best_r, best_c = 0, 0
            best_reward = -1.0
            for r in range(self.rows):
                for c in range(self.cols):
                    n = self._nodes[r][c]
                    if n.exploration_reward > best_reward:
                        best_reward = n.exploration_reward
                        best_r, best_c = r, c
        return best_r, best_c

    def most_attractive_node(self) -> Tuple[int, int]:
        """
        Return (row, col) with highest positive collective_valence.
        Exploitation-mode agents navigate here.
        """
        with self._lock:
            best_r, best_c = 0, 0
            best_val = -2.0
            for r in range(self.rows):
                for c in range(self.cols):
                    n = self._nodes[r][c]
                    score = n.collective_valence * n.attractor_count
                    if score > best_val:
                        best_val = score
                        best_r, best_c = r, c
        return best_r, best_c

    def nearest_virgin(self, from_r: int, from_c: int) -> Tuple[int, int]:
        """
        Return nearest virgin node to (from_r, from_c).
        Ties broken by highest exploration_reward.
        """
        with self._lock:
            best_r, best_c = from_r, from_c
            best_score = -1.0
            for r in range(self.rows):
                for c in range(self.cols):
                    n = self._nodes[r][c]
                    if not n.is_virgin:
                        continue
                    dist = math.sqrt((r - from_r) ** 2 + (c - from_c) ** 2)
                    score = n.exploration_reward / (1.0 + dist * 0.1)
                    if score > best_score:
                        best_score = score
                        best_r, best_c = r, c
        return best_r, best_c

    def danger_at(self, row: int, col: int) -> float:
        """Return collective danger level at node — [0, 1]."""
        with self._lock:
            return self._node(row, col).danger_level

    # ── Pulse tick ────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Slow cultural decay across all terrain nodes."""
        self._pulse += 1
        with self._lock:
            for row in self._nodes:
                for node in row:
                    node.decay_tick()

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Full terrain state — useful for visualisation."""
        with self._lock:
            return {
                "rows":   self.rows,
                "cols":   self.cols,
                "pulse":  self._pulse,
                "nodes":  [
                    self._nodes[r][c].to_dict()
                    for r in range(self.rows)
                    for c in range(self.cols)
                    if (self._nodes[r][c].attractor_count > 0
                        or self._nodes[r][c].cultural_deposit > 0)
                ],
                "hot_zones":    sum(1 for r in self._nodes
                                    for n in r if n.is_hot_zone),
                "cold_zones":   sum(1 for r in self._nodes
                                    for n in r if n.is_cold_zone),
                "sacred_places":sum(1 for r in self._nodes
                                    for n in r if n.is_sacred),
                "virgin_nodes": sum(1 for r in self._nodes
                                    for n in r if n.is_virgin),
            }

    def heatmap(self) -> List[List[float]]:
        """
        Return a 2D grid of collective_valence for visualisation.
        Positive = red / hot, Negative = blue / cold.
        """
        with self._lock:
            return [
                [self._nodes[r][c].collective_valence for c in range(self.cols)]
                for r in range(self.rows)
            ]


# ── Global terrain registry ───────────────────────────────────────────────────

class TerrainRegistry:
    """
    One SomTerrain per SOM map, keyed by (rows, cols) or an explicit map_id.
    Thread-safe.
    """

    def __init__(self):
        self._terrains: Dict[int, SomTerrain] = {}
        self._lock = threading.Lock()

    def get_or_create(self, map_id: int,
                      rows: int = 10, cols: int = 10) -> SomTerrain:
        with self._lock:
            if map_id not in self._terrains:
                self._terrains[map_id] = SomTerrain(rows, cols)
            return self._terrains[map_id]

    def tick_all(self) -> None:
        with self._lock:
            for t in self._terrains.values():
                t.tick()

    def snapshot(self) -> dict:
        with self._lock:
            return {mid: t.snapshot() for mid, t in self._terrains.items()}
