"""
runtime/som/som_scheduler.py — Phase 2 + 2.5: SOM-driven agent scheduler
=========================================================================

The SOM topology IS the scheduler.
Agents migrate toward high-activation regions automatically after each
SOM_TRAIN or SOM_WALK opcode — no OS scheduler intervention needed.

Wires into AgentRegistry + LiveSomMap.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

from runtime.som.som_map  import LiveSomMap
from runtime.som.emotion  import EmotionRegistry, ProtectMode
from runtime.som.memory   import MemoryManager, ConsolidationReport

if TYPE_CHECKING:
    from runtime.agent.agent_registry import AgentRegistry


class SomScheduler:
    """
    Manages the relationship between live agents and the SOM topology.

    Phase 2  : SOM_MAP / SOM_TRAIN / SOM_WALK / SOM_ELECT / SOM_BMU / SOM_NBHD
    Phase 2.5: DECAY_STEP / PRUNE_CHECK / EMOT_TAG / DECAY_PROTECT /
               PREDICT_ERR / MEMORY_CONSOLIDATE

    Responsibilities
    ----------------
    1. Place newly spawned agents on the SOM (SOM_MAP opcode).
    2. After SOM_TRAIN: update the spawning agent's activation.
    3. After SOM_WALK: migrate the agent to the highest-activation neighbour.
    4. For SOM_ELECT: find the agent whose node has the highest hit_count.
    5. Decay lr/sigma periodically (background thread, optional).
    """

    def __init__(self, som: LiveSomMap, registry: "AgentRegistry"):
        self.som          = som
        self.registry     = registry
        self.emotion_reg  = EmotionRegistry()
        self.memory_mgr   = MemoryManager(som, self.emotion_reg)
        self._decay_thread: Optional[threading.Thread] = None
        self._stop_decay   = threading.Event()

    # ── Placement ────────────────────────────────────────────────────────────

    def place_agent(self, agent_id: int, r: int, c: int):
        """
        SOM_MAP opcode: place agent at (r, c) on the topology.
        Updates AgentRegistry so ids_at_coord() stays accurate.
        """
        r = max(0, min(self.som.rows - 1, r))
        c = max(0, min(self.som.cols - 1, c))
        self.registry.set_som_coords(agent_id, r, c)

    # ── Train ────────────────────────────────────────────────────────────────

    def agent_train(self, agent_id: int, vec: List[float],
                    lr: Optional[float] = None,
                    sigma: Optional[float] = None) -> Tuple[int, int]:
        """
        SOM_TRAIN opcode.

        1. Find BMU for vec.
        2. Run Kohonen update.
        3. Move agent to BMU position.
        4. Return (bmu_r, bmu_c).
        """
        bmu_r, bmu_c = self.som.bmu(vec)
        self.som.train(vec, bmu_r, bmu_c, lr=lr, sigma=sigma)
        self.registry.set_som_coords(agent_id, bmu_r, bmu_c)
        return bmu_r, bmu_c

    # ── Walk ─────────────────────────────────────────────────────────────────

    def agent_walk(self, agent_id: int, gradient: bool = True) -> Tuple[int, int]:
        """
        SOM_WALK opcode.

        Moves the agent one step on the topology.
        gradient=True  → toward highest activation neighbour (SOM_WALK GRADIENT)
        gradient=False → random walk
        """
        handle = self.registry.get_or_none(agent_id)
        if handle is None:
            return 0, 0

        r, c = handle.som_x, handle.som_y

        if gradient:
            nr, nc = self.som.walk_gradient(r, c)
        else:
            nr, nc = self.som.walk_random(r, c)

        self.registry.set_som_coords(agent_id, nr, nc)
        return nr, nc

    # ── BMU (read-only) ──────────────────────────────────────────────────────

    def bmu(self, vec: List[float]) -> Tuple[int, int]:
        """SOM_BMU opcode — find closest node, don't train."""
        return self.som.bmu(vec)

    # ── Elect ────────────────────────────────────────────────────────────────

    def elect(self) -> int:
        """
        SOM_ELECT opcode.

        Returns the agent_id of the leader — the agent whose SOM node
        has the highest hit_count (most trained = most representative).
        """
        positions = []
        for handle in self.registry:
            if handle.is_alive:
                positions.append((handle.agent_id, handle.som_x, handle.som_y))

        if not positions:
            return 0

        return self.som.elect(positions)

    # ── Sense ────────────────────────────────────────────────────────────────

    def sense(self, r: int, c: int) -> float:
        """SOM_SENSE opcode — return activation at node (r, c)."""
        return self.som.sense(r, c)

    # ── Neighbourhood ────────────────────────────────────────────────────────

    def neighbourhood(self, r: int, c: int, sigma: float = 1.0) -> List[Tuple[int, int, float]]:
        """
        SOM_NBHD opcode.

        Returns list of (row, col, influence) for all nodes within
        Gaussian neighbourhood of (r, c).
        """
        result = []
        for nr in range(self.som.rows):
            for nc in range(self.som.cols):
                dist_sq = (nr - r) ** 2 + (nc - c) ** 2
                import math
                influence = math.exp(-dist_sq / (2.0 * sigma * sigma))
                if influence > 1e-4:
                    result.append((nr, nc, influence))
        return result

    # ── Background decay ─────────────────────────────────────────────────────

    def start_decay(self, interval_s: float = 1.0,
                    lr_rate: float = 0.01,
                    sigma_rate: float = 0.005):
        """
        Start a background thread that decays lr and sigma periodically.
        Call stop_decay() at HALT.
        """
        self._stop_decay.clear()

        def _loop():
            while not self._stop_decay.wait(timeout=interval_s):
                self.som.decay(lr_rate, sigma_rate)

        self._decay_thread = threading.Thread(
            target=_loop, daemon=True, name="soma-som-decay"
        )
        self._decay_thread.start()

    def stop_decay(self):
        """Stop the background decay thread."""
        self._stop_decay.set()
        if self._decay_thread is not None:
            self._decay_thread.join(timeout=2.0)
            self._decay_thread = None

    # ── Phase 2.5: DECAY_STEP ────────────────────────────────────────────────

    def decay_step(self, agent_id: int, rate: float = 0.001) -> int:
        """
        DECAY_STEP opcode.
        Erodes weights on all non-activated, non-protected nodes.
        Returns count of decayed nodes.
        """
        emotion_state = self.emotion_reg.get_or_create(agent_id)
        protected     = list(emotion_state.tags.keys())
        return self.som.decay_step(protected_coords=protected, base_rate=rate)

    # ── Phase 2.5: PRUNE_CHECK ───────────────────────────────────────────────

    def prune_check(self, threshold: float = 0.01) -> int:
        """
        PRUNE_CHECK opcode.
        Hard-deletes weights below threshold strength.
        Returns count of pruned nodes.
        """
        return self.som.prune_check(threshold=threshold)

    # ── Phase 2.5: EMOT_TAG ──────────────────────────────────────────────────

    def emot_tag(self, agent_id: int,
                 valence: float, intensity: float) -> tuple:
        """
        EMOT_TAG opcode.
        Tags the agent's current SOM position with emotion.
        Returns (bmu_r, bmu_c) that was tagged.
        """
        handle = self.registry.get_or_none(agent_id)
        r = handle.som_x if handle else 0
        c = handle.som_y if handle else 0
        emotion_state = self.emotion_reg.get_or_create(agent_id)
        emotion_state.emot_tag(r, c, valence=valence, intensity=intensity)
        return r, c

    # ── Phase 2.5: DECAY_PROTECT ─────────────────────────────────────────────

    def decay_protect(self, agent_id: int,
                      cycles: int = 100,
                      permanent: bool = False,
                      intensity: float = 1.0) -> None:
        """
        DECAY_PROTECT opcode.
        Shields the agent's current SOM node from decay for `cycles` pulses,
        or permanently if permanent=True.
        """
        handle = self.registry.get_or_none(agent_id)
        r = handle.som_x if handle else 0
        c = handle.som_y if handle else 0
        emotion_state = self.emotion_reg.get_or_create(agent_id)
        mode = ProtectMode.PERMANENT if permanent else ProtectMode.CYCLES
        emotion_state.decay_protect(r, c, mode=mode,
                                    cycles=cycles, intensity=intensity)

    # ── Phase 2.5: PREDICT_ERR ───────────────────────────────────────────────

    def predict_err(self, agent_id: int, vec: List[float]) -> float:
        """
        PREDICT_ERR opcode.
        Computes surprise: distance from agent's current position
        to actual BMU for `vec`. Returns normalised [0,1].
        """
        handle = self.registry.get_or_none(agent_id)
        pred_r = handle.som_x if handle else 0
        pred_c = handle.som_y if handle else 0

        bmu_r, bmu_c = self.som.bmu(vec)
        emotion_state = self.emotion_reg.get_or_create(agent_id)
        raw = emotion_state.predict_err(bmu_r, bmu_c, pred_r, pred_c)
        return emotion_state.normalise_err(raw, self.som.rows, self.som.cols)

    # ── Phase 2.5: MEMORY_CONSOLIDATE ────────────────────────────────────────

    def memory_consolidate(self, agent_id: int) -> "ConsolidationReport":
        """
        MEMORY_CONSOLIDATE opcode — the SOMA "REM sleep".
        Promotes high-salience weights to long-term storage,
        accelerates decay on low-salience nodes, prunes dead weights.
        """
        return self.memory_mgr.consolidate(agent_id)

    # ── Phase 2.5: PULSE tick ─────────────────────────────────────────────────

    def pulse_tick(self) -> None:
        """
        Called on every PULSE heartbeat.
        Advances emotion protection counters + memory pulse counters.
        """
        self.emotion_reg.tick_all()
        self.memory_mgr.tick_all()

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Full state dump — used by visualizer and tests."""
        agent_positions = []
        for handle in self.registry:
            if handle.is_alive:
                agent_positions.append({
                    "id":    handle.agent_id,
                    "som_r": handle.som_x,
                    "som_c": handle.som_y,
                })

        return {
            "som":    self.som.snapshot(),
            "agents": agent_positions,
        }
