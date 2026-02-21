"""
runtime/som/som_scheduler.py — Phase 2.5 + 2.6
================================================

Phase 2.5 (done):
  emot_tag(), decay_protect(), predict_err(), memory_consolidate(), pulse_tick()

Phase 2.6 (new):
  neighbor_sync()   — NEIGHBOR_SYNC: broadcast top-K memories to SOM neighbours
  memory_share()    — MEMORY_SHARE: directed memory transmission (legacy / death)
  memory_load()     — MEMORY_LOAD: restore consolidated memory into working SOM
  emot_recall()     — EMOT_RECALL: retrieve emotion tag for a SOM coord
  surprise_calc()   — SURPRISE_CALC: compute prediction error from raw vectors
  reorg_map()       — REORG_MAP: rebalance topology after pruning
  decay_rate_set()  — DECAY_RATE_SET: per-region custom decay rates
  msg_try_recv()    — MSG_TRY_RECV: non-blocking receive (no-op in scheduler layer)
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from runtime.agent import AgentRegistry

from runtime.som.som_map    import LiveSomMap
from runtime.som.emotion    import EmotionRegistry, EmotionTag, ProtectMode
from runtime.som.memory     import MemoryManager, ConsolidationReport, MemorySharePacket


class SomScheduler:
    """
    Wires SOM topology into AgentRegistry.

    Translates SOMA opcodes (as method calls) into coordinated operations
    across LiveSomMap + EmotionRegistry + MemoryManager.
    """

    def __init__(self, som: LiveSomMap, registry: "AgentRegistry"):
        self.som         = som
        self.registry    = registry
        self.emotion_reg = EmotionRegistry()
        self.memory_mgr  = MemoryManager(som, self.emotion_reg)

        self._decay_thread: Optional[threading.Thread] = None
        self._decay_stop   = threading.Event()

    # ── Agent placement ──────────────────────────────────────────────────────

    def place_agent(self, agent_id: int, r: int, c: int):
        """Place agent at SOM coordinate (r, c)."""
        r = max(0, min(self.som.rows - 1, r))
        c = max(0, min(self.som.cols - 1, c))
        self.registry.set_som_coords(agent_id, r, c)

    # ── BMU / Train / Walk ───────────────────────────────────────────────────

    def agent_train(self, agent_id: int, vec: List[float],
                    lr: Optional[float] = None,
                    sigma: Optional[float] = None) -> Tuple[int, int]:
        """SOM_BMU + SOM_TRAIN — find BMU and train. Returns new (r, c)."""
        bmu_r, bmu_c = self.som.bmu(vec)
        self.som.train(vec, bmu_r, bmu_c, lr=lr, sigma=sigma)
        self.registry.set_som_coords(agent_id, bmu_r, bmu_c)
        return bmu_r, bmu_c

    def agent_walk(self, agent_id: int, gradient: bool = True) -> Tuple[int, int]:
        """SOM_WALK — move agent along SOM. Returns new (r, c)."""
        handle = self.registry.get(agent_id)
        r, c   = handle.som_row, handle.som_col

        if gradient:
            new_r, new_c = self.som.walk_gradient(r, c)
        else:
            new_r, new_c = self.som.walk_random(r, c)

        self.registry.set_som_coords(agent_id, new_r, new_c)
        return new_r, new_c

    def bmu(self, vec: List[float]) -> Tuple[int, int]:
        """SOM_BMU without moving an agent."""
        return self.som.bmu(vec)

    def elect(self) -> int:
        """SOM_ELECT — find leader among all registered agents."""
        positions = [
            (h.agent_id, h.som_row, h.som_col)
            for h in self.registry
            if h.state.is_alive
        ]
        return self.som.elect(positions)

    def sense(self, r: int, c: int) -> float:
        """SOM_SENSE — mean weight at (r, c)."""
        return self.som.sense(r, c)

    def neighbourhood(self, r: int, c: int,
                      sigma: float = 1.0) -> List[Tuple[int, int, float]]:
        """Return nearby nodes with Gaussian weights."""
        import math
        result = []
        ir = int(math.ceil(sigma * 3))
        for dr in range(-ir, ir + 1):
            for dc in range(-ir, ir + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.som.rows and 0 <= nc < self.som.cols:
                    d  = math.sqrt(dr * dr + dc * dc)
                    h  = math.exp(-(d ** 2) / (2 * sigma ** 2))
                    if h > 1e-4:
                        result.append((nr, nc, h))
        return result

    # ── Background decay thread ──────────────────────────────────────────────

    def start_decay(self, interval_s: float = 1.0,
                    rate: float = 0.001) -> None:
        """Start background DECAY_STEP thread."""
        if self._decay_thread and self._decay_thread.is_alive():
            return
        self._decay_stop.clear()

        def _loop():
            import time
            while not self._decay_stop.wait(timeout=interval_s):
                protected = self._protected_coords()
                self.som.decay_step(protected_coords=protected, base_rate=rate)
                self.pulse_tick()

        self._decay_thread = threading.Thread(target=_loop, daemon=True,
                                              name="som-decay")
        self._decay_thread.start()

    def stop_decay(self) -> None:
        """Stop background decay thread."""
        self._decay_stop.set()
        if self._decay_thread:
            self._decay_thread.join(timeout=2.0)

    def decay_step(self, agent_id: int, rate: float = 0.001) -> int:
        """DECAY_STEP — manual single decay pulse for one agent."""
        state     = self.emotion_reg.get_or_create(agent_id)
        protected = list(state.tags.keys())
        return self.som.decay_step(protected_coords=protected, base_rate=rate)

    def prune_check(self, threshold: float = 0.01) -> int:
        """PRUNE_CHECK — hard-zero weak weights."""
        return self.som.prune_check(threshold)

    # ── Phase 2.5: Emotion operations ────────────────────────────────────────

    def emot_tag(self, agent_id: int,
                 valence: float, intensity: float) -> Optional[EmotionTag]:
        """EMOT_TAG — tag current SOM position of agent with emotion."""
        handle = self.registry.get_or_none(agent_id)
        if not handle:
            return None
        state = self.emotion_reg.get_or_create(agent_id)
        return state.emot_tag(handle.som_row, handle.som_col,
                              valence, intensity)

    def decay_protect(self, agent_id: int,
                      mode: ProtectMode = ProtectMode.CYCLES,
                      cycles: int = 100) -> None:
        """DECAY_PROTECT — shield current SOM position from decay."""
        handle = self.registry.get_or_none(agent_id)
        if not handle:
            return
        state = self.emotion_reg.get_or_create(agent_id)
        state.decay_protect(handle.som_row, handle.som_col,
                            mode=mode, cycles=cycles)

    def predict_err(self, agent_id: int, vec: List[float]) -> float:
        """PREDICT_ERR — measure surprise vs last SOM position."""
        handle = self.registry.get_or_none(agent_id)
        if not handle:
            return 0.0
        bmu_r, bmu_c = self.som.bmu(vec)
        state = self.emotion_reg.get_or_create(agent_id)
        err   = state.predict_err(
            bmu_r, bmu_c,
            handle.som_row, handle.som_col,
            self.som.rows, self.som.cols,
        )
        self.registry.set_som_coords(agent_id, bmu_r, bmu_c)
        return err

    def memory_consolidate(self, agent_id: int) -> "ConsolidationReport":
        """MEMORY_CONSOLIDATE — the SOMA REM sleep."""
        return self.memory_mgr.consolidate(agent_id)

    # ── Phase 2.6: EMOT_RECALL ───────────────────────────────────────────────

    def emot_recall(self, agent_id: int,
                    r: Optional[int] = None,
                    c: Optional[int] = None) -> Optional[EmotionTag]:
        """
        EMOT_RECALL opcode — retrieve emotion tag.

        If (r, c) omitted: uses agent's current SOM position.
        Returns EmotionTag or None if coord was never emotionally tagged.
        """
        handle = self.registry.get_or_none(agent_id)
        if not handle:
            return None
        row = r if r is not None else handle.som_row
        col = c if c is not None else handle.som_col
        state = self.emotion_reg.get_or_create(agent_id)
        return state.emot_recall(row, col)

    # ── Phase 2.6: SURPRISE_CALC ─────────────────────────────────────────────

    def surprise_calc(self, agent_id: int,
                      actual_vec:    List[float],
                      predicted_vec: List[float],
                      threshold:     float = 0.25,
                      auto_tag:      bool  = True) -> Tuple[float, bool]:
        """
        SURPRISE_CALC opcode — measure prediction error and optionally
        auto-fire EMOT_TAG if error exceeds threshold.

        Parameters
        ----------
        auto_tag  : if True and is_surprising, automatically call
                    EMOT_TAG with positive valence (discovery signal)

        Returns (error_magnitude, is_surprising)
        """
        state = self.emotion_reg.get_or_create(agent_id)
        err, is_surprising = state.surprise_calc(actual_vec, predicted_vec,
                                                 threshold=threshold)
        if is_surprising and auto_tag:
            self.emot_tag(agent_id, valence=0.7, intensity=min(err, 1.0))

        return err, is_surprising

    # ── Phase 2.6: NEIGHBOR_SYNC ─────────────────────────────────────────────

    def neighbor_sync(self, agent_id: int,
                      radius: float = 2.0,
                      top_n:  int   = 5,
                      attenuation: float = 0.4) -> Dict[int, int]:
        """
        NEIGHBOR_SYNC opcode — share top-N emotional memories with all
        agents currently within SOM radius.

        This is the cultural transmission mechanism for LIVING agents.
        (Dying agents use memory_share() instead.)

        Parameters
        ----------
        radius      : topological radius (SOM grid units)
        top_n       : how many memory clusters to broadcast
        attenuation : how strongly neighbours weight the foreign memories

        Returns {neighbour_agent_id: absorbed_count}
        """
        handle = self.registry.get_or_none(agent_id)
        if not handle:
            return {}

        # Build share packet from sender
        sender_mem = self.memory_mgr.get_or_create(agent_id)
        packet = sender_mem.build_share_packet(top_n=top_n,
                                               attenuation=attenuation)

        # Find agents within radius on the SOM grid
        my_r, my_c = handle.som_row, handle.som_col
        near_coords = set(self.som.neighbour_coords(my_r, my_c, radius))

        results: Dict[int, int] = {}
        for h in self.registry:
            if h.agent_id == agent_id:
                continue
            if not h.state.is_alive:
                continue
            if (h.som_row, h.som_col) in near_coords:
                receiver = self.memory_mgr.get_or_create(h.agent_id)
                absorbed = receiver.absorb_share_packet(packet)
                if absorbed > 0:
                    results[h.agent_id] = absorbed

        return results

    # ── Phase 2.6: MEMORY_SHARE ──────────────────────────────────────────────

    def memory_share(self, from_agent_id: int,
                     to_agent_id: int,
                     top_n:       int   = 10,
                     attenuation: float = 0.5) -> int:
        """
        MEMORY_SHARE opcode — directed experience transmission.

        Called when one agent explicitly shares with another —
        typically a dying agent leaving a legacy.

        Returns number of nodes/tags absorbed by receiver.
        """
        return self.memory_mgr.share(from_agent_id, to_agent_id,
                                     top_n=top_n, attenuation=attenuation)

    def memory_share_broadcast(self, from_agent_id: int,
                                top_n:       int   = 10,
                                attenuation: float = 0.3) -> Dict[int, int]:
        """
        MEMORY_SHARE ALL — dying agent broadcasts to all live agents.
        Auto-registers all live registry agents into MemoryManager first.
        Returns {agent_id: absorbed_count} for every recipient.
        """
        for h in self.registry:
            if h.state.is_alive and h.agent_id != from_agent_id:
                self.memory_mgr.get_or_create(h.agent_id)
        return self.memory_mgr.share_to_all(from_agent_id,
                                            top_n=top_n,
                                            attenuation=attenuation)

    # ── Phase 2.6: MEMORY_LOAD ───────────────────────────────────────────────

    def memory_load(self, agent_id: int,
                    min_salience: float = 0.0) -> int:
        """
        MEMORY_LOAD opcode — restore consolidated long-term memories
        back into the working SOM.

        Used on agent restart or after a wipe of the working SOM.
        Returns number of nodes restored.
        """
        return self.memory_mgr.load(agent_id, min_salience=min_salience)

    # ── Phase 2.6: REORG_MAP ─────────────────────────────────────────────────

    def reorg_map(self, dead_threshold: float = 1e-4,
                  spread_radius: int = 2) -> int:
        """
        REORG_MAP opcode — rebalance SOM topology after heavy pruning.

        Dead nodes are re-seeded from neighbours. Prevents permanent
        dead zones. Call after MEMORY_CONSOLIDATE for best results.

        Returns number of nodes reseeded.
        """
        return self.som.reorg_map(dead_threshold=dead_threshold,
                                  spread_radius=spread_radius)

    # ── Phase 2.6: DECAY_RATE_SET ────────────────────────────────────────────

    def decay_rate_set(self, agent_id: int,
                       radius: float = 2.0,
                       rate:   float = 0.001) -> int:
        """
        DECAY_RATE_SET opcode — set custom decay rate around agent's
        current SOM position.

        Low rate (0.0001) = expertise zone — build deep memory here.
        High rate (0.05)  = stale zone — flush this region quickly.

        Returns number of nodes updated.
        """
        handle = self.registry.get_or_none(agent_id)
        if not handle:
            return 0
        return self.som.set_region_decay_rate(
            handle.som_row, handle.som_col,
            radius=radius, rate=rate,
        )

    # ── Phase 2.6: MSG_TRY_RECV ──────────────────────────────────────────────

    def msg_try_recv(self, agent_id: int) -> Optional[object]:
        """
        MSG_TRY_RECV opcode — non-blocking receive.

        In the scheduler layer this is a no-op placeholder.
        Real implementation lives in the runtime mailbox system.
        Returns None if no message available (scheduler has no mailbox).
        """
        return None

    # ── Pulse tick ───────────────────────────────────────────────────────────

    def pulse_tick(self) -> None:
        """Called on every PULSE heartbeat — advances all counters."""
        self.emotion_reg.tick_all()
        self.memory_mgr.tick_all()

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "som": self.som.snapshot(),
            "emotion": self.emotion_reg.snapshot(),
            "memory":  self.memory_mgr.snapshot(),
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _protected_coords(self) -> List[Tuple[int, int]]:
        """Collect all emotion-protected (r, c) across all agents."""
        protected = set()
        for h in self.registry:
            state = self.emotion_reg.get_or_none(h.agent_id) \
                    if hasattr(self.emotion_reg, 'get_or_none') \
                    else self.emotion_reg.get_or_create(h.agent_id)
            for coord in state.tags:
                protected.add(coord)
        return list(protected)
