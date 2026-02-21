"""
runtime/som/memory.py — Phase 2.5: Two-Tier Memory System
==========================================================

Implements the hippocampus primitive for SOMA:
  - Working SOM    → volatile, fast, decays aggressively
  - Long-term SOM  → persistent, slow decay, promoted by MEMORY_CONSOLIDATE
  - MEMORY_CONSOLIDATE → slow "REM sleep" reorganisation

Paper reference: "A Path to AGI Part II: Liveliness"
  "Periodic MEMORY_CONSOLIDATE promotes highest-emotion-tagged weights
   into persistent SOM storage."

Consolidation schedule:
  - PULSE runs at ~100 Hz (every 10 ms)
  - MEMORY_CONSOLIDATE runs at slow interval (default: every 10,000 pulses)
  - Top 10% by emotion salience → promoted to long-term
  - Bottom 50% → accelerated decay
  - Below 0.5% weight strength → hard pruned
"""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from runtime.som.som_map  import LiveSomMap
    from runtime.som.emotion  import EmotionRegistry, AgentEmotionState


# ── Memory node — long-term storage cell ─────────────────────────────────────

@dataclass
class LongTermNode:
    """One node in the long-term SOM (post-consolidation)."""
    row:        int
    col:        int
    weights:    List[float]
    salience:   float = 0.0     # emotion salience at time of consolidation
    consolidated_at: float = field(default_factory=time.monotonic)
    decay_rate: float = 0.0001  # much slower than working SOM


# ── Consolidation result ──────────────────────────────────────────────────────

@dataclass
class ConsolidationReport:
    promoted:   int = 0    # nodes moved to long-term
    decayed:    int = 0    # nodes with accelerated decay applied
    pruned:     int = 0    # nodes hard-deleted
    rebalanced: bool = False
    duration_s: float = 0.0
    pulse_count: int = 0


# ── Two-tier memory ───────────────────────────────────────────────────────────

class TwoTierMemory:
    """
    Manages working SOM + long-term SOM for a single agent.

    Working SOM  → the live LiveSomMap shared across agents.
                   Decays aggressively (DECAY_STEP).
    Long-term SOM → private to this agent.
                   Only written during MEMORY_CONSOLIDATE.
                   Decays slowly, pruned rarely.
    """

    def __init__(self,
                 agent_id: int,
                 working_som: "LiveSomMap",
                 emotion_reg: "EmotionRegistry",
                 top_keep_pct:     float = 0.10,
                 decay_bottom_pct: float = 0.50,
                 prune_threshold:  float = 0.005):
        self.agent_id         = agent_id
        self.working_som      = working_som
        self.emotion_reg      = emotion_reg
        self.top_keep_pct     = top_keep_pct
        self.decay_bottom_pct = decay_bottom_pct
        self.prune_threshold  = prune_threshold

        # Long-term store: (row, col) → LongTermNode
        self._lt_store: Dict[Tuple[int,int], LongTermNode] = {}
        self._lock = threading.Lock()

        self.pulse_count    = 0
        self.consolidations = 0

    # ── MEMORY_CONSOLIDATE ───────────────────────────────────────────────────

    def consolidate(self) -> ConsolidationReport:
        """
        MEMORY_CONSOLIDATE opcode implementation.

        1. MEMORY_RANK   — rank all working SOM nodes by emotion salience.
        2. MEMORY_KEEP   — promote top 10% to long-term storage.
        3. MEMORY_DECAY  — accelerate decay on bottom 50%.
        4. MEMORY_PRUNE  — hard-delete weights below threshold.
        5. REORG_MAP     — rebalance SOM topology after pruning.
        """
        t0 = time.monotonic()
        report = ConsolidationReport(pulse_count=self.pulse_count)

        emotion_state = self.emotion_reg.get_or_create(self.agent_id)
        rows = self.working_som.rows
        cols = self.working_som.cols
        total_nodes = rows * cols

        # 1. Rank all nodes by salience
        ranked: List[Tuple[float, int, int]] = []   # (salience, r, c)
        for r in range(rows):
            for c in range(cols):
                sal = emotion_state.salience(r, c)
                ranked.append((sal, r, c))
        ranked.sort(reverse=True)

        n_keep  = max(1, int(total_nodes * self.top_keep_pct))
        n_decay = max(1, int(total_nodes * self.decay_bottom_pct))

        # 2. MEMORY_KEEP — promote top nodes to long-term
        with self.working_som._lock:
            for sal, r, c in ranked[:n_keep]:
                node = self.working_som.nodes[r][c]
                with self._lock:
                    self._lt_store[(r, c)] = LongTermNode(
                        row=r, col=c,
                        weights=list(node.weights),
                        salience=sal,
                    )
                report.promoted += 1

        # 3. MEMORY_DECAY — accelerate decay on bottom nodes
        bottom = ranked[-n_decay:]
        with self.working_som._lock:
            for sal, r, c in bottom:
                node = self.working_som.nodes[r][c]
                # Apply 10× decay factor
                node.weights = [w * 0.9 for w in node.weights]
                report.decayed += 1

        # 4. MEMORY_PRUNE — zero out weak weights
        with self.working_som._lock:
            for r in range(rows):
                for c in range(cols):
                    node = self.working_som.nodes[r][c]
                    strength = sum(w*w for w in node.weights) ** 0.5
                    if strength < self.prune_threshold:
                        node.weights = [0.0] * self.working_som.dims
                        node.activation = 0.0
                        report.pruned += 1

        # 5. REORG_MAP — decay long-term nodes slowly
        self._decay_longterm()
        report.rebalanced = True

        self.consolidations += 1
        report.duration_s = time.monotonic() - t0
        return report

    # ── Long-term decay ──────────────────────────────────────────────────────

    def _decay_longterm(self) -> None:
        """Slowly decay long-term nodes. High-salience nodes decay slower."""
        with self._lock:
            dead = []
            for key, node in self._lt_store.items():
                effective_rate = node.decay_rate * (1.0 - node.salience * 0.9)
                node.weights = [w * (1.0 - effective_rate) for w in node.weights]
                strength = sum(w*w for w in node.weights) ** 0.5
                if strength < self.prune_threshold * 0.1:
                    dead.append(key)
            for k in dead:
                del self._lt_store[k]

    # ── Recall ────────────────────────────────────────────────────────────────

    def recall(self, r: int, c: int) -> Optional[List[float]]:
        """
        Recall long-term weights for node (r, c).
        Returns None if not in long-term store.
        """
        with self._lock:
            node = self._lt_store.get((r, c))
            return list(node.weights) if node else None

    def recall_or_working(self, r: int, c: int) -> List[float]:
        """
        Prefer long-term recall; fall back to working SOM.
        Used for agent restart / warm-up after inactivity.
        """
        lt = self.recall(r, c)
        if lt:
            return lt
        with self.working_som._lock:
            return list(self.working_som.nodes[r][c].weights)

    # ── Restore ───────────────────────────────────────────────────────────────

    def restore_to_working(self, r: int, c: int) -> bool:
        """
        Write long-term weights back into the working SOM.
        Called when an agent re-enters a region it knew well.
        Returns True if long-term weights were available.
        """
        lt = self.recall(r, c)
        if lt is None:
            return False
        with self.working_som._lock:
            self.working_som.nodes[r][c].weights = lt
        return True

    # ── Pulse ─────────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Called on every PULSE — increments counter."""
        self.pulse_count += 1

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            lt_count = len(self._lt_store)
            mean_sal = (
                sum(n.salience for n in self._lt_store.values()) / lt_count
                if lt_count else 0.0
            )
        return {
            "agent_id":       self.agent_id,
            "pulse_count":    self.pulse_count,
            "consolidations": self.consolidations,
            "lt_nodes":       lt_count,
            "lt_mean_salience": mean_sal,
        }


# ── Memory manager: all agents ────────────────────────────────────────────────

class MemoryManager:
    """
    VM-level manager — one TwoTierMemory per agent.
    SomScheduler holds one instance.
    """

    def __init__(self, working_som: "LiveSomMap",
                 emotion_reg: "EmotionRegistry"):
        self.working_som = working_som
        self.emotion_reg = emotion_reg
        self._memories:  Dict[int, TwoTierMemory] = {}
        self._lock       = threading.Lock()

    def get_or_create(self, agent_id: int) -> TwoTierMemory:
        with self._lock:
            if agent_id not in self._memories:
                self._memories[agent_id] = TwoTierMemory(
                    agent_id=agent_id,
                    working_som=self.working_som,
                    emotion_reg=self.emotion_reg,
                )
            return self._memories[agent_id]

    def remove(self, agent_id: int) -> None:
        with self._lock:
            self._memories.pop(agent_id, None)

    def tick_all(self) -> None:
        with self._lock:
            mems = list(self._memories.values())
        for m in mems:
            m.tick()

    def consolidate(self, agent_id: int) -> ConsolidationReport:
        """MEMORY_CONSOLIDATE opcode — runs for one agent."""
        return self.get_or_create(agent_id).consolidate()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                aid: mem.snapshot()
                for aid, mem in self._memories.items()
            }
