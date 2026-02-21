"""
runtime/som/memory.py — Phase 2.5 + 2.6: Two-Tier Memory System
================================================================

Phase 2.5 (done):
  - Working SOM    → volatile, fast, decays aggressively
  - Long-term SOM  → persistent, slow decay, promoted by MEMORY_CONSOLIDATE
  - MEMORY_CONSOLIDATE → slow "REM sleep" reorganisation

Phase 2.6 (new):
  - MEMORY_SHARE   → share top-N salient clusters with another agent (culture)
  - MEMORY_LOAD    → load consolidated memory back into working SOM on restart
  - MemorySharePacket → the cultural transmission unit

Paper reference: "A Path to AGI Part II: Liveliness"

  "Periodic MEMORY_CONSOLIDATE promotes highest-emotion-tagged weights
   into persistent SOM storage."

  "An agent that shares its emotionally-tagged memories with neighbors
   before dying is doing something profound: it is transmitting experience.
   Run this across thousands of agents over thousands of cycles and you get
   something that looks remarkably like culture."
"""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from runtime.som.som_map  import LiveSomMap
    from runtime.som.emotion  import EmotionRegistry, AgentEmotionState, EmotionSnapshot


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

    def to_dict(self) -> dict:
        return {
            "row": self.row, "col": self.col,
            "weights": list(self.weights),
            "salience": self.salience,
            "consolidated_at": self.consolidated_at,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LongTermNode":
        return cls(**d)


# ── Consolidation result ──────────────────────────────────────────────────────

@dataclass
class ConsolidationReport:
    promoted:   int = 0    # nodes moved to long-term
    decayed:    int = 0    # nodes with accelerated decay applied
    pruned:     int = 0    # nodes hard-deleted
    rebalanced: bool = False
    duration_s: float = 0.0
    pulse_count: int = 0


# ── Memory share packet — the cultural transmission unit ─────────────────────

@dataclass
class MemorySharePacket:
    """
    MEMORY_SHARE output — a bundle of long-term nodes + emotion snapshots
    that one agent sends to another.

    This is what transmits experience between agents:
    - long_term_nodes  : the actual weight vectors worth keeping
    - emotion_snapshot : which of those nodes were emotionally significant
    - source_agent_id  : who this came from
    - top_n            : how many nodes are included
    - attenuation      : how much to weight foreign vs self experience

    When an agent dies and calls MEMORY_SHARE before AGENT_KILL, it is
    leaving a legacy. The receiving agents are slightly shaped by what
    the deceased agent found significant.
    """
    source_agent_id: int
    long_term_nodes: List[dict]             # LongTermNode.to_dict() each
    emotion_snapshot: Optional[dict] = None # EmotionSnapshot tags list
    top_n:       int   = 10
    attenuation: float = 0.5               # foreign memories weighted at 50%
    created_at:  float = field(default_factory=time.monotonic)

    def __len__(self) -> int:
        return len(self.long_term_nodes)


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

        # Long-term storage: (row, col) → LongTermNode
        self._longterm: Dict[Tuple[int,int], LongTermNode] = {}
        self._lt_lock   = threading.Lock()
        self.pulse_count = 0
        self._consolidation_count = 0

    # ── MEMORY_CONSOLIDATE ───────────────────────────────────────────────────

    def consolidate(self) -> ConsolidationReport:
        """
        MEMORY_CONSOLIDATE opcode implementation.

        1. MEMORY_RANK   — rank all working SOM nodes by emotion salience.
        2. MEMORY_KEEP   — promote top 10% to long-term storage.
        3. MEMORY_DECAY  — accelerate decay on bottom 50%.
        4. MEMORY_PRUNE  — hard-delete weights below threshold.
        """
        t0 = time.monotonic()
        report = ConsolidationReport(pulse_count=self.pulse_count)

        emotion_state = self.emotion_reg.get_or_create(self.agent_id)
        rows = self.working_som.rows
        cols = self.working_som.cols
        total_nodes = rows * cols

        # 1. Rank all nodes by salience
        ranked: List[Tuple[float, int, int]] = []
        for r in range(rows):
            for c in range(cols):
                sal = emotion_state.salience(r, c)
                ranked.append((sal, r, c))
        ranked.sort(reverse=True)

        # 2. Promote top N% → long-term
        keep_n = max(1, int(total_nodes * self.top_keep_pct))
        with self._lt_lock:
            for sal, r, c in ranked[:keep_n]:
                if sal > 0:
                    node = self.working_som.nodes[r][c]
                    self._longterm[(r, c)] = LongTermNode(
                        row=r, col=c,
                        weights=list(node.weights),
                        salience=sal,
                    )
                    report.promoted += 1

        # 3. Accelerate decay on bottom N%
        decay_start = total_nodes - int(total_nodes * self.decay_bottom_pct)
        with self.working_som._lock:
            for sal, r, c in ranked[decay_start:]:
                node = self.working_som.nodes[r][c]
                node.decay_rate = min(node.decay_rate * 5, 0.1)
                report.decayed += 1

        # 4. Hard-prune below threshold
        pruned = self.working_som.prune_check(self.prune_threshold)
        report.pruned = pruned

        report.duration_s = time.monotonic() - t0
        self._consolidation_count += 1
        return report

    # ── MEMORY_SHARE (Phase 2.6) ─────────────────────────────────────────────

    def build_share_packet(self, top_n: int = 10,
                           attenuation: float = 0.5) -> MemorySharePacket:
        """
        MEMORY_SHARE — build a packet of the top-N most salient long-term
        memories, ready to send to another agent.

        Called when an agent wants to share experience (legacy on death,
        or periodic NEIGHBOR_SYNC while alive).

        Parameters
        ----------
        top_n       : how many long-term nodes to include
        attenuation : how strongly the receiver should weight these
                      foreign memories (0 = ignore, 1 = full adoption)
        """
        emotion_state = self.emotion_reg.get_or_create(self.agent_id)
        em_snapshot   = emotion_state.snapshot_for_share(top_n=top_n)

        with self._lt_lock:
            # Sort long-term by salience descending
            candidates = sorted(
                self._longterm.values(),
                key=lambda n: n.salience,
                reverse=True
            )[:top_n]
            nodes_dicts = [n.to_dict() for n in candidates]

        return MemorySharePacket(
            source_agent_id = self.agent_id,
            long_term_nodes  = nodes_dicts,
            emotion_snapshot = {
                "tags": em_snapshot.tags
            },
            top_n        = top_n,
            attenuation  = attenuation,
        )

    def absorb_share_packet(self, packet: MemorySharePacket) -> int:
        """
        MEMORY_SHARE receive side — merge a foreign agent's memories.

        Long-term nodes: merged into this agent's long-term store at
                         reduced salience (packet.attenuation).
        Emotion tags:    absorbed via EmotionState.absorb_snapshot().

        Returns total number of nodes / tags absorbed.
        """
        absorbed = 0
        atten    = packet.attenuation

        # Absorb long-term weight clusters
        with self._lt_lock:
            for node_dict in packet.long_term_nodes:
                lt = LongTermNode.from_dict(node_dict)
                coord = (lt.row, lt.col)
                lt.salience *= atten   # attenuate foreign salience
                existing = self._longterm.get(coord)
                if existing is None:
                    self._longterm[coord] = lt
                    absorbed += 1
                elif lt.salience > existing.salience:
                    # Foreign is stronger — adopt weights, blend salience
                    existing.weights  = [
                        (a + b) / 2
                        for a, b in zip(lt.weights, existing.weights)
                    ]
                    existing.salience = lt.salience
                    absorbed += 1

        # Absorb emotion tags if present
        if packet.emotion_snapshot:
            from runtime.som.emotion import EmotionSnapshot
            snap = EmotionSnapshot(
                source_agent_id=packet.source_agent_id,
                tags=packet.emotion_snapshot.get("tags", []),
            )
            emotion_state = self.emotion_reg.get_or_create(self.agent_id)
            absorbed += emotion_state.absorb_snapshot(snap, weight=atten)

        return absorbed

    # ── MEMORY_LOAD (Phase 2.6) ──────────────────────────────────────────────

    def load_to_working(self, min_salience: float = 0.0) -> int:
        """
        MEMORY_LOAD opcode — restore long-term nodes back into the
        working SOM.

        Used on agent restart or after a cold-boot where the working SOM
        was wiped. Loads all long-term nodes with salience > min_salience
        back into the working SOM weights.

        Returns number of nodes restored.
        """
        restored = 0
        with self._lt_lock:
            nodes = [n for n in self._longterm.values()
                     if n.salience >= min_salience]

        with self.working_som._lock:
            for lt in nodes:
                r, c = lt.row, lt.col
                if 0 <= r < self.working_som.rows and \
                   0 <= c < self.working_som.cols:
                    self.working_som.nodes[r][c].weights = list(lt.weights)
                    restored += 1

        return restored

    # ── Long-term read ───────────────────────────────────────────────────────

    def recall(self, r: int, c: int) -> Optional[List[float]]:
        """Retrieve long-term weights at (r, c). None if not consolidated."""
        with self._lt_lock:
            lt = self._longterm.get((r, c))
        return list(lt.weights) if lt else None

    def recall_or_working(self, r: int, c: int) -> List[float]:
        """
        Retrieve weights from long-term if available, else working SOM.
        The 'best known' weights for a node.
        """
        lt = self.recall(r, c)
        if lt is not None:
            return lt
        return list(self.working_som.nodes[r][c].weights)

    def restore_to_working(self, r: int, c: int) -> bool:
        """Restore a single long-term node to the working SOM."""
        weights = self.recall(r, c)
        if weights is None:
            return False
        with self.working_som._lock:
            self.working_som.nodes[r][c].weights = weights
        return True

    # ── Long-term decay ──────────────────────────────────────────────────────

    def _decay_longterm(self) -> None:
        """Slow decay of long-term nodes. Called during MEMORY_CONSOLIDATE."""
        with self._lt_lock:
            dead = []
            for coord, lt in self._longterm.items():
                lt.weights = [w * (1.0 - lt.decay_rate) for w in lt.weights]
                if max(abs(w) for w in lt.weights) < 1e-6:
                    dead.append(coord)
            for coord in dead:
                del self._longterm[coord]

    # ── Pulse tick ───────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Advance one pulse. Increments counter; decay handled externally."""
        self.pulse_count += 1

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lt_lock:
            lt_count  = len(self._longterm)
            mean_sal  = (
                sum(n.salience for n in self._longterm.values()) / lt_count
            ) if lt_count else 0.0
        return {
            "agent_id":        self.agent_id,
            "pulse_count":     self.pulse_count,
            "longterm_nodes":  lt_count,
            "lt_nodes":        lt_count,   # alias expected by tests
            "mean_lt_salience": mean_sal,
            "consolidations":  self._consolidation_count,
        }


# ── Global memory manager ─────────────────────────────────────────────────────

class MemoryManager:
    """
    Global store of TwoTierMemory, keyed by agent_id.
    Thread-safe. Coordinates MEMORY_CONSOLIDATE + MEMORY_SHARE across agents.
    """

    def __init__(self, working_som: "LiveSomMap",
                 emotion_reg: "EmotionRegistry",
                 top_keep_pct:     float = 0.10,
                 decay_bottom_pct: float = 0.50,
                 prune_threshold:  float = 0.005):
        self.working_som      = working_som
        self.emotion_reg      = emotion_reg
        self.top_keep_pct     = top_keep_pct
        self.decay_bottom_pct = decay_bottom_pct
        self.prune_threshold  = prune_threshold

        self._memories: Dict[int, TwoTierMemory] = {}
        self._lock = threading.Lock()

    def get_or_create(self, agent_id: int) -> TwoTierMemory:
        with self._lock:
            if agent_id not in self._memories:
                self._memories[agent_id] = TwoTierMemory(
                    agent_id=agent_id,
                    working_som=self.working_som,
                    emotion_reg=self.emotion_reg,
                    top_keep_pct=self.top_keep_pct,
                    decay_bottom_pct=self.decay_bottom_pct,
                    prune_threshold=self.prune_threshold,
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

    # ── MEMORY_SHARE ─────────────────────────────────────────────────────────

    def share(self, from_agent_id: int,
              to_agent_id: int,
              top_n:       int   = 10,
              attenuation: float = 0.5) -> int:
        """
        MEMORY_SHARE opcode — transmit top-N memories from one agent to another.

        Parameters
        ----------
        from_agent_id : source agent (the sharer — e.g. dying agent)
        to_agent_id   : target agent (the receiver)
        top_n         : how many memory clusters to share
        attenuation   : how strongly the receiver weights foreign memory

        Returns number of nodes/tags absorbed by the receiver.
        """
        sender   = self.get_or_create(from_agent_id)
        receiver = self.get_or_create(to_agent_id)

        packet = sender.build_share_packet(top_n=top_n, attenuation=attenuation)
        return receiver.absorb_share_packet(packet)

    def share_to_all(self, from_agent_id: int,
                     top_n:       int   = 10,
                     attenuation: float = 0.3) -> Dict[int, int]:
        """
        MEMORY_SHARE broadcast — share with all registered agents.
        Used by a dying agent to leave a cultural legacy.

        Returns {agent_id: absorbed_count} dict.
        """
        with self._lock:
            recipients = [aid for aid in self._memories
                          if aid != from_agent_id]

        results = {}
        sender  = self.get_or_create(from_agent_id)
        packet  = sender.build_share_packet(top_n=top_n, attenuation=attenuation)

        for aid in recipients:
            receiver = self.get_or_create(aid)
            results[aid] = receiver.absorb_share_packet(packet)

        return results

    # ── MEMORY_LOAD ──────────────────────────────────────────────────────────

    def load(self, agent_id: int, min_salience: float = 0.0) -> int:
        """MEMORY_LOAD opcode — restore consolidated memory to working SOM."""
        return self.get_or_create(agent_id).load_to_working(min_salience)

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                aid: mem.snapshot()
                for aid, mem in self._memories.items()
            }
