"""
runtime/collective.py — Phase V Collective Intelligence for SOMA.

Opcodes implemented:
  0x73  NICHE_DECLARE   — agent broadcasts its specialisation vector
  0x74  SYMBOL_EMERGE   — repeated co-activation binds a symbol ID to SOM region
  0x75  HERITAGE_LOAD   — load parent soul top-K on birth
  0x76  NICHE_QUERY     — returns niche density; agent migrates if > threshold
  0x77  COLLECTIVE_SYNC — map-wide memory consolidation across all agents

Measurement:
  Shannon entropy of the niche distribution.
    H = 0          when all agents are in the same niche (identical)
    H → log2(64)   when each of 64 agents occupies a unique niche (max diversity)

Nobody programs the specialisation.  It emerges from:
  1. Agents sensing their local SOM region  (SOM_SENSE)
  2. Declaring their specialisation         (NICHE_DECLARE)
  3. Querying niche density                 (NICHE_QUERY)
  4. Migrating away from crowded niches
  5. Periodic global consolidation          (COLLECTIVE_SYNC)

Thread safety:
  CollectiveState uses threading.Lock for all shared structures.
  Agent threads call op_* functions directly; the interpreter routes here.
"""

from __future__ import annotations

import math
import threading
import time
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from runtime.bridge import (
    OP_NICHE_DECLARE,
    OP_SYMBOL_EMERGE,
    OP_HERITAGE_LOAD,
    OP_NICHE_QUERY,
    OP_COLLECTIVE_SYNC,
    NICHE_CAPACITY,
    NICHE_MIGRATE_THRESH,
    SYMBOL_BIND_THRESH,
    HERITAGE_TOP_K,
    COLLECTIVE_WINDOW,
    NICHE_IMM_DECLARE,
    NICHE_IMM_WITHDRAW,
    SomaInstruction,
)

if TYPE_CHECKING:
    from runtime.agent import AgentHandle

log = logging.getLogger(__name__)

VEC_DIM = 8   # dimensionality of weight / specialisation vectors

# ─────────────────────────────────────────────────────────────────────────────
# Soul store  (persists across agent lifetimes — HERITAGE_LOAD reads it)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SoulRecord:
    """Top-K weight vectors saved by a dying / forking agent."""
    agent_id:    int
    niche_id:    int
    vectors:     list[list[float]]   # top-K weight vectors, each VEC_DIM floats
    timestamp:   float = field(default_factory=time.time)


class SoulStore:
    """Thread-safe persistent store for agent soul records."""

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._souls: dict[int, SoulRecord] = {}   # agent_id → latest record

    def save(self, agent_id: int, niche_id: int, vectors: list[list[float]]) -> None:
        with self._lock:
            self._souls[agent_id] = SoulRecord(agent_id, niche_id, vectors[:HERITAGE_TOP_K])

    def load_parent(self, parent_id: int, top_k: int = HERITAGE_TOP_K) -> list[list[float]]:
        """Return up to top_k weight vectors from the parent's soul record."""
        with self._lock:
            record = self._souls.get(parent_id)
            if record is None:
                return []
            return record.vectors[:top_k]

    def all_records(self) -> list[SoulRecord]:
        with self._lock:
            return list(self._souls.values())


# ─────────────────────────────────────────────────────────────────────────────
# Symbol table  (SYMBOL_EMERGE binds SOM regions to symbol IDs)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolBinding:
    symbol_id: int
    som_x: int
    som_y: int
    activation_count: int
    bound_at: float = field(default_factory=time.time)
    agents: set[int] = field(default_factory=set)


class SymbolTable:
    """
    Tracks co-activations of SOM regions across agents.
    When a region is co-activated ≥ SYMBOL_BIND_THRESH times, it earns a symbol ID.
    """

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._counts:   Counter[tuple[int, int]] = Counter()
        self._agents:   defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        self._bindings: dict[tuple[int, int], SymbolBinding] = {}
        self._next_sym  = 1

    def record_activation(self, agent_id: int, som_x: int, som_y: int,
                          threshold: int = SYMBOL_BIND_THRESH) -> int | None:
        """
        Record that agent_id activated (som_x, som_y).
        Returns the symbol_id if a new binding is created, else None.
        """
        key = (som_x, som_y)
        with self._lock:
            self._counts[key] += 1
            self._agents[key].add(agent_id)

            if key in self._bindings:
                self._bindings[key].activation_count += 1
                self._bindings[key].agents.add(agent_id)
                return None   # already bound

            if self._counts[key] >= threshold:
                sym_id = self._next_sym
                self._next_sym += 1
                self._bindings[key] = SymbolBinding(
                    symbol_id=sym_id, som_x=som_x, som_y=som_y,
                    activation_count=self._counts[key],
                    agents=set(self._agents[key]),
                )
                log.info(
                    "SYMBOL EMERGED: id=%d at SOM(%d,%d) after %d co-activations by %d agents",
                    sym_id, som_x, som_y, self._counts[key], len(self._agents[key])
                )
                return sym_id
        return None

    def get_symbol(self, som_x: int, som_y: int) -> SymbolBinding | None:
        with self._lock:
            return self._bindings.get((som_x, som_y))

    def all_symbols(self) -> list[SymbolBinding]:
        with self._lock:
            return list(self._bindings.values())


# ─────────────────────────────────────────────────────────────────────────────
# Niche map  (NICHE_DECLARE / NICHE_QUERY)
# ─────────────────────────────────────────────────────────────────────────────

class NicheMap:
    """
    Global niche occupancy map.  Agents declare their niche; the map tracks
    how many agents occupy each niche so agents can migrate away from crowding.

    Niche IDs are integers 0–(NICHE_CAPACITY-1).  The niche an agent occupies
    is derived from its dominant SOM region (som_x, som_y mapped to niche_id).
    """

    def __init__(self, capacity: int = NICHE_CAPACITY) -> None:
        self._cap    = capacity
        self._lock   = threading.Lock()
        # agent_id → niche_id
        self._agent_niche: dict[int, int] = {}
        # niche_id → set of agent_ids
        self._niche_agents: defaultdict[int, set[int]] = defaultdict(set)

    # ── SOM coordinate → niche ID ─────────────────────────────────────────────
    @staticmethod
    def som_to_niche(som_x: int, som_y: int, grid_size: int = 16) -> int:
        """
        Map a SOM (x, y) position to a niche ID in [0, NICHE_CAPACITY).
        Default SOM is 16×16 = 256 cells; fold into NICHE_CAPACITY=64 buckets.
        """
        cell = (som_y * grid_size + som_x) % NICHE_CAPACITY
        return cell

    # ── Declare / withdraw ────────────────────────────────────────────────────
    def declare(self, agent_id: int, niche_id: int) -> None:
        niche_id = niche_id % self._cap
        with self._lock:
            old = self._agent_niche.get(agent_id)
            if old is not None and old != niche_id:
                self._niche_agents[old].discard(agent_id)
            self._agent_niche[agent_id] = niche_id
            self._niche_agents[niche_id].add(agent_id)

    def withdraw(self, agent_id: int) -> None:
        with self._lock:
            old = self._agent_niche.pop(agent_id, None)
            if old is not None:
                self._niche_agents[old].discard(agent_id)

    # ── Query ─────────────────────────────────────────────────────────────────
    def density(self, niche_id: int) -> float:
        """Fraction of total agents occupying niche_id (0.0–1.0)."""
        with self._lock:
            total = len(self._agent_niche)
            if total == 0:
                return 0.0
            return len(self._niche_agents[niche_id % self._cap]) / total

    def least_crowded_niche(self) -> int:
        """Return the niche_id with the fewest current occupants."""
        with self._lock:
            counts = {n: len(agents) for n, agents in self._niche_agents.items()}
            # Fill in empty niches
            for n in range(self._cap):
                counts.setdefault(n, 0)
            return min(counts, key=counts.__getitem__)

    def distribution(self) -> dict[int, int]:
        """Return {niche_id: occupant_count} for all niches that have occupants."""
        with self._lock:
            return {n: len(agents)
                    for n, agents in self._niche_agents.items()
                    if agents}

    # ── Shannon entropy of niche distribution ─────────────────────────────────
    def shannon_entropy(self) -> float:
        """
        H = -Σ p_i · log2(p_i)

        H = 0              → all agents in one niche (no specialisation)
        H = log2(64) ≈ 6   → one agent per niche (maximum specialisation)
        """
        with self._lock:
            total = len(self._agent_niche)
            if total == 0:
                return 0.0
            counts = [len(ag) for ag in self._niche_agents.values() if ag]
            if not counts:
                return 0.0
            H = 0.0
            for c in counts:
                p = c / total
                if p > 0:
                    H -= p * math.log2(p)
            return H

    @property
    def max_entropy(self) -> float:
        return math.log2(self._cap)


# ─────────────────────────────────────────────────────────────────────────────
# Collective memory  (COLLECTIVE_SYNC consolidates across all agents)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentSnapshot:
    agent_id: int
    niche_id: int
    registers: list[float]           # current R0 vector (VEC_DIM floats)
    som_x: int
    som_y: int
    pulse: int


class CollectiveMemory:
    """
    Shared memory pool written by all agents during COLLECTIVE_SYNC.
    After consolidation, each agent reads back the centroid of its niche.
    """

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._snapshots: dict[int, AgentSnapshot] = {}    # agent_id → snapshot
        self._centroids: dict[int, list[float]] = {}      # niche_id → centroid vector
        self._sync_count = 0

    def submit_snapshot(self, snap: AgentSnapshot) -> None:
        with self._lock:
            self._snapshots[snap.agent_id] = snap

    def consolidate(self, niche_map: NicheMap) -> dict[int, list[float]]:
        """
        Compute per-niche centroid from submitted snapshots.
        Returns {niche_id: centroid_vector}.
        """
        with self._lock:
            self._sync_count += 1
            niche_vecs: defaultdict[int, list[list[float]]] = defaultdict(list)

            for snap in self._snapshots.values():
                niche_vecs[snap.niche_id].append(snap.registers)

            centroids: dict[int, list[float]] = {}
            for niche_id, vecs in niche_vecs.items():
                if not vecs:
                    continue
                centroid = [0.0] * VEC_DIM
                for v in vecs:
                    for i, x in enumerate(v[:VEC_DIM]):
                        centroid[i] += x
                n = len(vecs)
                centroids[niche_id] = [x / n for x in centroid]

            self._centroids = centroids
            log.info(
                "COLLECTIVE_SYNC #%d: consolidated %d agents into %d niche centroids",
                self._sync_count, len(self._snapshots), len(centroids)
            )
            return dict(centroids)

    def get_centroid(self, niche_id: int) -> list[float] | None:
        with self._lock:
            return self._centroids.get(niche_id)

    @property
    def sync_count(self) -> int:
        with self._lock:
            return self._sync_count


# ─────────────────────────────────────────────────────────────────────────────
# Collective state singleton  (shared across all agents in one VM instance)
# ─────────────────────────────────────────────────────────────────────────────

class CollectiveState:
    """
    Per-VM singleton holding all Phase V shared data structures.

    Access via CollectiveState.instance().
    """
    _inst: "CollectiveState | None" = None
    _inst_lock = threading.Lock()

    def __init__(self) -> None:
        self.niche_map      = NicheMap()
        self.soul_store     = SoulStore()
        self.symbol_table   = SymbolTable()
        self.collective_mem = CollectiveMemory()
        self._pulse         = 0
        self._pulse_lock    = threading.Lock()

    @classmethod
    def instance(cls) -> "CollectiveState":
        if cls._inst is None:
            with cls._inst_lock:
                if cls._inst is None:
                    cls._inst = cls()
        return cls._inst

    @classmethod
    def reset(cls) -> None:
        """For testing — reset the singleton."""
        with cls._inst_lock:
            cls._inst = None

    def tick(self) -> int:
        with self._pulse_lock:
            self._pulse += 1
            return self._pulse

    @property
    def pulse(self) -> int:
        with self._pulse_lock:
            return self._pulse


# ─────────────────────────────────────────────────────────────────────────────
# Phase V opcode handlers
# ─────────────────────────────────────────────────────────────────────────────

def op_niche_declare(instr: SomaInstruction, agent_id: int,
                     registers: list[float], state: CollectiveState | None = None) -> int:
    """
    0x73  NICHE_DECLARE  N<reg>, #imm

    Broadcasts the agent's current specialisation to the global NicheMap.

    imm == NICHE_IMM_DECLARE (0x0001)  → declare niche at (som_x, som_y)
    imm == NICHE_IMM_WITHDRAW (0x0002) → withdraw from current niche

    Returns the niche_id the agent now occupies (or -1 on withdraw).
    """
    if state is None:
        state = CollectiveState.instance()

    if instr.imm == NICHE_IMM_WITHDRAW:
        state.niche_map.withdraw(agent_id)
        log.debug("Agent %d withdrew from niche", agent_id)
        return -1

    niche_id = NicheMap.som_to_niche(instr.som_x, instr.som_y)
    state.niche_map.declare(agent_id, niche_id)

    log.debug("Agent %d declared niche %d (som=%d,%d)",
              agent_id, niche_id, instr.som_x, instr.som_y)
    return niche_id


def op_symbol_emerge(instr: SomaInstruction, agent_id: int,
                     state: CollectiveState | None = None) -> int | None:
    """
    0x74  SYMBOL_EMERGE  R<reg>, #activation_count

    Records that agent_id co-activated SOM cell (som_x, som_y).
    If activation_count reaches SYMBOL_BIND_THRESH, a new symbol ID is minted
    and returned.  Returns None if no new symbol emerged.

    The symbol ID is written into reg (caller is responsible for the write-back).
    """
    if state is None:
        state = CollectiveState.instance()

    sym_id = state.symbol_table.record_activation(
        agent_id, instr.som_x, instr.som_y,
        threshold=max(instr.imm, SYMBOL_BIND_THRESH)
    )
    if sym_id is not None:
        log.info("Symbol %d emerged from agent %d at SOM(%d,%d)",
                 sym_id, agent_id, instr.som_x, instr.som_y)
    return sym_id


def op_heritage_load(instr: SomaInstruction, agent_id: int,
                     parent_id: int,
                     out_registers: list[list[float]],
                     state: CollectiveState | None = None) -> int:
    """
    0x75  HERITAGE_LOAD  PARENT, #top_k

    Loads the parent's soul top-K weight vectors into the child's registers
    (R0 … R_{top_k-1}).  Called immediately after SPAWN / FORK.

    Returns the number of vectors actually loaded (may be < top_k if parent
    soul record is sparse).
    """
    if state is None:
        state = CollectiveState.instance()

    top_k   = max(1, min(instr.imm or HERITAGE_TOP_K, HERITAGE_TOP_K))
    vectors = state.soul_store.load_parent(parent_id, top_k)

    loaded = 0
    for i, vec in enumerate(vectors[:top_k]):
        if i < len(out_registers):
            out_registers[i] = list(vec[:VEC_DIM])
            loaded += 1

    log.debug("Agent %d loaded %d/%d heritage vectors from parent %d",
              agent_id, loaded, top_k, parent_id)
    return loaded


def op_niche_query(instr: SomaInstruction, agent_id: int,
                   state: CollectiveState | None = None) -> tuple[float, int | None]:
    """
    0x76  NICHE_QUERY  N<reg>

    Queries the density of the agent's current niche.
    If density > NICHE_MIGRATE_THRESH, returns the least-crowded alternative
    niche so the interpreter can trigger a SOM walk / migration.

    Returns:
        (density: float, migrate_to: int | None)
        migrate_to is None if density is below threshold (no migration needed).
    """
    if state is None:
        state = CollectiveState.instance()

    # Determine which niche this agent is in
    cur_niche = NicheMap.som_to_niche(instr.som_x, instr.som_y)
    density   = state.niche_map.density(cur_niche)

    if density > NICHE_MIGRATE_THRESH:
        target = state.niche_map.least_crowded_niche()
        if target == cur_niche:
            return density, None   # already least crowded — stay
        log.info(
            "Agent %d migrating: niche %d density=%.2f > %.2f → niche %d",
            agent_id, cur_niche, density, NICHE_MIGRATE_THRESH, target
        )
        return density, target

    return density, None


def op_collective_sync(instr: SomaInstruction, agent_id: int,
                       registers: list[float],
                       niche_id: int,
                       som_x: int,
                       som_y: int,
                       pulse: int,
                       state: CollectiveState | None = None) -> list[float] | None:
    """
    0x77  COLLECTIVE_SYNC

    1. Agent submits its current snapshot to collective memory.
    2. If agent_id == 0 (the orchestrator), triggers consolidation and returns
       the centroid for this agent's niche.
    3. Other agents call this and block briefly until centroids are ready,
       then read back their niche centroid.

    Returns the niche centroid vector if available, else None.
    """
    if state is None:
        state = CollectiveState.instance()

    snap = AgentSnapshot(
        agent_id=agent_id,
        niche_id=niche_id,
        registers=list(registers[:VEC_DIM]),
        som_x=som_x,
        som_y=som_y,
        pulse=pulse,
    )
    state.collective_mem.submit_snapshot(snap)

    # Agent 0 triggers consolidation (it's the last to submit by convention)
    if agent_id == 0:
        state.collective_mem.consolidate(state.niche_map)
        log.info("Agent 0 triggered COLLECTIVE_SYNC at pulse %d; H=%.4f",
                 pulse, state.niche_map.shannon_entropy())

    centroid = state.collective_mem.get_centroid(niche_id)
    return centroid


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher  (called by interpreter._execute for Phase V opcodes)
# ─────────────────────────────────────────────────────────────────────────────

class PhaseVDispatcher:
    """
    Drop-in dispatcher for the SOMA interpreter.

    Usage in interpreter._execute:

        if instr.opcode in PHASE_V_OPS:
            result = self._phase_v.dispatch(instr, self._current_agent_id,
                                            self.r_registers, ...)
    """

    def __init__(self) -> None:
        self._state = CollectiveState.instance()

    def dispatch(self, instr: SomaInstruction, agent_id: int,
                 registers: list[float],
                 parent_id: int = 0,
                 out_registers: list[list[float]] | None = None,
                 niche_id: int = 0,
                 som_x: int = 0, som_y: int = 0,
                 pulse: int = 0) -> object:
        """
        Route a Phase V instruction to its handler.
        Returns handler-specific value (see individual op_ functions).
        """
        op = instr.opcode

        if op == OP_NICHE_DECLARE:
            return op_niche_declare(instr, agent_id, registers, self._state)

        elif op == OP_SYMBOL_EMERGE:
            return op_symbol_emerge(instr, agent_id, self._state)

        elif op == OP_HERITAGE_LOAD:
            regs = out_registers or []
            return op_heritage_load(instr, agent_id, parent_id, regs, self._state)

        elif op == OP_NICHE_QUERY:
            return op_niche_query(instr, agent_id, self._state)

        elif op == OP_COLLECTIVE_SYNC:
            return op_collective_sync(
                instr, agent_id, registers, niche_id,
                som_x, som_y, pulse, self._state
            )

        raise ValueError(f"Unknown Phase V opcode: 0x{op:02X}")


# ─────────────────────────────────────────────────────────────────────────────
# Entropy monitor  (attach to simulation loop)
# ─────────────────────────────────────────────────────────────────────────────

class EntropyMonitor:
    """
    Tracks Shannon entropy of the niche distribution over time.

    Attach to a simulation loop and call .record() each pulse (or every N pulses).
    Use .report() to get a formatted summary.

    Target: H → log2(64) ≈ 6.0 after 100 K pulses with 64 agents.
    """

    def __init__(self, sample_interval: int = 1000) -> None:
        self._interval = sample_interval
        self._history: list[tuple[int, float]] = []   # (pulse, entropy)
        self._state    = CollectiveState.instance()

    def record(self, pulse: int) -> float | None:
        if pulse % self._interval == 0:
            H = self._state.niche_map.shannon_entropy()
            self._history.append((pulse, H))
            return H
        return None

    def report(self) -> str:
        if not self._history:
            return "No entropy samples recorded."
        H_max = math.log2(NICHE_CAPACITY)
        last_pulse, last_H = self._history[-1]
        pct = 100 * last_H / H_max if H_max > 0 else 0
        lines = [
            f"Shannon Entropy of Niche Distribution",
            f"  Max possible: log2({NICHE_CAPACITY}) = {H_max:.4f} bits",
            f"  At pulse {last_pulse:,}: H = {last_H:.4f} bits ({pct:.1f}% of max)",
            f"  Samples collected: {len(self._history)}",
            "",
            "  Pulse        Entropy   % of max",
            "  ─────────────────────────────────",
        ]
        for pulse, H in self._history[::max(1, len(self._history) // 10)]:
            p = 100 * H / H_max if H_max > 0 else 0
            lines.append(f"  {pulse:>10,}    {H:>6.3f}    {p:>5.1f}%")
        return "\n".join(lines)

    @property
    def history(self) -> list[tuple[int, float]]:
        return list(self._history)

    @property
    def current_entropy(self) -> float:
        return self._state.niche_map.shannon_entropy()

    @property
    def max_entropy(self) -> float:
        return math.log2(NICHE_CAPACITY)
