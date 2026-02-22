#!/usr/bin/env python3
"""
examples/phase_v_emergence.py — Phase V Collective Intelligence demo.

Simulates 64 agents running for 100,000 pulses on a 16×16 SOM.
No agent is told which niche to occupy.
Specialisation emerges via:
  1. Each agent senses its local SOM BMU position
  2. It declares its niche                (NICHE_DECLARE  0x73)
  3. It records its SOM activation        (SYMBOL_EMERGE  0x74)
  4. Newborns inherit parent vectors      (HERITAGE_LOAD  0x75)
  5. It queries density and migrates      (NICHE_QUERY    0x76)
  6. Every COLLECTIVE_WINDOW pulses: sync (COLLECTIVE_SYNC 0x77)

Measurement: Shannon entropy H of niche distribution
  Start:  H ≈ 0  (all agents pile into niche 0)
  Target: H → log2(64) ≈ 6.0 after 100 K pulses
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.bridge import (
    SomaInstruction, OP_NICHE_DECLARE, OP_SYMBOL_EMERGE,
    OP_HERITAGE_LOAD, OP_NICHE_QUERY, OP_COLLECTIVE_SYNC,
    NICHE_IMM_DECLARE, COLLECTIVE_WINDOW, HERITAGE_TOP_K,
    encode_word, NICHE_CAPACITY,
)
from runtime.collective import (
    CollectiveState, EntropyMonitor, PhaseVDispatcher,
    NicheMap, SoulStore, VEC_DIM,
)

# ─────────────────────────────────────────────────────────────────────────────
# Simulated agent
# ─────────────────────────────────────────────────────────────────────────────

SOM_SIZE = 16   # 16×16 SOM grid


class SimAgent:
    """
    Lightweight agent stub that exercises all 5 Phase V opcodes.
    No threads — runs in a single-threaded simulation loop for speed.
    """

    def __init__(self, agent_id: int, parent_id: int | None,
                 dispatcher: PhaseVDispatcher, soul_store: SoulStore) -> None:
        self.id        = agent_id
        self.parent_id = parent_id
        self._d        = dispatcher
        self._souls    = soul_store

        # SOM position — starts random, drifts via migration
        self.som_x = random.randint(0, SOM_SIZE - 1)
        self.som_y = random.randint(0, SOM_SIZE - 1)

        # Weight registers R0–R7
        self.registers: list[float] = [random.gauss(0, 1) for _ in range(VEC_DIM)]

        # Current niche
        self.niche_id: int = NicheMap.som_to_niche(self.som_x, self.som_y)

        # Phase V state
        self._loaded_heritage = False

    # ── Heritage load (once, at birth) ────────────────────────────────────────
    def load_heritage(self, pulse: int) -> None:
        if self._loaded_heritage or self.parent_id is None:
            return
        instr = SomaInstruction(
            opcode=OP_HERITAGE_LOAD, agent_id=self.id,
            som_x=self.som_x, som_y=self.som_y,
            reg=0xFF01,   # PARENT register
            imm=HERITAGE_TOP_K,
        )
        out_regs: list[list[float]] = [self.registers[:]]   # current regs as base
        n_loaded = self._d.dispatch(
            instr, self.id, self.registers,
            parent_id=self.parent_id,
            out_registers=out_regs,
        )
        if n_loaded and out_regs:
            self.registers = out_regs[0]
        self._loaded_heritage = True

    # ── Per-pulse tick ─────────────────────────────────────────────────────────
    def tick(self, pulse: int) -> None:
        # 1. SOM walk: drift position slightly (simulates SOM_BMU / SOM_WALK)
        dx = random.choice([-1, 0, 0, 1])
        dy = random.choice([-1, 0, 0, 1])
        self.som_x = max(0, min(SOM_SIZE - 1, self.som_x + dx))
        self.som_y = max(0, min(SOM_SIZE - 1, self.som_y + dy))

        # 2. NICHE_DECLARE: broadcast specialisation
        nd_instr = SomaInstruction(
            opcode=OP_NICHE_DECLARE, agent_id=self.id,
            som_x=self.som_x, som_y=self.som_y,
            reg=0x0300,   # N0
            imm=NICHE_IMM_DECLARE,
        )
        self.niche_id = self._d.dispatch(
            nd_instr, self.id, self.registers,
            niche_id=self.niche_id,
            som_x=self.som_x, som_y=self.som_y, pulse=pulse,
        )

        # 3. SYMBOL_EMERGE: record co-activation
        se_instr = SomaInstruction(
            opcode=OP_SYMBOL_EMERGE, agent_id=self.id,
            som_x=self.som_x, som_y=self.som_y,
            reg=0x0000,   # R0
            imm=3,        # bind threshold
        )
        self._d.dispatch(se_instr, self.id, self.registers,
                         som_x=self.som_x, som_y=self.som_y)

        # 4. NICHE_QUERY: migrate if crowded
        nq_instr = SomaInstruction(
            opcode=OP_NICHE_QUERY, agent_id=self.id,
            som_x=self.som_x, som_y=self.som_y,
            reg=0x0300,   # N0
            imm=0,
        )
        result = self._d.dispatch(nq_instr, self.id, self.registers,
                                  som_x=self.som_x, som_y=self.som_y)
        density, migrate_to = result
        if migrate_to is not None:
            # Migrate: jump SOM position toward the target niche
            target_x = (migrate_to % SOM_SIZE)
            target_y = (migrate_to // SOM_SIZE) % SOM_SIZE
            # Move 1 step toward target
            self.som_x += int(math.copysign(1, target_x - self.som_x)) if target_x != self.som_x else 0
            self.som_y += int(math.copysign(1, target_y - self.som_y)) if target_y != self.som_y else 0
            self.som_x = max(0, min(SOM_SIZE - 1, self.som_x))
            self.som_y = max(0, min(SOM_SIZE - 1, self.som_y))

        # 5. Update registers slightly (simulates SOM_TRAIN influence)
        centroid = CollectiveState.instance().collective_mem.get_centroid(self.niche_id)
        if centroid:
            lr = 0.01
            for i in range(VEC_DIM):
                self.registers[i] += lr * (centroid[i] - self.registers[i])

        # 6. Occasionally save soul
        if pulse % 500 == 0:
            CollectiveState.instance().soul_store.save(
                self.id, self.niche_id, [self.registers[:]]
            )

    # ── COLLECTIVE_SYNC ────────────────────────────────────────────────────────
    def collective_sync(self, pulse: int) -> None:
        cs_instr = SomaInstruction(
            opcode=OP_COLLECTIVE_SYNC, agent_id=self.id,
            som_x=self.som_x, som_y=self.som_y,
            reg=0, imm=COLLECTIVE_WINDOW,
        )
        self._d.dispatch(
            cs_instr, self.id, self.registers,
            niche_id=self.niche_id,
            som_x=self.som_x, som_y=self.som_y, pulse=pulse,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    n_agents: int = 64,
    n_pulses: int = 100_000,
    sample_interval: int = 5_000,
    seed: int | None = 42,
) -> EntropyMonitor:
    """
    Run the Phase V collective intelligence simulation.

    Returns an EntropyMonitor with the full entropy history.
    """
    if seed is not None:
        random.seed(seed)

    # Fresh collective state for this run
    CollectiveState.reset()
    state      = CollectiveState.instance()
    dispatcher = PhaseVDispatcher()
    monitor    = EntropyMonitor(sample_interval=sample_interval)

    print(f"Phase V Simulation: {n_agents} agents × {n_pulses:,} pulses")
    print(f"Max entropy: log2({NICHE_CAPACITY}) = {math.log2(NICHE_CAPACITY):.4f} bits")
    print(f"Sample interval: every {sample_interval:,} pulses")
    print()

    # Spawn initial agents (no parents)
    agents = [
        SimAgent(i, parent_id=None, dispatcher=dispatcher, soul_store=state.soul_store)
        for i in range(n_agents)
    ]

    t0 = time.perf_counter()

    for pulse in range(1, n_pulses + 1):
        state.tick()

        # Heritage load on pulse 1 (parents don't exist yet, so noop)
        if pulse == 1:
            for ag in agents:
                ag.load_heritage(pulse)

        # Every agent ticks
        for ag in agents:
            ag.tick(pulse)

        # COLLECTIVE_SYNC every COLLECTIVE_WINDOW pulses
        if pulse % COLLECTIVE_WINDOW == 0:
            # All agents submit, agent 0 consolidates last
            for ag in sorted(agents, key=lambda a: a.id, reverse=True):
                ag.collective_sync(pulse)

        # Sample entropy
        H = monitor.record(pulse)
        if H is not None:
            elapsed = time.perf_counter() - t0
            pct     = 100 * H / monitor.max_entropy
            bar_len = int(40 * H / monitor.max_entropy)
            bar     = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  pulse={pulse:>7,}  H={H:5.3f} [{bar}] {pct:5.1f}%  ({elapsed:.1f}s)")

    print()
    print(monitor.report())
    print()

    # Final niche distribution
    dist = state.niche_map.distribution()
    print(f"Final niche distribution ({len(dist)} occupied niches):")
    for niche_id in sorted(dist)[:20]:
        count = dist[niche_id]
        bar   = "▪" * count
        print(f"  niche {niche_id:>3d}: {bar} ({count})")
    if len(dist) > 20:
        print(f"  … and {len(dist) - 20} more niches")

    # Symbol table
    symbols = state.symbol_table.all_symbols()
    print(f"\nSymbols emerged: {len(symbols)}")
    for sym in symbols[:10]:
        print(f"  sym#{sym.symbol_id} @ SOM({sym.som_x},{sym.som_y})"
              f"  activations={sym.activation_count}  agents={len(sym.agents)}")

    return monitor


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SOMA Phase V Collective Intelligence Demo")
    parser.add_argument("--agents",   type=int, default=64,      help="Number of agents (default 64)")
    parser.add_argument("--pulses",   type=int, default=100_000, help="Number of pulses (default 100000)")
    parser.add_argument("--interval", type=int, default=5_000,   help="Entropy sample interval (default 5000)")
    parser.add_argument("--seed",     type=int, default=42,      help="RNG seed (default 42)")
    args = parser.parse_args()

    monitor = run_simulation(
        n_agents=args.agents,
        n_pulses=args.pulses,
        sample_interval=args.interval,
        seed=args.seed,
    )

    final_H   = monitor.current_entropy
    max_H     = monitor.max_entropy
    pct       = 100 * final_H / max_H
    threshold = 0.80   # 80% of max is a reasonable "emerged" threshold

    print()
    if pct >= threshold * 100:
        print(f"✅  Specialisation EMERGED: H = {final_H:.4f} ({pct:.1f}% of log2({NICHE_CAPACITY}))")
    else:
        print(f"⚠️   Specialisation in progress: H = {final_H:.4f} ({pct:.1f}% of log2({NICHE_CAPACITY}))")
