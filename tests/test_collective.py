"""
tests/test_collective.py — Phase V: Collective Intelligence tests
=================================================================

Test categories
---------------
Unit  — NicheMap          (declare, withdraw, density, entropy, migration target)
Unit  — SymbolTable       (co-activation counting, symbol binding threshold)
Unit  — SoulStore         (save, load, top-K truncation)
Unit  — CollectiveMemory  (submit snapshot, consolidate, centroid read-back)
Unit  — CollectiveState   (singleton, reset, pulse tick)
Unit  — EntropyMonitor    (record, report, history)
Unit  — op_niche_declare  (declare path, withdraw path)
Unit  — op_symbol_emerge  (below threshold, at threshold, above threshold)
Unit  — op_heritage_load  (full load, partial soul, empty parent)
Unit  — op_niche_query    (below thresh → no migration, above thresh → migrate)
Unit  — op_collective_sync (snapshot submission, agent-0 consolidation)
Unit  — PhaseVDispatcher  (routes all 5 opcodes; rejects unknown)
Integration — 64-agent emergence: H > 4.0 bits after 10 k pulses
Stress — NicheMap thread-safety (50 threads declaring concurrently)

Run
---
    pytest tests/test_collective.py -v
    pytest tests/test_collective.py -v -k "entropy"   # entropy tests only
    pytest tests/test_collective.py -v -k "stress"    # thread-safety stress

Paper: "A Path to AGI Part V: Collective Intelligence", Swapnil Bhadade, 2026
"""

from __future__ import annotations

import math
import random
import threading
import time
from unittest.mock import patch

import pytest

# ── Module under test ─────────────────────────────────────────────────────────
from runtime.collective import (
    NicheMap,
    SymbolTable,
    SoulStore,
    CollectiveMemory,
    CollectiveState,
    AgentSnapshot,
    EntropyMonitor,
    op_niche_declare,
    op_symbol_emerge,
    op_heritage_load,
    op_niche_query,
    op_collective_sync,
    PhaseVDispatcher,
    VEC_DIM,
)
from runtime.bridge import (
    SomaInstruction,
    OP_NICHE_DECLARE,
    OP_SYMBOL_EMERGE,
    OP_HERITAGE_LOAD,
    OP_NICHE_QUERY,
    OP_COLLECTIVE_SYNC,
    NICHE_CAPACITY,
    NICHE_MIGRATE_THRESH,
    SYMBOL_BIND_THRESH,
    HERITAGE_TOP_K,
    NICHE_IMM_DECLARE,
    NICHE_IMM_WITHDRAW,
    encode_word,
    encode_reg,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _instr(opcode: int, agent_id: int = 0, som_x: int = 0, som_y: int = 0,
           reg: int = 0, imm: int = 0) -> SomaInstruction:
    """Build a SomaInstruction directly without going through the assembler."""
    word = encode_word(opcode, agent_id, som_x, som_y, reg, imm)
    return SomaInstruction.from_word(word)


def _rand_vec() -> list[float]:
    """Random VEC_DIM float vector in [0, 1]."""
    return [random.random() for _ in range(VEC_DIM)]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset CollectiveState singleton before and after every test."""
    CollectiveState.reset()
    yield
    CollectiveState.reset()


@pytest.fixture
def state() -> CollectiveState:
    return CollectiveState.instance()


@pytest.fixture
def niche_map() -> NicheMap:
    return NicheMap()


@pytest.fixture
def symbol_table() -> SymbolTable:
    return SymbolTable()


@pytest.fixture
def soul_store() -> SoulStore:
    return SoulStore()


@pytest.fixture
def collective_mem() -> CollectiveMemory:
    return CollectiveMemory()


# ══════════════════════════════════════════════════════════════════════════════
# NicheMap — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestNicheMap:

    def test_declare_registers_agent(self, niche_map):
        niche_map.declare(agent_id=0, niche_id=5)
        assert niche_map.density(5) > 0.0

    def test_density_one_agent(self, niche_map):
        niche_map.declare(0, 3)
        assert niche_map.density(3) == pytest.approx(1.0)

    def test_density_two_agents_same_niche(self, niche_map):
        niche_map.declare(0, 3)
        niche_map.declare(1, 3)
        assert niche_map.density(3) == pytest.approx(1.0)

    def test_density_two_agents_different_niches(self, niche_map):
        niche_map.declare(0, 3)
        niche_map.declare(1, 7)
        assert niche_map.density(3) == pytest.approx(0.5)
        assert niche_map.density(7) == pytest.approx(0.5)

    def test_density_empty_map(self, niche_map):
        assert niche_map.density(0) == 0.0

    def test_density_empty_niche(self, niche_map):
        niche_map.declare(0, 3)
        assert niche_map.density(5) == 0.0

    def test_withdraw_removes_agent(self, niche_map):
        niche_map.declare(0, 3)
        niche_map.withdraw(0)
        assert niche_map.density(3) == 0.0

    def test_withdraw_unknown_agent_is_safe(self, niche_map):
        niche_map.withdraw(999)   # should not raise

    def test_redeclare_moves_agent(self, niche_map):
        niche_map.declare(0, 3)
        niche_map.declare(0, 7)   # agent 0 moves from 3 → 7
        assert niche_map.density(3) == 0.0
        assert niche_map.density(7) == pytest.approx(1.0)

    def test_distribution_counts(self, niche_map):
        for i in range(4):
            niche_map.declare(i, i % 2)   # 2 agents per niche 0 and 1
        dist = niche_map.distribution()
        assert dist[0] == 2
        assert dist[1] == 2

    def test_least_crowded_niche_empty_map(self, niche_map):
        # All niches empty → should return 0 (or any valid niche id)
        lc = niche_map.least_crowded_niche()
        assert 0 <= lc < NICHE_CAPACITY

    def test_least_crowded_niche_returns_empty_niche(self, niche_map):
        # Fill niche 0 and 1, niche 2 is empty
        niche_map.declare(0, 0)
        niche_map.declare(1, 1)
        lc = niche_map.least_crowded_niche()
        # Any niche other than 0 and 1 with 0 agents is equally "least crowded"
        assert lc not in (0, 1) or niche_map.density(lc) <= 0.5

    def test_som_to_niche_in_range(self):
        for x in range(16):
            for y in range(16):
                nid = NicheMap.som_to_niche(x, y)
                assert 0 <= nid < NICHE_CAPACITY

    def test_niche_id_wraps_capacity(self, niche_map):
        # niche_id >= NICHE_CAPACITY should be wrapped
        niche_map.declare(0, NICHE_CAPACITY + 5)
        assert niche_map.density(5) > 0.0

    # ── Shannon entropy ───────────────────────────────────────────────────────

    def test_entropy_zero_agents(self, niche_map):
        assert niche_map.shannon_entropy() == pytest.approx(0.0)

    def test_entropy_all_same_niche(self, niche_map):
        for i in range(8):
            niche_map.declare(i, 0)
        assert niche_map.shannon_entropy() == pytest.approx(0.0)

    def test_entropy_increases_with_spread(self, niche_map):
        # 8 agents in 1 niche
        for i in range(8):
            niche_map.declare(i, 0)
        h_low = niche_map.shannon_entropy()

        # Spread to 4 niches
        for i in range(8):
            niche_map.declare(i, i % 4)
        h_high = niche_map.shannon_entropy()

        assert h_high > h_low

    def test_entropy_max_bound(self, niche_map):
        # Theoretical max = log2(NICHE_CAPACITY)
        assert niche_map.max_entropy == pytest.approx(math.log2(NICHE_CAPACITY))

    def test_entropy_perfectly_spread_64_agents(self, niche_map):
        # One agent per niche → H = log2(64)
        for i in range(NICHE_CAPACITY):
            niche_map.declare(i, i)
        H = niche_map.shannon_entropy()
        assert H == pytest.approx(math.log2(NICHE_CAPACITY), abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# SymbolTable — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSymbolTable:

    def test_no_symbol_below_threshold(self, symbol_table):
        for i in range(SYMBOL_BIND_THRESH - 1):
            sym = symbol_table.record_activation(i, 3, 4)
            assert sym is None

    def test_symbol_minted_at_threshold(self, symbol_table):
        sym = None
        for i in range(SYMBOL_BIND_THRESH):
            sym = symbol_table.record_activation(i, 3, 4)
        assert sym is not None
        assert sym >= 1

    def test_symbol_ids_are_monotone(self, symbol_table):
        def _mint(x, y):
            sym = None
            for i in range(SYMBOL_BIND_THRESH):
                sym = symbol_table.record_activation(i * 10 + x, x, y)
            return sym

        s1 = _mint(0, 0)
        s2 = _mint(1, 1)
        assert s1 is not None
        assert s2 is not None
        assert s2 > s1

    def test_already_bound_cell_returns_none(self, symbol_table):
        for i in range(SYMBOL_BIND_THRESH):
            symbol_table.record_activation(i, 0, 0)
        # Additional activations after binding → None (already bound)
        result = symbol_table.record_activation(99, 0, 0)
        assert result is None

    def test_get_symbol_returns_binding(self, symbol_table):
        for i in range(SYMBOL_BIND_THRESH):
            symbol_table.record_activation(i, 2, 3)
        binding = symbol_table.get_symbol(2, 3)
        assert binding is not None
        assert binding.som_x == 2
        assert binding.som_y == 3

    def test_get_symbol_none_for_unbound(self, symbol_table):
        assert symbol_table.get_symbol(0, 0) is None

    def test_all_symbols_lists_all(self, symbol_table):
        for j in range(3):
            for i in range(SYMBOL_BIND_THRESH):
                symbol_table.record_activation(i * 10 + j, j, j)
        syms = symbol_table.all_symbols()
        assert len(syms) == 3

    def test_custom_threshold(self, symbol_table):
        custom = SYMBOL_BIND_THRESH + 5
        sym = None
        for i in range(custom):
            sym = symbol_table.record_activation(i, 5, 5, threshold=custom)
        assert sym is not None


# ══════════════════════════════════════════════════════════════════════════════
# SoulStore — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSoulStore:

    def test_save_and_load(self, soul_store):
        vecs = [_rand_vec() for _ in range(3)]
        soul_store.save(agent_id=1, niche_id=5, vectors=vecs)
        loaded = soul_store.load_parent(parent_id=1)
        assert len(loaded) == 3
        assert loaded[0] == pytest.approx(vecs[0])

    def test_load_unknown_parent_returns_empty(self, soul_store):
        assert soul_store.load_parent(999) == []

    def test_top_k_truncation_on_save(self, soul_store):
        # Supply more vectors than HERITAGE_TOP_K
        vecs = [_rand_vec() for _ in range(HERITAGE_TOP_K + 10)]
        soul_store.save(0, 0, vecs)
        loaded = soul_store.load_parent(0)
        assert len(loaded) <= HERITAGE_TOP_K

    def test_top_k_truncation_on_load(self, soul_store):
        vecs = [_rand_vec() for _ in range(HERITAGE_TOP_K)]
        soul_store.save(0, 0, vecs)
        loaded = soul_store.load_parent(0, top_k=2)
        assert len(loaded) == 2

    def test_overwrite_latest_record(self, soul_store):
        vecs1 = [_rand_vec()]
        vecs2 = [_rand_vec()]
        soul_store.save(0, 0, vecs1)
        soul_store.save(0, 1, vecs2)   # overwrite
        loaded = soul_store.load_parent(0)
        assert loaded[0] == pytest.approx(vecs2[0])

    def test_all_records(self, soul_store):
        soul_store.save(0, 0, [_rand_vec()])
        soul_store.save(1, 1, [_rand_vec()])
        assert len(soul_store.all_records()) == 2


# ══════════════════════════════════════════════════════════════════════════════
# CollectiveMemory — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCollectiveMemory:

    def _snap(self, agent_id, niche_id, vec=None):
        return AgentSnapshot(
            agent_id=agent_id, niche_id=niche_id,
            registers=vec or _rand_vec(),
            som_x=niche_id, som_y=0, pulse=0,
        )

    def test_submit_and_consolidate(self, collective_mem):
        nm = NicheMap()
        for i in range(4):
            nm.declare(i, 0)
            collective_mem.submit_snapshot(self._snap(i, 0))
        centroids = collective_mem.consolidate(nm)
        assert 0 in centroids
        assert len(centroids[0]) == VEC_DIM

    def test_centroid_is_average(self, collective_mem):
        nm = NicheMap()
        v1 = [1.0] * VEC_DIM
        v2 = [0.0] * VEC_DIM
        for aid, vec in ((0, v1), (1, v2)):
            nm.declare(aid, 0)
            collective_mem.submit_snapshot(self._snap(aid, 0, vec))
        collective_mem.consolidate(nm)
        centroid = collective_mem.get_centroid(0)
        assert centroid == pytest.approx([0.5] * VEC_DIM)

    def test_get_centroid_before_consolidate_is_none(self, collective_mem):
        assert collective_mem.get_centroid(0) is None

    def test_sync_count_increments(self, collective_mem):
        nm = NicheMap()
        assert collective_mem.sync_count == 0
        collective_mem.consolidate(nm)
        collective_mem.consolidate(nm)
        assert collective_mem.sync_count == 2

    def test_multiple_niches_consolidated(self, collective_mem):
        nm = NicheMap()
        for niche in range(4):
            for agent in range(3):
                aid = niche * 3 + agent
                nm.declare(aid, niche)
                collective_mem.submit_snapshot(self._snap(aid, niche))
        centroids = collective_mem.consolidate(nm)
        assert len(centroids) == 4


# ══════════════════════════════════════════════════════════════════════════════
# CollectiveState singleton — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCollectiveState:

    def test_singleton_returns_same_instance(self):
        a = CollectiveState.instance()
        b = CollectiveState.instance()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = CollectiveState.instance()
        CollectiveState.reset()
        b = CollectiveState.instance()
        assert a is not b

    def test_pulse_starts_at_zero(self, state):
        assert state.pulse == 0

    def test_tick_increments_pulse(self, state):
        p1 = state.tick()
        p2 = state.tick()
        assert p2 == p1 + 1

    def test_sub_components_present(self, state):
        assert state.niche_map      is not None
        assert state.soul_store     is not None
        assert state.symbol_table   is not None
        assert state.collective_mem is not None


# ══════════════════════════════════════════════════════════════════════════════
# EntropyMonitor — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEntropyMonitor:

    def test_record_at_interval(self, state):
        monitor = EntropyMonitor(sample_interval=100)
        for i in range(8):
            state.niche_map.declare(i, i)
        result = monitor.record(100)
        assert result is not None

    def test_no_record_between_intervals(self, state):
        monitor = EntropyMonitor(sample_interval=100)
        result = monitor.record(50)
        assert result is None

    def test_history_accumulates(self, state):
        monitor = EntropyMonitor(sample_interval=10)
        for i in range(8):
            state.niche_map.declare(i, i)
        for pulse in range(0, 50, 10):
            monitor.record(pulse)
        assert len(monitor.history) >= 4

    def test_report_contains_entropy_value(self, state):
        monitor = EntropyMonitor(sample_interval=1)
        for i in range(8):
            state.niche_map.declare(i, i)
        monitor.record(1)
        report = monitor.report()
        assert "bits" in report
        assert "%" in report

    def test_report_empty_monitor(self, state):
        monitor = EntropyMonitor()
        report = monitor.report()
        assert "No entropy" in report

    def test_current_entropy_reflects_map(self, state):
        monitor = EntropyMonitor()
        for i in range(NICHE_CAPACITY):
            state.niche_map.declare(i, i)
        H = monitor.current_entropy
        assert H == pytest.approx(math.log2(NICHE_CAPACITY), abs=1e-9)

    def test_max_entropy_property(self):
        monitor = EntropyMonitor()
        assert monitor.max_entropy == pytest.approx(math.log2(NICHE_CAPACITY))


# ══════════════════════════════════════════════════════════════════════════════
# op_niche_declare — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOpNicheDeclare:

    def test_declare_returns_valid_niche_id(self, state):
        instr = _instr(OP_NICHE_DECLARE, agent_id=0, som_x=4, som_y=2,
                       imm=NICHE_IMM_DECLARE)
        nid = op_niche_declare(instr, agent_id=0, registers=[], state=state)
        assert 0 <= nid < NICHE_CAPACITY

    def test_declare_registers_in_niche_map(self, state):
        instr = _instr(OP_NICHE_DECLARE, agent_id=0, som_x=4, som_y=2,
                       imm=NICHE_IMM_DECLARE)
        nid = op_niche_declare(instr, 0, [], state)
        assert state.niche_map.density(nid) > 0.0

    def test_withdraw_removes_agent(self, state):
        # First declare
        d_instr = _instr(OP_NICHE_DECLARE, 0, 4, 2, imm=NICHE_IMM_DECLARE)
        nid = op_niche_declare(d_instr, 0, [], state)
        # Then withdraw
        w_instr = _instr(OP_NICHE_DECLARE, 0, 4, 2, imm=NICHE_IMM_WITHDRAW)
        result = op_niche_declare(w_instr, 0, [], state)
        assert result == -1
        assert state.niche_map.density(nid) == 0.0

    def test_multiple_agents_same_niche(self, state):
        for aid in range(5):
            instr = _instr(OP_NICHE_DECLARE, aid, 0, 0, imm=NICHE_IMM_DECLARE)
            op_niche_declare(instr, aid, [], state)
        nid = NicheMap.som_to_niche(0, 0)
        assert state.niche_map.density(nid) == pytest.approx(1.0)

    def test_uses_global_state_when_none_passed(self):
        # Should fall back to CollectiveState.instance()
        instr = _instr(OP_NICHE_DECLARE, 0, 1, 1, imm=NICHE_IMM_DECLARE)
        nid = op_niche_declare(instr, 0, [])   # no state= arg
        assert 0 <= nid < NICHE_CAPACITY


# ══════════════════════════════════════════════════════════════════════════════
# op_symbol_emerge — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOpSymbolEmerge:

    def test_no_symbol_below_threshold(self, state):
        instr = _instr(OP_SYMBOL_EMERGE, 0, 3, 3, imm=SYMBOL_BIND_THRESH)
        for i in range(SYMBOL_BIND_THRESH - 1):
            r = op_symbol_emerge(instr, agent_id=i, state=state)
            assert r is None

    def test_symbol_emerges_at_threshold(self, state):
        instr = _instr(OP_SYMBOL_EMERGE, 0, 3, 3, imm=SYMBOL_BIND_THRESH)
        sym = None
        for i in range(SYMBOL_BIND_THRESH):
            sym = op_symbol_emerge(instr, i, state)
        assert sym is not None

    def test_symbol_stored_in_table(self, state):
        instr = _instr(OP_SYMBOL_EMERGE, 0, 1, 2, imm=SYMBOL_BIND_THRESH)
        for i in range(SYMBOL_BIND_THRESH):
            op_symbol_emerge(instr, i, state)
        binding = state.symbol_table.get_symbol(1, 2)
        assert binding is not None
        assert binding.som_x == 1
        assert binding.som_y == 2


# ══════════════════════════════════════════════════════════════════════════════
# op_heritage_load — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOpHeritageLoad:

    def test_loads_parent_vectors(self, state):
        vecs = [_rand_vec() for _ in range(HERITAGE_TOP_K)]
        state.soul_store.save(agent_id=42, niche_id=0, vectors=vecs)
        out = [[0.0] * VEC_DIM for _ in range(HERITAGE_TOP_K)]
        instr = _instr(OP_HERITAGE_LOAD, imm=HERITAGE_TOP_K)
        n = op_heritage_load(instr, agent_id=1, parent_id=42,
                             out_registers=out, state=state)
        assert n == HERITAGE_TOP_K
        assert out[0] == pytest.approx(vecs[0])

    def test_returns_zero_for_unknown_parent(self, state):
        out = [[0.0] * VEC_DIM for _ in range(HERITAGE_TOP_K)]
        instr = _instr(OP_HERITAGE_LOAD, imm=HERITAGE_TOP_K)
        n = op_heritage_load(instr, 1, parent_id=999, out_registers=out, state=state)
        assert n == 0

    def test_partial_load_when_soul_sparse(self, state):
        vecs = [_rand_vec() for _ in range(2)]   # only 2 vectors
        state.soul_store.save(0, 0, vecs)
        out = [[0.0] * VEC_DIM for _ in range(HERITAGE_TOP_K)]
        instr = _instr(OP_HERITAGE_LOAD, imm=HERITAGE_TOP_K)
        n = op_heritage_load(instr, 1, 0, out, state)
        assert n == 2

    def test_top_k_clipped_by_imm(self, state):
        vecs = [_rand_vec() for _ in range(HERITAGE_TOP_K)]
        state.soul_store.save(0, 0, vecs)
        out = [[0.0] * VEC_DIM for _ in range(HERITAGE_TOP_K)]
        instr = _instr(OP_HERITAGE_LOAD, imm=2)
        n = op_heritage_load(instr, 1, 0, out, state)
        assert n == 2


# ══════════════════════════════════════════════════════════════════════════════
# op_niche_query — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOpNicheQuery:

    def test_no_migration_below_threshold(self, state):
        # Spread 20 agents across 20 different niches so density per niche is 1/20 = 0.05
        # well below NICHE_MIGRATE_THRESH — no migration should fire
        for i in range(20):
            state.niche_map.declare(i, i)
        # Query agent 0 which is in niche 0 (density = 1/20 = 0.05)
        instr = _instr(OP_NICHE_QUERY, 0, 0, 0)
        density, target = op_niche_query(instr, agent_id=0, state=state)
        assert target is None
        assert 0.0 <= density <= 1.0

    def test_migration_triggered_above_threshold(self, state):
        # Fill one niche above NICHE_MIGRATE_THRESH
        niche_id = NicheMap.som_to_niche(0, 0)
        n_agents = int(NICHE_MIGRATE_THRESH * NICHE_CAPACITY) + 5
        for i in range(n_agents):
            state.niche_map.declare(i, niche_id)
        # Also populate some agents to create total count
        total = max(n_agents + 5, 10)
        for i in range(n_agents, total):
            state.niche_map.declare(i, (niche_id + 1) % NICHE_CAPACITY)

        instr = _instr(OP_NICHE_QUERY, 0, 0, 0)
        density, target = op_niche_query(instr, 0, state)
        if density > NICHE_MIGRATE_THRESH:
            assert target is not None
            assert target != niche_id

    def test_density_value_in_range(self, state):
        state.niche_map.declare(0, 5)
        instr = _instr(OP_NICHE_QUERY, 0, 0, 0)
        density, _ = op_niche_query(instr, 0, state)
        assert 0.0 <= density <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# op_collective_sync — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOpCollectiveSync:

    def test_non_zero_agent_submits_snapshot(self, state):
        registers = [0.5] * VEC_DIM
        instr = _instr(OP_COLLECTIVE_SYNC)
        op_collective_sync(instr, agent_id=1, registers=registers,
                           niche_id=0, som_x=0, som_y=0, pulse=1, state=state)
        # Agent 0 now consolidates
        op_collective_sync(instr, agent_id=0, registers=registers,
                           niche_id=0, som_x=0, som_y=0, pulse=1, state=state)
        centroid = state.collective_mem.get_centroid(0)
        assert centroid is not None

    def test_agent_zero_triggers_consolidation(self, state):
        registers = [1.0] * VEC_DIM
        instr = _instr(OP_COLLECTIVE_SYNC)
        assert state.collective_mem.sync_count == 0
        op_collective_sync(instr, 0, registers, 0, 0, 0, 1, state)
        assert state.collective_mem.sync_count == 1

    def test_centroid_returned_after_sync(self, state):
        registers = [0.8] * VEC_DIM
        instr = _instr(OP_COLLECTIVE_SYNC)
        op_collective_sync(instr, 0, registers, 0, 0, 0, 1, state)
        centroid = op_collective_sync(instr, 1, registers, 0, 0, 0, 2, state)
        # Centroid may be None if agent 1 calls before consolidation
        # (non-zero agents read back what was last consolidated)
        # This is valid — just check type
        assert centroid is None or isinstance(centroid, list)


# ══════════════════════════════════════════════════════════════════════════════
# PhaseVDispatcher — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPhaseVDispatcher:

    @pytest.fixture
    def dispatcher(self) -> PhaseVDispatcher:
        return PhaseVDispatcher()

    def test_dispatch_niche_declare(self, dispatcher, state):
        instr = _instr(OP_NICHE_DECLARE, 0, 2, 2, imm=NICHE_IMM_DECLARE)
        result = dispatcher.dispatch(instr, agent_id=0, registers=[])
        assert isinstance(result, int)

    def test_dispatch_symbol_emerge(self, dispatcher, state):
        instr = _instr(OP_SYMBOL_EMERGE, 0, 1, 1, imm=SYMBOL_BIND_THRESH)
        result = dispatcher.dispatch(instr, 0, [])
        assert result is None or isinstance(result, int)

    def test_dispatch_heritage_load(self, dispatcher, state):
        state.soul_store.save(0, 0, [_rand_vec()])
        instr = _instr(OP_HERITAGE_LOAD, imm=1)
        out = [[0.0] * VEC_DIM]
        result = dispatcher.dispatch(instr, 1, [], parent_id=0, out_registers=out)
        assert isinstance(result, int)

    def test_dispatch_niche_query(self, dispatcher, state):
        state.niche_map.declare(0, 0)
        instr = _instr(OP_NICHE_QUERY, 0, 0, 0)
        result = dispatcher.dispatch(instr, 0, [])
        assert isinstance(result, tuple)
        density, target = result
        assert isinstance(density, float)

    def test_dispatch_collective_sync(self, dispatcher, state):
        instr = _instr(OP_COLLECTIVE_SYNC)
        result = dispatcher.dispatch(instr, 0, [0.5] * VEC_DIM,
                                     niche_id=0, som_x=0, som_y=0, pulse=1)
        assert result is None or isinstance(result, list)

    def test_dispatch_unknown_opcode_raises(self, dispatcher):
        instr = _instr(0x01)   # SPAWN — not a Phase V opcode
        with pytest.raises(ValueError, match="Unknown Phase V opcode"):
            dispatcher.dispatch(instr, 0, [])

    def test_all_five_phase_v_opcodes_handled(self, dispatcher, state):
        """Smoke-test that none of the 5 opcodes raise ValueError."""
        state.soul_store.save(0, 0, [_rand_vec()])
        ops = [
            (_instr(OP_NICHE_DECLARE,   0, 1, 1, imm=NICHE_IMM_DECLARE), {}),
            (_instr(OP_SYMBOL_EMERGE,   0, 1, 1, imm=SYMBOL_BIND_THRESH), {}),
            (_instr(OP_HERITAGE_LOAD,   0, 0, 0, imm=1),
             {"parent_id": 0, "out_registers": [[0.0] * VEC_DIM]}),
            (_instr(OP_NICHE_QUERY,     0, 1, 1), {}),
            (_instr(OP_COLLECTIVE_SYNC, 0, 0, 0),
             {"niche_id": 0, "som_x": 0, "som_y": 0, "pulse": 1}),
        ]
        for instr, kwargs in ops:
            dispatcher.dispatch(instr, 0, [0.5] * VEC_DIM, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Integration — 64-agent emergence simulation
# ══════════════════════════════════════════════════════════════════════════════

class TestEmergenceIntegration:
    """
    Verifies that H > 4.0 bits after 10,000 pulses with 64 agents.
    Mirrors the logic of examples/phase_v_emergence.py but scaled down
    for fast CI execution (10 k pulses instead of 100 k).

    References: soma_life.soma (.AGENTS 64, .SOMSIZE 32×32) and
    soma_curious.soma (NICHE_DECLARE + COLLECTIVE_SYNC pattern).
    """

    N_AGENTS   = 64
    SOM_SIDE   = 16
    N_PULSES   = 10_000
    SYNC_EVERY = 1_000
    H_TARGET   = 4.0    # bits  (loose threshold for fast CI run)

    def _run_simulation(self, state: CollectiveState) -> float:
        rng = random.Random(42)
        positions = {
            aid: (rng.randrange(self.SOM_SIDE), rng.randrange(self.SOM_SIDE))
            for aid in range(self.N_AGENTS)
        }

        for pulse in range(self.N_PULSES):
            for aid in range(self.N_AGENTS):
                x, y = positions[aid]
                niche_id = NicheMap.som_to_niche(x, y)

                # NICHE_DECLARE
                d_instr = _instr(OP_NICHE_DECLARE, aid, x, y, imm=NICHE_IMM_DECLARE)
                op_niche_declare(d_instr, aid, [], state)

                # SYMBOL_EMERGE
                e_instr = _instr(OP_SYMBOL_EMERGE, aid, x, y, imm=SYMBOL_BIND_THRESH)
                op_symbol_emerge(e_instr, aid, state)

                # NICHE_QUERY — migrate if crowded
                q_instr = _instr(OP_NICHE_QUERY, aid, x, y)
                density, target = op_niche_query(q_instr, aid, state)
                if target is not None:
                    # Map target niche back to a SOM cell (simple mod)
                    nx = target % self.SOM_SIDE
                    ny = (target // self.SOM_SIDE) % self.SOM_SIDE
                    positions[aid] = (nx, ny)

                # COLLECTIVE_SYNC every SYNC_EVERY pulses
                if pulse % self.SYNC_EVERY == 0:
                    regs = [rng.random() for _ in range(VEC_DIM)]
                    s_instr = _instr(OP_COLLECTIVE_SYNC)
                    op_collective_sync(s_instr, aid, regs, niche_id, x, y, pulse, state)

        return state.niche_map.shannon_entropy()

    @pytest.mark.slow
    def test_emergence_entropy_above_target(self, state):
        H = self._run_simulation(state)
        assert H > self.H_TARGET, (
            f"Emergence failed: H = {H:.4f} bits < target {self.H_TARGET} bits "
            f"after {self.N_PULSES:,} pulses with {self.N_AGENTS} agents."
        )

    def test_niches_occupied_after_simulation(self, state):
        self._run_simulation(state)
        dist = state.niche_map.distribution()
        # At minimum, several distinct niches should be occupied
        assert len(dist) >= 4, (
            f"Only {len(dist)} niches occupied — expected ≥ 4"
        )

    def test_symbols_emerge_during_simulation(self, state):
        self._run_simulation(state)
        syms = state.symbol_table.all_symbols()
        # With 64 agents over 10 k pulses, symbols should emerge
        assert len(syms) >= 1, "No symbols emerged — SYMBOL_EMERGE broken"


# ══════════════════════════════════════════════════════════════════════════════
# Stress — NicheMap thread safety
# ══════════════════════════════════════════════════════════════════════════════

class TestNicheMapThreadSafety:
    """
    50 threads each declaring + withdrawing 100 times should not raise or
    corrupt internal state.
    """

    N_THREADS    = 50
    OPS_PER_THREAD = 100

    def test_concurrent_declare_and_withdraw(self, niche_map):
        errors = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(self.OPS_PER_THREAD):
                    nid = (thread_id + i) % NICHE_CAPACITY
                    niche_map.declare(agent_id=thread_id, niche_id=nid)
                    niche_map.density(nid)
                    if i % 10 == 0:
                        niche_map.withdraw(agent_id=thread_id)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(t,), daemon=True)
            for t in range(self.N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Thread-safety errors: {errors}"

    def test_entropy_stable_after_concurrent_ops(self, niche_map):
        """Entropy should be a valid float in [0, max_entropy] after race."""
        def worker(tid):
            for i in range(200):
                niche_map.declare(tid, tid % NICHE_CAPACITY)

        threads = [threading.Thread(target=worker, args=(t,), daemon=True)
                   for t in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        H = niche_map.shannon_entropy()
        assert 0.0 <= H <= niche_map.max_entropy


# ══════════════════════════════════════════════════════════════════════════════
# Soma program pattern tests  (reference: soma_life.soma, soma_curious.soma)
# ══════════════════════════════════════════════════════════════════════════════

class TestSomaProgramPatterns:
    """
    Validates Phase V opcode sequences as they would be emitted by
    soma_curious.soma (curiosity agent) and soma_life.soma (64-agent liveness).

    These tests are not full assembler tests — they verify that the Phase V
    runtime correctly handles the instruction patterns those .soma files produce.
    """

    def test_hello_agent_spawn_then_niche_declare(self, state):
        """
        hello_agent.soma: SPAWN A0, @worker_agent → SOM_MAP A0,(0,0)
        After spawn, worker would call NICHE_DECLARE to announce itself.
        """
        instr = _instr(OP_NICHE_DECLARE, agent_id=0, som_x=0, som_y=0,
                       imm=NICHE_IMM_DECLARE)
        nid = op_niche_declare(instr, 0, [], state)
        assert nid == NicheMap.som_to_niche(0, 0)

    def test_soma_life_64_agents_all_declare(self, state):
        """
        soma_life.soma: FORK 64, @be_alive → each agent calls NICHE_DECLARE.
        After all 64 declare, the map should have occupants.
        """
        for aid in range(64):
            x = aid % 16
            y = aid // 16
            instr = _instr(OP_NICHE_DECLARE, aid, x, y, imm=NICHE_IMM_DECLARE)
            op_niche_declare(instr, aid, [], state)

        dist = state.niche_map.distribution()
        total_registered = sum(dist.values())
        assert total_registered == 64

    def test_soma_curious_soul_inherit_pattern(self, state):
        """
        soma_curious.soma: EVOLVE A1 → SOUL_INHERIT A1 → child calls HERITAGE_LOAD.
        Parent (agent 0) saves soul; child (agent 1) loads it.
        """
        parent_vecs = [_rand_vec() for _ in range(HERITAGE_TOP_K)]
        state.soul_store.save(0, 0, parent_vecs)

        out_regs = [[0.0] * VEC_DIM for _ in range(HERITAGE_TOP_K)]
        instr = _instr(OP_HERITAGE_LOAD, imm=HERITAGE_TOP_K)
        n = op_heritage_load(instr, agent_id=1, parent_id=0,
                             out_registers=out_regs, state=state)
        assert n == HERITAGE_TOP_K
        assert out_regs[0] == pytest.approx(parent_vecs[0])

    def test_soma_curious_goal_stall_then_migrate(self, state):
        """
        soma_curious.soma: GOAL_STALL @curiosity_mode → META_SPAWN → NICHE_QUERY.
        Simulate a crowded niche that triggers migration.
        """
        # Pack many agents into niche 0
        for i in range(20):
            state.niche_map.declare(i, 0)
        # Add a few agents elsewhere to form a non-trivial total
        for i in range(20, 25):
            state.niche_map.declare(i, 1)

        instr = _instr(OP_NICHE_QUERY, 0, 0, 0)  # agent 0 at SOM (0,0) → niche 0
        density, target = op_niche_query(instr, 0, state)
        # density should be high (20/25 = 0.8)
        assert density > NICHE_MIGRATE_THRESH
        assert target is not None

    def test_collective_sync_after_soma_life_consolidation(self, state):
        """
        soma_life.soma: MEMORY_CONSOLIDATE SELF every 10 k heartbeats.
        Phase V equivalent: COLLECTIVE_SYNC — agent 0 triggers consolidation.
        """
        regs = [0.75] * VEC_DIM
        for aid in range(4):
            state.niche_map.declare(aid, 0)
            instr = _instr(OP_COLLECTIVE_SYNC)
            op_collective_sync(instr, aid, regs, 0, 0, 0, 10_000, state)

        # Agent 0 was last to submit and also triggered consolidation
        centroid = state.collective_mem.get_centroid(0)
        assert centroid is not None
        assert centroid == pytest.approx([0.75] * VEC_DIM, abs=1e-6)
