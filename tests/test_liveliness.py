"""
tests/test_liveliness.py — Phase 2.5: Liveliness tests
=======================================================
55 tests across:
  - EmotionTag / AgentEmotionState / EmotionRegistry  (unit)
  - DECAY_STEP / PRUNE_CHECK on LiveSomMap            (unit)
  - EMOT_TAG / DECAY_PROTECT / PREDICT_ERR            (unit + integration)
  - TwoTierMemory / MemoryManager / MEMORY_CONSOLIDATE (unit + integration)
  - SomScheduler Phase 2.5 wiring                     (integration)
  - Concurrency stress                                 (stress)

Paper: "A Path to AGI Part II: Liveliness"
"""
from __future__ import annotations

import math
import threading
import time
from unittest.mock import MagicMock

import pytest

from runtime.som.som_map    import LiveSomMap
from runtime.som.emotion    import (EmotionTag, AgentEmotionState,
                                    EmotionRegistry, ProtectMode, Valence)
from runtime.som.memory     import TwoTierMemory, MemoryManager, ConsolidationReport
from runtime.som.som_scheduler import SomScheduler


# ── Helpers ──────────────────────────────────────────────────────────────────

def vec(vals=None):
    v = list(vals) if vals else [0.5] * 8
    return (v + [0.0] * 8)[:8]

def make_registry(n=4):
    from runtime.agent.agent_registry import AgentRegistry
    from runtime.agent.lifecycle import AgentState
    AgentRegistry.reset()
    reg = AgentRegistry.get_instance()
    for i in range(n):
        reg.register(agent_id=i, parent_id=None, som_x=0, som_y=0)
        reg.set_state(i, AgentState.RUNNING)
    return reg

def make_scheduler(rows=8, cols=8, n_agents=4):
    reg = make_registry(n_agents)
    som = LiveSomMap(rows, cols)
    return SomScheduler(som, reg), som, reg


# ══════════════════════════════════════════════════════════════════════════════
# EmotionTag — unit
# ══════════════════════════════════════════════════════════════════════════════

class TestEmotionTag:
    def test_emotion_score(self):
        tag = EmotionTag(valence=0.8, intensity=0.5)
        assert abs(tag.emotion_score - 0.4) < 1e-9

    def test_salience_unsigned(self):
        tag = EmotionTag(valence=-1.0, intensity=0.7)
        assert abs(tag.salience - 0.7) < 1e-9

    def test_is_protected_false_default(self):
        tag = EmotionTag()
        assert not tag.is_protected

    def test_is_protected_cycles(self):
        tag = EmotionTag(protect_cycles_remaining=10)
        assert tag.is_protected

    def test_is_protected_permanent(self):
        tag = EmotionTag(protect_cycles_remaining=-1)
        assert tag.is_protected

    def test_tick_decrements_counter(self):
        tag = EmotionTag(protect_cycles_remaining=5)
        tag.tick()
        assert tag.protect_cycles_remaining == 4

    def test_tick_permanent_stays_minus_one(self):
        tag = EmotionTag(protect_cycles_remaining=-1)
        for _ in range(100):
            tag.tick()
        assert tag.protect_cycles_remaining == -1

    def test_tick_expires_at_zero(self):
        tag = EmotionTag(protect_cycles_remaining=1)
        tag.tick()
        assert tag.protect_cycles_remaining == 0
        assert not tag.is_protected


# ══════════════════════════════════════════════════════════════════════════════
# AgentEmotionState — unit
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentEmotionState:
    def setup_method(self):
        self.state = AgentEmotionState(agent_id=0)

    def test_emot_tag_creates_entry(self):
        self.state.emot_tag(2, 3, valence=0.9, intensity=0.8)
        assert (2, 3) in self.state.tags

    def test_emot_tag_clamps_valence(self):
        tag = self.state.emot_tag(0, 0, valence=5.0, intensity=0.5)
        assert tag.valence <= 1.0

    def test_emot_tag_clamps_intensity(self):
        tag = self.state.emot_tag(0, 0, valence=0.5, intensity=-1.0)
        assert tag.intensity >= 0.0

    def test_decay_protect_cycles(self):
        self.state.emot_tag(1, 1, valence=0.5, intensity=0.5)
        self.state.decay_protect(1, 1, mode=ProtectMode.CYCLES, cycles=50)
        assert self.state.is_protected(1, 1)
        assert self.state.tags[(1, 1)].protect_cycles_remaining == 50

    def test_decay_protect_permanent(self):
        self.state.decay_protect(2, 2, mode=ProtectMode.PERMANENT)
        assert self.state.is_protected(2, 2)
        assert self.state.tags[(2, 2)].protect_cycles_remaining == -1

    def test_decay_protect_scaled(self):
        self.state.decay_protect(3, 3, mode=ProtectMode.SCALED,
                                 cycles=100, intensity=0.5)
        assert self.state.tags[(3, 3)].protect_cycles_remaining == 50

    def test_predict_err_records(self):
        err = self.state.predict_err(5, 5, 3, 3)
        assert len(self.state.prediction_errors) == 1
        assert err > 0

    def test_predict_err_same_position_zero(self):
        err = self.state.predict_err(4, 4, 4, 4)
        assert err == 0.0

    def test_normalise_err(self):
        norm = self.state.normalise_err(10.0, rows=8, cols=8)
        assert 0.0 <= norm <= 1.0

    def test_top_salient_nodes(self):
        self.state.emot_tag(0, 0, valence=1.0, intensity=1.0)
        self.state.emot_tag(1, 1, valence=0.5, intensity=0.5)
        top = self.state.top_salient_nodes(n=1)
        assert top[0][:2] == (0, 0)

    def test_tick_decrements_all(self):
        self.state.decay_protect(0, 0, mode=ProtectMode.CYCLES, cycles=3)
        self.state.tick()
        assert self.state.tags[(0, 0)].protect_cycles_remaining == 2

    def test_snapshot_has_keys(self):
        snap = self.state.snapshot()
        assert "agent_id" in snap
        assert "tagged_nodes" in snap
        assert "mean_salience" in snap


# ══════════════════════════════════════════════════════════════════════════════
# EmotionRegistry — unit
# ══════════════════════════════════════════════════════════════════════════════

class TestEmotionRegistry:
    def test_get_or_create_returns_state(self):
        reg = EmotionRegistry()
        state = reg.get_or_create(42)
        assert state.agent_id == 42

    def test_get_or_create_idempotent(self):
        reg = EmotionRegistry()
        s1 = reg.get_or_create(1)
        s2 = reg.get_or_create(1)
        assert s1 is s2

    def test_remove(self):
        reg = EmotionRegistry()
        reg.get_or_create(7)
        reg.remove(7)
        # new object created after removal
        s = reg.get_or_create(7)
        assert s.agent_id == 7

    def test_tick_all_no_crash(self):
        reg = EmotionRegistry()
        for i in range(5):
            s = reg.get_or_create(i)
            s.decay_protect(0, 0, mode=ProtectMode.CYCLES, cycles=10)
        reg.tick_all()   # should not raise

    def test_snapshot_all_agents(self):
        reg = EmotionRegistry()
        reg.get_or_create(0)
        reg.get_or_create(1)
        snap = reg.snapshot()
        assert 0 in snap and 1 in snap


# ══════════════════════════════════════════════════════════════════════════════
# DECAY_STEP on LiveSomMap — unit
# ══════════════════════════════════════════════════════════════════════════════

class TestDecayStep:
    def test_decay_step_reduces_weights(self):
        som = LiveSomMap(4, 4)
        # Force known weights
        som.nodes[0][0].weights = [1.0] * 8
        old_w = list(som.nodes[0][0].weights)
        som.decay_step(base_rate=0.01)
        new_w = som.nodes[0][0].weights
        assert sum(new_w) < sum(old_w)

    def test_protected_node_not_decayed(self):
        som = LiveSomMap(4, 4)
        som.nodes[2][2].weights = [1.0] * 8
        before = list(som.nodes[2][2].weights)
        som.decay_step(protected_coords=[(2, 2)])
        assert som.nodes[2][2].weights == before

    def test_activated_node_not_decayed(self):
        som = LiveSomMap(4, 4)
        som.nodes[1][1].weights = [1.0] * 8
        som.nodes[1][1].activated_this_pulse = True
        before = list(som.nodes[1][1].weights)
        som.decay_step()
        assert som.nodes[1][1].weights == before

    def test_train_marks_bmu_activated(self):
        som = LiveSomMap(4, 4)
        # Force (0,0) to be BMU
        target = [1.0] * 8
        som.nodes[0][0].weights = list(target)
        for r in range(4):
            for c in range(4):
                if not (r == 0 and c == 0):
                    som.nodes[r][c].weights = [0.0] * 8
        som.train(target, 0, 0)
        assert som.nodes[0][0].activated_this_pulse

    def test_decay_step_returns_count(self):
        som = LiveSomMap(2, 2)
        n = som.decay_step()
        assert isinstance(n, int)
        assert n >= 0


# ══════════════════════════════════════════════════════════════════════════════
# PRUNE_CHECK on LiveSomMap — unit
# ══════════════════════════════════════════════════════════════════════════════

class TestPruneCheck:
    def test_prune_zeroes_weak_node(self):
        som = LiveSomMap(2, 2)
        som.nodes[0][0].weights = [0.001] * 8   # very weak
        pruned = som.prune_check(threshold=0.1)
        assert pruned > 0
        assert som.nodes[0][0].weights == [0.0] * 8

    def test_prune_keeps_strong_node(self):
        som = LiveSomMap(2, 2)
        som.nodes[1][1].weights = [1.0] * 8
        before = list(som.nodes[1][1].weights)
        som.prune_check(threshold=0.01)
        assert som.nodes[1][1].weights == before

    def test_prune_returns_count(self):
        som = LiveSomMap(2, 2)
        for r in range(2):
            for c in range(2):
                som.nodes[r][c].weights = [0.0001] * 8
        n = som.prune_check(threshold=0.1)
        assert n == 4


# ══════════════════════════════════════════════════════════════════════════════
# TwoTierMemory / MemoryManager — unit
# ══════════════════════════════════════════════════════════════════════════════

class TestTwoTierMemory:
    def setup_method(self):
        self.som      = LiveSomMap(4, 4)
        self.em_reg   = EmotionRegistry()
        self.mem      = TwoTierMemory(
            agent_id=0,
            working_som=self.som,
            emotion_reg=self.em_reg,
        )

    def test_consolidate_returns_report(self):
        report = self.mem.consolidate()
        assert isinstance(report, ConsolidationReport)

    def test_consolidate_promotes_nodes(self):
        # Tag a node with high salience
        es = self.em_reg.get_or_create(0)
        es.emot_tag(0, 0, valence=1.0, intensity=1.0)
        report = self.mem.consolidate()
        assert report.promoted > 0

    def test_consolidate_prunes_weak(self):
        # Set all weights to zero → should be pruned
        for r in range(4):
            for c in range(4):
                self.som.nodes[r][c].weights = [0.0001] * 8
        report = self.mem.consolidate()
        assert report.pruned > 0

    def test_recall_after_consolidate(self):
        # Promote node (0,0) to long-term
        es = self.em_reg.get_or_create(0)
        es.emot_tag(0, 0, valence=1.0, intensity=1.0)
        self.som.nodes[0][0].weights = [0.9] * 8
        self.mem.consolidate()
        recalled = self.mem.recall(0, 0)
        assert recalled is not None
        assert len(recalled) == 8

    def test_recall_returns_none_if_not_promoted(self):
        result = self.mem.recall(3, 3)
        assert result is None

    def test_recall_or_working_fallback(self):
        weights = self.mem.recall_or_working(2, 2)
        assert len(weights) == 8

    def test_restore_to_working(self):
        es = self.em_reg.get_or_create(0)
        es.emot_tag(1, 1, valence=1.0, intensity=1.0)
        self.som.nodes[1][1].weights = [0.7] * 8
        self.mem.consolidate()
        # Trash the working SOM node
        self.som.nodes[1][1].weights = [0.0] * 8
        ok = self.mem.restore_to_working(1, 1)
        assert ok
        assert any(w > 0 for w in self.som.nodes[1][1].weights)

    def test_tick_increments_pulse(self):
        self.mem.tick()
        assert self.mem.pulse_count == 1

    def test_snapshot_has_keys(self):
        snap = self.mem.snapshot()
        assert "pulse_count" in snap
        assert "consolidations" in snap
        assert "lt_nodes" in snap


# ══════════════════════════════════════════════════════════════════════════════
# SomScheduler Phase 2.5 integration
# ══════════════════════════════════════════════════════════════════════════════

class TestSchedulerLiveliness:
    def setup_method(self):
        self.sched, self.som, self.reg = make_scheduler()

    def test_decay_step_via_scheduler(self):
        n = self.sched.decay_step(agent_id=0)
        assert isinstance(n, int)

    def test_prune_check_via_scheduler(self):
        n = self.sched.prune_check(threshold=0.0)
        assert n == 0   # all weights > 0 by default

    def test_emot_tag_via_scheduler(self):
        r, c = self.sched.emot_tag(0, valence=0.9, intensity=0.8)
        assert 0 <= r < self.som.rows
        assert 0 <= c < self.som.cols

    def test_decay_protect_via_scheduler(self):
        self.sched.emot_tag(0, valence=1.0, intensity=1.0)
        self.sched.decay_protect(0, cycles=200)
        es = self.sched.emotion_reg.get_or_create(0)
        # at least one protected node
        assert any(t.is_protected for t in es.tags.values())

    def test_decay_protect_permanent(self):
        self.sched.decay_protect(0, permanent=True)
        es = self.sched.emotion_reg.get_or_create(0)
        assert any(t.protect_cycles_remaining == -1 for t in es.tags.values())

    def test_predict_err_range(self):
        err = self.sched.predict_err(0, vec([0.9]*8))
        assert 0.0 <= err <= 1.0

    def test_memory_consolidate_via_scheduler(self):
        report = self.sched.memory_consolidate(0)
        assert isinstance(report, ConsolidationReport)

    def test_pulse_tick_no_crash(self):
        self.sched.pulse_tick()

    def test_protected_node_survives_decay(self):
        # Tag and protect agent 0's current node
        self.sched.emot_tag(0, valence=1.0, intensity=1.0)
        self.sched.decay_protect(0, cycles=500)

        handle = self.reg.get(0)
        r, c   = handle.som_x, handle.som_y
        self.som.nodes[r][c].weights = [1.0] * 8
        before = list(self.som.nodes[r][c].weights)

        # Run decay — protected node should be skipped
        self.sched.decay_step(agent_id=0, rate=0.5)
        assert self.som.nodes[r][c].weights == before


# ══════════════════════════════════════════════════════════════════════════════
# Concurrency stress
# ══════════════════════════════════════════════════════════════════════════════

class TestLivelinessConcurrency:
    def test_concurrent_emot_tag_no_crash(self):
        sched, som, reg = make_scheduler(n_agents=8)
        errors = []

        def worker(aid):
            try:
                for _ in range(20):
                    sched.emot_tag(aid, valence=0.5, intensity=0.5)
                    sched.decay_protect(aid, cycles=10)
                    sched.pulse_tick()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []

    def test_concurrent_decay_and_train(self):
        sched, som, reg = make_scheduler(n_agents=4)
        errors = []

        def decay_worker():
            try:
                for _ in range(30):
                    sched.decay_step(agent_id=0)
                    sched.prune_check(threshold=1e-9)
            except Exception as e:
                errors.append(e)

        def train_worker(aid):
            try:
                for _ in range(30):
                    sched.agent_train(aid, vec([0.5]*8))
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=decay_worker)] +
            [threading.Thread(target=train_worker, args=(i,)) for i in range(4)]
        )
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []

    def test_concurrent_consolidate(self):
        sched, som, reg = make_scheduler(n_agents=4)
        errors = []

        def worker(aid):
            try:
                for _ in range(5):
                    sched.emot_tag(aid, valence=1.0, intensity=1.0)
                    sched.memory_consolidate(aid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []
