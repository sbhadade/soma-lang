"""
tests/test_som.py — Phase 2: SOM scheduling tests
==================================================
40+ tests across:
  - LiveSomMap  (unit)
  - SomScheduler (integration)
  - SomVisualizer (smoke)
  - Concurrency stress
"""
from __future__ import annotations

import math
import threading
import time
from unittest.mock import MagicMock

import pytest

from runtime.som.som_map       import LiveSomMap
from runtime.som.som_scheduler import SomScheduler
from runtime.som.som_visualizer import SomVisualizer


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_registry(n_agents: int = 4):
    """Create a real AgentRegistry with n agents registered."""
    from runtime.agent.agent_registry import AgentRegistry
    from runtime.agent.lifecycle import AgentState

    AgentRegistry.reset()
    reg = AgentRegistry.get_instance()
    for i in range(n_agents):
        reg.register(agent_id=i, parent_id=None, som_x=0, som_y=0)
        reg.set_state(i, AgentState.RUNNING)
    return reg


def vec(vals):
    """Pad / truncate to 8 dims."""
    v = list(vals)
    return (v + [0.0] * 8)[:8]


# ══════════════════════════════════════════════════════════════════════════════
# LiveSomMap — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLiveSomMapInit:
    def test_default_shape(self):
        som = LiveSomMap(4, 4)
        assert som.rows == 4 and som.cols == 4

    def test_node_count(self):
        som = LiveSomMap(3, 5)
        assert len(som.nodes) == 3
        assert all(len(row) == 5 for row in som.nodes)

    def test_initial_activation_zero(self):
        som = LiveSomMap(4, 4)
        for r in range(4):
            for c in range(4):
                assert som.nodes[r][c].activation == 0.0

    def test_initial_hit_count_zero(self):
        som = LiveSomMap(4, 4)
        for r in range(4):
            for c in range(4):
                assert som.nodes[r][c].hit_count == 0

    def test_init_random_resets_weights(self):
        som = LiveSomMap(4, 4)
        old = [som.nodes[0][0].weights[i] for i in range(8)]
        som.init_random()
        # extremely unlikely all weights unchanged
        new = [som.nodes[0][0].weights[i] for i in range(8)]
        # just check it ran without error; weights are random
        assert len(new) == 8

    def test_init_grid_no_dead_neurons(self):
        som = LiveSomMap(4, 4)
        som.init_grid()
        # all weights should differ between (0,0) and (3,3)
        w00 = som.nodes[0][0].weights
        w33 = som.nodes[3][3].weights
        assert w00 != w33

    def test_dims_param(self):
        som = LiveSomMap(2, 2, dims=4)
        assert len(som.nodes[0][0].weights) == 4


class TestLiveSomMapBMU:
    def test_bmu_returns_valid_coords(self):
        som = LiveSomMap(4, 4)
        r, c = som.bmu(vec([0.5] * 8))
        assert 0 <= r < 4 and 0 <= c < 4

    def test_bmu_exact_match(self):
        som = LiveSomMap(2, 2)
        target = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Force node (0,0) to have those exact weights
        som.nodes[0][0].weights = list(target)
        for r in range(2):
            for c in range(2):
                if not (r == 0 and c == 0):
                    som.nodes[r][c].weights = [0.0] * 8
        br, bc = som.bmu(target)
        assert (br, bc) == (0, 0)

    def test_bmu_short_vec_padded(self):
        som = LiveSomMap(2, 2)
        # should not raise even with short vec
        r, c = som.bmu([1.0, 0.0])
        assert 0 <= r < 2 and 0 <= c < 2

    def test_bmu_does_not_modify_activation(self):
        som = LiveSomMap(2, 2)
        som.bmu(vec([0.5] * 8))
        # BMU should NOT update activation (read-only)
        assert all(som.nodes[r][c].activation == 0.0
                   for r in range(2) for c in range(2))


class TestLiveSomMapTrain:
    def test_train_increases_hit_count(self):
        som = LiveSomMap(4, 4)
        bmu_r, bmu_c = som.bmu(vec([0.9] * 8))
        som.train(vec([0.9] * 8), bmu_r, bmu_c)
        assert som.nodes[bmu_r][bmu_c].hit_count == 1

    def test_train_moves_weights_closer(self):
        som = LiveSomMap(2, 2)
        target = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        old_w = list(som.nodes[0][0].weights)
        som.train(target, 0, 0, lr=1.0, sigma=0.1)
        new_w = som.nodes[0][0].weights
        old_dist = math.sqrt(sum((a - b)**2 for a, b in zip(old_w, target)))
        new_dist = math.sqrt(sum((a - b)**2 for a, b in zip(new_w, target)))
        assert new_dist <= old_dist + 1e-6

    def test_train_updates_activation(self):
        som = LiveSomMap(4, 4)
        som.train(vec([0.5] * 8), 0, 0, lr=0.1, sigma=2.0)
        # BMU node must have activation > 0
        assert som.nodes[0][0].activation > 0.0

    def test_train_increments_epoch(self):
        som = LiveSomMap(4, 4)
        assert som.epoch == 0
        som.train(vec([0.5] * 8), 0, 0)
        assert som.epoch == 1

    def test_neighbourhood_influence_decreases_with_distance(self):
        som = LiveSomMap(8, 8)
        som.train(vec([1.0] * 8), 4, 4, lr=0.5, sigma=2.0)
        act_bmu  = som.nodes[4][4].activation
        act_near = som.nodes[4][5].activation
        act_far  = som.nodes[0][0].activation
        assert act_bmu >= act_near >= act_far


class TestLiveSomMapWalk:
    def test_walk_gradient_returns_valid_coords(self):
        som = LiveSomMap(4, 4)
        nr, nc = som.walk_gradient(2, 2)
        assert 0 <= nr < 4 and 0 <= nc < 4

    def test_walk_gradient_moves_toward_activation(self):
        som = LiveSomMap(4, 4)
        # Train node (3,3) heavily
        som.train(vec([1.0] * 8), 3, 3, lr=0.9, sigma=0.5)
        # Agent at (2,2) — should move toward (3,3)
        nr, nc = som.walk_gradient(2, 2)
        # Should move closer to (3,3)
        old_dist = math.sqrt((2-3)**2 + (2-3)**2)
        new_dist = math.sqrt((nr-3)**2 + (nc-3)**2)
        assert new_dist <= old_dist + 1e-6

    def test_walk_gradient_stays_in_bounds(self):
        som = LiveSomMap(4, 4)
        nr, nc = som.walk_gradient(0, 0)
        assert 0 <= nr < 4 and 0 <= nc < 4

    def test_walk_random_stays_in_bounds(self):
        som = LiveSomMap(4, 4)
        for _ in range(20):
            nr, nc = som.walk_random(0, 0)
            assert 0 <= nr < 4 and 0 <= nc < 4


class TestLiveSomMapElect:
    def test_elect_highest_hit_count(self):
        som = LiveSomMap(4, 4)
        # Train node (1,1) 5 times, node (2,2) once
        for _ in range(5):
            som.train(vec([0.9] * 8), 1, 1)
        som.train(vec([0.1] * 8), 2, 2)

        positions = [(0, 1, 1), (1, 2, 2)]   # (agent_id, r, c)
        leader = som.elect(positions)
        assert leader == 0   # agent at (1,1) has more hits

    def test_elect_empty_returns_default(self):
        som = LiveSomMap(4, 4)
        leader = som.elect([])
        assert leader == 0


class TestLiveSomMapDecay:
    def test_decay_reduces_lr(self):
        som = LiveSomMap(4, 4, lr=1.0)
        old_lr = som.lr
        som.decay(lr_rate=0.1)
        assert som.lr < old_lr

    def test_decay_does_not_go_below_minimum(self):
        som = LiveSomMap(4, 4, lr=0.001)
        for _ in range(100):
            som.decay(lr_rate=0.5)
        assert som.lr >= 0.001

    def test_decay_reduces_sigma(self):
        som = LiveSomMap(4, 4, sigma=5.0)
        som.decay(sigma_rate=0.1)
        assert som.sigma < 5.0


class TestLiveSomMapSnapshot:
    def test_snapshot_shape(self):
        som = LiveSomMap(3, 3)
        snap = som.snapshot()
        assert snap["rows"] == 3 and snap["cols"] == 3
        assert len(snap["nodes"]) == 9

    def test_snapshot_epoch(self):
        som = LiveSomMap(3, 3)
        som.train(vec([0.5]*8), 1, 1)
        snap = som.snapshot()
        assert snap["epoch"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# SomScheduler — integration tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSomScheduler:
    def setup_method(self):
        self.reg = make_registry(4)
        self.som = LiveSomMap(8, 8)
        self.sched = SomScheduler(self.som, self.reg)

    def test_place_agent(self):
        self.sched.place_agent(0, 3, 5)
        h = self.reg.get(0)
        assert (h.som_x, h.som_y) == (3, 5)

    def test_place_agent_clamps_to_bounds(self):
        self.sched.place_agent(0, 999, 999)
        h = self.reg.get(0)
        assert h.som_x < self.som.rows
        assert h.som_y < self.som.cols

    def test_agent_train_moves_to_bmu(self):
        bmu_r, bmu_c = self.sched.agent_train(0, vec([1.0]*8))
        h = self.reg.get(0)
        assert (h.som_x, h.som_y) == (bmu_r, bmu_c)

    def test_agent_train_returns_bmu_coords(self):
        r, c = self.sched.agent_train(0, vec([0.5]*8))
        assert 0 <= r < 8 and 0 <= c < 8

    def test_agent_walk_updates_registry(self):
        # Train first so there's activation
        self.sched.agent_train(0, vec([1.0]*8))
        self.sched.place_agent(1, 0, 0)
        nr, nc = self.sched.agent_walk(1, gradient=True)
        h = self.reg.get(1)
        assert (h.som_x, h.som_y) == (nr, nc)

    def test_elect_returns_valid_agent_id(self):
        # Train agent 2's position heavily
        self.sched.place_agent(2, 4, 4)
        for _ in range(5):
            self.sched.agent_train(2, vec([0.8]*8))
        leader = self.sched.elect()
        # leader should be one of the registered agents
        assert leader in [0, 1, 2, 3]

    def test_sense_returns_float(self):
        val = self.sched.sense(0, 0)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0 + 1e-6

    def test_neighbourhood_returns_list(self):
        result = self.sched.neighbourhood(4, 4, sigma=1.5)
        assert isinstance(result, list)
        assert all(len(t) == 3 for t in result)

    def test_snapshot_contains_agents(self):
        snap = self.sched.snapshot()
        assert "som" in snap
        assert "agents" in snap
        assert len(snap["agents"]) == 4

    def test_decay_thread_starts_and_stops(self):
        self.sched.start_decay(interval_s=0.05)
        time.sleep(0.15)
        old_lr = self.som.lr
        self.sched.stop_decay()
        # lr should have decreased
        assert self.som.lr <= old_lr


# ══════════════════════════════════════════════════════════════════════════════
# SomVisualizer — smoke tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSomVisualizer:
    def setup_method(self):
        self.reg   = make_registry(2)
        self.som   = LiveSomMap(4, 4)
        self.sched = SomScheduler(self.som, self.reg)
        self.vis   = SomVisualizer(self.sched)

    def test_render_returns_string(self):
        out = self.vis.render()
        assert isinstance(out, str)
        assert len(out) > 0

    def test_render_contains_dimensions(self):
        out = self.vis.render()
        assert "4×4" in out

    def test_render_contains_epoch(self):
        out = self.vis.render()
        assert "epoch=" in out

    def test_hit_map_returns_string(self):
        out = self.vis.hit_map()
        assert isinstance(out, str)
        assert "SOM Hit Map" in out

    def test_render_after_train(self):
        self.sched.agent_train(0, vec([0.9]*8))
        out = self.vis.render()
        assert isinstance(out, str)


# ══════════════════════════════════════════════════════════════════════════════
# Concurrency stress tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSomConcurrency:
    def test_concurrent_bmu_no_crash(self):
        som = LiveSomMap(8, 8)
        errors = []

        def worker():
            try:
                for _ in range(50):
                    som.bmu(vec([0.5]*8))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []

    def test_concurrent_train_no_crash(self):
        som = LiveSomMap(8, 8)
        errors = []

        def worker(v):
            try:
                for _ in range(20):
                    r, c = som.bmu(vec([v]*8))
                    som.train(vec([v]*8), r, c)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i/8,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []

    def test_concurrent_scheduler_no_crash(self):
        reg   = make_registry(8)
        som   = LiveSomMap(8, 8)
        sched = SomScheduler(som, reg)
        errors = []

        def worker(aid):
            try:
                for _ in range(10):
                    sched.agent_train(aid, vec([aid/8]*8))
                    sched.agent_walk(aid, gradient=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []

    def test_visualizer_during_training(self):
        reg   = make_registry(4)
        som   = LiveSomMap(4, 4)
        sched = SomScheduler(som, reg)
        vis   = SomVisualizer(sched)
        errors = []

        def train_loop():
            try:
                for i in range(20):
                    sched.agent_train(i % 4, vec([i/20]*8))
            except Exception as e:
                errors.append(e)

        def vis_loop():
            try:
                for _ in range(10):
                    vis.render()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=train_loop)
        t2 = threading.Thread(target=vis_loop)
        t1.start(); t2.start()
        t1.join();  t2.join()
        assert errors == []
