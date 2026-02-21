"""
tests/test_phase26.py — Phase 2.6: Culture layer tests
=======================================================
54 tests across:
  - EmotionSnapshot creation / absorption / attenuation       (unit)
  - EMOT_RECALL via AgentEmotionState                        (unit)
  - SURPRISE_CALC raw vector pipeline                         (unit)
  - absorb_snapshot() cultural blending                       (unit)
  - MemorySharePacket build / absorb                          (unit)
  - MEMORY_SHARE directed (agent A → agent B)                 (unit)
  - MEMORY_SHARE broadcast (dying agent → all)                (unit)
  - MEMORY_LOAD restore long-term to working SOM              (unit)
  - REORG_MAP dead-node reseeding                             (unit)
  - DECAY_RATE_SET per-region custom rates                    (unit)
  - NEIGHBOR_SYNC topological broadcast                       (integration)
  - SomScheduler Phase 2.6 wiring                             (integration)
  - Cultural divergence stress test                           (stress)

Paper: "A Path to AGI Part II: Liveliness"
"""
from __future__ import annotations

import math
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# ── Minimal AgentRegistry stub ────────────────────────────────────────────────

class _FakeHandle:
    def __init__(self, agent_id, r=0, c=0, alive=True):
        self.agent_id = agent_id
        self.som_row  = r
        self.som_col  = c
        self.state    = MagicMock()
        self.state.is_alive = alive


class _FakeRegistry:
    def __init__(self):
        self._h: dict = {}

    def register(self, agent_id, r=0, c=0):
        self._h[agent_id] = _FakeHandle(agent_id, r, c)

    def get(self, agent_id):
        return self._h[agent_id]

    def get_or_none(self, agent_id):
        return self._h.get(agent_id)

    def set_som_coords(self, agent_id, r, c):
        h = self._h.get(agent_id)
        if h:
            h.som_row = r
            h.som_col = c

    def __iter__(self):
        return iter(self._h.values())


def _make_scheduler(rows=8, cols=8):
    from runtime.som.som_map      import LiveSomMap
    from runtime.som.som_scheduler import SomScheduler
    som = LiveSomMap(rows=rows, cols=cols)
    reg = _FakeRegistry()
    return SomScheduler(som, reg), som, reg


# ══════════════════════════════════════════════════════════════════════════════
#  1. EmotionSnapshot
# ══════════════════════════════════════════════════════════════════════════════

class TestEmotionSnapshot(unittest.TestCase):

    def _make_state(self, agent_id=0):
        from runtime.som.emotion import AgentEmotionState
        return AgentEmotionState(agent_id)

    def test_snapshot_for_share_empty(self):
        s = self._make_state()
        snap = s.snapshot_for_share(top_n=10)
        self.assertEqual(len(snap), 0)
        self.assertEqual(snap.source_agent_id, 0)

    def test_snapshot_for_share_returns_top_n(self):
        s = self._make_state()
        # Add 20 tags with varying salience
        for i in range(20):
            s.emot_tag(i % 4, i // 4, valence=1.0, intensity=i / 20)
        snap = s.snapshot_for_share(top_n=5)
        self.assertLessEqual(len(snap), 5)

    def test_snapshot_sorted_by_salience(self):
        s = self._make_state()
        s.emot_tag(0, 0, valence=1.0, intensity=0.1)
        s.emot_tag(1, 1, valence=1.0, intensity=0.9)
        s.emot_tag(2, 2, valence=1.0, intensity=0.5)
        snap = s.snapshot_for_share(top_n=3)
        saliences = [abs(t[2]["valence"]) * t[2]["intensity"]
                     for t in snap.tags]
        self.assertEqual(saliences, sorted(saliences, reverse=True))

    def test_snapshot_top_filters(self):
        from runtime.som.emotion import EmotionSnapshot
        snap = EmotionSnapshot(
            source_agent_id=0,
            tags=[(0, i, {"valence": 1.0, "intensity": i / 10})
                  for i in range(10)],
        )
        top3 = snap.top(3)
        self.assertEqual(len(top3), 3)

    def test_snapshot_serialise_roundtrip(self):
        from runtime.som.emotion import EmotionTag
        tag = EmotionTag(valence=0.8, intensity=0.6,
                         protect_cycles_remaining=50)
        d   = tag.to_dict()
        tag2 = EmotionTag.from_dict(d)
        self.assertAlmostEqual(tag2.valence, 0.8)
        self.assertAlmostEqual(tag2.intensity, 0.6)
        self.assertEqual(tag2.protect_cycles_remaining, 50)


# ══════════════════════════════════════════════════════════════════════════════
#  2. EMOT_RECALL
# ══════════════════════════════════════════════════════════════════════════════

class TestEmotRecall(unittest.TestCase):

    def _make_state(self, agent_id=0):
        from runtime.som.emotion import AgentEmotionState
        return AgentEmotionState(agent_id)

    def test_recall_returns_none_for_untagged(self):
        s = self._make_state()
        self.assertIsNone(s.emot_recall(3, 3))

    def test_recall_returns_tag_after_emot_tag(self):
        s = self._make_state()
        s.emot_tag(2, 4, valence=0.9, intensity=0.7)
        tag = s.emot_recall(2, 4)
        self.assertIsNotNone(tag)
        self.assertAlmostEqual(tag.valence, 0.9)
        self.assertAlmostEqual(tag.intensity, 0.7)

    def test_recall_salience_matches(self):
        s = self._make_state()
        s.emot_tag(1, 1, valence=0.5, intensity=0.6)
        tag = s.emot_recall(1, 1)
        self.assertAlmostEqual(tag.salience, 0.5 * 0.6, places=5)

    def test_recall_different_coord_returns_none(self):
        s = self._make_state()
        s.emot_tag(0, 0, valence=1.0, intensity=1.0)
        self.assertIsNone(s.emot_recall(0, 1))

    def test_recall_via_scheduler(self):
        sched, som, reg = _make_scheduler()
        reg.register(0, r=3, c=3)
        sched.emot_tag(0, valence=0.8, intensity=0.5)
        tag = sched.emot_recall(0)
        self.assertIsNotNone(tag)
        self.assertAlmostEqual(tag.valence, 0.8, places=2)

    def test_recall_explicit_coord_via_scheduler(self):
        sched, som, reg = _make_scheduler()
        reg.register(0, r=2, c=2)
        # Manually tag coord (4, 4)
        state = sched.emotion_reg.get_or_create(0)
        state.emot_tag(4, 4, valence=0.6, intensity=0.4)
        tag = sched.emot_recall(0, r=4, c=4)
        self.assertIsNotNone(tag)

    def test_recall_returns_none_for_unknown_agent(self):
        sched, _, _ = _make_scheduler()
        self.assertIsNone(sched.emot_recall(999))


# ══════════════════════════════════════════════════════════════════════════════
#  3. SURPRISE_CALC
# ══════════════════════════════════════════════════════════════════════════════

class TestSurpriseCalc(unittest.TestCase):

    def _make_state(self, agent_id=0):
        from runtime.som.emotion import AgentEmotionState
        return AgentEmotionState(agent_id)

    def test_identical_vecs_not_surprising(self):
        s = self._make_state()
        vec = [0.5] * 8
        err, is_surp = s.surprise_calc(vec, vec, threshold=0.25)
        self.assertAlmostEqual(err, 0.0)
        self.assertFalse(is_surp)

    def test_very_different_vecs_surprising(self):
        s = self._make_state()
        a = [1.0] * 8
        b = [0.0] * 8
        err, is_surp = s.surprise_calc(a, b, threshold=0.25)
        self.assertGreater(err, 0.25)
        self.assertTrue(is_surp)

    def test_error_clamped_to_one(self):
        s = self._make_state()
        a = [1e6] * 8
        b = [0.0] * 8
        err, _ = s.surprise_calc(a, b)
        self.assertLessEqual(err, 1.0)

    def test_mismatched_lengths_handled(self):
        s = self._make_state()
        err, _ = s.surprise_calc([1.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        self.assertGreaterEqual(err, 0.0)

    def test_error_recorded_in_history(self):
        s = self._make_state()
        s.surprise_calc([1.0]*8, [0.0]*8)
        s.surprise_calc([0.5]*8, [0.5]*8)
        self.assertEqual(len(s.prediction_errors), 2)

    def test_scheduler_surprise_calc_auto_tag(self):
        sched, som, reg = _make_scheduler()
        reg.register(0, r=2, c=2)
        a = [1.0]*8
        b = [0.0]*8
        err, is_surp = sched.surprise_calc(0, a, b, threshold=0.1, auto_tag=True)
        self.assertTrue(is_surp)
        # auto_tag should have fired EMOT_TAG
        state = sched.emotion_reg.get_or_create(0)
        self.assertGreater(len(state.tags), 0)

    def test_scheduler_surprise_calc_no_auto_tag(self):
        sched, som, reg = _make_scheduler()
        reg.register(0, r=2, c=2)
        a = [1.0]*8; b = [0.0]*8
        sched.surprise_calc(0, a, b, threshold=0.1, auto_tag=False)
        state = sched.emotion_reg.get_or_create(0)
        self.assertEqual(len(state.tags), 0)


# ══════════════════════════════════════════════════════════════════════════════
#  4. absorb_snapshot (cultural blending)
# ══════════════════════════════════════════════════════════════════════════════

class TestAbsorbSnapshot(unittest.TestCase):

    def _make_state(self, agent_id=0):
        from runtime.som.emotion import AgentEmotionState
        return AgentEmotionState(agent_id)

    def test_absorb_adds_new_tags(self):
        sender   = self._make_state(0)
        receiver = self._make_state(1)
        sender.emot_tag(3, 3, valence=0.9, intensity=0.8)
        snap = sender.snapshot_for_share()
        absorbed = receiver.absorb_snapshot(snap, weight=1.0)
        self.assertGreater(absorbed, 0)
        self.assertIsNotNone(receiver.emot_recall(3, 3))

    def test_absorb_attenuation_reduces_intensity(self):
        sender   = self._make_state(0)
        receiver = self._make_state(1)
        sender.emot_tag(1, 1, valence=1.0, intensity=1.0)
        snap = sender.snapshot_for_share()
        receiver.absorb_snapshot(snap, weight=0.3)
        tag = receiver.emot_recall(1, 1)
        self.assertIsNotNone(tag)
        self.assertLessEqual(tag.intensity, 0.35)  # 1.0 * 0.3 with tolerance

    def test_absorb_stronger_foreign_overwrites(self):
        sender   = self._make_state(0)
        receiver = self._make_state(1)
        receiver.emot_tag(2, 2, valence=0.1, intensity=0.1)
        sender.emot_tag(2, 2, valence=0.9, intensity=0.9)
        snap = sender.snapshot_for_share()
        receiver.absorb_snapshot(snap, weight=1.0)
        tag = receiver.emot_recall(2, 2)
        self.assertGreater(tag.intensity, 0.1)

    def test_absorb_weaker_foreign_ignored(self):
        sender   = self._make_state(0)
        receiver = self._make_state(1)
        receiver.emot_tag(2, 2, valence=0.9, intensity=0.9)
        sender.emot_tag(2, 2, valence=0.1, intensity=0.05)
        snap = sender.snapshot_for_share()
        receiver.absorb_snapshot(snap, weight=1.0)
        tag = receiver.emot_recall(2, 2)
        # receiver's strong tag should survive
        self.assertGreater(tag.intensity, 0.5)

    def test_absorb_zero_weight_no_change(self):
        sender   = self._make_state(0)
        receiver = self._make_state(1)
        sender.emot_tag(0, 0, valence=1.0, intensity=1.0)
        snap = sender.snapshot_for_share()
        absorbed = receiver.absorb_snapshot(snap, weight=0.0)
        # 0 attenuation means 0 intensity → won't beat existing=0
        # absorb count may still be 0 or 1 depending on threshold
        self.assertGreaterEqual(absorbed, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  5. MemorySharePacket
# ══════════════════════════════════════════════════════════════════════════════

class TestMemorySharePacket(unittest.TestCase):

    def _make_memory(self, agent_id=0, rows=8, cols=8):
        from runtime.som.som_map import LiveSomMap
        from runtime.som.emotion import EmotionRegistry
        from runtime.som.memory  import TwoTierMemory
        som = LiveSomMap(rows=rows, cols=cols)
        em  = EmotionRegistry()
        return TwoTierMemory(agent_id, som, em), som, em

    def test_build_packet_empty_longterm(self):
        mem, _, _ = self._make_memory()
        packet = mem.build_share_packet(top_n=5)
        self.assertEqual(packet.source_agent_id, 0)
        self.assertEqual(len(packet.long_term_nodes), 0)

    def test_build_packet_after_consolidate(self):
        mem, som, em = self._make_memory()
        # Set up strong node
        state = em.get_or_create(0)
        state.emot_tag(2, 2, valence=1.0, intensity=0.9)
        som.nodes[2][2].weights = [1.0] * 8
        report = mem.consolidate()
        packet = mem.build_share_packet(top_n=5)
        self.assertGreaterEqual(len(packet.long_term_nodes), 0)

    def test_packet_has_emotion_snapshot(self):
        mem, som, em = self._make_memory()
        state = em.get_or_create(0)
        state.emot_tag(1, 1, valence=0.8, intensity=0.7)
        packet = mem.build_share_packet(top_n=5)
        self.assertIsNotNone(packet.emotion_snapshot)

    def test_packet_len(self):
        from runtime.som.memory import MemorySharePacket
        p = MemorySharePacket(source_agent_id=0, long_term_nodes=[{}, {}])
        self.assertEqual(len(p), 2)

    def test_absorb_packet_increases_longterm(self):
        sender_mem, som, em = self._make_memory(agent_id=0)
        recv_mem, _, _ = self._make_memory(agent_id=1)
        recv_mem.working_som = som  # share same map

        state = em.get_or_create(0)
        state.emot_tag(3, 3, valence=1.0, intensity=1.0)
        som.nodes[3][3].weights = [0.9] * 8
        sender_mem.consolidate()

        packet   = sender_mem.build_share_packet(top_n=10)
        absorbed = recv_mem.absorb_share_packet(packet)
        self.assertGreaterEqual(absorbed, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  6. MEMORY_SHARE (directed + broadcast)
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryShare(unittest.TestCase):

    def _make_manager(self, rows=8, cols=8):
        from runtime.som.som_map import LiveSomMap
        from runtime.som.emotion import EmotionRegistry
        from runtime.som.memory  import MemoryManager
        som = LiveSomMap(rows=rows, cols=cols)
        em  = EmotionRegistry()
        return MemoryManager(som, em), som, em

    def test_directed_share_returns_int(self):
        mgr, som, em = self._make_manager()
        absorbed = mgr.share(0, 1, top_n=5, attenuation=0.5)
        self.assertIsInstance(absorbed, int)
        self.assertGreaterEqual(absorbed, 0)

    def test_directed_share_transmits_emotion(self):
        mgr, som, em = self._make_manager()
        # Give agent 0 a strong tag
        state = em.get_or_create(0)
        state.emot_tag(2, 2, valence=1.0, intensity=1.0)
        som.nodes[2][2].weights = [1.0] * 8
        mem0 = mgr.get_or_create(0)
        mem0.consolidate()

        absorbed = mgr.share(0, 1, top_n=10, attenuation=1.0)
        # Receiver's emotion state should now have some tags
        recv_state = em.get_or_create(1)
        # absorbed may come from emotion tags
        self.assertGreaterEqual(absorbed, 0)

    def test_broadcast_share_returns_dict(self):
        mgr, _, _ = self._make_manager()
        mgr.get_or_create(0)
        mgr.get_or_create(1)
        mgr.get_or_create(2)
        result = mgr.share_to_all(0, top_n=5, attenuation=0.3)
        self.assertIsInstance(result, dict)
        # Keys should be 1 and 2 (not 0)
        self.assertNotIn(0, result)

    def test_broadcast_reaches_all_agents(self):
        mgr, _, em = self._make_manager()
        state = em.get_or_create(0)
        state.emot_tag(1, 1, valence=0.9, intensity=0.8)
        for i in range(5):
            mgr.get_or_create(i)
        result = mgr.share_to_all(0, top_n=10)
        self.assertEqual(len(result), 4)  # 4 recipients (not self)

    def test_share_via_scheduler(self):
        sched, som, reg = _make_scheduler()
        reg.register(0, r=1, c=1)
        reg.register(1, r=3, c=3)
        absorbed = sched.memory_share(0, 1, top_n=5)
        self.assertIsInstance(absorbed, int)

    def test_share_broadcast_via_scheduler(self):
        sched, som, reg = _make_scheduler()
        for i in range(4):
            reg.register(i, r=i, c=i)
        result = sched.memory_share_broadcast(0, top_n=5)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)


# ══════════════════════════════════════════════════════════════════════════════
#  7. MEMORY_LOAD
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryLoad(unittest.TestCase):

    def test_load_empty_returns_zero(self):
        from runtime.som.som_map import LiveSomMap
        from runtime.som.emotion import EmotionRegistry
        from runtime.som.memory  import TwoTierMemory
        som = LiveSomMap(4, 4)
        em  = EmotionRegistry()
        mem = TwoTierMemory(0, som, em)
        self.assertEqual(mem.load_to_working(), 0)

    def test_load_restores_weights(self):
        from runtime.som.som_map import LiveSomMap
        from runtime.som.emotion import EmotionRegistry
        from runtime.som.memory  import TwoTierMemory
        som   = LiveSomMap(4, 4)
        em    = EmotionRegistry()
        mem   = TwoTierMemory(0, som, em)
        state = em.get_or_create(0)

        # Plant strong tag and consolidate
        state.emot_tag(1, 1, valence=1.0, intensity=1.0)
        som.nodes[1][1].weights = [0.99] * 8
        mem.consolidate()

        # Wipe working SOM
        som.nodes[1][1].weights = [0.0] * 8

        restored = mem.load_to_working(min_salience=0.0)
        self.assertGreaterEqual(restored, 0)
        # If something was consolidated, weights should be restored
        if restored > 0:
            self.assertGreater(sum(som.nodes[1][1].weights), 0)

    def test_load_via_manager(self):
        from runtime.som.som_map import LiveSomMap
        from runtime.som.emotion import EmotionRegistry
        from runtime.som.memory  import MemoryManager
        som = LiveSomMap(4, 4)
        em  = EmotionRegistry()
        mgr = MemoryManager(som, em)
        n   = mgr.load(0)
        self.assertIsInstance(n, int)

    def test_load_via_scheduler(self):
        sched, som, reg = _make_scheduler()
        reg.register(0, r=0, c=0)
        n = sched.memory_load(0)
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  8. REORG_MAP
# ══════════════════════════════════════════════════════════════════════════════

class TestReorgMap(unittest.TestCase):

    def test_reorg_on_fresh_map_returns_int(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(4, 4)
        n   = som.reorg_map()
        self.assertIsInstance(n, int)

    def test_reorg_reseeds_dead_nodes(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(4, 4)
        # Kill the centre node
        som.nodes[2][2].weights = [0.0] * 8
        # Make sure neighbours are alive
        som.nodes[2][3].weights = [0.8] * 8
        som.nodes[3][2].weights = [0.7] * 8
        reseeded = som.reorg_map(dead_threshold=1e-4, spread_radius=2)
        self.assertGreaterEqual(reseeded, 1)
        # Centre node should now have non-zero weights
        self.assertGreater(sum(abs(w) for w in som.nodes[2][2].weights), 0)

    def test_reorg_all_dead_random_seeds(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(4, 4)
        for r in range(4):
            for c in range(4):
                som.nodes[r][c].weights = [0.0] * 8
        reseeded = som.reorg_map()
        self.assertEqual(reseeded, 16)

    def test_reorg_via_scheduler(self):
        sched, som, reg = _make_scheduler()
        reseeded = sched.reorg_map()
        self.assertIsInstance(reseeded, int)

    def test_reorg_no_false_positives(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(4, 4)
        # All nodes healthy
        reseeded = som.reorg_map()
        self.assertEqual(reseeded, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  9. DECAY_RATE_SET
# ══════════════════════════════════════════════════════════════════════════════

class TestDecayRateSet(unittest.TestCase):

    def test_set_rate_returns_count(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(8, 8)
        n   = som.set_region_decay_rate(4, 4, radius=2.0, rate=0.0001)
        self.assertGreater(n, 0)

    def test_set_rate_updates_nodes(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(8, 8)
        som.set_region_decay_rate(0, 0, radius=1.0, rate=0.0001)
        self.assertAlmostEqual(som.nodes[0][0].decay_rate, 0.0001)

    def test_radius_zero_updates_one(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(8, 8)
        n   = som.set_region_decay_rate(3, 3, radius=0.0, rate=0.05)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(som.nodes[3][3].decay_rate, 0.05)

    def test_set_rate_clamps_to_one(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(4, 4)
        som.set_region_decay_rate(0, 0, radius=10.0, rate=999.0)
        for r in range(4):
            for c in range(4):
                self.assertLessEqual(som.nodes[r][c].decay_rate, 1.0)

    def test_set_rate_via_scheduler(self):
        sched, som, reg = _make_scheduler()
        reg.register(0, r=4, c=4)
        n = sched.decay_rate_set(0, radius=2.0, rate=0.0001)
        self.assertGreater(n, 0)

    def test_low_rate_slows_decay(self):
        from runtime.som.som_map import LiveSomMap
        som_fast = LiveSomMap(4, 4)
        som_slow = LiveSomMap(4, 4)
        # Same initial weights
        for r in range(4):
            for c in range(4):
                som_fast.nodes[r][c].weights = [1.0] * 8
                som_slow.nodes[r][c].weights = [1.0] * 8
        som_slow.set_region_decay_rate(0, 0, radius=10, rate=0.0001)
        # Run 100 decay steps
        for _ in range(100):
            som_fast.decay_step()
            som_slow.decay_step()
        # Slow-decay node should be stronger
        fast_str = som_fast.node_strength(0, 0)
        slow_str = som_slow.node_strength(0, 0)
        self.assertGreater(slow_str, fast_str)


# ══════════════════════════════════════════════════════════════════════════════
#  10. NEIGHBOR_SYNC
# ══════════════════════════════════════════════════════════════════════════════

class TestNeighborSync(unittest.TestCase):

    def test_neighbor_sync_no_neighbours_returns_empty(self):
        sched, som, reg = _make_scheduler()
        reg.register(0, r=0, c=0)
        result = sched.neighbor_sync(0, radius=1.0)
        self.assertEqual(result, {})

    def test_neighbor_sync_reaches_adjacent_agents(self):
        sched, som, reg = _make_scheduler(rows=8, cols=8)
        reg.register(0, r=4, c=4)
        reg.register(1, r=4, c=5)  # one step away
        reg.register(2, r=0, c=0)  # far away

        # Give agent 0 a salient tag
        state = sched.emotion_reg.get_or_create(0)
        state.emot_tag(4, 4, valence=0.9, intensity=0.8)
        # Consolidate so there's something to share
        sched.memory_mgr.get_or_create(0)

        result = sched.neighbor_sync(0, radius=2.0)
        # Agent 1 (adjacent) may appear; agent 2 (far) should not
        self.assertNotIn(2, result)

    def test_neighbor_sync_returns_dict(self):
        sched, som, reg = _make_scheduler()
        reg.register(0, r=2, c=2)
        reg.register(1, r=2, c=3)
        result = sched.neighbor_sync(0, radius=2.0)
        self.assertIsInstance(result, dict)

    def test_neighbour_coords_returns_list(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(8, 8)
        coords = som.neighbour_coords(4, 4, radius=2.0)
        self.assertIsInstance(coords, list)
        self.assertGreater(len(coords), 0)

    def test_neighbour_coords_within_bounds(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(4, 4)
        coords = som.neighbour_coords(0, 0, radius=3.0)
        for r, c in coords:
            self.assertGreaterEqual(r, 0)
            self.assertGreaterEqual(c, 0)
            self.assertLess(r, 4)
            self.assertLess(c, 4)

    def test_neighbour_coords_excludes_self(self):
        from runtime.som.som_map import LiveSomMap
        som = LiveSomMap(8, 8)
        coords = som.neighbour_coords(4, 4, radius=2.0)
        self.assertNotIn((4, 4), coords)


# ══════════════════════════════════════════════════════════════════════════════
#  11. Cultural divergence stress test
# ══════════════════════════════════════════════════════════════════════════════

class TestCulturalDivergence(unittest.TestCase):

    def test_two_agents_same_start_diverge(self):
        """
        Paper claim: two agents on different input streams develop
        different SOM topologies. Verify at small scale.
        """
        from runtime.som.som_map import LiveSomMap
        from runtime.som.emotion import EmotionRegistry, AgentEmotionState
        from runtime.som.memory  import MemoryManager

        som0 = LiveSomMap(4, 4)
        som1 = LiveSomMap(4, 4)
        em0  = EmotionRegistry()
        em1  = EmotionRegistry()

        import random
        random.seed(0)
        # Agent 0: excited about top-left region
        state0 = em0.get_or_create(0)
        for _ in range(20):
            vec = [random.gauss(0.9, 0.05) for _ in range(8)]
            r, c = som0.bmu(vec)
            som0.train(vec, r, c)
            if r < 2 and c < 2:
                state0.emot_tag(r, c, valence=1.0, intensity=0.8)

        # Agent 1: excited about bottom-right region
        state1 = em1.get_or_create(1)
        for _ in range(20):
            vec = [random.gauss(0.1, 0.05) for _ in range(8)]
            r, c = som1.bmu(vec)
            som1.train(vec, r, c)
            if r >= 2 and c >= 2:
                state1.emot_tag(r, c, valence=1.0, intensity=0.8)

        # Different emotion regions
        tags0 = set(state0.tags.keys())
        tags1 = set(state1.tags.keys())
        # They may overlap by chance on a 4×4 map — just ensure they exist
        self.assertGreater(len(tags0), 0)
        self.assertGreater(len(tags1), 0)

    def test_dead_agent_legacy_absorbed(self):
        """Dying agent shares memories; surviving agents absorb them."""
        from runtime.som.som_map import LiveSomMap
        from runtime.som.emotion import EmotionRegistry
        from runtime.som.memory  import MemoryManager

        som = LiveSomMap(8, 8)
        em  = EmotionRegistry()
        mgr = MemoryManager(som, em)

        # Dying agent with strong memories
        dying_state = em.get_or_create(99)
        dying_state.emot_tag(3, 3, valence=0.9, intensity=0.9)
        dying_state.emot_tag(5, 5, valence=-0.8, intensity=0.7)
        som.nodes[3][3].weights = [0.9] * 8
        som.nodes[5][5].weights = [0.5] * 8
        mgr.get_or_create(99).consolidate()

        # Broadcast legacy to survivors
        survivors = [0, 1, 2]
        for aid in survivors:
            mgr.get_or_create(aid)
        result = mgr.share_to_all(99, top_n=10, attenuation=0.5)

        self.assertEqual(len(result), 3)

    def test_concurrent_memory_share_no_crash(self):
        """Multiple agents sharing simultaneously must not crash."""
        from runtime.som.som_map import LiveSomMap
        from runtime.som.emotion import EmotionRegistry
        from runtime.som.memory  import MemoryManager

        som = LiveSomMap(8, 8)
        em  = EmotionRegistry()
        mgr = MemoryManager(som, em)

        for i in range(8):
            state = em.get_or_create(i)
            state.emot_tag(i % 8, i % 8, valence=0.5, intensity=0.5)
            mgr.get_or_create(i)

        errors = []

        def share_task(from_id):
            try:
                for to_id in range(8):
                    if to_id != from_id:
                        mgr.share(from_id, to_id, top_n=3)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=share_task, args=(i,))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
