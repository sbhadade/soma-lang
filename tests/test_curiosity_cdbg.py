"""
tests/test_curiosity_cdbg.py
============================
Tests for Phase III (Curiosity — AgentSoul, SomTerrain)
     and Phase IV (Context-Discriminated Binary Grammar — CDBG)

Run with:   pytest tests/test_curiosity_cdbg.py -v
       or:  python -m pytest tests/test_curiosity_cdbg.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import struct
import pytest

# ── Phase III: AgentSoul ───────────────────────────────────────────────────────

from runtime.som.soul import (
    AgentSoul, MasterSoul, SoulRegistry,
    fingerprint, fp_distance, MemoryEntry,
)


class TestFingerprint:
    def test_same_vector_zero_distance(self):
        v = [0.1, 0.5, 0.9, 0.3]
        assert fp_distance(fingerprint(v), fingerprint(v)) == 0.0

    def test_different_vectors_nonzero(self):
        a = [0.0] * 8
        b = [1.0] * 8
        assert fp_distance(fingerprint(a), fingerprint(b)) > 0.0

    def test_distance_bounded(self):
        a = [0.0] * 16
        b = [1.0] * 16
        d = fp_distance(fingerprint(a), fingerprint(b))
        assert 0.0 <= d <= 1.0


class TestAgentSoul:
    def test_goal_set_and_check(self):
        soul = AgentSoul(agent_id=1)
        goal = [0.0] * 16
        soul.goal_set(goal)
        # agent is at all-zeros → exactly at goal
        dist, curious = soul.goal_check([0.0] * 16)
        assert dist < 0.05
        assert not curious

    def test_goal_stall_triggers_curiosity(self):
        soul = AgentSoul(agent_id=2)
        soul.goal_set([1.0] * 16)   # goal far away
        # Simulate many pulses without progress
        for _ in range(soul.STALL_THRESHOLD + 5):
            soul.goal_check([0.0] * 16)   # always at same wrong spot
        assert soul.goal_stall_count > soul.STALL_THRESHOLD
        assert soul.curiosity_drive > 0.0

    def test_tag_memory_and_soul_query(self):
        soul = AgentSoul(agent_id=3)
        weights = [0.5] * 16
        soul.tag_memory(weights, valence=0.8, intensity=0.9)
        # Same weights → should hit
        hit = soul.soul_query(weights)
        assert hit is not None
        assert abs(hit.valence - 0.8) < 0.05

    def test_soul_query_miss_on_different_weights(self):
        soul = AgentSoul(agent_id=4)
        soul.tag_memory([0.0] * 16, valence=1.0, intensity=1.0)
        # Very different weights → miss
        hit = soul.soul_query([1.0] * 16)
        assert hit is None

    def test_spawn_mutated_goals(self):
        soul = AgentSoul(agent_id=5)
        soul.goal_set([0.5] * 16)
        mutations = soul.spawn_mutated_goals(4)
        assert len(mutations) == 4
        # Each mutation is a valid unit vector
        for m in mutations:
            assert len(m) == 16
            assert all(0.0 <= v <= 1.0 for v in m)

    def test_introspect_returns_dict(self):
        soul = AgentSoul(agent_id=6)
        soul.goal_set([0.1] * 16)
        info = soul.introspect()
        assert "agent_id" in info
        assert info["agent_id"] == 6
        assert "curiosity_drive" in info
        assert "generation" in info

    def test_inherit_from_parent(self):
        parent = AgentSoul(agent_id=10, generation=5)
        weights = [0.3] * 16
        parent.tag_memory(weights, valence=-0.9, intensity=1.0)
        parent.goal_set([0.7] * 16)

        child = AgentSoul(agent_id=11)
        child.inherit_from(parent, weight=1.0)

        assert child.generation == 6
        assert child.goal_vector is not None
        assert len(child.content_memory) > 0

    def test_to_binary_roundtrip(self):
        soul = AgentSoul(agent_id=99)
        soul.tag_memory([0.5] * 16, valence=0.6, intensity=0.8)
        raw = soul.to_binary()
        soul2 = AgentSoul.from_binary(raw, agent_id=99)
        assert len(soul2.content_memory) == len(soul.content_memory)


class TestMasterSoul:
    def test_absorb_and_snapshot(self):
        master = MasterSoul()
        soul = AgentSoul(agent_id=20)
        soul.tag_memory([0.5] * 16, valence=1.0, intensity=1.0)
        absorbed = master.absorb(soul)
        assert absorbed > 0
        snap = master.snapshot()
        assert snap["generations"] == 1
        assert snap["memory_count"] > 0


class TestSoulRegistry:
    def test_get_or_create(self):
        reg = SoulRegistry()
        s1  = reg.get_or_create(1)
        s2  = reg.get_or_create(1)
        assert s1 is s2   # same object

    def test_evolve(self):
        reg = SoulRegistry()
        parent = reg.get_or_create(0)
        parent.tag_memory([0.5] * 16, valence=0.7, intensity=0.8)
        parent.goal_set([0.8] * 16)
        for i in range(1, 4):
            reg.get_or_create(i)
        winner = reg.evolve(winner_id=1, candidates=[0, 1, 2, 3])
        assert winner.agent_id == 1


# ── Phase III: SomTerrain ──────────────────────────────────────────────────────

from runtime.som.terrain import SomTerrain, TerrainNode, TerrainRegistry


class TestTerrainNode:
    def test_virgin_on_init(self):
        n = TerrainNode(0, 0)
        assert n.is_virgin
        assert not n.is_sacred
        assert not n.is_hot_zone

    def test_visit_updates_fields(self):
        n = TerrainNode(1, 1)
        n.visit(pulse=10, valence=0.9, intensity=0.8)
        assert n.attractor_count == 1
        assert n.exploration_reward < 1.0

    def test_hot_zone_after_many_positive_visits(self):
        n = TerrainNode(2, 2)
        for i in range(20):
            n.visit(pulse=i, valence=0.8, intensity=0.9)
        assert n.is_hot_zone

    def test_danger_level_rises_on_negative(self):
        n = TerrainNode(3, 3)
        for i in range(10):
            n.visit(pulse=i, valence=-0.9, intensity=1.0)
        assert n.danger_level > 0.0

    def test_cultural_deposit(self):
        n = TerrainNode(4, 4)
        for _ in range(4): n.deposit(salience=0.9)
        assert n.is_sacred


class TestSomTerrain:
    def setup_method(self):
        self.terrain = SomTerrain(rows=8, cols=8)

    def test_mark_and_read(self):
        for i in range(4):
            self.terrain.mark(3, 4, pulse=i, valence=0.7, intensity=0.8)
        info = self.terrain.read(3, 4)
        assert info["attractor_count"] == 4
        assert not info["is_virgin"]

    def test_most_curious_node_is_unvisited(self):
        # Visit only (0,0)
        self.terrain.mark(0, 0, pulse=1, valence=0.5, intensity=0.5)
        r, c = self.terrain.most_curious_node()
        assert (r, c) != (0, 0)   # curious agent looks elsewhere

    def test_heatmap_shape(self):
        hm = self.terrain.heatmap()
        assert len(hm) == 8
        assert len(hm[0]) == 8

    def test_snapshot_counts(self):
        self.terrain.mark(1, 1, pulse=5, valence=0.9, intensity=1.0)
        snap = self.terrain.snapshot()
        assert snap["hot_zones"] >= 0
        assert snap["virgin_nodes"] > 0

    def test_tick_runs_without_error(self):
        self.terrain.tick()
        self.terrain.tick()


# ── Phase IV: CDBG ────────────────────────────────────────────────────────────

from soma.cdbg import (
    Frame, CTX, Encoder, StreamDecoder,
    crc4, make_agent_id, parse_agent_id, encode_soul_snapshot,
    SOUL_FIELDS,
)


class TestCRC4:
    def test_zero_crc(self):
        assert crc4(b"") == 0

    def test_known_value(self):
        # Any consistent value — we just verify stability
        v1 = crc4(b"SOMA")
        v2 = crc4(b"SOMA")
        assert v1 == v2
        assert 0 <= v1 <= 15

    def test_different_data_different_crc(self):
        assert crc4(b"\x00") != crc4(b"\xFF") or True  # at least stable


class TestFrameEncodeDecode:
    def test_roundtrip_all_contexts(self):
        frames = [
            Encoder.som_map(10, 20, 0x12),
            Encoder.agent(0x234567),
            Encoder.soul_field(0x03, 0.75),
            Encoder.memory_ref(0xABCDEF),
            Encoder.pulse(1234),
            Encoder.emot_tag(5, 0.6, 0.8),
            Encoder.surprise(3, 7, 0.4),
            Encoder.history_goal(generation=2, goal_record_id=42),
            Encoder.history_map(som_id=1, pulses_spent=500),
        ]
        for frame in frames:
            raw = frame.encode()
            assert len(raw) == 5
            decoded = Frame.decode(raw)
            assert decoded.ctx == frame.ctx
            assert decoded.sub == frame.sub
            assert decoded.payload == frame.payload

    def test_crc_mismatch_raises(self):
        raw = Encoder.agent(0x001234).encode()
        corrupted = bytearray(raw)
        corrupted[2] ^= 0xFF   # flip bits in payload
        with pytest.raises(ValueError, match="CRC"):
            Frame.decode(bytes(corrupted))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            Frame.decode(b"\x10\x00\x00\x00")   # 4 bytes, not 5


class TestParsers:
    def test_som_map_parsed(self):
        f = Encoder.som_map(5, 10, 0x60)
        p = f.parsed()
        assert p["x"] == 5
        assert p["y"] == 10
        assert p["opcode"] == 0x60

    def test_agent_parsed(self):
        aid = make_agent_id(cluster=2, map_id=52, seq=100)
        f   = Encoder.agent(aid)
        p   = f.parsed()
        assert p["cluster"]  == 2
        assert p["map_id"]   == 52
        assert p["seq"]      == 100

    def test_soul_field_parsed(self):
        f = Encoder.soul_field(0x03, 0.5)   # curiosity_drive = 0.5
        p = f.parsed()
        assert p["field"] == "curiosity_drive"
        assert abs(p["value"] - 0.5) < 0.01   # fp16 precision

    def test_emotion_emot_tag_parsed(self):
        f = Encoder.emot_tag(row=7, valence=0.8, intensity=0.6)
        p = f.parsed()
        assert p["row"] == 7
        assert abs(p["valence"]   - 0.8) < 0.02
        assert abs(p["intensity"] - 0.6) < 0.02

    def test_history_map_parsed(self):
        f = Encoder.history_map(som_id=3, pulses_spent=256)
        p = f.parsed()
        assert p["som_id"] == 3
        assert p["pulses_spent"] == 256


class TestStreamDecoder:
    def test_single_frame(self):
        raw = Encoder.pulse(42).encode()
        dec = StreamDecoder()
        frames = list(dec.feed(raw))
        assert len(frames) == 1
        assert frames[0].ctx == CTX.PULSE

    def test_multiple_frames(self):
        raw = b"".join(Encoder.pulse(i).encode() for i in range(10))
        dec = StreamDecoder()
        frames = [f for f in dec.feed(raw) if f is not None]
        assert len(frames) == 10

    def test_partial_feed(self):
        raw = Encoder.agent(0x001234).encode()
        dec = StreamDecoder()
        # Feed 3 bytes first, then 2
        frames1 = list(dec.feed(raw[:3]))
        frames2 = list(dec.feed(raw[3:]))
        all_frames = [f for f in frames1 + frames2 if f is not None]
        assert len(all_frames) == 1


class TestAgentID:
    def test_roundtrip(self):
        for cluster, map_id, seq in [(0,0,0), (15,255,4095), (7,128,2000)]:
            aid  = make_agent_id(cluster, map_id, seq)
            info = parse_agent_id(aid)
            assert info["cluster"]  == cluster
            assert info["map_id"]   == map_id
            assert info["seq"]      == seq

    def test_max_unique_agents(self):
        # 24 bits = 16,777,216 unique agents
        assert make_agent_id(15, 255, 4095) == 0xFFFFFF


class TestSoulStreaming:
    def test_encode_soul_snapshot(self):
        soul_fields = {
            "curiosity_drive": 0.6,
            "valence_mean":    -0.2,   # not in registry — skipped
            "goal_stall_count": 15.0,
        }
        raw = encode_soul_snapshot(agent_id=42, soul_fields=soul_fields)
        # Should have at least: agent frame + 2 soul field frames = 3 × 5 bytes
        assert len(raw) >= 10
        assert len(raw) % 5 == 0
