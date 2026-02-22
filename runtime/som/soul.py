"""
runtime/som/soul.py — Phase 3: AgentSoul
=========================================

The AgentSoul is the portable identity of an agent.  It travels between
SOM maps, through EVOLVE selection, and across generations.

Key ideas from Part III:
  - content_memory : emotion tags indexed by SHA-256 fingerprint of weight
    vector — NOT by coordinate.  Survives map migration.
  - goal : target weight-vector in SOM space.  Intentionality, not reaction.
  - curiosity_drive : activates when goal_stall_count > threshold.
  - EVOLVE inheritance : MasterSoul accumulates fingerprints across generations.

Opcodes implemented here (dispatched from vm.py / soma_runtime.py):
    GOAL_SET    0x60  — set agent goal vector
    GOAL_CHECK  0x61  — distance from current BMU to goal; updates stall count
    SOUL_QUERY  0x62  — query content memory for fingerprint match
    META_SPAWN  0x63  — spawn N agents with mutated goal vectors
    EVOLVE      0x64  — select survivor by goal proximity; inherit soul
    INTROSPECT  0x65  — export own state snapshot as data

Paper: "A Path to AGI Part III: Curiosity", Swapnil Bhadade, Feb 2026
"""
from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Content fingerprint helpers ───────────────────────────────────────────────

def fingerprint(weights: List[float]) -> bytes:
    """SHA-256 of a weight vector, normalised to 32-byte digest."""
    raw = b"".join(hash(w).to_bytes(8, "big", signed=True)
                   for w in weights)
    return hashlib.sha256(raw).digest()


def fp_distance(a: bytes, b: bytes) -> float:
    """
    Hamming-style similarity between two 256-bit fingerprints.
    Returns 0.0 = identical, 1.0 = maximally different.
    """
    if a == b:
        return 0.0
    xor_bits = sum(bin(x ^ y).count("1") for x, y in zip(a, b))
    return xor_bits / 256.0


# ── Memory entry ──────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """
    One emotionally-tagged content memory.

    Stored against the SHA-256 fingerprint of the weight vector that was
    active when EMOT_TAG fired.  Survives SOM migration because pattern,
    not address, is the key.
    """
    fingerprint:  bytes        # 32-byte SHA-256
    valence:      float        # [-1, +1]
    intensity:    float        # [0, 1]
    birth_pulse:  int   = 0
    last_access:  int   = 0
    access_count: int   = 0
    flags:        int   = 0    # bit0=protect bit1=share bit2=curiosity

    @property
    def salience(self) -> float:
        return abs(self.valence) * self.intensity

    @property
    def is_protected(self) -> bool:
        return bool(self.flags & 0x01)

    def matches(self, fp: bytes, threshold: float = 0.15) -> bool:
        """True if this memory's fingerprint is close enough to fp."""
        return fp_distance(self.fingerprint, fp) <= threshold

    def to_dict(self) -> dict:
        return {
            "fingerprint":  self.fingerprint.hex(),
            "valence":      self.valence,
            "intensity":    self.intensity,
            "birth_pulse":  self.birth_pulse,
            "last_access":  self.last_access,
            "access_count": self.access_count,
            "flags":        self.flags,
        }


# ── AgentSoul ─────────────────────────────────────────────────────────────────

class AgentSoul:
    """
    Complete portable identity of one SOMA agent.

    Fields
    ------
    agent_id          : int — unique identifier
    generation        : int — incremented by EVOLVE each cycle
    content_memory    : list[MemoryEntry] — sorted by salience (top-K)
    goal_vector       : list[float] | None — desired SOM weight-space target
    goal_stall_count  : int — pulses without goal progress
    curiosity_drive   : float [0,1] — activates exploration when stall > thresh
    valence_mean      : float — rolling mean of recent emotional valence
    surprise_sensitivity: float [0,1] — how easily PREDICT_ERR triggers EMOT_TAG
    inheritance_depth : int — how many EVOLVE generations contributed
    """

    MAX_MEMORIES    = 10_000
    TOP_K_INHERIT   = 50        # memories passed to next generation
    STALL_THRESHOLD = 20        # pulses before curiosity activates
    MATCH_THRESHOLD = 0.15      # fingerprint similarity radius

    def __init__(self, agent_id: int, generation: int = 0):
        self.agent_id           = agent_id
        self.generation         = generation
        self.content_memory:    List[MemoryEntry] = []
        self.goal_vector:       Optional[List[float]] = None
        self.goal_stall_count:  int   = 0
        self.curiosity_drive:   float = 0.0
        self.valence_mean:      float = 0.0
        self.surprise_sensitivity: float = 0.25
        self.inheritance_depth: int   = 0
        self._pulse:            int   = 0
        self._lock = threading.Lock()

    # ── GOAL_SET ──────────────────────────────────────────────────────────────

    def goal_set(self, goal_vector: List[float]) -> None:
        """
        GOAL_SET opcode — encode a desired future as a weight vector.
        Resets stall counter and curiosity drive.
        """
        with self._lock:
            self.goal_vector      = list(goal_vector)
            self.goal_stall_count = 0
            self.curiosity_drive  = 0.0

    # ── GOAL_CHECK ────────────────────────────────────────────────────────────

    def goal_check(self, current_weights: List[float]) -> Tuple[float, bool]:
        """
        GOAL_CHECK opcode — measure distance between current SOM position
        (by weight vector) and goal vector.

        Returns (distance, curiosity_activated).

        Side-effects:
          - Increments goal_stall_count when distance barely changes.
          - Sets curiosity_drive when stall > STALL_THRESHOLD.
        """
        with self._lock:
            if self.goal_vector is None:
                return 1.0, False

            n   = min(len(current_weights), len(self.goal_vector))
            sq  = sum((a - b) ** 2
                      for a, b in zip(current_weights[:n], self.goal_vector[:n]))
            dist = math.sqrt(sq) / math.sqrt(max(n, 1))
            dist = min(dist, 1.0)

            # Stall detection — if delta is tiny, increment stall counter
            if dist > 0.05:   # still far — count as stall only if not improving
                self.goal_stall_count += 1
            else:
                self.goal_stall_count = max(0, self.goal_stall_count - 2)

            curiosity = self.goal_stall_count > self.STALL_THRESHOLD
            self.curiosity_drive = min(1.0, self.goal_stall_count / (self.STALL_THRESHOLD * 2))
            return dist, curiosity

    # ── SOUL_QUERY ────────────────────────────────────────────────────────────

    def soul_query(self, weights: List[float]) -> Optional[MemoryEntry]:
        """
        SOUL_QUERY opcode — given a weight vector, find the closest
        emotionally-tagged memory in content_memory.

        Returns the best matching MemoryEntry, or None.
        This is the computational definition of intuition:
        the agent recognises the pattern without recognising the place.
        """
        fp = fingerprint(weights)
        with self._lock:
            best: Optional[MemoryEntry] = None
            best_dist = self.MATCH_THRESHOLD + 1.0
            for entry in self.content_memory:
                d = fp_distance(entry.fingerprint, fp)
                if d < best_dist:
                    best_dist = d
                    best = entry
            if best is not None:
                best.last_access  = self._pulse
                best.access_count += 1
            return best if best_dist <= self.MATCH_THRESHOLD else None

    # ── Tag a memory entry ────────────────────────────────────────────────────

    def tag_memory(self, weights: List[float],
                   valence: float, intensity: float,
                   flags: int = 0) -> MemoryEntry:
        """
        Called by EMOT_TAG (Phase 2) when content-addressing is active.
        Stores the emotion against the SHA-256 fingerprint of `weights`.
        """
        fp = fingerprint(weights)
        valence   = max(-1.0, min(1.0, valence))
        intensity = max(0.0,  min(1.0, intensity))

        with self._lock:
            # Update existing entry if fingerprint matches
            for entry in self.content_memory:
                if fp_distance(entry.fingerprint, fp) < 0.05:
                    if intensity > entry.intensity:
                        entry.valence   = valence
                        entry.intensity = intensity
                        entry.flags    |= flags
                    return entry

            # New entry
            entry = MemoryEntry(
                fingerprint=fp,
                valence=valence,
                intensity=intensity,
                birth_pulse=self._pulse,
                last_access=self._pulse,
                flags=flags,
            )
            self.content_memory.append(entry)

            # Trim to MAX_MEMORIES keeping highest salience
            if len(self.content_memory) > self.MAX_MEMORIES:
                self.content_memory.sort(key=lambda e: e.salience, reverse=True)
                self.content_memory = self.content_memory[:self.MAX_MEMORIES]

            # Update rolling valence mean
            n = len(self.content_memory)
            self.valence_mean = sum(e.valence * e.intensity
                                    for e in self.content_memory) / max(n, 1)
            return entry

    # ── META_SPAWN ────────────────────────────────────────────────────────────

    def spawn_mutated_goals(self, n: int,
                            mutation_scale: float = 0.1) -> List[List[float]]:
        """
        META_SPAWN helper — produce N mutated copies of the current goal.
        Each child gets a slightly perturbed goal vector.
        If no goal exists, returns N random unit vectors.
        """
        base = self.goal_vector if self.goal_vector else [random.random()
                                                           for _ in range(16)]
        result = []
        for _ in range(n):
            mutated = [
                max(0.0, min(1.0, v + random.gauss(0, mutation_scale)))
                for v in base
            ]
            result.append(mutated)
        return result

    # ── EVOLVE / inheritance ──────────────────────────────────────────────────

    def inherit_from(self, parent: "AgentSoul",
                     weight: float = 1.0) -> None:
        """
        EVOLVE opcode — copy the top-K memories from parent into this soul.

        The `weight` parameter attenuates intensities (0–1).
        Only memories with salience > 0.1 transfer.
        Called when a new agent is selected as EVOLVE winner.
        """
        with parent._lock:
            top_k = sorted(parent.content_memory,
                            key=lambda e: e.salience,
                            reverse=True)[:self.TOP_K_INHERIT]

        with self._lock:
            for entry in top_k:
                if entry.salience < 0.1:
                    continue
                new_entry = MemoryEntry(
                    fingerprint=entry.fingerprint,
                    valence=entry.valence,
                    intensity=entry.intensity * weight,
                    birth_pulse=entry.birth_pulse,
                    last_access=self._pulse,
                    flags=entry.flags,
                )
                self.content_memory.append(new_entry)

            # De-duplicate
            seen: Dict[bytes, MemoryEntry] = {}
            for e in self.content_memory:
                key = e.fingerprint
                if key not in seen or e.salience > seen[key].salience:
                    seen[key] = e
            self.content_memory = sorted(seen.values(),
                                          key=lambda e: e.salience,
                                          reverse=True)[:self.MAX_MEMORIES]

            self.generation        = parent.generation + 1
            self.inheritance_depth = parent.inheritance_depth + 1
            self.valence_mean      = parent.valence_mean * weight
            self.surprise_sensitivity = parent.surprise_sensitivity
            # Carry forward goal if parent had one
            if parent.goal_vector is not None:
                self.goal_vector = list(parent.goal_vector)

    # ── INTROSPECT ────────────────────────────────────────────────────────────

    def introspect(self) -> dict:
        """
        INTROSPECT opcode — return a full state snapshot as a dict.
        Agents can pass this to MSG_SEND to broadcast their identity,
        or use it for self-aware decision making.
        """
        with self._lock:
            top_10 = sorted(self.content_memory,
                             key=lambda e: e.salience,
                             reverse=True)[:10]
            return {
                "agent_id":           self.agent_id,
                "generation":         self.generation,
                "inheritance_depth":  self.inheritance_depth,
                "pulse":              self._pulse,
                "valence_mean":       round(self.valence_mean, 4),
                "curiosity_drive":    round(self.curiosity_drive, 4),
                "goal_stall_count":   self.goal_stall_count,
                "has_goal":           self.goal_vector is not None,
                "memory_count":       len(self.content_memory),
                "surprise_sensitivity": self.surprise_sensitivity,
                "top_memories":       [m.to_dict() for m in top_10],
            }

    # ── Pulse tick ────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Advance one PULSE."""
        self._pulse += 1

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_binary(self) -> bytes:
        """
        Serialise to .cmem binary format (Part IV §5.3).
        HEADER: 4B magic + 4B agent_id + 4B entry_count + 4B version
        ENTRY:  32B fingerprint + 1B valence(s8) + 1B intensity(u8) + 1B flags + 1B reserved
                + 4B birth_pulse + 4B last_access + 4B access_count
        Total per entry: 48 bytes
        """
        import struct
        MAGIC   = b"SOMA"
        VERSION = 4
        entries = self.content_memory
        header  = struct.pack(">4sIII", MAGIC, self.agent_id,
                              len(entries), VERSION)
        body = b""
        for e in entries:
            v8  = max(-128, min(127, int(e.valence * 127)))
            i8  = max(0, min(255, int(e.intensity * 255)))
            body += e.fingerprint
            body += struct.pack(">bBBBIII",
                                v8, i8, e.flags, 0,
                                e.birth_pulse, e.last_access, e.access_count)
        return header + body

    @classmethod
    def from_binary(cls, data: bytes, agent_id: int) -> "AgentSoul":
        """Deserialise from .cmem bytes."""
        import struct
        soul = cls(agent_id)
        if len(data) < 16:
            return soul
        magic, aid, count, ver = struct.unpack_from(">4sIII", data, 0)
        if magic != b"SOMA":
            raise ValueError("Invalid .cmem magic")
        offset = 16
        for _ in range(count):
            if offset + 48 > len(data):
                break
            fp = data[offset:offset+32]
            v8, i8, flags, _, bp, la, ac = struct.unpack_from(">bBBBIII",
                                                               data, offset+32)
            entry = MemoryEntry(
                fingerprint=fp,
                valence=v8 / 127.0,
                intensity=i8 / 255.0,
                birth_pulse=bp,
                last_access=la,
                access_count=ac,
                flags=flags,
            )
            soul.content_memory.append(entry)
            offset += 48
        return soul


# ── MasterSoul — accumulated across EVOLVE generations ───────────────────────

class MasterSoul:
    """
    Accumulates the distilled wisdom of many EVOLVE generations.
    Each time EVOLVE fires, the winner's soul merges into MasterSoul.
    MasterSoul is what the lineage inherits — not the specific goal,
    not the specific coordinates, but the emotionally significant patterns
    that survived 100 rounds of selection pressure.
    """

    def __init__(self):
        self.generations: int = 0
        self._memories:   List[MemoryEntry] = []
        self._lock = threading.Lock()

    def absorb(self, soul: AgentSoul, attenuation: float = 0.9) -> int:
        """
        Merge the top-K memories from `soul` into the master record.
        Returns number of new or updated memories.
        """
        count = 0
        with soul._lock:
            top = sorted(soul.content_memory,
                          key=lambda e: e.salience,
                          reverse=True)[:AgentSoul.TOP_K_INHERIT]

        with self._lock:
            existing = {e.fingerprint: e for e in self._memories}
            for entry in top:
                if entry.salience < 0.05:
                    continue
                fp = entry.fingerprint
                if fp in existing:
                    ex = existing[fp]
                    if entry.salience * attenuation > ex.salience:
                        ex.valence   = (ex.valence + entry.valence) / 2
                        ex.intensity = entry.intensity * attenuation
                        count += 1
                else:
                    new_e = MemoryEntry(
                        fingerprint=fp,
                        valence=entry.valence,
                        intensity=entry.intensity * attenuation,
                        birth_pulse=entry.birth_pulse,
                        last_access=entry.last_access,
                        flags=entry.flags,
                    )
                    self._memories.append(new_e)
                    existing[fp] = new_e
                    count += 1

            # Trim
            self._memories.sort(key=lambda e: e.salience, reverse=True)
            self._memories = self._memories[:AgentSoul.MAX_MEMORIES]
            self.generations += 1
        return count

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "generations":    self.generations,
                "memory_count":   len(self._memories),
                "mean_salience":  (
                    sum(e.salience for e in self._memories) / max(len(self._memories), 1)
                ),
                "top_memories": [
                    e.to_dict() for e in
                    sorted(self._memories, key=lambda e: e.salience, reverse=True)[:5]
                ],
            }


# ── Global soul registry ──────────────────────────────────────────────────────

class SoulRegistry:
    """Thread-safe registry: agent_id → AgentSoul."""

    def __init__(self):
        self._souls: Dict[int, AgentSoul] = {}
        self._master = MasterSoul()
        self._lock   = threading.Lock()

    def get_or_create(self, agent_id: int, generation: int = 0) -> AgentSoul:
        with self._lock:
            if agent_id not in self._souls:
                self._souls[agent_id] = AgentSoul(agent_id, generation)
            return self._souls[agent_id]

    def remove(self, agent_id: int) -> Optional[AgentSoul]:
        with self._lock:
            return self._souls.pop(agent_id, None)

    def evolve(self, winner_id: int, candidates: List[int]) -> AgentSoul:
        """
        EVOLVE opcode — winner inherits from all candidates via MasterSoul.
        Returns the evolved winner soul.
        """
        with self._lock:
            souls = [self._souls.get(cid) for cid in candidates]
            souls = [s for s in souls if s is not None]

        for soul in souls:
            self._master.absorb(soul)

        winner = self.get_or_create(winner_id)
        for soul in souls:
            if soul.agent_id != winner_id:
                winner.inherit_from(soul, weight=0.7)

        return winner

    def tick_all(self) -> None:
        with self._lock:
            for soul in self._souls.values():
                soul.tick()

    @property
    def master(self) -> MasterSoul:
        return self._master

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "agent_count": len(self._souls),
                "master":      self._master.snapshot(),
            }
