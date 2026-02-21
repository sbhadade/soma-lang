"""
runtime/som/emotion.py — Phase 2.5 + 2.6: Emotional Memory Stack
=================================================================

Phase 2.5 (done):
  - EMOT_TAG      → attach valence + intensity to current SOM activation
  - DECAY_PROTECT → shield emotionally tagged weights from decay
  - PREDICT_ERR   → compute surprise (BMU distance before vs prediction)

Phase 2.6 (new):
  - EMOT_RECALL   → retrieve emotional tag of any coord (decision weighting)
  - SURPRISE_CALC → formalised prediction-error → EMOT_TAG pipeline
  - EmotionSnapshot → serialisable state for MEMORY_SHARE packets

Paper reference: "A Path to AGI Part II: Liveliness"
  High delta (surprise) = high emotional tag = slow decay = strong memory.
  Low delta             = low tag            = fast decay = forgotten.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ── Emotion valence ───────────────────────────────────────────────────────────

class Valence(float, Enum):
    STRONGLY_POSITIVE =  1.0   # reward, success, discovery
    POSITIVE          =  0.5
    NEUTRAL           =  0.0
    NEGATIVE          = -0.5
    STRONGLY_NEGATIVE = -1.0   # failure, loss, danger


# ── Protection mode ──────────────────────────────────────────────────────────

class ProtectMode(Enum):
    CYCLES    = "cycles"      # protect for N more pulses
    PERMANENT = "permanent"   # never decay (core skill / trauma)
    SCALED    = "scaled"      # duration ∝ intensity


# ── Emotion tag on a single SOM node ─────────────────────────────────────────

@dataclass
class EmotionTag:
    """
    Attached to a SOM node when EMOT_TAG fires.

    valence   : float  — positive good, negative bad. Range [-1, +1].
    intensity : float  — how much world-model changed. Range [0, 1].
    timestamp : float  — time.monotonic() when tagged.
    protect_cycles_remaining : int  — pulses until protection expires.
                                      -1 = permanent.
    """
    valence:   float = 0.0
    intensity: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)
    protect_cycles_remaining: int = 0   # 0 = not protected

    @property
    def is_protected(self) -> bool:
        return self.protect_cycles_remaining != 0  # -1 = permanent

    @property
    def emotion_score(self) -> float:
        """Signed emotional weight: valence × intensity. Range [-1, +1]."""
        return self.valence * self.intensity

    @property
    def salience(self) -> float:
        """Unsigned importance score. Range [0, 1]."""
        return abs(self.valence) * self.intensity

    def tick(self) -> None:
        """Called on every PULSE. Decrements protection counter."""
        if self.protect_cycles_remaining > 0:
            self.protect_cycles_remaining -= 1
        # -1 stays -1 (permanent)

    def to_dict(self) -> dict:
        """Serialisable snapshot for MEMORY_SHARE packets."""
        return {
            "valence":   self.valence,
            "intensity": self.intensity,
            "timestamp": self.timestamp,
            "protect_cycles_remaining": self.protect_cycles_remaining,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionTag":
        return cls(
            valence=d["valence"],
            intensity=d["intensity"],
            timestamp=d["timestamp"],
            protect_cycles_remaining=d["protect_cycles_remaining"],
        )


# ── Emotion snapshot — shareable packet ─────────────────────────────────────

@dataclass
class EmotionSnapshot:
    """
    Serialisable snapshot of an agent's top-N emotion tags.
    Used by MEMORY_SHARE to transmit experience between agents.

    This is what an agent 'wills' to its neighbours on death,
    or broadcasts during NEIGHBOR_SYNC.
    """
    source_agent_id: int
    tags: List[Tuple[int, int, dict]]   # [(row, col, tag_dict), ...]
    created_at: float = field(default_factory=time.monotonic)

    def top(self, n: int) -> "EmotionSnapshot":
        """Return a new snapshot with only the top-N salient tags."""
        ranked = sorted(
            self.tags,
            key=lambda t: abs(t[2].get("valence", 0)) * t[2].get("intensity", 0),
            reverse=True
        )
        return EmotionSnapshot(
            source_agent_id=self.source_agent_id,
            tags=ranked[:n],
            created_at=self.created_at,
        )

    def __len__(self) -> int:
        return len(self.tags)


# ── Per-agent emotion state ───────────────────────────────────────────────────

@dataclass
class AgentEmotionState:
    """
    Emotional memory state for a single agent.
    Stored in AgentHandle (or AgentRegistry) by agent_id.
    """
    agent_id:  int
    # Sparse map: (som_r, som_c) → EmotionTag
    tags: Dict[Tuple[int,int], EmotionTag] = field(default_factory=dict)
    # Running prediction error history (for PREDICT_ERR / SURPRISE_CALC)
    prediction_errors: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock,
                                   repr=False, compare=False)

    # ── EMOT_TAG ─────────────────────────────────────────────────────────────

    def emot_tag(self, r: int, c: int,
                 valence: float, intensity: float) -> EmotionTag:
        """
        EMOT_TAG opcode implementation.
        Attaches or updates an emotion tag at SOM node (r, c).
        Clamps valence to [-1, +1] and intensity to [0, 1].
        """
        valence   = max(-1.0, min(1.0,  valence))
        intensity = max( 0.0, min(1.0, intensity))
        with self._lock:
            tag = self.tags.get((r, c))
            if tag is None:
                tag = EmotionTag(valence=valence, intensity=intensity)
                self.tags[(r, c)] = tag
            else:
                # Blend: new tag wins if stronger
                if intensity > tag.intensity:
                    tag.valence   = valence
                    tag.intensity = intensity
                    tag.timestamp = time.monotonic()
        return tag

    # ── DECAY_PROTECT ────────────────────────────────────────────────────────

    def decay_protect(self, r: int, c: int,
                      mode: ProtectMode = ProtectMode.CYCLES,
                      cycles: int = 100,
                      intensity: float = 1.0) -> None:
        """
        DECAY_PROTECT opcode implementation.
        Sets protection on the tag at (r, c), creating a neutral tag
        if none exists.
        """
        with self._lock:
            tag = self.tags.setdefault(
                (r, c), EmotionTag(valence=0.0, intensity=intensity)
            )
            if mode == ProtectMode.PERMANENT:
                tag.protect_cycles_remaining = -1
            elif mode == ProtectMode.SCALED:
                tag.protect_cycles_remaining = max(1, int(intensity * cycles))
            else:   # CYCLES
                tag.protect_cycles_remaining = max(1, cycles)

    # ── PREDICT_ERR ──────────────────────────────────────────────────────────

    def predict_err(self, bmu_r: int, bmu_c: int,
                    prev_bmu_r: int, prev_bmu_c: int,
                    rows: int, cols: int) -> float:
        """
        PREDICT_ERR opcode — compute normalised surprise.
        Surprise = topological distance between predicted BMU
                   (= last BMU) and actual BMU, normalised to [0, 1].
        """
        dr = bmu_r - prev_bmu_r
        dc = bmu_c - prev_bmu_c
        dist = math.sqrt(dr * dr + dc * dc)
        err  = self.normalise_err(dist, rows, cols)
        with self._lock:
            self.prediction_errors.append(err)
            if len(self.prediction_errors) > 1000:
                self.prediction_errors = self.prediction_errors[-500:]
        return err

    # ── EMOT_RECALL (Phase 2.6) ──────────────────────────────────────────────

    def emot_recall(self, r: int, c: int) -> Optional[EmotionTag]:
        """
        EMOT_RECALL opcode — retrieve the emotion tag for node (r, c).

        Returns None if no tag exists (node never emotionally significant).
        Use salience / emotion_score properties for decision weighting.

        Example:
            tag = state.emot_recall(3, 7)
            if tag and tag.salience > 0.5:
                # this region was highly significant — treat accordingly
        """
        with self._lock:
            return self.tags.get((r, c))

    # ── SURPRISE_CALC (Phase 2.6) ────────────────────────────────────────────

    def surprise_calc(self,
                      actual_vec:    List[float],
                      predicted_vec: List[float],
                      threshold:     float = 0.25) -> Tuple[float, bool]:
        """
        SURPRISE_CALC opcode — compute prediction error from raw vectors.

        Returns (error_magnitude, is_surprising) where is_surprising is
        True when error > threshold.  If surprising, caller should fire
        EMOT_TAG automatically.

        Parameters
        ----------
        actual_vec    : the input that actually arrived
        predicted_vec : what the agent expected (e.g. last BMU weights)
        threshold     : normalised error above which we call it surprising
        """
        if len(actual_vec) != len(predicted_vec):
            n = min(len(actual_vec), len(predicted_vec))
            actual_vec    = actual_vec[:n]
            predicted_vec = predicted_vec[:n]

        sq_sum = sum((a - b) ** 2
                     for a, b in zip(actual_vec, predicted_vec))
        err = math.sqrt(sq_sum) / math.sqrt(max(len(actual_vec), 1))
        err = min(err, 1.0)   # clamp to [0, 1]

        with self._lock:
            self.prediction_errors.append(err)
            if len(self.prediction_errors) > 1000:
                self.prediction_errors = self.prediction_errors[-500:]

        return err, err > threshold

    # ── Snapshot for MEMORY_SHARE ────────────────────────────────────────────

    def snapshot_for_share(self, top_n: int = 10) -> EmotionSnapshot:
        """
        Build an EmotionSnapshot of top-N most salient tags.
        Used by MEMORY_SHARE to send experience to another agent.
        """
        with self._lock:
            ranked = sorted(
                self.tags.items(),
                key=lambda kv: kv[1].salience,
                reverse=True
            )[:top_n]
        return EmotionSnapshot(
            source_agent_id=self.agent_id,
            tags=[(r, c, tag.to_dict()) for (r, c), tag in ranked],
        )

    def absorb_snapshot(self, snap: EmotionSnapshot,
                        weight: float = 0.5) -> int:
        """
        Receive an EmotionSnapshot from another agent (MEMORY_SHARE / NEIGHBOR_SYNC).

        Merges foreign emotion tags into this agent's state.
        Foreign tags are attenuated by `weight` (0–1) — full-weight = full
        adoption, 0.5 = blended (the default for neighbour influence).

        Returns number of tags absorbed.
        """
        absorbed = 0
        with self._lock:
            for r, c, tag_dict in snap.tags:
                foreign_intensity = tag_dict.get("intensity", 0.0) * weight
                foreign_valence   = tag_dict.get("valence", 0.0)
                coord = (r, c)

                existing = self.tags.get(coord)
                if existing is None:
                    # New tag — adopt at attenuated intensity
                    self.tags[coord] = EmotionTag(
                        valence=foreign_valence,
                        intensity=foreign_intensity,
                    )
                    absorbed += 1
                elif foreign_intensity > existing.intensity:
                    # Foreign is stronger — blend toward it
                    existing.valence   = (existing.valence + foreign_valence) / 2
                    existing.intensity = foreign_intensity
                    absorbed += 1
        return absorbed

    # ── Helpers ───────────────────────────────────────────────────────────────

    def normalise_err(self, err: float, rows: int, cols: int) -> float:
        """Normalise absolute topological distance to [0, 1]."""
        max_dist = math.sqrt(rows ** 2 + cols ** 2)
        return min(err / max_dist, 1.0) if max_dist > 0 else 0.0

    def tick(self) -> None:
        """Called on every PULSE. Decrements all protection counters."""
        with self._lock:
            expired = []
            for coord, tag in self.tags.items():
                tag.tick()
                # Remove fully-expired, unprotected, low-salience tags
                if (tag.protect_cycles_remaining == 0
                        and tag.salience < 0.01):
                    expired.append(coord)
            for coord in expired:
                del self.tags[coord]

    def is_protected(self, r: int, c: int) -> bool:
        """True if (r, c) has active decay protection."""
        with self._lock:
            tag = self.tags.get((r, c))
            return tag.is_protected if tag else False

    def emotion_score(self, r: int, c: int) -> float:
        """Signed emotion score at (r, c). 0.0 if no tag."""
        with self._lock:
            tag = self.tags.get((r, c))
            return tag.emotion_score if tag else 0.0

    def salience(self, r: int, c: int) -> float:
        """Unsigned salience at (r, c). 0.0 if no tag."""
        with self._lock:
            tag = self.tags.get((r, c))
            return tag.salience if tag else 0.0

    def top_salient_nodes(self, n: int = 10) -> List[Tuple[int, int, float]]:
        """Return top-N (r, c, salience) tuples, descending."""
        with self._lock:
            ranked = sorted(
                ((r, c, t.salience) for (r, c), t in self.tags.items()),
                key=lambda x: x[2],
                reverse=True
            )
        return ranked[:n]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "agent_id":  self.agent_id,
                "tag_count": len(self.tags),
                "protected": sum(1 for t in self.tags.values()
                                 if t.is_protected),
                "mean_salience": (
                    sum(t.salience for t in self.tags.values())
                    / len(self.tags)
                ) if self.tags else 0.0,
                "mean_prediction_error": (
                    sum(self.prediction_errors) / len(self.prediction_errors)
                ) if self.prediction_errors else 0.0,
            }


# ── Global emotion registry ───────────────────────────────────────────────────

class EmotionRegistry:
    """
    Global store of AgentEmotionState, keyed by agent_id.
    Thread-safe singleton-style registry.
    """

    def __init__(self):
        self._states: Dict[int, AgentEmotionState] = {}
        self._lock   = threading.Lock()

    def get_or_create(self, agent_id: int) -> AgentEmotionState:
        with self._lock:
            if agent_id not in self._states:
                self._states[agent_id] = AgentEmotionState(agent_id)
            return self._states[agent_id]

    def remove(self, agent_id: int) -> None:
        with self._lock:
            self._states.pop(agent_id, None)

    def tick_all(self) -> None:
        """Advance one PULSE for all agents."""
        with self._lock:
            states = list(self._states.values())
        for s in states:
            s.tick()

    def snapshot(self) -> dict:
        with self._lock:
            return {aid: s.snapshot() for aid, s in self._states.items()}
