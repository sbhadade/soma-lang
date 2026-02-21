"""
runtime/som/emotion.py — Phase 2.5: Emotional Memory Stack
===========================================================

Implements the amygdala primitive for SOMA:
  - EMOT_TAG      → attach valence + intensity to current SOM activation
  - DECAY_PROTECT → shield emotionally tagged weights from decay
  - PREDICT_ERR   → compute surprise (BMU distance before vs prediction)

Paper reference: "A Path to AGI Part II: Liveliness"
  High delta (surprise) = high emotional tag = slow decay = strong memory.
  Low delta             = low tag            = fast decay = forgotten.
"""
from __future__ import annotations

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
    # Running prediction error history (for PREDICT_ERR)
    prediction_errors: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock,
                                   repr=False, compare=False)

    # ── EMOT_TAG ─────────────────────────────────────────────────────────────

    def emot_tag(self, r: int, c: int,
                 valence: float, intensity: float) -> EmotionTag:
        """
        EMOT_TAG opcode implementation.
        Attaches (or updates) an emotion tag on node (r, c).
        """
        valence   = max(-1.0, min(1.0, valence))
        intensity = max(0.0,  min(1.0, intensity))
        tag = EmotionTag(valence=valence, intensity=intensity)
        with self._lock:
            self.tags[(r, c)] = tag
        return tag

    # ── DECAY_PROTECT ────────────────────────────────────────────────────────

    def decay_protect(self, r: int, c: int,
                      mode: ProtectMode = ProtectMode.CYCLES,
                      cycles: int = 100,
                      intensity: float = 1.0) -> None:
        """
        DECAY_PROTECT opcode implementation.
        Shields the emotion tag at (r, c) from being overwritten by decay.

        mode=CYCLES:    protect for `cycles` pulses.
        mode=PERMANENT: protect forever (-1).
        mode=SCALED:    protect for int(cycles * intensity) pulses.
        """
        with self._lock:
            tag = self.tags.get((r, c))
            if tag is None:
                # Create a neutral tag just to carry the protection flag
                tag = EmotionTag(valence=0.0, intensity=intensity)
                self.tags[(r, c)] = tag

            if mode == ProtectMode.PERMANENT:
                tag.protect_cycles_remaining = -1
            elif mode == ProtectMode.SCALED:
                tag.protect_cycles_remaining = max(1, int(cycles * intensity))
            else:  # CYCLES
                tag.protect_cycles_remaining = cycles

    # ── PREDICT_ERR ──────────────────────────────────────────────────────────

    def predict_err(self, bmu_r: int, bmu_c: int,
                    predicted_r: int, predicted_c: int) -> float:
        """
        PREDICT_ERR opcode implementation.
        Returns normalised surprise: Euclidean distance between
        predicted BMU and actual BMU, divided by map diagonal.
        Range [0, 1].
        """
        import math
        err = math.sqrt((bmu_r - predicted_r)**2 + (bmu_c - predicted_c)**2)
        # Store for history
        with self._lock:
            self.prediction_errors.append(err)
            if len(self.prediction_errors) > 1000:
                self.prediction_errors.pop(0)
        return err

    def normalise_err(self, err: float, rows: int, cols: int) -> float:
        """Normalise raw BMU distance to [0, 1] using map diagonal."""
        import math
        diagonal = math.sqrt(rows**2 + cols**2)
        return min(1.0, err / diagonal) if diagonal > 0 else 0.0

    # ── Pulse tick ───────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Called on every PULSE — decrements all protection counters."""
        with self._lock:
            for tag in self.tags.values():
                tag.tick()

    # ── Queries ──────────────────────────────────────────────────────────────

    def is_protected(self, r: int, c: int) -> bool:
        with self._lock:
            tag = self.tags.get((r, c))
            return tag.is_protected if tag else False

    def emotion_score(self, r: int, c: int) -> float:
        with self._lock:
            tag = self.tags.get((r, c))
            return tag.emotion_score if tag else 0.0

    def salience(self, r: int, c: int) -> float:
        with self._lock:
            tag = self.tags.get((r, c))
            return tag.salience if tag else 0.0

    def top_salient_nodes(self, n: int = 10) -> List[Tuple[int, int, float]]:
        """Return top-N nodes by salience: [(r, c, salience), ...]."""
        with self._lock:
            ranked = [
                (r, c, tag.salience)
                for (r, c), tag in self.tags.items()
            ]
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked[:n]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "agent_id": self.agent_id,
                "tagged_nodes": len(self.tags),
                "protected_nodes": sum(
                    1 for t in self.tags.values() if t.is_protected
                ),
                "mean_salience": (
                    sum(t.salience for t in self.tags.values()) / len(self.tags)
                    if self.tags else 0.0
                ),
                "recent_prediction_error": (
                    sum(self.prediction_errors[-10:]) / len(self.prediction_errors[-10:])
                    if self.prediction_errors else 0.0
                ),
            }


# ── Registry: emotion states for all agents ──────────────────────────────────

class EmotionRegistry:
    """
    Singleton-per-VM store of AgentEmotionState objects.
    SomScheduler holds one instance and passes it to agents.
    """

    def __init__(self):
        self._states: Dict[int, AgentEmotionState] = {}
        self._lock   = threading.Lock()

    def get_or_create(self, agent_id: int) -> AgentEmotionState:
        with self._lock:
            if agent_id not in self._states:
                self._states[agent_id] = AgentEmotionState(agent_id=agent_id)
            return self._states[agent_id]

    def remove(self, agent_id: int) -> None:
        with self._lock:
            self._states.pop(agent_id, None)

    def tick_all(self) -> None:
        """Advance protection counters for all agents. Call on every PULSE."""
        with self._lock:
            states = list(self._states.values())
        for s in states:
            s.tick()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                aid: state.snapshot()
                for aid, state in self._states.items()
            }
