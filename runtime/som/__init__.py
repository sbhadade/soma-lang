"""
runtime.som
===========
Phase 2   — Live SOM topology and SOM-driven agent scheduler.
Phase 2.5 — Liveliness: Decay, Emotional Memory, Memory Consolidation.
Phase 2.6 — Culture: Memory Share, Neighbor Sync, Reorg Map, Emot Recall.

Exports:
    LiveSomMap       — thread-safe SOM grid (BMU/TRAIN/WALK/ELECT)
    SomScheduler     — wires SOM topology into AgentRegistry
    SomVisualizer    — ASCII terminal heatmap renderer

    EmotionRegistry  — per-agent EMOT_TAG / DECAY_PROTECT / PREDICT_ERR
    EmotionTag       — single-node emotion record (valence, intensity, protection)
    AgentEmotionState— per-agent emotion state (tag store + EMOT_RECALL + SURPRISE_CALC)
    EmotionSnapshot  — shareable packet of top-N emotion tags (NEIGHBOR_SYNC / MEMORY_SHARE)
    ProtectMode      — CYCLES / PERMANENT / SCALED decay protection
    Valence          — strongly_positive .. strongly_negative enum

    MemoryManager    — two-tier working + long-term SOM (MEMORY_CONSOLIDATE)
    TwoTierMemory    — per-agent working + long-term SOM
    MemorySharePacket— cultural transmission unit (MEMORY_SHARE output)
    ConsolidationReport — result of MEMORY_CONSOLIDATE
"""

from .som_map        import LiveSomMap
from .som_scheduler  import SomScheduler
from .som_visualizer import SomVisualizer
from .emotion        import (EmotionRegistry, EmotionTag, EmotionSnapshot,
                             AgentEmotionState, ProtectMode, Valence)
from .memory         import (MemoryManager, TwoTierMemory,
                             MemorySharePacket, ConsolidationReport)

__all__ = [
    # SOM core
    "LiveSomMap",
    "SomScheduler",
    "SomVisualizer",
    # Emotion
    "EmotionRegistry",
    "EmotionTag",
    "EmotionSnapshot",
    "AgentEmotionState",
    "ProtectMode",
    "Valence",
    # Memory
    "MemoryManager",
    "TwoTierMemory",
    "MemorySharePacket",
    "ConsolidationReport",
]
