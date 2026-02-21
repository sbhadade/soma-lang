"""
runtime.som
===========
Phase 2   — Live SOM topology and SOM-driven agent scheduler.
Phase 2.5 — Liveliness: Decay, Emotional Memory, Memory Consolidation.

Exports:
    LiveSomMap       — thread-safe SOM grid (BMU/TRAIN/WALK/ELECT)
    SomScheduler     — wires SOM topology into AgentRegistry
    SomVisualizer    — ASCII terminal heatmap renderer
    EmotionRegistry  — per-agent EMOT_TAG / DECAY_PROTECT / PREDICT_ERR
    MemoryManager    — two-tier working + long-term SOM (MEMORY_CONSOLIDATE)
    ProtectMode      — CYCLES / PERMANENT / SCALED decay protection
"""

from .som_map        import LiveSomMap
from .som_scheduler  import SomScheduler
from .som_visualizer import SomVisualizer
from .emotion        import (EmotionRegistry, EmotionTag,
                             AgentEmotionState, ProtectMode, Valence)
from .memory         import MemoryManager, TwoTierMemory, ConsolidationReport

__all__ = [
    "LiveSomMap",
    "SomScheduler",
    "SomVisualizer",
    "EmotionRegistry",
    "EmotionTag",
    "AgentEmotionState",
    "ProtectMode",
    "Valence",
    "MemoryManager",
    "TwoTierMemory",
    "ConsolidationReport",
]
