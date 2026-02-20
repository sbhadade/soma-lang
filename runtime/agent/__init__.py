"""
runtime.agent
=============
Phase 1 — True concurrency layer for SOMA.

Exports:
    AgentState      — lifecycle enum (SPAWNING | RUNNING | BLOCKED | DEAD)
    AgentHandle     — lightweight struct stored in the registry
    AgentRegistry   — global thread-safe table of all live agents
    ThreadAgent     — one OS thread per SOMA agent
"""

from .agent_registry import AgentRegistry
from .thread_agent import ThreadAgent
from .lifecycle import AgentState, AgentHandle

__all__ = ["AgentState", "AgentHandle", "AgentRegistry", "ThreadAgent"]
