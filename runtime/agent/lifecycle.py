"""
runtime.agent.lifecycle
=======================
Shared enums and data types for agent lifecycle management.

These are imported by both agent_registry and thread_agent to avoid
circular imports.

ISA reference (SOMBIN.spec):
    0x01  SPAWN      — create agent       → AgentState.SPAWNING → RUNNING
    0x02  AGENT_KILL — terminate agent    → AgentState.DEAD
    0x03  FORK       — duplicate N times  (Phase 1, Step 3)
    0x04  MERGE      — join N results     (Phase 1, Step 3)
    0x05  BARRIER    — sync N agents      (Phase 1, Step 3)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ─────────────────────────────────────────────
# Agent lifecycle states
# ─────────────────────────────────────────────

class AgentState(Enum):
    """
    State machine for a SOMA agent thread.

        SPAWNING ──► RUNNING ──► DEAD
                        │
                        ▼
                     BLOCKED   (on MSG_RECV / BARRIER)
                        │
                        └──► RUNNING  (when unblocked)
    """
    SPAWNING = auto()   # thread created, not yet executing code
    RUNNING  = auto()   # actively executing SOMA instructions
    BLOCKED  = auto()   # waiting on MSG_RECV or BARRIER
    DEAD     = auto()   # thread finished or AGENT_KILL called


# ─────────────────────────────────────────────
# Per-agent handle (stored in AgentRegistry)
# ─────────────────────────────────────────────

@dataclass
class AgentHandle:
    """
    Lightweight descriptor for one SOMA agent.

    Stored inside AgentRegistry.  ThreadAgent holds a reference to its own
    handle and updates state/timestamps as it runs.

    Fields
    ------
    agent_id : int
        8-bit agent identifier matching the AGENT-ID field in the 64-bit
        SOMA instruction word.  Range: 0–255.  Agent 0 is always the main
        (entry-point) thread.

    som_x, som_y : int
        SOM topology coordinates set by SOM_MAP opcode.  Used by the SOM
        scheduler (Phase 2) to pin agents to NUMA nodes and compute
        neighbourhoods.  Default (0, 0) means "unplaced".

    state : AgentState
        Current lifecycle state.  Written by the owning ThreadAgent;
        read by the registry and other agents under the registry lock.

    thread : threading.Thread | None
        The OS thread executing this agent.  None for Agent 0 (main thread).

    parent_id : int | None
        The agent_id that spawned this agent (set at SPAWN time).
        Used by AGENT_KILL SELF to notify the parent.

    mailbox : queue.Queue | None
        Per-agent message queue (wired in Step 2).  Kept here so the
        registry can expose a single lookup for both thread and mailbox.

    born_at : float
        Epoch timestamp (time.monotonic()) when the handle was registered.

    died_at : float | None
        Epoch timestamp when state transitioned to DEAD.
    """
    agent_id  : int
    som_x     : int             = 0
    som_y     : int             = 0
    state     : AgentState      = AgentState.SPAWNING
    thread    : Optional[threading.Thread] = field(default=None, repr=False)
    parent_id : Optional[int]   = None
    mailbox   : object          = field(default=None, repr=False)  # queue.Queue (Step 2)
    born_at   : float           = field(default_factory=time.monotonic)
    died_at   : Optional[float] = None

    # ── convenience ──────────────────────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        return self.state in (AgentState.SPAWNING, AgentState.RUNNING,
                              AgentState.BLOCKED)

    @property
    def lifetime_ms(self) -> float:
        """Wall-clock lifetime in milliseconds (up to now, or until death)."""
        end = self.died_at if self.died_at else time.monotonic()
        return (end - self.born_at) * 1_000

    def mark_dead(self) -> None:
        self.state    = AgentState.DEAD
        self.died_at  = time.monotonic()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AgentHandle(id={self.agent_id}, state={self.state.name}, "
            f"som=({self.som_x},{self.som_y}), "
            f"parent={self.parent_id}, alive={self.is_alive})"
        )
