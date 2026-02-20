"""
runtime.agent.agent_registry
=============================
Global, thread-safe agent table.

Design principles
-----------------
*  One process-wide singleton — ``AgentRegistry.get_instance()``
*  Agents are registered/deregistered under an ``RLock`` so any thread can
   safely read or mutate the table.
*  Agent IDs are 8-bit (0–255) to match the AGENT-ID field in the 64-bit
   SOMA instruction word.  ID 0 is always the main (entry) thread.
*  The registry hands out the *next available* ID automatically so SPAWN
   never needs to specify one explicitly from Python.
*  ``wait_for_agent(agent_id, timeout)`` lets ``WAIT A0`` block the caller
   without busy-polling.

Thread safety
-------------
All public methods acquire ``self._lock`` (an ``RLock``) before touching
``self._agents``.  Reads of individual ``AgentHandle`` fields (e.g. state)
are safe because Python's GIL guarantees atomic attribute reads for simple
types; complex mutations (state transitions) are still done under the lock
for correctness across all Python implementations.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Iterator, List, Optional

from .lifecycle import AgentHandle, AgentState


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class AgentNotFoundError(KeyError):
    """Raised when an agent_id is not present in the registry."""


class AgentIDExhaustedError(RuntimeError):
    """Raised when all 256 agent IDs (0–255) are occupied."""


class AgentAlreadyExistsError(ValueError):
    """Raised when registering an agent_id that already exists."""


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Singleton global table of SOMA agents.

    Usage
    -----
    ::

        registry = AgentRegistry.get_instance()

        # Register main thread (Agent 0) at startup
        handle = registry.register(agent_id=0)
        handle.state = AgentState.RUNNING

        # Register a spawned agent (auto ID)
        new_id = registry.next_id()
        handle = registry.register(agent_id=new_id, parent_id=0, som_x=2, som_y=3)

        # Look up a handle
        h = registry.get(new_id)

        # Wait for an agent to die (WAIT opcode)
        registry.wait_for_agent(new_id, timeout=10.0)

        # Deregister after death
        registry.deregister(new_id)
    """

    _instance: Optional["AgentRegistry"] = None
    _instance_lock = threading.Lock()

    # ── singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Destroy and recreate the singleton.

        **Only for use in tests.**  Tears down all registered agents and
        resets the registry to a clean state.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._shutdown()
            cls._instance = None

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._agents: Dict[int, AgentHandle] = {}
        # Condition variable used by wait_for_agent() — notified whenever any
        # agent's state changes to DEAD.
        self._death_event: threading.Condition = threading.Condition(self._lock)

    # ── ID allocation ─────────────────────────────────────────────────────────

    def next_id(self) -> int:
        """
        Return the lowest unused agent ID in [1, 255].

        Agent 0 is reserved for the main thread and must be registered
        explicitly via ``register(agent_id=0)``.

        Raises
        ------
        AgentIDExhaustedError
            If all 255 non-main IDs are occupied.
        """
        with self._lock:
            for candidate in range(1, 256):
                if candidate not in self._agents:
                    return candidate
            raise AgentIDExhaustedError(
                "All 255 agent IDs (1–255) are occupied. "
                "Increase SOMSIZE or reduce concurrent agent count."
            )

    # ── registration ──────────────────────────────────────────────────────────

    def register(
        self,
        agent_id: int,
        *,
        parent_id: Optional[int] = None,
        som_x: int = 0,
        som_y: int = 0,
    ) -> AgentHandle:
        """
        Create and store a new AgentHandle for ``agent_id``.

        Parameters
        ----------
        agent_id : int
            Must be unique and in range [0, 255].
        parent_id : int | None
            ID of the spawning agent, if any.
        som_x, som_y : int
            Initial SOM topology coordinates (updated by SOM_MAP opcode).

        Returns
        -------
        AgentHandle
            The newly created handle (state = SPAWNING).

        Raises
        ------
        ValueError
            If ``agent_id`` is out of range [0, 255].
        AgentAlreadyExistsError
            If an *alive* agent with ``agent_id`` already exists.
        """
        if not (0 <= agent_id <= 255):
            raise ValueError(f"agent_id must be 0–255, got {agent_id!r}")

        with self._lock:
            existing = self._agents.get(agent_id)
            if existing is not None and existing.is_alive:
                raise AgentAlreadyExistsError(
                    f"Agent {agent_id} is already alive ({existing.state.name}). "
                    "AGENT_KILL it before re-using the ID."
                )
            handle = AgentHandle(
                agent_id=agent_id,
                parent_id=parent_id,
                som_x=som_x,
                som_y=som_y,
            )
            self._agents[agent_id] = handle
            return handle

    def deregister(self, agent_id: int) -> None:
        """
        Remove a DEAD agent from the registry.

        It is safe to call this multiple times for the same ``agent_id``.
        Raises ``AgentNotFoundError`` if the ID has never been registered.
        Raises ``RuntimeError`` if the agent is still alive (not DEAD).
        """
        with self._lock:
            handle = self._agents.get(agent_id)
            if handle is None:
                raise AgentNotFoundError(
                    f"Agent {agent_id} not found in registry."
                )
            if handle.is_alive:
                raise RuntimeError(
                    f"Cannot deregister alive agent {agent_id} "
                    f"(state={handle.state.name}). Call kill() first."
                )
            del self._agents[agent_id]

    # ── lookup ────────────────────────────────────────────────────────────────

    def get(self, agent_id: int) -> AgentHandle:
        """
        Return the handle for ``agent_id``.

        Raises
        ------
        AgentNotFoundError
        """
        with self._lock:
            handle = self._agents.get(agent_id)
            if handle is None:
                raise AgentNotFoundError(
                    f"Agent {agent_id} not found in registry."
                )
            return handle

    def get_or_none(self, agent_id: int) -> Optional[AgentHandle]:
        """Return the handle or ``None`` if not found."""
        with self._lock:
            return self._agents.get(agent_id)

    # ── state mutation (always under lock) ────────────────────────────────────

    def set_state(self, agent_id: int, state: AgentState) -> None:
        """
        Transition ``agent_id`` to ``state``.

        If transitioning to DEAD, notifies all threads waiting in
        ``wait_for_agent()``.
        """
        with self._death_event:  # _death_event wraps _lock (same RLock)
            handle = self._agents.get(agent_id)
            if handle is None:
                raise AgentNotFoundError(agent_id)
            handle.state = state
            if state is AgentState.DEAD:
                handle.mark_dead()
                self._death_event.notify_all()

    def set_som_coords(self, agent_id: int, x: int, y: int) -> None:
        """Update SOM coordinates for ``agent_id`` (SOM_MAP opcode)."""
        with self._lock:
            handle = self._agents.get(agent_id)
            if handle is None:
                raise AgentNotFoundError(agent_id)
            handle.som_x = x
            handle.som_y = y

    # ── WAIT primitive (used by WAIT opcode in interpreter) ───────────────────

    def wait_for_agent(
        self,
        agent_id: int,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Block the calling thread until ``agent_id`` reaches DEAD state.

        Implements the ``WAIT A<n>`` SOMA opcode:  the calling agent blocks
        here until the target agent finishes (HALT or AGENT_KILL).

        Parameters
        ----------
        agent_id : int
            The agent to wait on.
        timeout : float | None
            Seconds to wait before giving up.  ``None`` = wait forever.

        Returns
        -------
        bool
            ``True`` if the agent died within the timeout, ``False`` otherwise.
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None

        with self._death_event:
            while True:
                handle = self._agents.get(agent_id)
                if handle is None or handle.state is AgentState.DEAD:
                    return True

                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False

                self._death_event.wait(timeout=remaining)

    # ── iteration & stats ─────────────────────────────────────────────────────

    def all_handles(self) -> List[AgentHandle]:
        """Snapshot of all registered handles (alive + dead) at call time."""
        with self._lock:
            return list(self._agents.values())

    def alive_handles(self) -> List[AgentHandle]:
        """Snapshot of all currently alive handles."""
        with self._lock:
            return [h for h in self._agents.values() if h.is_alive]

    def count(self) -> int:
        """Total registered agents (alive + dead not yet deregistered)."""
        with self._lock:
            return len(self._agents)

    def alive_count(self) -> int:
        """Number of currently alive agents."""
        with self._lock:
            return sum(1 for h in self._agents.values() if h.is_alive)

    def ids_at_coord(self, x: int, y: int) -> List[int]:
        """
        Return IDs of alive agents at SOM coordinate (x, y).

        Used by SOM_NBHD opcode to find agents in a neighbourhood.
        """
        with self._lock:
            return [
                h.agent_id
                for h in self._agents.values()
                if h.is_alive and h.som_x == x and h.som_y == y
            ]

    def __iter__(self) -> Iterator[AgentHandle]:
        """Iterate over a snapshot of all handles."""
        return iter(self.all_handles())

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, agent_id: int) -> bool:
        with self._lock:
            return agent_id in self._agents

    def __repr__(self) -> str:  # pragma: no cover
        alive = self.alive_count()
        total = self.count()
        return f"AgentRegistry(alive={alive}, total={total})"

    # ── internal ──────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        """
        Force-kill all alive agents and clear the registry.

        Called only by ``reset()`` during test teardown.
        """
        with self._lock:
            for handle in list(self._agents.values()):
                if handle.is_alive and handle.thread is not None:
                    try:
                        # Best-effort: daemon threads will die with the process
                        handle.thread.join(timeout=0.1)
                    except Exception:  # noqa: BLE001
                        pass
                handle.mark_dead()
            self._agents.clear()
