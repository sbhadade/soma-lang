"""
runtime.agent.thread_agent
===========================
One real OS thread per SOMA agent.

Design
------
``ThreadAgent`` wraps Python's ``threading.Thread`` and ties it to an
``AgentHandle`` in the ``AgentRegistry``.  It manages the full lifecycle:

    SPAWNING  →  start()  →  RUNNING  →  (fn returns or kill())  →  DEAD

Key decisions
-------------
*  Threads are daemon threads so they never block interpreter shutdown.
*  The agent function ``fn`` receives the agent's own ``AgentHandle`` as its
   first argument — this is how SOMA opcodes like ``MSG_RECV`` and
   ``AGENT_KILL SELF`` know which agent they belong to.
*  If ``fn`` raises an uncaught exception the thread logs it and transitions
   to DEAD (never silently stalls).
*  ``kill()`` sets a ``_stop_event`` that cooperative SOMA instructions
   (MSG_RECV, SOM_WALK) check via ``is_stopped()``.  For truly unresponsive
   threads we fall back to ``join(timeout)`` and accept daemon-thread cleanup
   at process exit — we do NOT use ``ctypes`` thread cancellation which is
   unsafe in CPython.

ISA opcodes wired here (Step 1)
---------------------------------
    0x01  SPAWN        →  ThreadAgent(…).start()
    0x02  AGENT_KILL   →  ThreadAgent.kill()
    WAIT               →  ThreadAgent.join() / registry.wait_for_agent()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .lifecycle import AgentHandle, AgentState
from .agent_registry import AgentRegistry

logger = logging.getLogger(__name__)

# Default thread stack size in bytes.
# Python's threading module uses the OS default (~8 MB) unless we override.
# 256 KB is enough for SOMA control-flow; SOM weight arrays live on the heap.
_DEFAULT_STACK_KB = 256


class ThreadAgent:
    """
    A SOMA agent backed by a real OS thread.

    Parameters
    ----------
    agent_id : int
        8-bit agent identifier [0–255].  Must be unique in the registry.
        Normally obtained via ``AgentRegistry.next_id()``.
    fn : Callable[[AgentHandle], None]
        The function to execute in the new thread.  It receives the agent's
        own ``AgentHandle`` so it can query its SOM coordinates, send
        messages, etc.
    parent_id : int | None
        The agent_id that is spawning this agent (for PARENT pseudo-register).
    som_x, som_y : int
        Initial SOM coordinates (can be updated later via SOM_MAP).
    stack_kb : int
        Stack size in KB.  Defaults to 256 KB.
    registry : AgentRegistry | None
        The registry to use.  Defaults to the process-wide singleton.
        Pass an explicit instance in unit tests.

    Example
    -------
    ::

        registry = AgentRegistry.get_instance()

        def worker(handle: AgentHandle) -> None:
            print(f"Agent {handle.agent_id} running at "
                  f"SOM ({handle.som_x}, {handle.som_y})")

        agent_id = registry.next_id()
        ta = ThreadAgent(agent_id, fn=worker, parent_id=0, som_x=1, som_y=2)
        ta.start()
        ta.join()          # blocks until worker() returns
    """

    def __init__(
        self,
        agent_id: int,
        fn: Callable[[AgentHandle], None],
        *,
        parent_id: Optional[int] = None,
        som_x: int = 0,
        som_y: int = 0,
        stack_kb: int = _DEFAULT_STACK_KB,
        registry: Optional[AgentRegistry] = None,
    ) -> None:
        self._registry = registry or AgentRegistry.get_instance()
        self._agent_id = agent_id
        self._fn = fn
        self._stack_kb = stack_kb

        # Register the handle *before* the thread starts so the registry is
        # always the source of truth even during the SPAWNING window.
        self._handle: AgentHandle = self._registry.register(
            agent_id,
            parent_id=parent_id,
            som_x=som_x,
            som_y=som_y,
        )

        # Cooperative stop flag — checked by long-running SOMA instructions
        self._stop_event = threading.Event()

        # The OS thread (created lazily in start())
        self._thread: Optional[threading.Thread] = None

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def agent_id(self) -> int:
        return self._agent_id

    @property
    def handle(self) -> AgentHandle:
        return self._handle

    @property
    def is_stopped(self) -> bool:
        """True once kill() has been called or the fn has returned."""
        return self._stop_event.is_set()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> "ThreadAgent":
        """
        Spawn the OS thread and transition to RUNNING.

        Returns ``self`` for chaining::

            ta = ThreadAgent(...).start()

        Raises
        ------
        RuntimeError
            If called more than once.
        """
        if self._thread is not None:
            raise RuntimeError(
                f"ThreadAgent {self._agent_id} has already been started."
            )

        # Set stack size before creating the thread.
        # threading.stack_size() is process-global, so we restore it after.
        old_stack = threading.stack_size()
        try:
            threading.stack_size(self._stack_kb * 1024)
            self._thread = threading.Thread(
                target=self._run,
                name=f"soma-agent-{self._agent_id}",
                daemon=True,   # never blocks interpreter shutdown
            )
        finally:
            threading.stack_size(old_stack)

        # Store thread reference in the handle so the registry can join it
        self._handle.thread = self._thread

        # Transition to RUNNING before the thread actually starts — this way
        # any caller that checks the registry immediately after start() sees
        # the correct state.
        self._registry.set_state(self._agent_id, AgentState.RUNNING)
        self._thread.start()

        logger.debug(
            "SPAWN agent=%d parent=%s som=(%d,%d) tid=%s",
            self._agent_id,
            self._handle.parent_id,
            self._handle.som_x,
            self._handle.som_y,
            self._thread.ident,
        )
        return self

    def join(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the agent thread to finish.

        Mirrors the ``WAIT A<n>`` SOMA opcode.

        Parameters
        ----------
        timeout : float | None
            Maximum seconds to wait.  ``None`` = wait forever.

        Returns
        -------
        bool
            ``True`` if the thread has finished, ``False`` if timed out.

        Raises
        ------
        RuntimeError
            If ``start()`` was never called.
        """
        if self._thread is None:
            raise RuntimeError(
                f"ThreadAgent {self._agent_id}: start() was never called."
            )
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def kill(self) -> None:
        """
        Request cooperative termination of this agent.

        Sets ``_stop_event`` so any SOMA instruction that checks
        ``agent.is_stopped`` will return cleanly.  Then waits up to 2 s for
        the thread to finish.  After that we give up — daemon threads will
        be cleaned up at process exit.

        Transitions the agent to DEAD and deregisters it from the registry.
        """
        if self._stop_event.is_set():
            return  # idempotent

        logger.debug("AGENT_KILL agent=%d", self._agent_id)
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(
                    "Agent %d thread did not stop within 2 s after kill(). "
                    "It will be cleaned up at process exit (daemon thread).",
                    self._agent_id,
                )

        # Ensure DEAD state is set in registry (may already be set if fn
        # returned naturally — _run() calls _finish() first).
        handle = self._registry.get_or_none(self._agent_id)
        if handle and handle.state is not AgentState.DEAD:
            self._registry.set_state(self._agent_id, AgentState.DEAD)

    # ── internal thread target ─────────────────────────────────────────────

    def _run(self) -> None:
        """
        Entry point executed inside the new OS thread.

        Calls the user-supplied ``fn(handle)`` and ensures the agent
        transitions to DEAD regardless of how ``fn`` exits.
        """
        try:
            self._fn(self._handle)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Uncaught exception in agent %d — marking DEAD",
                self._agent_id,
            )
        finally:
            self._finish()

    def _finish(self) -> None:
        """Mark the agent DEAD and signal any waiters."""
        self._stop_event.set()
        handle = self._registry.get_or_none(self._agent_id)
        if handle and handle.state is not AgentState.DEAD:
            self._registry.set_state(self._agent_id, AgentState.DEAD)
        logger.debug(
            "Agent %d finished. lifetime=%.2f ms",
            self._agent_id,
            self._handle.lifetime_ms,
        )

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        state = self._handle.state.name
        tid = self._thread.ident if self._thread else None
        return (
            f"ThreadAgent(id={self._agent_id}, state={state}, "
            f"som=({self._handle.som_x},{self._handle.som_y}), tid={tid})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory used by the interpreter's SPAWN dispatch
# ─────────────────────────────────────────────────────────────────────────────

def spawn_agent(
    fn: Callable[[AgentHandle], None],
    *,
    parent_id: int = 0,
    som_x: int = 0,
    som_y: int = 0,
    registry: Optional[AgentRegistry] = None,
) -> ThreadAgent:
    """
    Allocate a new agent ID, create a ThreadAgent, and start it.

    This is the one-liner called by the interpreter when it dispatches
    opcode ``0x01 SPAWN``::

        # In interpreter.py dispatch loop:
        if opcode == 0x01:  # SPAWN
            target_fn = resolve_label(instruction.imm)
            ta = spawn_agent(target_fn, parent_id=current_agent_id,
                             som_x=instr.som_x, som_y=instr.som_y)
            registers[instr.reg] = ta.agent_id   # store handle in A-register

    Parameters
    ----------
    fn : Callable[[AgentHandle], None]
        The agent's entry function.
    parent_id : int
        ID of the calling agent.
    som_x, som_y : int
        Initial SOM placement.
    registry : AgentRegistry | None
        Defaults to singleton.

    Returns
    -------
    ThreadAgent
        Already started (state = RUNNING).
    """
    reg = registry or AgentRegistry.get_instance()
    agent_id = reg.next_id()
    ta = ThreadAgent(
        agent_id,
        fn=fn,
        parent_id=parent_id,
        som_x=som_x,
        som_y=som_y,
        registry=reg,
    )
    ta.start()
    return ta
