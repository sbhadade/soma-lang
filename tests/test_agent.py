"""
tests.test_agent
================
Phase 1 / Step 1 — AgentRegistry + ThreadAgent tests.

Test categories
---------------
Unit — AgentState / AgentHandle
Unit — AgentRegistry  (ID allocation, register, deregister, state machine)
Unit — ThreadAgent    (start, join, kill, error handling)
Integration — SPAWN + WAIT + HALT flow
Stress — race-condition detection (50× repetitions)
Thread-leak detection — active_count() guard on every test

Run
---
    pytest tests/test_agent.py -v
    pytest tests/test_agent.py -v --timeout=30    # with pytest-timeout
"""

from __future__ import annotations

import threading
import time
from typing import List

import pytest

from runtime.agent.lifecycle import AgentHandle, AgentState
from runtime.agent.agent_registry import (
    AgentRegistry,
    AgentAlreadyExistsError,
    AgentIDExhaustedError,
    AgentNotFoundError,
)
from runtime.agent.thread_agent import ThreadAgent, spawn_agent


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_registry():
    """
    Reset the singleton registry before *and* after every test.

    This guarantees:
    1. Tests are isolated — no agent from a previous test bleeds in.
    2. Thread-leak detection: we assert active_count() returns to its
       pre-test baseline after the test (daemon threads die quickly).
    """
    AgentRegistry.reset()
    baseline_threads = threading.active_count()
    yield AgentRegistry.get_instance()
    # Give daemon threads up to 1 s to die naturally after test teardown
    AgentRegistry.reset()
    deadline = time.monotonic() + 1.0
    while threading.active_count() > baseline_threads and time.monotonic() < deadline:
        time.sleep(0.01)
    assert threading.active_count() <= baseline_threads, (
        f"Thread leak detected: {threading.active_count()} threads alive, "
        f"expected ≤ {baseline_threads}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# AgentState / AgentHandle unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentState:
    def test_enum_values_exist(self):
        for name in ("SPAWNING", "RUNNING", "BLOCKED", "DEAD"):
            assert hasattr(AgentState, name)

    def test_is_alive_for_live_states(self):
        handle = AgentHandle(agent_id=1)
        for live in (AgentState.SPAWNING, AgentState.RUNNING, AgentState.BLOCKED):
            handle.state = live
            assert handle.is_alive

    def test_is_alive_false_when_dead(self):
        handle = AgentHandle(agent_id=1, state=AgentState.DEAD)
        assert not handle.is_alive

    def test_mark_dead_sets_state_and_timestamp(self):
        handle = AgentHandle(agent_id=1)
        assert handle.died_at is None
        handle.mark_dead()
        assert handle.state is AgentState.DEAD
        assert handle.died_at is not None
        assert handle.lifetime_ms >= 0

    def test_lifetime_ms_increases_over_time(self):
        handle = AgentHandle(agent_id=1)
        t0 = handle.lifetime_ms
        time.sleep(0.05)
        t1 = handle.lifetime_ms
        assert t1 > t0

    def test_lifetime_ms_frozen_after_death(self):
        handle = AgentHandle(agent_id=1)
        handle.mark_dead()
        t0 = handle.lifetime_ms
        time.sleep(0.05)
        t1 = handle.lifetime_ms
        assert abs(t1 - t0) < 1.0  # frozen within 1 ms tolerance


# ─────────────────────────────────────────────────────────────────────────────
# AgentRegistry unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentRegistry:

    # ── singleton ──────────────────────────────────────────────────────────

    def test_singleton_returns_same_instance(self):
        a = AgentRegistry.get_instance()
        b = AgentRegistry.get_instance()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = AgentRegistry.get_instance()
        AgentRegistry.reset()
        b = AgentRegistry.get_instance()
        assert a is not b

    # ── register ──────────────────────────────────────────────────────────

    def test_register_agent_0_for_main_thread(self, fresh_registry):
        handle = fresh_registry.register(0)
        assert handle.agent_id == 0
        assert handle.state is AgentState.SPAWNING
        assert 0 in fresh_registry

    def test_register_stores_som_coords(self, fresh_registry):
        handle = fresh_registry.register(1, som_x=3, som_y=7)
        assert handle.som_x == 3
        assert handle.som_y == 7

    def test_register_stores_parent_id(self, fresh_registry):
        fresh_registry.register(0)
        handle = fresh_registry.register(1, parent_id=0)
        assert handle.parent_id == 0

    def test_register_duplicate_alive_raises(self, fresh_registry):
        fresh_registry.register(1)
        with pytest.raises(AgentAlreadyExistsError):
            fresh_registry.register(1)

    def test_register_after_death_reuses_id(self, fresh_registry):
        """Dead agents can have their ID recycled."""
        h = fresh_registry.register(1)
        fresh_registry.set_state(1, AgentState.DEAD)
        fresh_registry.deregister(1)
        # Should not raise
        h2 = fresh_registry.register(1)
        assert h2.agent_id == 1

    def test_register_invalid_id_raises(self, fresh_registry):
        with pytest.raises(ValueError):
            fresh_registry.register(-1)
        with pytest.raises(ValueError):
            fresh_registry.register(256)

    # ── next_id ───────────────────────────────────────────────────────────

    def test_next_id_starts_at_1(self, fresh_registry):
        assert fresh_registry.next_id() == 1

    def test_next_id_skips_occupied(self, fresh_registry):
        fresh_registry.register(1)
        assert fresh_registry.next_id() == 2

    def test_next_id_exhausted_raises(self, fresh_registry):
        # Fill IDs 1–255
        for i in range(1, 256):
            fresh_registry.register(i)
        with pytest.raises(AgentIDExhaustedError):
            fresh_registry.next_id()

    # ── deregister ────────────────────────────────────────────────────────

    def test_deregister_dead_agent(self, fresh_registry):
        fresh_registry.register(1)
        fresh_registry.set_state(1, AgentState.DEAD)
        fresh_registry.deregister(1)
        assert 1 not in fresh_registry

    def test_deregister_alive_agent_raises(self, fresh_registry):
        fresh_registry.register(1)
        fresh_registry.set_state(1, AgentState.RUNNING)
        with pytest.raises(RuntimeError, match="alive"):
            fresh_registry.deregister(1)

    def test_deregister_unknown_raises(self, fresh_registry):
        with pytest.raises(AgentNotFoundError):
            fresh_registry.deregister(99)

    # ── get ───────────────────────────────────────────────────────────────

    def test_get_returns_handle(self, fresh_registry):
        fresh_registry.register(5, som_x=1, som_y=2)
        h = fresh_registry.get(5)
        assert h.agent_id == 5

    def test_get_unknown_raises(self, fresh_registry):
        with pytest.raises(AgentNotFoundError):
            fresh_registry.get(42)

    def test_get_or_none_returns_none_for_unknown(self, fresh_registry):
        assert fresh_registry.get_or_none(99) is None

    # ── set_state ─────────────────────────────────────────────────────────

    def test_set_state_transitions(self, fresh_registry):
        fresh_registry.register(1)
        for state in (AgentState.RUNNING, AgentState.BLOCKED,
                      AgentState.RUNNING, AgentState.DEAD):
            fresh_registry.set_state(1, state)
            assert fresh_registry.get(1).state is state

    def test_set_state_dead_notifies_waiters(self, fresh_registry):
        """wait_for_agent() must unblock when state → DEAD."""
        fresh_registry.register(1)
        fresh_registry.set_state(1, AgentState.RUNNING)

        results: List[bool] = []

        def waiter():
            ok = fresh_registry.wait_for_agent(1, timeout=2.0)
            results.append(ok)

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.05)
        fresh_registry.set_state(1, AgentState.DEAD)
        t.join(timeout=1.0)
        assert results == [True]

    # ── set_som_coords ────────────────────────────────────────────────────

    def test_set_som_coords(self, fresh_registry):
        fresh_registry.register(1)
        fresh_registry.set_som_coords(1, 4, 9)
        h = fresh_registry.get(1)
        assert h.som_x == 4
        assert h.som_y == 9

    # ── wait_for_agent ────────────────────────────────────────────────────

    def test_wait_for_agent_times_out(self, fresh_registry):
        """Agent never dies → wait must return False within timeout."""
        fresh_registry.register(1)
        fresh_registry.set_state(1, AgentState.RUNNING)
        ok = fresh_registry.wait_for_agent(1, timeout=0.1)
        assert ok is False

    def test_wait_for_already_dead_returns_immediately(self, fresh_registry):
        fresh_registry.register(1)
        fresh_registry.set_state(1, AgentState.DEAD)
        t0 = time.monotonic()
        ok = fresh_registry.wait_for_agent(1, timeout=5.0)
        elapsed = time.monotonic() - t0
        assert ok is True
        assert elapsed < 0.5  # should be nearly instant

    # ── counts & iteration ────────────────────────────────────────────────

    def test_count_and_alive_count(self, fresh_registry):
        assert fresh_registry.count() == 0
        assert fresh_registry.alive_count() == 0
        fresh_registry.register(1)
        fresh_registry.set_state(1, AgentState.RUNNING)
        fresh_registry.register(2)
        assert fresh_registry.count() == 2
        assert fresh_registry.alive_count() == 2
        fresh_registry.set_state(1, AgentState.DEAD)
        assert fresh_registry.count() == 2
        assert fresh_registry.alive_count() == 1

    def test_ids_at_coord(self, fresh_registry):
        fresh_registry.register(1, som_x=2, som_y=3)
        fresh_registry.set_state(1, AgentState.RUNNING)
        fresh_registry.register(2, som_x=2, som_y=3)
        fresh_registry.set_state(2, AgentState.RUNNING)
        fresh_registry.register(3, som_x=0, som_y=0)
        fresh_registry.set_state(3, AgentState.RUNNING)

        ids = fresh_registry.ids_at_coord(2, 3)
        assert set(ids) == {1, 2}
        assert fresh_registry.ids_at_coord(0, 0) == [3]
        assert fresh_registry.ids_at_coord(9, 9) == []

    def test_iter_returns_all_handles(self, fresh_registry):
        for i in range(1, 6):
            fresh_registry.register(i)
        handles = list(fresh_registry)
        assert len(handles) == 5

    def test_len(self, fresh_registry):
        for i in range(1, 4):
            fresh_registry.register(i)
        assert len(fresh_registry) == 3

    # ── thread safety ─────────────────────────────────────────────────────

    def test_concurrent_register_no_id_collision(self, fresh_registry):
        """
        50 threads each calling next_id() + register() concurrently must
        produce 50 unique IDs with no collisions or exceptions.
        """
        N = 50
        allocated: List[int] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker():
            try:
                aid = fresh_registry.next_id()
                fresh_registry.register(aid)
                with lock:
                    allocated.append(aid)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Errors during concurrent register: {errors}"
        assert len(allocated) == N
        assert len(set(allocated)) == N, "ID collision detected!"


# ─────────────────────────────────────────────────────────────────────────────
# ThreadAgent unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestThreadAgent:

    def _make_agent(self, fn, *, fresh_registry, parent_id=0,
                    som_x=0, som_y=0) -> ThreadAgent:
        """Helper: allocate ID, create ThreadAgent (not yet started)."""
        fresh_registry.register(0)                    # main thread
        fresh_registry.set_state(0, AgentState.RUNNING)
        aid = fresh_registry.next_id()
        return ThreadAgent(
            aid, fn,
            parent_id=parent_id,
            som_x=som_x, som_y=som_y,
            registry=fresh_registry,
        )

    # ── basic start / join ────────────────────────────────────────────────

    def test_spawn_join_simple(self, fresh_registry):
        """SPAWN + WAIT: agent runs fn, returns, state → DEAD."""
        ran: List[int] = []

        def worker(handle: AgentHandle):
            ran.append(handle.agent_id)

        ta = self._make_agent(worker, fresh_registry=fresh_registry)
        ta.start()
        finished = ta.join(timeout=5.0)

        assert finished is True
        assert len(ran) == 1
        assert fresh_registry.get(ta.agent_id).state is AgentState.DEAD

    def test_agent_state_is_running_during_execution(self, fresh_registry):
        """State must be RUNNING while fn is executing."""
        observed_states: List[AgentState] = []
        ready = threading.Event()
        proceed = threading.Event()

        def worker(handle: AgentHandle):
            ready.set()
            proceed.wait(timeout=2.0)
            observed_states.append(handle.state)

        ta = self._make_agent(worker, fresh_registry=fresh_registry)
        ta.start()
        ready.wait(timeout=2.0)
        # Sample state while thread is blocking on proceed.wait()
        observed_states.append(fresh_registry.get(ta.agent_id).state)
        proceed.set()
        ta.join(timeout=5.0)

        assert AgentState.RUNNING in observed_states

    def test_state_becomes_dead_after_fn_returns(self, fresh_registry):
        def worker(handle: AgentHandle):
            pass

        ta = self._make_agent(worker, fresh_registry=fresh_registry)
        ta.start()
        ta.join(timeout=5.0)
        assert fresh_registry.get(ta.agent_id).state is AgentState.DEAD

    def test_som_coords_stored_in_handle(self, fresh_registry):
        def worker(handle: AgentHandle):
            pass

        ta = self._make_agent(worker, fresh_registry=fresh_registry,
                              som_x=3, som_y=7)
        ta.start()
        ta.join(timeout=5.0)
        h = fresh_registry.get(ta.agent_id)
        assert h.som_x == 3
        assert h.som_y == 7

    def test_parent_id_stored(self, fresh_registry):
        def worker(handle: AgentHandle):
            pass

        ta = self._make_agent(worker, fresh_registry=fresh_registry,
                              parent_id=0)
        assert ta.handle.parent_id == 0

    # ── join behaviour ────────────────────────────────────────────────────

    def test_join_timeout_returns_false_for_slow_agent(self, fresh_registry):
        """join() with a short timeout returns False if agent hasn't finished."""
        blocker = threading.Event()

        def worker(handle: AgentHandle):
            blocker.wait(timeout=10.0)

        ta = self._make_agent(worker, fresh_registry=fresh_registry)
        ta.start()
        finished = ta.join(timeout=0.1)
        assert finished is False
        blocker.set()          # unblock so daemon thread can clean up
        ta.join(timeout=2.0)

    def test_join_before_start_raises(self, fresh_registry):
        def worker(handle):
            pass

        fresh_registry.register(0)
        fresh_registry.set_state(0, AgentState.RUNNING)
        aid = fresh_registry.next_id()
        ta = ThreadAgent(aid, worker, registry=fresh_registry)
        with pytest.raises(RuntimeError, match="never called"):
            ta.join()

    def test_start_twice_raises(self, fresh_registry):
        def worker(handle):
            pass

        ta = self._make_agent(worker, fresh_registry=fresh_registry)
        ta.start()
        with pytest.raises(RuntimeError, match="already been started"):
            ta.start()
        ta.join(timeout=2.0)

    # ── kill ──────────────────────────────────────────────────────────────

    def test_kill_transitions_to_dead(self, fresh_registry):
        blocker = threading.Event()

        def worker(handle: AgentHandle):
            blocker.wait(timeout=10.0)

        ta = self._make_agent(worker, fresh_registry=fresh_registry)
        ta.start()
        time.sleep(0.02)    # give thread time to enter blocker.wait()
        ta.kill()
        blocker.set()
        ta.join(timeout=2.0)
        assert fresh_registry.get(ta.agent_id).state is AgentState.DEAD

    def test_kill_is_idempotent(self, fresh_registry):
        def worker(handle):
            pass

        ta = self._make_agent(worker, fresh_registry=fresh_registry)
        ta.start()
        ta.join(timeout=2.0)
        # Calling kill on already-dead agent must not raise
        ta.kill()
        ta.kill()

    def test_is_stopped_after_kill(self, fresh_registry):
        blocker = threading.Event()

        def worker(handle):
            blocker.wait(timeout=5.0)

        ta = self._make_agent(worker, fresh_registry=fresh_registry)
        ta.start()
        assert not ta.is_stopped
        ta.kill()
        blocker.set()
        assert ta.is_stopped

    # ── exception safety ──────────────────────────────────────────────────

    def test_uncaught_exception_marks_dead(self, fresh_registry):
        """If fn raises, agent must still transition to DEAD."""
        def bad_worker(handle: AgentHandle):
            raise ValueError("intentional error for testing")

        ta = self._make_agent(bad_worker, fresh_registry=fresh_registry)
        ta.start()
        ta.join(timeout=5.0)
        assert fresh_registry.get(ta.agent_id).state is AgentState.DEAD


# ─────────────────────────────────────────────────────────────────────────────
# spawn_agent() factory
# ─────────────────────────────────────────────────────────────────────────────

class TestSpawnAgent:

    def test_spawn_agent_returns_running_thread_agent(self, fresh_registry):
        done = threading.Event()

        def worker(handle):
            done.wait(timeout=5.0)

        ta = spawn_agent(worker, parent_id=0, registry=fresh_registry)
        assert isinstance(ta, ThreadAgent)
        assert fresh_registry.get(ta.agent_id).state is AgentState.RUNNING
        done.set()
        ta.join(timeout=2.0)

    def test_spawn_agent_auto_assigns_id(self, fresh_registry):
        results: List[int] = []

        def worker(handle):
            results.append(handle.agent_id)

        ta1 = spawn_agent(worker, registry=fresh_registry)
        ta2 = spawn_agent(worker, registry=fresh_registry)
        ta1.join(timeout=2.0)
        ta2.join(timeout=2.0)

        assert ta1.agent_id != ta2.agent_id
        assert set(results) == {ta1.agent_id, ta2.agent_id}

    def test_spawn_agent_som_coords(self, fresh_registry):
        def worker(handle):
            pass

        ta = spawn_agent(worker, som_x=5, som_y=6, registry=fresh_registry)
        ta.join(timeout=2.0)
        h = fresh_registry.get(ta.agent_id)
        assert h.som_x == 5
        assert h.som_y == 6


# ─────────────────────────────────────────────────────────────────────────────
# Integration — SPAWN + WAIT + HALT flow
# ─────────────────────────────────────────────────────────────────────────────

class TestSpawnWaitHaltIntegration:

    def test_hello_agent_flow(self, fresh_registry):
        """
        Simulate the hello_agent.soma execution:

            @_start:
              SPAWN  A0, @worker     ← spawn_agent(worker_fn, parent_id=0)
              SOM_MAP A0, (0,0)      ← set_som_coords(A0, 0, 0)
              WAIT   A0              ← join() / wait_for_agent()
              HALT                   ← main returns

            @worker:
              ; does some work
              AGENT_KILL SELF        ← fn returns → DEAD
        """
        worker_ran = threading.Event()

        def worker_fn(handle: AgentHandle):
            worker_ran.set()
            # Simulate SOM_TRAIN work
            time.sleep(0.02)

        # SPAWN
        ta = spawn_agent(worker_fn, parent_id=0, registry=fresh_registry)
        worker_id = ta.agent_id

        # SOM_MAP
        fresh_registry.set_som_coords(worker_id, 0, 0)

        # WAIT
        finished = ta.join(timeout=5.0)

        assert finished is True
        assert worker_ran.is_set()
        assert fresh_registry.get(worker_id).state is AgentState.DEAD

    def test_multiple_agents_sequential_wait(self, fresh_registry):
        """
        Spawn 4 agents, WAIT on each in order.  All must finish cleanly.
        """
        N = 4
        finished_ids: List[int] = []
        lock = threading.Lock()

        def worker_fn(handle: AgentHandle):
            time.sleep(0.01)
            with lock:
                finished_ids.append(handle.agent_id)

        agents = [
            spawn_agent(worker_fn, parent_id=0, registry=fresh_registry)
            for _ in range(N)
        ]
        for ta in agents:
            ok = ta.join(timeout=5.0)
            assert ok is True, f"Agent {ta.agent_id} did not finish in time"

        assert len(finished_ids) == N
        assert len(set(finished_ids)) == N  # all unique IDs

    def test_registry_wait_for_agent_matches_join(self, fresh_registry):
        """
        ``registry.wait_for_agent()`` and ``ta.join()`` must agree on
        whether the agent has finished.
        """
        def worker_fn(handle):
            time.sleep(0.02)

        ta = spawn_agent(worker_fn, registry=fresh_registry)

        # Use registry-level wait (this is what WAIT opcode uses directly)
        ok = fresh_registry.wait_for_agent(ta.agent_id, timeout=5.0)
        assert ok is True
        assert fresh_registry.get(ta.agent_id).state is AgentState.DEAD


# ─────────────────────────────────────────────────────────────────────────────
# Stress tests — race-condition detection
# ─────────────────────────────────────────────────────────────────────────────

class TestStress:

    @pytest.mark.parametrize("iteration", range(50))
    def test_spawn_join_stress(self, fresh_registry, iteration):
        """
        Spawn an agent, let it finish, assert DEAD — 50 iterations.
        Any race condition in state transitions will surface here.
        """
        done = threading.Event()

        def worker(handle):
            done.set()

        ta = spawn_agent(worker, registry=fresh_registry)
        done.wait(timeout=2.0)
        ta.join(timeout=2.0)
        assert fresh_registry.get(ta.agent_id).state is AgentState.DEAD

    def test_concurrent_spawn_all_get_unique_ids(self, fresh_registry):
        """
        50 threads simultaneously call spawn_agent().
        All 50 must get unique IDs and finish cleanly.
        """
        N = 50
        spawned: List[ThreadAgent] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def spawner():
            try:
                ta = spawn_agent(
                    lambda h: time.sleep(0.01),
                    registry=fresh_registry,
                )
                with lock:
                    spawned.append(ta)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=spawner, daemon=True) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Errors during concurrent spawn: {errors}"
        assert len(spawned) == N

        for ta in spawned:
            ta.join(timeout=5.0)

        ids = [ta.agent_id for ta in spawned]
        assert len(set(ids)) == N, "Duplicate agent IDs detected!"

    def test_concurrent_state_reads_while_writing(self, fresh_registry):
        """
        One thread writes state; 10 readers continuously poll.
        No exception must be raised.  (Tests RLock correctness.)
        """
        fresh_registry.register(1)
        fresh_registry.set_state(1, AgentState.RUNNING)
        stop = threading.Event()
        errors: List[Exception] = []

        def reader():
            while not stop.is_set():
                try:
                    h = fresh_registry.get_or_none(1)
                    if h:
                        _ = h.state
                        _ = h.is_alive
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

        def writer():
            states = [AgentState.RUNNING, AgentState.BLOCKED,
                      AgentState.RUNNING, AgentState.DEAD]
            for state in states:
                time.sleep(0.01)
                try:
                    h = fresh_registry.get_or_none(1)
                    if h and h.state is not AgentState.DEAD:
                        fresh_registry.set_state(1, state)
                except Exception:  # noqa: BLE001
                    pass

        readers = [threading.Thread(target=reader, daemon=True) for _ in range(10)]
        writer_t = threading.Thread(target=writer, daemon=True)

        for r in readers:
            r.start()
        writer_t.start()
        writer_t.join(timeout=5.0)
        stop.set()
        for r in readers:
            r.join(timeout=2.0)

        assert not errors, f"Concurrent read/write errors: {errors}"
