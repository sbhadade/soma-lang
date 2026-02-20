"""
runtime/interpreter.py  —  PATCH for Phase 1 / Step 1
=======================================================

Apply this diff to the existing interpreter.py.

The existing dispatch loop already handles opcodes as integers decoded from
the 64-bit instruction word.  We wire THREE opcodes here:

    0x01  SPAWN        →  spawn_agent()
    0x02  AGENT_KILL   →  ThreadAgent.kill() / handle.mark_dead()
    0x37  HALT         →  join all agents, then sys.exit / return

Nothing else changes in this step.  MSG_SEND/RECV (0x20/0x21) will be wired
in Step 2 once the mailbox system is in place.

HOW TO APPLY
------------
1. Open runtime/interpreter.py
2. Add the import block below to the top-level imports section.
3. Find the opcode dispatch section (likely a dict, if/elif chain, or match
   statement) and add/replace the three cases shown.
4. In __init__ (or wherever the interpreter is constructed), add the two
   lines that register Agent 0 and initialise the agent_threads dict.

If your interpreter uses a DIFFERENT dispatch pattern (e.g. a handler dict),
adapt accordingly — the logic is identical.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Add these imports at the top of runtime/interpreter.py
# ─────────────────────────────────────────────────────────────────────────────

# ADD to imports:
from runtime.agent import AgentRegistry, AgentState, ThreadAgent
from runtime.agent.thread_agent import spawn_agent


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: In Interpreter.__init__  (or wherever execution state lives)
# ─────────────────────────────────────────────────────────────────────────────

# ADD these two lines after the existing register/memory initialisation:
#
#   self.registry = AgentRegistry.get_instance()
#   self._agent_threads: dict[int, ThreadAgent] = {}
#   # Register main thread as Agent 0
#   main_handle = self.registry.register(0)
#   self.registry.set_state(0, AgentState.RUNNING)
#   main_handle.thread = None   # main thread has no ThreadAgent wrapper


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: In the opcode dispatch section — add/replace these three cases
# ─────────────────────────────────────────────────────────────────────────────
#
# Your existing dispatch probably looks like:
#
#   def _execute(self, instruction):
#       opcode = instruction.opcode          # e.g. 0x01
#       agent_id = instruction.agent_id      # e.g. 0
#       som_x = instruction.som_x
#       som_y = instruction.som_y
#       reg   = instruction.reg              # register index
#       imm   = instruction.imm              # immediate value
#
# Replace / add the three blocks below:

def _handle_spawn(self, instruction, resolve_label):
    """
    0x01 SPAWN A<reg>, @label

    Spawns a new agent that begins execution at @label.
    Stores the new agent_id into the A-register specified by instruction.reg.

    Parameters
    ----------
    instruction : SomaInstruction
        The decoded 64-bit SOMA instruction.
    resolve_label : Callable[[int], Callable]
        Function that converts an immediate (label address) into a Python
        callable the new thread will execute.  The callable must accept
        one argument: the agent's AgentHandle.

    Example (from hello_agent.soma):
        SPAWN A0, @worker     ; A0 ← new agent_id running @worker
    """
    target_fn = resolve_label(instruction.imm)   # imm = label address
    parent_id = self._current_agent_id           # who is doing the spawning

    ta = spawn_agent(
        target_fn,
        parent_id=parent_id,
        som_x=instruction.som_x,
        som_y=instruction.som_y,
        registry=self.registry,
    )

    # Store new agent_id in the A-register named by instruction.reg
    # A-registers are 64-bit; agent_id fits in the lower 8 bits.
    self.a_registers[instruction.reg] = ta.agent_id
    self._agent_threads[ta.agent_id] = ta


def _handle_agent_kill(self, instruction):
    """
    0x02 AGENT_KILL SELF | A<reg>

    Terminates the specified agent.

    If the target is SELF (conventional encoding: agent_id == 0xFF or
    agent_id field == current agent), the current agent's thread simply
    returns — _run() in ThreadAgent will call _finish() automatically.

    For other targets, call ta.kill().

    Example:
        AGENT_KILL SELF
        AGENT_KILL A1
    """
    SELF_SENTINEL = 0xFF   # conventional encoding of SELF in ISA

    target_id = instruction.agent_id
    if target_id == SELF_SENTINEL or target_id == self._current_agent_id:
        # Returning from the agent function is the cleanest termination.
        # Raise a dedicated exception that _run() catches and treats as
        # normal exit (not an error).
        raise _AgentSelfKill()

    ta = self._agent_threads.get(target_id)
    if ta is not None:
        ta.kill()
    else:
        # Agent was spawned elsewhere or already dead — update registry only
        handle = self.registry.get_or_none(target_id)
        if handle and handle.is_alive:
            self.registry.set_state(target_id, AgentState.DEAD)


def _handle_wait(self, instruction):
    """
    WAIT A<reg>   (no dedicated opcode — synthesised by assembler)

    Blocks the current agent until the target agent's state → DEAD.
    Mirrors the WAIT pseudo-opcode / join behaviour.

    Timeout of 30 s is a safety guard against deadlocks during development.
    Raise RuntimeError on timeout so the interpreter surfaces it clearly.
    """
    target_id = self.a_registers[instruction.reg]
    ta = self._agent_threads.get(target_id)

    if ta is not None:
        finished = ta.join(timeout=30.0)
    else:
        finished = self.registry.wait_for_agent(target_id, timeout=30.0)

    if not finished:
        raise RuntimeError(
            f"WAIT timed out after 30 s waiting for agent {target_id}. "
            "Check for deadlocks."
        )


def _handle_halt(self, instruction):
    """
    0x37 HALT

    Graceful program termination.

    1. Join all spawned agent threads (with a 10 s safety timeout each).
    2. Transition main agent (Agent 0) to DEAD.
    3. Return from execute() — caller handles sys.exit if needed.
    """
    for agent_id, ta in list(self._agent_threads.items()):
        finished = ta.join(timeout=10.0)
        if not finished:
            import logging
            logging.getLogger(__name__).warning(
                "HALT: agent %d did not finish within 10 s — killing.",
                agent_id,
            )
            ta.kill()

    self.registry.set_state(0, AgentState.DEAD)
    # Return to caller; interpreter's execute() returns normally.


# ─────────────────────────────────────────────────────────────────────────────
# Helper exception for AGENT_KILL SELF
# ─────────────────────────────────────────────────────────────────────────────

class _AgentSelfKill(Exception):
    """
    Raised inside an agent thread to signal clean self-termination.

    Caught by ThreadAgent._run() — treated as normal return, not an error.
    ThreadAgent._run() must be patched to catch this:

        def _run(self) -> None:
            try:
                self._fn(self._handle)
            except _AgentSelfKill:
                pass          # ← clean AGENT_KILL SELF — not an error
            except Exception:
                logger.exception("Uncaught exception in agent %d", self._agent_id)
            finally:
                self._finish()
    """


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Wire into the dispatch loop
# ─────────────────────────────────────────────────────────────────────────────
#
# If your dispatch is a dict:
#
#   DISPATCH = {
#       ...existing opcodes...
#       0x01: self._handle_spawn,
#       0x02: self._handle_agent_kill,
#       0x37: self._handle_halt,
#   }
#
# If your dispatch is if/elif:
#
#   elif opcode == 0x01:
#       self._handle_spawn(instruction, resolve_label)
#   elif opcode == 0x02:
#       self._handle_agent_kill(instruction)
#   elif opcode == 0x37:
#       self._handle_halt(instruction)
#       return   # stop the execution loop
#
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Handle _AgentSelfKill in ThreadAgent._run()
# ─────────────────────────────────────────────────────────────────────────────
#
# In runtime/agent/thread_agent.py, update _run():
#
#   from runtime.interpreter import _AgentSelfKill  # avoids circular: use lazy import
#
#   def _run(self) -> None:
#       try:
#           self._fn(self._handle)
#       except Exception as e:
#           if type(e).__name__ == "_AgentSelfKill":
#               pass   # clean self-kill — not an error
#           else:
#               logger.exception("Uncaught exception in agent %d", self._agent_id)
#       finally:
#           self._finish()
#
# NOTE: We use type(e).__name__ to avoid a circular import.
# Alternatively, define _AgentSelfKill in runtime/agent/lifecycle.py
# (which has no dependencies) and import it from there in both places.
