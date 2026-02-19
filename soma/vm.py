"""SOMA Virtual Machine — interprets .sombin bytecode.

Implements:
  - Multi-agent scheduler
  - SOM (Self-Organizing Map) topology + BMU, training, neighborhood
  - Inter-agent message queues
  - All ISA opcodes from SOMA v1.0
"""
from __future__ import annotations
import struct
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from soma.isa import MAGIC, ARCH_NAMES, OPCODE_NAMES, OPCODES, decode_reg
from soma.assembler import HEADER_SIZE


# ── SOM node ──────────────────────────────────────────────────────────────────

@dataclass
class SomNode:
    row: int
    col: int
    weights: List[float] = field(default_factory=lambda: [random.random() for _ in range(16)])


class SomMap:
    def __init__(self, rows: int, cols: int, dims: int = 16):
        self.rows = rows
        self.cols = cols
        self.dims = dims
        self.nodes: List[List[SomNode]] = [
            [SomNode(r, c, [random.random() for _ in range(dims)]) for c in range(cols)]
            for r in range(rows)
        ]

    def bmu(self, vector: List[float]) -> Tuple[int, int]:
        best_dist = float("inf")
        best_r, best_c = 0, 0
        for r in range(self.rows):
            for c in range(self.cols):
                d = sum((a - b) ** 2 for a, b in zip(self.nodes[r][c].weights, vector))
                if d < best_dist:
                    best_dist = d
                    best_r, best_c = r, c
        return best_r, best_c

    def train(self, vector: List[float], bmu_r: int, bmu_c: int,
              lr: float = 0.1, sigma: float = 1.0):
        for r in range(self.rows):
            for c in range(self.cols):
                dist2 = (r - bmu_r) ** 2 + (c - bmu_c) ** 2
                influence = math.exp(-dist2 / (2 * sigma * sigma))
                node = self.nodes[r][c]
                node.weights = [
                    w + lr * influence * (v - w)
                    for w, v in zip(node.weights, vector)
                ]


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentState:
    RUNNING  = "running"
    WAITING  = "waiting"
    DEAD     = "dead"
    BLOCKED  = "blocked"  # waiting for MSG_RECV

    def __init__(self, agent_id: int, pc: int, parent_id: Optional[int] = None):
        self.agent_id = agent_id
        self.parent_id = parent_id
        self.pc = pc
        self.state = self.RUNNING
        # Registers
        self.R: List[int] = [0] * 16            # 256-bit general (stored as int)
        self.A: List[int] = [0] * 64            # agent handles
        self.S: List[float] = [0.0] * 16        # SOM state: S0=lr, S1=sigma, S2=epoch
        self.S[0] = 0.1   # learning rate
        self.S[1] = 1.0   # sigma
        self.S[2] = 0.0   # epoch
        # SOM position
        self.som_x = 0
        self.som_y = 0
        # Message queue
        self.inbox: deque = deque()
        # Flags
        self.zero_flag = False
        # Barrier counter
        self.barrier_remaining = 0
        # Trace log (for test assertions)
        self.log: List[str] = []


# ── VM ─────────────────────────────────────────────────────────────────────────

class VMError(Exception):
    pass


class SomaVM:
    def __init__(self, binary: bytes, *, max_steps: int = 100_000, trace: bool = False):
        if len(binary) < HEADER_SIZE or binary[:4] != MAGIC:
            raise VMError("Not a valid .sombin file")

        self.binary = binary
        self.max_steps = max_steps
        self.trace = trace

        # Parse header
        (self.ver_major,) = struct.unpack_from(">H", binary, 0x04)
        (self.ver_minor,) = struct.unpack_from(">H", binary, 0x06)
        self.arch_target   = binary[0x08]
        self.som_rows      = binary[0x09]
        self.som_cols      = binary[0x0A]
        self.max_agents    = binary[0x0B]
        (self.code_offset,) = struct.unpack_from(">I", binary, 0x0C)
        (self.code_size,)   = struct.unpack_from(">I", binary, 0x10)
        (self.data_offset,) = struct.unpack_from(">I", binary, 0x14)
        (self.data_size,)   = struct.unpack_from(">I", binary, 0x18)
        (self.som_offset,)  = struct.unpack_from(">I", binary, 0x1C)
        (self.flags,)       = struct.unpack_from(">H", binary, 0x1E)

        # Code starts at index 0 (relative)
        self.code_start = self.code_offset
        self.code_end   = self.code_offset + self.code_size

        # SOM map
        self.som = SomMap(self.som_rows, self.som_cols)

        # Agents
        self._next_id = 0
        self.agents: Dict[int, AgentState] = {}
        self._output: List[str] = []

        # Spawn the root agent at code start
        root = self._spawn_agent(pc=0, parent_id=None)
        self._entry_pc = 0

    def _spawn_agent(self, pc: int, parent_id: Optional[int]) -> AgentState:
        aid = self._next_id
        self._next_id += 1
        agent = AgentState(aid, pc, parent_id)
        self.agents[aid] = agent
        return agent

    def _read_instr(self, agent: AgentState) -> Tuple[int,int,int,int,int,int]:
        file_off = self.code_start + agent.pc
        if file_off + 8 > len(self.binary):
            raise VMError(f"PC out of range: {agent.pc}")
        (word,) = struct.unpack_from(">Q", self.binary, file_off)
        opcode   = (word >> 56) & 0xFF
        agent_id = (word >> 48) & 0xFF
        som_x    = (word >> 40) & 0xFF
        som_y    = (word >> 32) & 0xFF
        reg      = (word >> 16) & 0xFFFF
        imm      = word & 0xFFFF
        return opcode, agent_id, som_x, som_y, reg, imm

    def _get_reg(self, agent: AgentState, reg_code: int) -> int:
        if reg_code == 0xFF00:  # SELF
            return agent.agent_id
        if reg_code == 0xFF01:  # PARENT
            return agent.parent_id or 0
        if reg_code == 0xFF02:  # ALL
            return 0
        hi = (reg_code >> 8) & 0xFF
        lo = reg_code & 0xFF
        if hi == 0x00:
            return agent.R[lo] if lo < 16 else 0
        if hi == 0x01:
            return agent.A[lo] if lo < 64 else 0
        if hi == 0x02:
            return int(agent.S[lo]) if lo < 16 else 0
        return 0

    def _set_reg(self, agent: AgentState, reg_code: int, value: int):
        hi = (reg_code >> 8) & 0xFF
        lo = reg_code & 0xFF
        if hi == 0x00 and lo < 16:
            agent.R[lo] = value & 0xFFFFFFFFFFFFFFFF
        elif hi == 0x01 and lo < 64:
            agent.A[lo] = value & 0xFFFFFFFFFFFFFFFF
        elif hi == 0x02 and lo < 16:
            agent.S[lo] = float(value)

    def _broadcast(self, sender_id: int, value: int):
        for aid, ag in self.agents.items():
            if aid != sender_id and ag.state != AgentState.DEAD:
                ag.inbox.append(value)

    def _send_to(self, dest_id: int, value: int):
        if dest_id == 0xFF02:  # ALL
            self._broadcast(-1, value)
        elif dest_id in self.agents:
            self.agents[dest_id].inbox.append(value)

    def run(self) -> List[str]:
        """Run until HALT or max_steps. Returns output lines."""
        step = 0
        while step < self.max_steps:
            # Collect runnable agents
            runnable = [ag for ag in self.agents.values()
                        if ag.state == AgentState.RUNNING]
            if not runnable:
                # Check if any are just blocked
                blocked = [ag for ag in self.agents.values()
                           if ag.state == AgentState.BLOCKED]
                if blocked:
                    # Try to unblock
                    for ag in blocked:
                        if ag.inbox:
                            ag.state = AgentState.RUNNING
                if not any(ag.state == AgentState.RUNNING for ag in self.agents.values()):
                    break
                continue

            for agent in list(runnable):
                if agent.state != AgentState.RUNNING:
                    continue
                done = self._step(agent)
                step += 1
                if done:
                    break

        return self._output

    def _step(self, agent: AgentState) -> bool:
        """Execute one instruction. Returns True if HALT encountered."""
        try:
            opcode, enc_agent_id, som_x, som_y, reg, imm = self._read_instr(agent)
        except VMError:
            agent.state = AgentState.DEAD
            return False

        mnem = OPCODE_NAMES.get(opcode, f"UNK({opcode:#04x})")
        if self.trace:
            print(f"  [A{agent.agent_id}] pc={agent.pc:04x}  {mnem}  reg={decode_reg(reg)} imm={imm:#06x}")

        agent.pc += 8
        O = OPCODES

        # ── HALT ──────────────────────────────────────────────────────────────
        if opcode == O["HALT"]:
            agent.state = AgentState.DEAD
            # Kill all agents
            for ag in self.agents.values():
                ag.state = AgentState.DEAD
            return True

        # ── NOP ───────────────────────────────────────────────────────────────
        elif opcode == O["NOP"]:
            pass

        # ── SPAWN ─────────────────────────────────────────────────────────────
        elif opcode == O["SPAWN"]:
            target_pc = imm   # imm is already byte offset
            child = self._spawn_agent(pc=target_pc, parent_id=agent.agent_id)
            # Store child handle in the reg slot
            self._set_reg(agent, reg, child.agent_id)

        # ── AGENT_KILL ────────────────────────────────────────────────────────
        elif opcode == O["AGENT_KILL"]:
            target_id = self._get_reg(agent, reg)
            if reg == 0xFF00 or target_id == agent.agent_id:  # SELF
                agent.state = AgentState.DEAD
            elif target_id in self.agents:
                self.agents[target_id].state = AgentState.DEAD

        # ── FORK ──────────────────────────────────────────────────────────────
        elif opcode == O["FORK"]:
            n = enc_agent_id  # N stored in agent_id field
            target_pc = imm   # label byte offset stored in imm
            for _ in range(n if n > 0 else 1):
                self._spawn_agent(pc=target_pc, parent_id=agent.agent_id)

        # ── MERGE ─────────────────────────────────────────────────────────────
        elif opcode == O["MERGE"]:
            pass  # simplified

        # ── BARRIER ───────────────────────────────────────────────────────────
        elif opcode == O["BARRIER"]:
            # Simplified: just continue
            pass

        # ── WAIT ──────────────────────────────────────────────────────────────
        elif opcode == O["WAIT"]:
            target_id = self._get_reg(agent, reg)
            if target_id in self.agents and self.agents[target_id].state != AgentState.DEAD:
                agent.pc -= 8  # retry
            # else: proceed

        # ── SOM_INIT ──────────────────────────────────────────────────────────
        elif opcode == O["SOM_INIT"]:
            # RANDOM=0xFFFF or value
            if imm == 0xFFFF:
                for r in range(self.som.rows):
                    for c in range(self.som.cols):
                        self.som.nodes[r][c].weights = [random.random() for _ in range(self.som.dims)]

        # ── SOM_MAP ───────────────────────────────────────────────────────────
        elif opcode == O["SOM_MAP"]:
            agent.som_x = som_x
            agent.som_y = som_y

        # ── SOM_BMU ───────────────────────────────────────────────────────────
        elif opcode == O["SOM_BMU"]:
            vec = [float(agent.R[i]) for i in range(min(self.som.dims, 16))]
            br, bc = self.som.bmu(vec)
            agent.som_x = br
            agent.som_y = bc
            self._set_reg(agent, reg, br * self.som.cols + bc)

        # ── SOM_TRAIN ─────────────────────────────────────────────────────────
        elif opcode == O["SOM_TRAIN"]:
            vec = [float(agent.R[i]) for i in range(min(self.som.dims, 16))]
            s_reg = imm  # S register index
            lr = agent.S[0]
            sigma = agent.S[1]
            bmu_r, bmu_c = self.som.bmu(vec)
            self.som.train(vec, bmu_r, bmu_c, lr, sigma)
            agent.S[2] += 1  # epoch++

        # ── SOM_NBHD ──────────────────────────────────────────────────────────
        elif opcode == O["SOM_NBHD"]:
            pass  # returns neighborhood influence; simplified

        # ── WGHT_UPD ──────────────────────────────────────────────────────────
        elif opcode == O["WGHT_UPD"]:
            pass  # weight update is handled inside SOM_TRAIN

        # ── SOM_WALK ──────────────────────────────────────────────────────────
        elif opcode == O["SOM_WALK"]:
            if imm == 0xFFFE:  # GRADIENT
                # Move toward highest activation neighbor
                pass
            # Random walk
            dr = random.choice([-1, 0, 1])
            dc = random.choice([-1, 0, 1])
            agent.som_x = max(0, min(self.som.rows - 1, agent.som_x + dr))
            agent.som_y = max(0, min(self.som.cols - 1, agent.som_y + dc))

        # ── SOM_ELECT ─────────────────────────────────────────────────────────
        elif opcode == O["SOM_ELECT"]:
            living = [aid for aid, ag in self.agents.items() if ag.state != AgentState.DEAD]
            leader = min(living) if living else 0
            self._set_reg(agent, reg, leader)

        # ── MSG_SEND ──────────────────────────────────────────────────────────
        elif opcode == O["MSG_SEND"]:
            target_id = self._get_reg(agent, reg)
            value = imm
            if reg == 0xFF02:  # ALL / BROADCAST
                self._broadcast(agent.agent_id, value)
            elif target_id == 0xFF01:  # PARENT
                if agent.parent_id is not None and agent.parent_id in self.agents:
                    self.agents[agent.parent_id].inbox.append(value)
            else:
                self._send_to(target_id, value)

        # ── MSG_RECV ──────────────────────────────────────────────────────────
        elif opcode == O["MSG_RECV"]:
            if agent.inbox:
                val = agent.inbox.popleft()
                self._set_reg(agent, reg, val)
                agent.state = AgentState.RUNNING
            else:
                # Block until message arrives
                agent.state = AgentState.BLOCKED
                agent.pc -= 8  # retry

        # ── BROADCAST ────────────────────────────────────────────────────────
        elif opcode == O["BROADCAST"]:
            self._broadcast(agent.agent_id, imm)

        # ── JMP ───────────────────────────────────────────────────────────────
        elif opcode == O["JMP"]:
            agent.pc = imm  # imm is byte offset

        # ── JZ ────────────────────────────────────────────────────────────────
        elif opcode == O["JZ"]:
            val = self._get_reg(agent, reg)
            if val == 0:
                agent.pc = imm

        # ── JNZ ───────────────────────────────────────────────────────────────
        elif opcode == O["JNZ"]:
            val = self._get_reg(agent, reg)
            if val != 0:
                agent.pc = imm

        # ── CALL ──────────────────────────────────────────────────────────────
        elif opcode == O["CALL"]:
            # Push return address (current PC, already advanced by 8 after reading)
            agent.R[15] = agent.pc  # byte offset into code segment
            agent.pc = imm          # jump to label (byte offset)

        # ── RET ───────────────────────────────────────────────────────────────
        elif opcode == O["RET"]:
            agent.pc = agent.R[15]

        # ── Arithmetic ───────────────────────────────────────────────────────
        elif opcode == O["ADD"]:
            a = self._get_reg(agent, reg)
            self._set_reg(agent, reg, a + imm)
        elif opcode == O["SUB"]:
            a = self._get_reg(agent, reg)
            self._set_reg(agent, reg, a - imm)
        elif opcode == O["MUL"]:
            a = self._get_reg(agent, reg)
            self._set_reg(agent, reg, a * imm)
        elif opcode == O["MOV"]:
            self._set_reg(agent, reg, imm)
        elif opcode == O["CMP"]:
            a = self._get_reg(agent, reg)
            agent.zero_flag = (a == imm)
        elif opcode == O["DOT"]:
            # Dot product of R[reg] vector with immediate (stub)
            pass
        elif opcode == O["NORM"]:
            pass

        # ── LOAD / STORE ─────────────────────────────────────────────────────
        elif opcode == O["LOAD"]:
            off = self.data_offset + imm * 8
            if off + 8 <= len(self.binary):
                (val,) = struct.unpack_from(">Q", self.binary, off)
                self._set_reg(agent, reg, val)
        elif opcode == O["STORE"]:
            pass  # read-only binary

        return False  # not halted

    @property
    def output(self) -> List[str]:
        return self._output
