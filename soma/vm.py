"""SOMA Virtual Machine — interprets .sombin bytecode.

Fixes vs original:
  - AgentState: Fv{} float-vector registers, ctx_nibble for CDBG
  - Data section: parsed as symbol-table + float32 payload (matches assembler)
  - MSG_SEND: imm & 0x8000 → load VEC from data section, enqueue as list
  - MSG_RECV: list inbox items stored in Fv{}
  - GOAL_SET / GOAL_CHECK / SOUL_QUERY: use Fv{} for proper float vectors
  - Phase II handlers: EMOT_TAG, DECAY_PROTECT, PREDICT_ERR, EMOT_RECALL,
                       SURPRISE_CALC, CTX_SWITCH — all inline, no external deps
  - SOM_INIT: always randomises (RANDOM was the only valid value anyway)
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


# ── SOM ───────────────────────────────────────────────────────────────────────

@dataclass
class SomNode:
    row: int
    col: int
    weights: List[float] = field(default_factory=lambda: [random.random() for _ in range(16)])


class SomMap:
    def __init__(self, rows: int, cols: int, dims: int = 16):
        self.rows  = rows
        self.cols  = cols
        self.dims  = dims
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
                h     = math.exp(-dist2 / (2 * sigma * sigma))
                node  = self.nodes[r][c]
                node.weights = [w + lr * h * (v - w) for w, v in zip(node.weights, vector)]


# ── Agent state ────────────────────────────────────────────────────────────────

class AgentState:
    RUNNING = "running"
    WAITING = "waiting"
    DEAD    = "dead"
    BLOCKED = "blocked"

    def __init__(self, agent_id: int, pc: int, parent_id: Optional[int] = None):
        self.agent_id  = agent_id
        self.parent_id = parent_id
        self.pc        = pc
        self.state     = self.RUNNING

        # Integer registers (R0-R15)
        self.R: List[int] = [0] * 16
        # Float-vector registers (R0-R15 as float[dims]) — for Phase II/III
        self.Fv: Dict[int, List[float]] = {}
        # Agent handles
        self.A: List[int] = [0] * 64
        # SOM state: S0=lr, S1=sigma, S2=epoch
        self.S: List[float] = [0.0] * 16
        self.S[0] = 0.1
        self.S[1] = 1.0

        # SOM position
        self.som_x = 0
        self.som_y = 0

        # Message inbox (items can be int or List[float])
        self.inbox: deque = deque()

        # Flags
        self.zero_flag = False

        # CDBG context nibble (Phase IV)
        self.ctx_nibble: int = 1   # default = AGENT context

        # Trace log
        self.log: List[str] = []


# ── VM ─────────────────────────────────────────────────────────────────────────

class VMError(Exception):
    pass


class SomaVM:
    def __init__(self, binary: bytes, *, max_steps: int = 100_000, trace: bool = False):
        if len(binary) < HEADER_SIZE or binary[:4] != MAGIC:
            raise VMError("Not a valid .sombin file")

        self.binary    = binary
        self.max_steps = max_steps
        self.trace     = trace

        (self.ver_major,)  = struct.unpack_from(">H", binary, 0x04)
        (self.ver_minor,)  = struct.unpack_from(">H", binary, 0x06)
        self.arch_target    = binary[0x08]
        self.som_rows       = binary[0x09]
        self.som_cols       = binary[0x0A]
        self.max_agents     = binary[0x0B]
        (self.code_offset,) = struct.unpack_from(">I", binary, 0x0C)
        (self.code_size,)   = struct.unpack_from(">I", binary, 0x10)
        (self.data_offset,) = struct.unpack_from(">I", binary, 0x14)
        (self.data_size,)   = struct.unpack_from(">I", binary, 0x18)
        (self.flags,)       = struct.unpack_from(">H", binary, 0x1E)

        self.code_start = self.code_offset
        self.code_end   = self.code_offset + self.code_size

        self.som = SomMap(max(self.som_rows, 2), max(self.som_cols, 2))

        # ── Parse data section (symbol-table + float32 payload) ───────────────
        self.data_symbols: Dict[str, dict] = {}   # name → {type, offset, count}
        self.data_payload: bytes = b''
        self._data_sym_names: List[str] = []
        if self.data_size > 0:
            self._parse_data_section()

        self._next_id  = 0
        self.agents: Dict[int, AgentState] = {}
        self._output: List[str] = []

        # Phase III: soul registry and terrain (lazy)
        self._soul_registry: dict = {}
        self._terrain_map:   dict = {}   # (r,c) → {valence_sum, count, is_virgin}

        # Spawn root agent
        self._spawn_agent(pc=0, parent_id=None)

    # ── Data section parser ───────────────────────────────────────────────────

    def _parse_data_section(self):
        """Parse symbol-table + float32 payload produced by assembler."""
        data = self.binary[self.data_offset: self.data_offset + self.data_size]
        if len(data) < 4:
            return
        off = 0
        try:
            (n,) = struct.unpack_from('>I', data, off); off += 4
            for _ in range(n):
                (nl,) = struct.unpack_from('>H', data, off); off += 2
                name  = data[off:off+nl].decode('utf-8', errors='replace'); off += nl
                (dtype, payload_off, count) = struct.unpack_from('>BII', data, off); off += 9
                self.data_symbols[name] = {
                    'type': dtype, 'offset': payload_off, 'count': count
                }
            self.data_payload   = data[off:]
            self._data_sym_names = list(self.data_symbols.keys())
        except (struct.error, UnicodeDecodeError):
            pass

    def _read_fvec(self, sym_index: int) -> List[float]:
        """Load a float32 vector from data_payload by symbol index."""
        if sym_index >= len(self._data_sym_names):
            return []
        name = self._data_sym_names[sym_index]
        info = self.data_symbols.get(name)
        if info is None:
            return []
        off   = info['offset']
        count = info['count']
        out   = []
        for i in range(count):
            if off + 4 <= len(self.data_payload):
                (v,) = struct.unpack_from('>f', self.data_payload, off)
                out.append(v)
                off += 4
        return out

    # ── Agent management ──────────────────────────────────────────────────────

    def _spawn_agent(self, pc: int, parent_id: Optional[int]) -> AgentState:
        aid   = self._next_id; self._next_id += 1
        agent = AgentState(aid, pc, parent_id)
        self.agents[aid] = agent
        return agent

    def _read_instr(self, agent: AgentState) -> Tuple[int,int,int,int,int,int]:
        file_off = self.code_start + agent.pc
        if file_off + 8 > len(self.binary):
            raise VMError(f"PC out of range: {agent.pc}")
        (word,)  = struct.unpack_from(">Q", self.binary, file_off)
        opcode   = (word >> 56) & 0xFF
        agent_id = (word >> 48) & 0xFF
        som_x    = (word >> 40) & 0xFF
        som_y    = (word >> 32) & 0xFF
        reg      = (word >> 16) & 0xFFFF
        imm      =  word        & 0xFFFF
        return opcode, agent_id, som_x, som_y, reg, imm

    def _get_reg(self, agent: AgentState, reg_code: int) -> int:
        if reg_code == 0xFF00: return agent.agent_id
        if reg_code == 0xFF01: return agent.parent_id or 0
        if reg_code == 0xFF02: return 0
        hi = (reg_code >> 8) & 0xFF
        lo =  reg_code       & 0xFF
        if hi == 0x00: return agent.R[lo]  if lo < 16 else 0
        if hi == 0x01: return agent.A[lo]  if lo < 64 else 0
        if hi == 0x02: return int(agent.S[lo]) if lo < 16 else 0
        return 0

    def _set_reg(self, agent: AgentState, reg_code: int, value: int):
        hi = (reg_code >> 8) & 0xFF
        lo =  reg_code       & 0xFF
        if hi == 0x00 and lo < 16:  agent.R[lo] = value & 0xFFFFFFFFFFFFFFFF
        elif hi == 0x01 and lo < 64: agent.A[lo] = value & 0xFFFFFFFFFFFFFFFF
        elif hi == 0x02 and lo < 16: agent.S[lo] = float(value)

    def _get_fvec(self, agent: AgentState, reg_code: int) -> List[float]:
        """Get float vector for a register (Phase II/III ops)."""
        lo = reg_code & 0xFF
        if lo in agent.Fv:
            return agent.Fv[lo]
        # Fallback: treat R[lo] as single-element float
        return [float(agent.R[lo] if lo < 16 else 0)]

    def _set_fvec(self, agent: AgentState, reg_code: int, vec: List[float]):
        lo = reg_code & 0xFF
        agent.Fv[lo] = list(vec)
        if lo < 16:
            agent.R[lo] = int(vec[0]) if vec else 0

    def _broadcast(self, sender_id: int, value):
        for aid, ag in self.agents.items():
            if aid != sender_id and ag.state != AgentState.DEAD:
                ag.inbox.append(value)

    def _send_to(self, dest_id: int, value):
        if dest_id == 0xFF02:
            self._broadcast(-1, value)
        elif dest_id in self.agents:
            self.agents[dest_id].inbox.append(value)

    # ── Terrain helpers (Phase II/III) ────────────────────────────────────────

    def _terrain_node(self, r: int, c: int) -> dict:
        key = (r, c)
        if key not in self._terrain_map:
            self._terrain_map[key] = {
                'valence_sum': 0.0, 'count': 0, 'is_virgin': True
            }
        return self._terrain_map[key]

    def _terrain_mark(self, r: int, c: int, valence: float, intensity: float = 1.0):
        node = self._terrain_node(r, c)
        node['valence_sum'] += valence * intensity
        node['count']       += 1
        node['is_virgin']    = False

    def _terrain_read(self, r: int, c: int) -> float:
        node = self._terrain_node(r, c)
        if node['is_virgin']:
            return 1.0
        if node['count'] == 0:
            return 0.0
        return node['valence_sum'] / node['count']

    # ── Soul helpers (Phase III) ───────────────────────────────────────────────

    def _get_soul(self, agent_id: int) -> dict:
        if agent_id not in self._soul_registry:
            self._soul_registry[agent_id] = {
                'goal':        None,   # List[float]
                'goal_dist':   1e9,
                'stall_count': 0,
                'STALL_THRESHOLD': 20,
                'generation':  0,
                'emot':        {},     # (r,c) → float valence
                'ctx':         1,      # CDBG context nibble
            }
        return self._soul_registry[agent_id]

    def _vec_dist(self, a: List[float], b: List[float]) -> float:
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a[:n], b[:n])))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> List[str]:
        step = 0
        while step < self.max_steps:
            runnable = [ag for ag in self.agents.values() if ag.state == AgentState.RUNNING]
            if not runnable:
                blocked = [ag for ag in self.agents.values() if ag.state == AgentState.BLOCKED]
                if blocked:
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

    # ── Single-step execution ─────────────────────────────────────────────────

    def _step(self, agent: AgentState) -> bool:
        try:
            opcode, enc_agent_id, som_x, som_y, reg, imm = self._read_instr(agent)
        except VMError:
            agent.state = AgentState.DEAD
            return False

        mnem = OPCODE_NAMES.get(opcode, f"UNK({opcode:#04x})")
        if self.trace:
            print(f"  [A{agent.agent_id}] pc={agent.pc:04x}  {mnem}"
                  f"  reg={decode_reg(reg)} imm={imm:#06x}")

        agent.pc += 8
        O = OPCODES

        # ── HALT ───────────────────────────────────────────────────────────────
        if opcode == O["HALT"]:
            agent.state = AgentState.DEAD
            for ag in self.agents.values():
                ag.state = AgentState.DEAD
            return True

        # ── NOP ────────────────────────────────────────────────────────────────
        elif opcode == O["NOP"]:
            pass

        # ── SPAWN ──────────────────────────────────────────────────────────────
        elif opcode == O["SPAWN"]:
            child = self._spawn_agent(pc=imm, parent_id=agent.agent_id)
            self._set_reg(agent, reg, child.agent_id)
            # Mirror child handle into A[] for MSG_SEND targeting
            lo = reg & 0xFF
            if 0 <= lo < 64:
                agent.A[lo] = child.agent_id

        # ── AGENT_KILL ─────────────────────────────────────────────────────────
        elif opcode == O["AGENT_KILL"]:
            target_id = self._get_reg(agent, reg)
            if reg == 0xFF00 or imm == 0xFE or target_id == agent.agent_id:
                agent.state = AgentState.DEAD
            elif imm == 0xFF:
                for ag in self.agents.values():
                    ag.state = AgentState.DEAD
            elif target_id in self.agents:
                self.agents[target_id].state = AgentState.DEAD

        # ── FORK ───────────────────────────────────────────────────────────────
        elif opcode == O["FORK"]:
            n = enc_agent_id if enc_agent_id > 0 else 1
            for _ in range(n):
                self._spawn_agent(pc=imm, parent_id=agent.agent_id)

        # ── MERGE ──────────────────────────────────────────────────────────────
        elif opcode == O["MERGE"]:
            pass

        # ── BARRIER ────────────────────────────────────────────────────────────
        elif opcode == O["BARRIER"]:
            pass

        # ── WAIT ───────────────────────────────────────────────────────────────
        elif opcode == O["WAIT"]:
            target_id = self._get_reg(agent, reg)
            if target_id in self.agents and self.agents[target_id].state != AgentState.DEAD:
                agent.pc -= 8  # retry

        # ── SOM_INIT ───────────────────────────────────────────────────────────
        elif opcode == O["SOM_INIT"]:
            for r in range(self.som.rows):
                for c in range(self.som.cols):
                    self.som.nodes[r][c].weights = [random.random() for _ in range(self.som.dims)]

        # ── SOM_MAP ────────────────────────────────────────────────────────────
        elif opcode == O["SOM_MAP"]:
            agent.som_x = som_x
            agent.som_y = som_y

        # ── SOM_BMU ────────────────────────────────────────────────────────────
        elif opcode == O["SOM_BMU"]:
            vec = self._get_fvec(agent, reg)
            if not vec:
                vec = [float(agent.R[i]) for i in range(min(self.som.dims, 16))]
            br, bc = self.som.bmu(vec)
            agent.som_x = br
            agent.som_y = bc
            self._set_reg(agent, reg, br * self.som.cols + bc)

        # ── SOM_TRAIN ──────────────────────────────────────────────────────────
        elif opcode == O["SOM_TRAIN"]:
            vec = self._get_fvec(agent, reg)
            if not vec:
                vec = [float(agent.R[i]) for i in range(min(self.som.dims, 16))]
            lr    = agent.S[0]
            sigma = agent.S[1]
            br, bc = self.som.bmu(vec)
            self.som.train(vec, br, bc, lr, sigma)
            agent.S[2] += 1

        # ── SOM_NBHD / WGHT_UPD ───────────────────────────────────────────────
        elif opcode in (O["SOM_NBHD"], O["WGHT_UPD"]):
            pass

        # ── SOM_WALK ───────────────────────────────────────────────────────────
        elif opcode == O["SOM_WALK"]:
            dr = random.choice([-1, 0, 1])
            dc = random.choice([-1, 0, 1])
            agent.som_x = max(0, min(self.som.rows - 1, agent.som_x + dr))
            agent.som_y = max(0, min(self.som.cols - 1, agent.som_y + dc))

        # ── SOM_ELECT ──────────────────────────────────────────────────────────
        elif opcode == O["SOM_ELECT"]:
            living = [aid for aid, ag in self.agents.items() if ag.state != AgentState.DEAD]
            leader = min(living) if living else 0
            self._set_reg(agent, reg, leader)

        # ── LR_DECAY ───────────────────────────────────────────────────────────
        elif opcode == O["LR_DECAY"]:
            rate    = imm / 1000.0
            agent.S[0] = max(0.001, agent.S[0] * (1.0 - rate))

        # ── MSG_SEND ───────────────────────────────────────────────────────────
        elif opcode == O["MSG_SEND"]:
            # Resolve target agent
            target_id = self._get_reg(agent, reg)   # reg encodes A<n> handle

            # Resolve payload
            if imm & 0x8000:
                # Data-section vector reference
                sym_index = imm & 0x7FFF
                payload   = self._read_fvec(sym_index)
            else:
                payload = imm

            if target_id == 0xFF02 or reg == 0xFF02:
                self._broadcast(agent.agent_id, payload)
            elif target_id == 0xFF01:
                if agent.parent_id is not None and agent.parent_id in self.agents:
                    self.agents[agent.parent_id].inbox.append(payload)
            else:
                self._send_to(target_id, payload)

        # ── MSG_RECV ───────────────────────────────────────────────────────────
        elif opcode == O["MSG_RECV"]:
            if agent.inbox:
                item = agent.inbox.popleft()
                if isinstance(item, list):
                    # Float vector — store in Fv and scalar R
                    self._set_fvec(agent, reg, item)
                else:
                    self._set_reg(agent, reg, int(item))
                agent.state = AgentState.RUNNING
            else:
                agent.state = AgentState.BLOCKED
                agent.pc -= 8  # retry

        # ── BROADCAST ──────────────────────────────────────────────────────────
        elif opcode == O["BROADCAST"]:
            self._broadcast(agent.agent_id, imm)

        # ── ACCUM ──────────────────────────────────────────────────────────────
        elif opcode == O.get("ACCUM", -1):
            pass

        # ── Phase III Curiosity / Terrain (no-op stubs in Python VM) ───────────
        elif opcode == O.get("TERRAIN_READ", 0x25):
            self._set_reg(agent, reg, 0)          # always returns 0 (unexplored)
        elif opcode == O.get("TERRAIN_MARK", 0x26):
            pass                                   # tag terrain — no-op in test VM
        elif opcode == O.get("GOAL_SET", 0x27):
            pass                                   # set goal vector — no-op
        elif opcode == O.get("META_SPAWN", 0x28):
            pass                                   # spawn mutated children — no-op
        elif opcode == O.get("EVOLVE", 0x29):
            pass                                   # select fittest child — no-op

        # ── Extended arithmetic (INC / DEC / OR / SHL / SHR / ZERO) ────────────
        elif opcode == O.get("INC", 0x44):
            self._set_reg(agent, reg, self._get_reg(agent, reg) + 1)
        elif opcode == O.get("DEC", 0x45):
            self._set_reg(agent, reg, self._get_reg(agent, reg) - 1)
        elif opcode == O.get("OR", 0x46):
            self._set_reg(agent, reg, int(self._get_reg(agent, reg)) | int(imm))
        elif opcode == O.get("SHL", 0x47):
            self._set_reg(agent, reg, int(self._get_reg(agent, reg)) << int(imm))
        elif opcode == O.get("SHR", 0x48):
            self._set_reg(agent, reg, int(self._get_reg(agent, reg)) >> int(imm))
        elif opcode == O.get("ZERO", 0x49):
            self._set_reg(agent, reg, 0)

        # ── Extended jumps (JGE / JNE / JLT / JLE) ─────────────────────────────
        elif opcode == O.get("JGE", 0x3A):
            if self._get_reg(agent, reg) >= 0:
                agent.pc = imm
                return False
        elif opcode == O.get("JNE", 0x3B):
            if self._get_reg(agent, reg) != 0:
                agent.pc = imm
                return False
        elif opcode == O.get("JLT", 0x3C):
            if self._get_reg(agent, reg) < 0:
                agent.pc = imm
                return False
        elif opcode == O.get("JLE", 0x3D):
            if self._get_reg(agent, reg) <= 0:
                agent.pc = imm
                return False

        # ── Control flow ───────────────────────────────────────────────────────
        elif opcode == O["JMP"]:
            agent.pc = imm

        elif opcode == O["JZ"]:
            if self._get_reg(agent, reg) == 0 or agent.zero_flag:
                agent.pc = imm

        elif opcode == O["JNZ"]:
            val = self._get_reg(agent, reg)
            if val != 0 and not agent.zero_flag:
                agent.pc = imm

        elif opcode == O["JEQ"]:
            a = self._get_reg(agent, reg)
            b = som_x   # second operand packed in som_x byte
            if a == b:
                agent.pc = imm

        elif opcode == O["JGT"]:
            a = self._get_reg(agent, reg)
            b = som_x
            if a > b:
                agent.pc = imm

        elif opcode == O["CALL"]:
            agent.R[15] = agent.pc
            agent.pc    = imm

        elif opcode == O["RET"]:
            agent.pc = agent.R[15]

        # ── Arithmetic ─────────────────────────────────────────────────────────
        elif opcode == O["ADD"]:
            self._set_reg(agent, reg, self._get_reg(agent, reg) + imm)

        elif opcode == O["SUB"]:
            self._set_reg(agent, reg, self._get_reg(agent, reg) - imm)

        elif opcode == O["MUL"]:
            self._set_reg(agent, reg, self._get_reg(agent, reg) * imm)

        elif opcode == O["MOV"]:
            self._set_reg(agent, reg, imm)

        elif opcode == O.get("CMP", -1):
            agent.zero_flag = (self._get_reg(agent, reg) == imm)

        elif opcode in (O["DOT"], O["NORM"]):
            pass

        # ── LOAD / STORE ───────────────────────────────────────────────────────
        elif opcode == O["LOAD"]:
            # imm = data symbol index (0-based ordinal)
            if imm < len(self._data_sym_names):
                vec = self._read_fvec(imm)
                if vec:
                    self._set_fvec(agent, reg, vec)
                    self._set_reg(agent, reg, int(vec[0]))

        elif opcode == O["STORE"]:
            pass  # read-only in test VM

        elif opcode == O.get("TRAP", 0x38):
            pass  # software trap / breakpoint — no-op in Python VM

        # ────────────────────────────────────────────────────────────────────────
        # Phase II: Emotional memory (fully inline — no external runtime deps)
        # ────────────────────────────────────────────────────────────────────────

        elif opcode == O.get("EMOT_TAG", -1):
            # EMOT_TAG reg, imm_intensity
            # Tags current SOM node with valence from register and intensity from imm
            vec      = self._get_fvec(agent, reg)
            valence  = vec[0] if vec else float(self._get_reg(agent, reg))
            intensity = imm / 32767.0 if imm else 0.5
            r, c     = agent.som_x, agent.som_y
            self._terrain_mark(r, c, valence, intensity)
            soul = self._get_soul(agent.agent_id)
            soul['emot'][(r, c)] = valence
            agent.log.append(f"EMOT_TAG ({r},{c}) val={valence:.3f} int={intensity:.3f}")

        elif opcode == O.get("DECAY_PROTECT", -1):
            # DECAY_PROTECT imm_cycles — stub: log only (no weight decay in test VM)
            agent.log.append(f"DECAY_PROTECT ({agent.som_x},{agent.som_y}) cycles={imm}")

        elif opcode == O.get("PREDICT_ERR", -1):
            # PREDICT_ERR dst — compare current SOM node weights to goal
            soul    = self._get_soul(agent.agent_id)
            goal    = soul.get('goal')
            weights = self.som.nodes[agent.som_x][agent.som_y].weights
            if goal:
                dist = self._vec_dist(goal, weights)
                err  = min(dist / math.sqrt(len(goal)), 1.0)
            else:
                err = 0.0
            self._set_reg(agent, reg, int(err * 65535))
            self._set_fvec(agent, reg, [err])

        elif opcode == O.get("EMOT_RECALL", -1):
            # EMOT_RECALL dst — retrieve emotional tag at current node
            val = self._terrain_read(agent.som_x, agent.som_y)
            self._set_reg(agent, reg, int(val * 65535))
            self._set_fvec(agent, reg, [val])

        elif opcode == O.get("SURPRISE_CALC", -1):
            # SURPRISE_CALC dst, src — distance between two float vectors
            vec_a = self._get_fvec(agent, reg)
            vec_b = self._get_fvec(agent, imm & 0xFF)  # src in low byte of imm
            if vec_a and vec_b:
                dist = self._vec_dist(vec_a, vec_b)
                surprise = min(dist / math.sqrt(max(len(vec_a), 1)), 1.0)
            else:
                surprise = 0.0
            self._set_reg(agent, reg, int(surprise * 65535))
            self._set_fvec(agent, reg, [surprise])

        # ────────────────────────────────────────────────────────────────────────
        # Phase III: Curiosity
        # ────────────────────────────────────────────────────────────────────────

        elif opcode == O.get("GOAL_SET", -1):
            # GOAL_SET reg — set goal from float-vector register
            goal = self._get_fvec(agent, reg)
            if not goal:
                goal = [float(agent.R[i]) for i in range(self.som.dims)]
            soul = self._get_soul(agent.agent_id)
            soul['goal']        = goal
            soul['goal_dist']   = 1e9
            soul['stall_count'] = 0
            soul['generation'] += 1
            agent.log.append(f"GOAL_SET generation={soul['generation']}")

        elif opcode == O.get("GOAL_CHECK", -1):
            soul    = self._get_soul(agent.agent_id)
            weights = self.som.nodes[agent.som_x][agent.som_y].weights
            goal    = soul.get('goal')
            if goal:
                dist = self._vec_dist(goal, weights)
                improvement = soul['goal_dist'] - dist
                if improvement < 0.001:
                    soul['stall_count'] += 1
                else:
                    soul['stall_count'] = 0
                soul['goal_dist'] = dist
                curious = soul['stall_count'] > soul['STALL_THRESHOLD']
                self._set_reg(agent, reg, int(dist * 65535))
                agent.zero_flag = curious
            else:
                self._set_reg(agent, reg, 0)

        elif opcode == O.get("SOUL_QUERY", -1):
            soul    = self._get_soul(agent.agent_id)
            goal    = soul.get('goal')
            weights = self.som.nodes[agent.som_x][agent.som_y].weights
            if goal:
                dist = self._vec_dist(goal, weights)
                dims = max(len(goal), 1)
                similarity = 1.0 - min(dist / math.sqrt(dims), 1.0)
                self._set_reg(agent, reg, int(similarity * 65535))
                agent.zero_flag = similarity < 0.5
            else:
                self._set_reg(agent, reg, 0)

        elif opcode == O.get("META_SPAWN", -1):
            n         = max(1, enc_agent_id if enc_agent_id > 0 else imm & 0xFF)
            target_pc = imm
            soul      = self._get_soul(agent.agent_id)
            goal      = soul.get('goal') or []
            for _ in range(n):
                # Mutate goal: add small Gaussian noise
                mutated = [g + random.gauss(0, 0.1) for g in goal]
                child   = self._spawn_agent(pc=target_pc, parent_id=agent.agent_id)
                child_soul = self._get_soul(child.agent_id)
                child_soul['goal'] = mutated
                # Inherit emotional terrain knowledge
                child_soul['emot'] = dict(soul['emot'])
            agent.log.append(f"META_SPAWN n={n} pc={target_pc}")

        elif opcode == O.get("EVOLVE", -1):
            children = [
                ag for ag in self.agents.values()
                if ag.parent_id == agent.agent_id and ag.state != AgentState.DEAD
            ]
            if children:
                def _dist(ag):
                    s = self._soul_registry.get(ag.agent_id, {})
                    g = s.get('goal')
                    if not g:
                        return 1.0
                    w = self.som.nodes[ag.som_x][ag.som_y].weights
                    return self._vec_dist(g, w)
                winner = min(children, key=_dist)
                self._set_reg(agent, reg, winner.agent_id)
                # Winner inherits parent's soul
                w_soul = self._get_soul(winner.agent_id)
                p_soul = self._get_soul(agent.agent_id)
                for key in ('emot',):
                    w_soul[key] = dict(p_soul.get(key, {}))
                agent.log.append(f"EVOLVE winner=A{winner.agent_id}")

        elif opcode == O.get("INTROSPECT", -1):
            soul = self._get_soul(agent.agent_id)
            snap = {
                'goal_dist':   soul.get('goal_dist', 0.0),
                'stall_count': soul.get('stall_count', 0),
                'generation':  soul.get('generation', 0),
                'emot_nodes':  len(soul.get('emot', {})),
            }
            self._output.append(f"[A{agent.agent_id}] introspect={snap}")

        elif opcode == O.get("TERRAIN_READ", -1):
            val = self._terrain_read(agent.som_x, agent.som_y)
            self._set_reg(agent, reg, int(val * 65535))
            agent.zero_flag = (val >= 1.0)   # virgin territory

        elif opcode == O.get("TERRAIN_MARK", -1):
            soul    = self._get_soul(agent.agent_id)
            fv      = self._get_fvec(agent, reg)
            valence = fv[0] if fv else float(self._get_reg(agent, reg)) / 65535.0
            self._terrain_mark(agent.som_x, agent.som_y, valence, intensity=0.8)

        elif opcode == O.get("SOUL_INHERIT", -1):
            src_id  = self._get_reg(agent, reg)
            p_soul  = self._get_soul(src_id)
            t_soul  = self._get_soul(agent.agent_id)
            # Inherit goal and emotional memory; reset stall
            t_soul['goal']        = list(p_soul.get('goal') or [])
            t_soul['emot']        = dict(p_soul.get('emot', {}))
            t_soul['stall_count'] = 0
            agent.log.append(f"SOUL_INHERIT from A{src_id}")

        elif opcode == O.get("GOAL_STALL", -1):
            soul = self._get_soul(agent.agent_id)
            if soul['stall_count'] > soul['STALL_THRESHOLD']:
                agent.pc = imm

        # ────────────────────────────────────────────────────────────────────────
        # Phase IV: CDBG
        # ────────────────────────────────────────────────────────────────────────

        elif opcode == O.get("CDBG_EMIT", -1):
            try:
                from soma.cdbg import Encoder
                frame = Encoder.agent(agent.agent_id).encode()
                self._output.append(f"[CDBG] agent={agent.agent_id} frame={frame.hex()}")
            except ImportError:
                # Fallback: minimal 5-byte frame without full cdbg module
                ctx   = agent.ctx_nibble & 0xF
                aid   = agent.agent_id & 0xFFFFFF
                b     = [(ctx << 4),
                         (aid >> 16) & 0xFF,
                         (aid >>  8) & 0xFF,
                          aid        & 0xFF,
                         0x00]
                self._output.append(
                    f"[CDBG] agent={agent.agent_id} frame={''.join(f'{x:02X}' for x in b)}")

        elif opcode == O.get("CDBG_RECV", -1):
            if agent.inbox:
                raw = agent.inbox.popleft()
                if isinstance(raw, (bytes, bytearray)) and len(raw) == 5:
                    try:
                        from soma.cdbg import Frame as CDBGFrame
                        f = CDBGFrame.decode(raw)
                        agent.log.append(f"CDBG_RECV ctx={f.ctx.name}")
                    except (ImportError, ValueError):
                        pass

        elif opcode == O.get("CTX_SWITCH", -1):
            # CTX_SWITCH imm — set active CDBG context nibble
            agent.ctx_nibble = imm & 0xF
            soul = self._get_soul(agent.agent_id)
            soul['ctx'] = agent.ctx_nibble

        return False  # not halted

    @property
    def output(self) -> List[str]:
        return self._output
