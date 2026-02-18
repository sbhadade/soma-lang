#!/usr/bin/env python3
"""
SOMA Runtime — v2.0 Production
Phase 1: Real DATA section loading (VEC, WGHT, COORD, INT, FLOAT, BYTES)
Phase 2: Real multi-agent threading via Python threading.Thread
Phase 3: JIT — hot-path compilation to native Python bytecode via compile()
"""

import sys
import struct
import math
import threading
import time
import random
import types
import io
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADER_SIZE    = 32
MAX_CALL_DEPTH = 1024
MEM_FLAG       = 0x80000000   # bit 31 set in imm = data-section reference

DTYPE_INT=0; DTYPE_FLOAT=1; DTYPE_VEC=2; DTYPE_WGHT=3; DTYPE_COORD=4; DTYPE_BYTES=5
DTYPE_NAME = {0:'INT',1:'FLOAT',2:'VEC',3:'WGHT',4:'COORD',5:'BYTES'}


# ---------------------------------------------------------------------------
# Phase 1 — DATA section loader
# ---------------------------------------------------------------------------

class DataMemory:
    """
    Parses the binary data section and provides named symbol access.
    Supports read and write of individual floats and vectors.
    """

    def __init__(self, data_bytes: bytes):
        self.symbols  = {}   # name -> {'type','offset','count'}
        self.payload  = bytearray()
        if data_bytes:
            self._parse(data_bytes)

    def _parse(self, data: bytes):
        off = 0
        (num_syms,) = struct.unpack_from('>I', data, off); off += 4

        for _ in range(num_syms):
            (name_len,) = struct.unpack_from('>H', data, off); off += 2
            name = data[off:off+name_len].decode('utf-8'); off += name_len
            (dtype, payload_off, count) = struct.unpack_from('>BII', data, off); off += 9
            self.symbols[name] = {
                'type': dtype, 'offset': payload_off, 'count': count
            }

        # Rest is float32 payload
        self.payload = bytearray(data[off:])

    def read_float(self, payload_off: int) -> float:
        if payload_off + 4 > len(self.payload):
            return 0.0
        (v,) = struct.unpack_from('>f', self.payload, payload_off)
        return v

    def write_float(self, payload_off: int, val: float):
        if payload_off + 4 <= len(self.payload):
            struct.pack_into('>f', self.payload, payload_off, float(val))

    def read_vec(self, name: str) -> list:
        """Read all float values for a named symbol."""
        if name not in self.symbols:
            return []
        s = self.symbols[name]
        return [self.read_float(s['offset'] + i*4) for i in range(s['count'])]

    def write_vec(self, name: str, values: list):
        if name not in self.symbols:
            return
        s = self.symbols[name]
        for i, v in enumerate(values[:s['count']]):
            self.write_float(s['offset'] + i*4, v)

    def read_by_offset(self, payload_off: int, count: int = 1) -> list:
        return [self.read_float(payload_off + i*4) for i in range(count)]

    def write_by_offset(self, payload_off: int, values: list):
        for i, v in enumerate(values):
            self.write_float(payload_off + i*4, v)

    def dump(self):
        print("  [DATA] Loaded symbols:")
        for name, s in self.symbols.items():
            vals = self.read_by_offset(s['offset'], min(s['count'], 4))
            ellipsis = '...' if s['count'] > 4 else ''
            print(f"    {name:20s} {DTYPE_NAME[s['type']]:6s}  "
                  f"count={s['count']:4d}  [{', '.join(f'{v:.4f}' for v in vals)}{ellipsis}]")


# ---------------------------------------------------------------------------
# SOM — real implementation
# ---------------------------------------------------------------------------

class SOM:
    def __init__(self, rows=16, cols=16, dim=8, lr=0.5, nbhd=3.0):
        self.rows = rows
        self.cols = cols
        self.dim  = dim
        self.lr   = lr
        self.nbhd_radius = nbhd
        self.weights = [
            [[random.random() for _ in range(dim)] for _ in range(cols)]
            for _ in range(rows)
        ]
        self._lock = threading.Lock()

    def bmu(self, vec: list) -> tuple:
        if not vec: return (0, 0)
        best, br, bc = float('inf'), 0, 0
        for r in range(self.rows):
            for c in range(self.cols):
                w = self.weights[r][c]
                d = sum((a-b)**2 for a,b in zip(vec, w[:len(vec)]))
                if d < best:
                    best = d; br = r; bc = c
        return br, bc

    def train(self, vec: list, bmu_r: int, bmu_c: int):
        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    d2 = (r-bmu_r)**2 + (c-bmu_c)**2
                    h  = math.exp(-d2 / (2.0 * self.nbhd_radius**2))
                    w  = self.weights[r][c]
                    for k in range(min(len(vec), self.dim)):
                        w[k] += self.lr * h * (vec[k] - w[k])

    def decay_lr(self, rate: float):
        self.lr = max(0.001, self.lr * (1.0 - rate))

    def decay_nbhd(self, rate: float):
        self.nbhd_radius = max(0.5, self.nbhd_radius * (1.0 - rate))

    def sense(self, r: int, c: int) -> float:
        w = self.weights[r][c]
        return sum(w) / len(w) if w else 0.0

    def elect_leader(self) -> tuple:
        best, br, bc = -1.0, 0, 0
        for r in range(self.rows):
            for c in range(self.cols):
                v = self.sense(r, c)
                if v > best:
                    best = v; br = r; bc = c
        return br, bc

    def init_random(self):
        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    self.weights[r][c] = [random.random() for _ in range(self.dim)]

    def node_distance(self, r1, c1, r2, c2) -> float:
        return math.sqrt((r1-r2)**2 + (c1-c2)**2)


# ---------------------------------------------------------------------------
# Phase 3 — JIT compiler
# ---------------------------------------------------------------------------

class JITCache:
    """
    Compiles frequently-executed straight-line basic blocks into
    native Python functions via compile() + exec().
    A basic block is a contiguous sequence of non-branch instructions.
    """
    HOT_THRESHOLD = 10   # executions before compiling

    def __init__(self):
        self.hit_count  = defaultdict(int)
        self.cache      = {}   # pc -> compiled function

    def record(self, pc: int) -> bool:
        """Returns True if this block is now hot and should be compiled."""
        self.hit_count[pc] += 1
        return self.hit_count[pc] == self.HOT_THRESHOLD

    def is_cached(self, pc: int) -> bool:
        return pc in self.cache

    def get(self, pc):
        return self.cache.get(pc)

    def compile_block(self, pc: int, code: bytes, opcode_map: dict) -> 'callable | None':
        """
        Decompile a basic block starting at `pc` into Python source,
        then compile it to a native function.
        Returns the function, or None if block is not safe to JIT.
        """
        lines  = ['def _jit_block(R, S, mem, som):']
        offset = pc
        jitted = 0

        while offset + 8 <= len(code):
            raw     = struct.unpack_from('>Q', code, offset)[0]
            op      = (raw >> 56) & 0xFF
            src_reg = (raw >> 40) & 0xFF
            dst_reg = (raw >> 32) & 0xFF
            imm     = raw & 0xFFFFFFFF

            # Only JIT simple arithmetic and register moves
            if op == 0x40:   # MOV
                if imm & MEM_FLAG:
                    break    # memory ref — stop block
                lines.append(f'  R[{dst_reg}] = {imm}')
            elif op == 0x50:  # ADD
                lines.append(f'  R[{dst_reg}] = (R.get({dst_reg},0) + R.get({src_reg},0) + {imm}) & 0xFFFFFFFF')
            elif op == 0x51:  # SUB
                lines.append(f'  R[{dst_reg}] = (R.get({dst_reg},0) - R.get({src_reg},0) - {imm}) & 0xFFFFFFFF')
            elif op == 0x38:  # NOP
                lines.append(f'  pass')
            elif op == 0x1F:  # LR_DECAY
                rate = imm / 1000.0
                lines.append(f'  som.decay_lr({rate})')
            else:
                break         # non-JITable instruction — end block

            jitted  += 1
            offset  += 8

            # Branch/halt terminates block
            if op in (0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37):
                break

        if jitted == 0:
            return None, 0

        lines.append(f'  return {jitted}')  # return instructions consumed
        src = '\n'.join(lines)

        try:
            ns = {}
            exec(compile(src, '<jit>', 'exec'), ns)
            fn = ns['_jit_block']
            self.cache[pc] = (fn, jitted)
            return fn, jitted
        except Exception:
            return None, 0


# ---------------------------------------------------------------------------
# Phase 2 — Agent context (runs in its own thread)
# ---------------------------------------------------------------------------

class AgentContext:
    def __init__(self, agent_id: int, entry_pc: int, parent_id: int,
                 shared: 'SharedState'):
        self.agent_id  = agent_id
        self.pc        = entry_pc - HEADER_SIZE   # convert to code offset
        self.parent_id = parent_id
        self.shared    = shared
        self.R         = {}    # registers
        self.S         = [0.0] * 16
        self.call_stack = []
        self.running   = True
        self.result    = None
        self._inbox    = []
        self._inbox_lock = threading.Lock()
        self._inbox_event = threading.Event()

    def send_msg(self, val: int):
        with self._inbox_lock:
            self._inbox.append(val)
        self._inbox_event.set()

    def recv_msg(self, timeout: float = 2.0) -> int:
        self._inbox_event.wait(timeout=timeout)
        with self._inbox_lock:
            if self._inbox:
                val = self._inbox.pop(0)
                if not self._inbox:
                    self._inbox_event.clear()
                return val
        return 0

    def rget(self, idx): return self.R.get(idx & 0xFF, 0)
    def rset(self, idx, val): self.R[idx & 0xFF] = val & 0xFFFFFFFF

    def run(self):
        rt = self.shared.runtime
        try:
            while self.running and self.pc + 8 <= len(rt.code):
                # JIT check
                if rt.jit.is_cached(self.pc):
                    fn, count = rt.jit.get(self.pc)
                    consumed = fn(self.R, self.S, rt.mem, rt.som)
                    self.pc += consumed * 8
                    continue

                if rt.jit.record(self.pc):
                    fn, count = rt.jit.compile_block(self.pc, rt.code, OPCODES_INV)
                    if fn:
                        consumed = fn(self.R, self.S, rt.mem, rt.som)
                        self.pc += consumed * 8
                        continue

                raw     = struct.unpack_from('>Q', rt.code, self.pc)[0]
                op      = (raw >> 56) & 0xFF
                agent   = (raw >> 48) & 0xFF
                src_reg = (raw >> 40) & 0xFF
                dst_reg = (raw >> 32) & 0xFF
                imm     = raw & 0xFFFFFFFF

                jumped = self._execute(op, agent, src_reg, dst_reg, imm, rt)
                if self.running and not jumped:
                    self.pc += 8
        except Exception as e:
            print(f"  [Agent {self.agent_id}] CRASH: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
        finally:
            self.shared.agent_done(self.agent_id)

    def _execute(self, op, agent, src_reg, dst_reg, imm, rt) -> bool:
        jumped = False

        if op == 0x01:    # SPAWN
            child_id = agent
            entry    = imm
            rt._spawn_agent(child_id, entry, parent_id=self.agent_id)

        elif op == 0x07:  # WAIT
            target = agent
            rt.shared.wait_for_agent(target)

        elif op == 0x02:  # AGENT_KILL
            if imm == 0xFF:
                rt.shared.kill_all_agents()
            else:
                rt.shared.kill_agent(agent if imm != 0xFE else self.agent_id)
            if imm == 0xFE:
                self.running = False; jumped = True

        elif op == 0x03:  # FORK
            count = src_reg
            entry = imm
            for i in range(count):
                rt._spawn_agent(i, entry, parent_id=self.agent_id)

        elif op == 0x04:  # MERGE
            self.rset(dst_reg, rt.shared.merge_acc)
            rt.shared.merge_acc = 0

        elif op == 0x05:  # BARRIER
            rt.shared.barrier(imm)

        elif op == 0x1A:  # SOM_MAP
            r = (imm >> 8) & 0xFF; c = imm & 0xFF
            self.som_pos = (r, c)

        elif op == 0x1C:  # SOM_INIT
            if imm == 0x01:
                rt.som.init_random()   # PCA stub → random
            else:
                rt.som.init_random()

        elif op == 0x1D:  # SOM_WALK
            pass  # stub — topology walk

        elif op == 0x11:  # SOM_BMU
            vec = self._load_vec_reg(src_reg, rt)
            br, bc = rt.som.bmu(vec)
            self.bmu_pos = (br, bc)
            self.rset(dst_reg, (br << 8) | bc)

        elif op == 0x12:  # SOM_TRAIN
            vec = self._load_vec_reg(src_reg, rt)
            br, bc = getattr(self, 'bmu_pos', (0,0))
            rt.som.train(vec, br, bc)

        elif op == 0x1B:  # SOM_SENSE
            r, c = getattr(self, 'som_pos', (0,0))
            val  = rt.som.sense(r, c)
            self.rset(dst_reg, int(val * 1000) & 0xFFFFFFFF)

        elif op == 0x13:  # SOM_NBHD
            bmu_val = self.rget(src_reg)
            r, c = (bmu_val >> 8) & 0xFF, bmu_val & 0xFF
            self.S[0] = rt.som.sense(r, c)

        elif op == 0x14:  # WGHT_UPD
            vec = self._load_vec_reg(src_reg, rt)
            br, bc = getattr(self, 'bmu_pos', (0,0))
            rt.som.train(vec, br, bc)

        elif op == 0x19:  # SOM_ELECT
            br, bc = rt.som.elect_leader()
            self.rset(dst_reg, (br << 8) | bc)

        elif op == 0x1E:  # SOM_DIST
            c1 = self.rget(src_reg)
            c2 = self.rget(dst_reg)
            r1,cc1 = (c1>>8)&0xFF, c1&0xFF
            r2,cc2 = (c2>>8)&0xFF, c2&0xFF
            d = rt.som.node_distance(r1,cc1,r2,cc2)
            self.rset(dst_reg, int(d * 1000) & 0xFFFFFFFF)

        elif op == 0x1F:  # LR_DECAY
            rt.som.decay_lr(imm / 1000.0)

        elif op == 0x20:  # MSG_SEND
            val = self.rget(src_reg) if src_reg else imm
            if agent == 0xFF:   # PARENT
                parent = rt.shared.get_agent(self.parent_id)
                if parent: parent.send_msg(val)
            elif agent == 0xFE: # SELF
                self.send_msg(val)
            else:
                target = rt.shared.get_agent(agent)
                if target: target.send_msg(val)
                else: rt.shared.main_inbox.append(val)

        elif op == 0x21:  # MSG_RECV
            val = self.recv_msg()
            self.rset(dst_reg, val)

        elif op == 0x23:  # BROADCAST
            rt.shared.broadcast(imm, exclude=self.agent_id)

        elif op == 0x24:  # ACCUM
            sv = self.rget(src_reg); dv = self.rget(dst_reg)
            res = (sv + dv) & 0xFFFFFFFF
            self.rset(dst_reg, res)
            with rt.shared._acc_lock:
                rt.shared.merge_acc = (rt.shared.merge_acc + sv) & 0xFFFFFFFF

        elif op == 0x30:  # JMP
            tp = imm - HEADER_SIZE
            if 0 <= tp <= len(rt.code)-8: self.pc = tp; jumped = True
            else: self.running = False; jumped = True

        elif op == 0x31:  # JZ
            val = self.rget(src_reg)
            if val == 0:
                tp = imm - HEADER_SIZE
                if 0 <= tp <= len(rt.code)-8: self.pc = tp; jumped = True

        elif op == 0x32:  # JNZ
            val = self.rget(src_reg)
            if val != 0:
                tp = imm - HEADER_SIZE
                if 0 <= tp <= len(rt.code)-8: self.pc = tp; jumped = True

        elif op == 0x33:  # JEQ
            va = self.rget(src_reg); vb = self.rget(dst_reg)
            if va == vb:
                tp = imm - HEADER_SIZE
                if 0 <= tp <= len(rt.code)-8: self.pc = tp; jumped = True

        elif op == 0x34:  # JGT
            va = self.rget(src_reg); vb = self.rget(dst_reg)
            if va > vb:
                tp = imm - HEADER_SIZE
                if 0 <= tp <= len(rt.code)-8: self.pc = tp; jumped = True

        elif op == 0x35:  # CALL
            if len(self.call_stack) >= MAX_CALL_DEPTH:
                print(f"  [A{self.agent_id}] Stack overflow", file=sys.stderr)
                self.running = False; jumped = True
            else:
                self.call_stack.append(self.pc + 8)
                tp = imm - HEADER_SIZE
                if 0 <= tp <= len(rt.code)-8: self.pc = tp; jumped = True

        elif op == 0x36:  # RET
            if self.call_stack:
                self.pc = self.call_stack.pop(); jumped = True
            else:
                self.running = False; jumped = True

        elif op == 0x37:  # HALT
            self.running = False; jumped = True

        elif op == 0x38:  # NOP
            pass

        elif op == 0x40:  # MOV
            if src_reg:
                val = self.rget(src_reg)
            elif imm & MEM_FLAG:
                payload_off = imm & ~MEM_FLAG
                val = int(rt.mem.read_float(payload_off) * 1000) & 0xFFFFFFFF
            else:
                val = imm
            self.rset(dst_reg, val)

        elif op == 0x41:  # STORE
            val = self.rget(src_reg)
            if imm & MEM_FLAG:
                payload_off = imm & ~MEM_FLAG
                rt.mem.write_float(payload_off, val / 1000.0)

        elif op == 0x42:  # LOAD
            # imm encodes a data_sym payload offset
            val = int(rt.mem.read_float(imm & ~MEM_FLAG) * 1000) & 0xFFFFFFFF
            self.rset(dst_reg, val)

        elif op == 0x43:  # TRAP
            self._trap(imm, rt)

        elif op == 0x50:  # ADD
            dv = self.rget(dst_reg); sv = self.rget(src_reg)
            self.rset(dst_reg, (dv + sv + imm) & 0xFFFFFFFF)

        elif op == 0x51:  # SUB
            dv = self.rget(dst_reg); sv = self.rget(src_reg)
            self.rset(dst_reg, (dv - sv - imm) & 0xFFFFFFFF)

        elif op == 0x52:  # MUL
            dv = self.rget(dst_reg); sv = self.rget(src_reg)
            self.rset(dst_reg, (dv * (sv + imm)) & 0xFFFFFFFF)

        elif op == 0x54:  # DOT
            pass  # stub

        elif op == 0x55:  # NORM
            val = self.rget(dst_reg)
            self.rset(dst_reg, int(math.sqrt(max(0,val))))

        return jumped

    def _load_vec_reg(self, reg_idx_val: int, rt: 'SomaRuntime') -> list:
        """
        If R[reg] holds a data-section ref (bit 31 set), load the vector.
        Otherwise treat R[reg] as a single-element vector.
        """
        val = self.rget(reg_idx_val)
        if val & MEM_FLAG:
            payload_off = val & ~MEM_FLAG
            # try to load a reasonable vector length (check sym table)
            for sym in rt.mem.symbols.values():
                if sym['offset'] == payload_off:
                    return rt.mem.read_by_offset(payload_off, sym['count'])
            return rt.mem.read_by_offset(payload_off, 8)
        return [val / 1000.0]

    def _trap(self, code: int, rt: 'SomaRuntime'):
        if code == 0x01:    # read_file stub
            self.rset(0, 0)
        elif code == 0x02:  # write_file stub
            pass
        elif code == 0x20:  # get_input_vector
            self.rset(0, 0)
        elif code == 0x30:  # read_stream → EOF
            self.rset(0, 0)
        elif code == 0xFF:  # error
            self.running = False


OPCODES_INV = {}  # not used in JIT directly


# ---------------------------------------------------------------------------
# Phase 2 — Shared state (thread-safe)
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self, runtime: 'SomaRuntime'):
        self.runtime    = runtime
        self._agents    = {}    # agent_id -> AgentContext
        self._lock      = threading.Lock()
        self._acc_lock  = threading.Lock()
        self.merge_acc  = 0
        self.main_inbox = []
        self._done_events = {}  # agent_id -> Event

    def register_agent(self, ctx: 'AgentContext'):
        with self._lock:
            self._agents[ctx.agent_id] = ctx
            self._done_events[ctx.agent_id] = threading.Event()

    def get_agent(self, agent_id: int) -> 'AgentContext | None':
        with self._lock:
            return self._agents.get(agent_id)

    def agent_done(self, agent_id: int):
        with self._lock:
            self._agents.pop(agent_id, None)
        ev = self._done_events.get(agent_id)
        if ev: ev.set()

    def wait_for_agent(self, agent_id: int, timeout: float = 5.0):
        ev = self._done_events.get(agent_id)
        if ev: ev.wait(timeout=timeout)

    def kill_agent(self, agent_id: int):
        with self._lock:
            ctx = self._agents.get(agent_id)
            if ctx: ctx.running = False

    def kill_all_agents(self):
        with self._lock:
            for ctx in self._agents.values():
                ctx.running = False

    def broadcast(self, val: int, exclude: int = -1):
        with self._lock:
            targets = list(self._agents.values())
        for ctx in targets:
            if ctx.agent_id != exclude:
                ctx.send_msg(val)

    def barrier(self, count: int, timeout: float = 10.0):
        """Simple barrier — waits until agent count drops to ≤ (registered - count)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                active = len(self._agents)
            if active <= 1:  # only main context left
                break
            time.sleep(0.01)

    def active_count(self) -> int:
        with self._lock:
            return len(self._agents)


# ---------------------------------------------------------------------------
# Main runtime
# ---------------------------------------------------------------------------

class SomaRuntime:
    def __init__(self, binpath: str, verbose: bool = True):
        self.verbose = verbose
        data = self._load_file(binpath)
        self._parse_header(data, binpath)

        # Phase 1 — load data section
        self.mem = DataMemory(self.data_bytes)

        # SOM
        self.som = SOM(self.som_rows, self.som_cols, dim=8, lr=0.5)

        # Phase 2 — thread infrastructure
        self.shared = SharedState(self)
        self._threads = []

        # Phase 3 — JIT
        self.jit = JITCache()

        # Main execution context (agent 0)
        self.main_ctx = AgentContext(0xFF, HEADER_SIZE, -1, self.shared)
        self.main_ctx.pc = 0

        if verbose:
            print(f"[SOMA] Loaded '{binpath}'")
            print(f"[SOMA] Version 0x{self.version:08x}  "
                  f"SOM {self.som_rows}×{self.som_cols}  "
                  f"MaxAgents={self.max_agents}")
            print(f"[SOMA] Code: {len(self.code)} bytes  "
                  f"({len(self.code)//8} instructions)")
            if self.mem.symbols:
                self.mem.dump()
            print()

    # ── file loading ────────────────────────────────────────────────── #

    def _load_file(self, path: str) -> bytes:
        try:
            return open(path, 'rb').read()
        except OSError as e:
            print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

    def _parse_header(self, data: bytes, path: str):
        if len(data) < HEADER_SIZE or data[:4] != b'SOMA':
            print(f"ERROR: not a valid .sombin file: '{path}'", file=sys.stderr)
            sys.exit(1)

        (self.version,) = struct.unpack_from('>I', data, 4)
        self.som_rows   = data[9]  if data[9]  > 0 else 16
        self.som_cols   = data[10] if data[10] > 0 else 16
        self.max_agents = data[11] if data[11] > 0 else 64

        code_off  = struct.unpack_from('>I', data, 12)[0]
        code_size = struct.unpack_from('>I', data, 16)[0]
        data_off  = struct.unpack_from('>I', data, 20)[0]
        data_size = struct.unpack_from('>I', data, 24)[0]

        if code_size == 0 or code_off + code_size > len(data):
            code_size = max(0, len(data) - code_off - data_size)

        self.code = data[code_off: code_off + code_size]
        rem = len(self.code) % 8
        if rem: self.code = self.code + bytes(8 - rem)

        if data_size > 0 and data_off + data_size <= len(data):
            self.data_bytes = data[data_off: data_off + data_size]
        else:
            self.data_bytes = b''

    # ── agent spawning ───────────────────────────────────────────────── #

    def _spawn_agent(self, agent_id: int, entry_abs: int, parent_id: int):
        ctx = AgentContext(agent_id, entry_abs, parent_id, self.shared)
        self.shared.register_agent(ctx)
        if self.verbose:
            print(f"  [SOMA] Spawning agent A{agent_id} "
                  f"at entry=0x{entry_abs:08x}")
        t = threading.Thread(target=ctx.run, daemon=True,
                             name=f"soma-agent-{agent_id}")
        self._threads.append(t)
        t.start()
        return ctx

    # ── main run loop ────────────────────────────────────────────────── #

    def run(self):
        ctx = self.main_ctx

        while ctx.running and ctx.pc + 8 <= len(self.code):
            # Phase 3 — JIT hot path
            if self.jit.is_cached(ctx.pc):
                fn, count = self.jit.get(ctx.pc)
                consumed = fn(ctx.R, ctx.S, self.mem, self.som)
                ctx.pc += consumed * 8
                continue

            if self.jit.record(ctx.pc):
                fn, count = self.jit.compile_block(ctx.pc, self.code, {})
                if fn:
                    consumed = fn(ctx.R, ctx.S, self.mem, self.som)
                    ctx.pc += consumed * 8
                    continue

            raw     = struct.unpack_from('>Q', self.code, ctx.pc)[0]
            op      = (raw >> 56) & 0xFF
            agent   = (raw >> 48) & 0xFF
            src_reg = (raw >> 40) & 0xFF
            dst_reg = (raw >> 32) & 0xFF
            imm     = raw & 0xFFFFFFFF

            if self.verbose:
                self._log(op, agent, src_reg, dst_reg, imm, ctx)

            jumped = ctx._execute(op, agent, src_reg, dst_reg, imm, self)
            if ctx.running and not jumped:
                ctx.pc += 8

        # Wait for all spawned agents
        for t in self._threads:
            t.join(timeout=10.0)

        if self.verbose:
            self._print_summary(ctx)

    # ── verbose logging ──────────────────────────────────────────────── #

    OPNAME = {
        0x01:'SPAWN',0x02:'AGENT_KILL',0x03:'FORK',0x04:'MERGE',0x05:'BARRIER',
        0x06:'SPAWN_MAP',0x07:'WAIT',
        0x11:'SOM_BMU',0x12:'SOM_TRAIN',0x13:'SOM_NBHD',0x14:'WGHT_UPD',
        0x19:'SOM_ELECT',0x1A:'SOM_MAP',0x1B:'SOM_SENSE',0x1C:'SOM_INIT',
        0x1D:'SOM_WALK',0x1E:'SOM_DIST',0x1F:'LR_DECAY',
        0x20:'MSG_SEND',0x21:'MSG_RECV',0x23:'BROADCAST',0x24:'ACCUM',
        0x30:'JMP',0x31:'JZ',0x32:'JNZ',0x33:'JEQ',0x34:'JGT',
        0x35:'CALL',0x36:'RET',0x37:'HALT',0x38:'NOP',
        0x40:'MOV',0x41:'STORE',0x42:'LOAD',0x43:'TRAP',
        0x50:'ADD',0x51:'SUB',0x52:'MUL',0x53:'DIV',0x54:'DOT',0x55:'NORM',
    }

    def _log(self, op, agent, src_reg, dst_reg, imm, ctx):
        name = self.OPNAME.get(op, f'0x{op:02x}')
        mem_flag = '→[DATA]' if imm & MEM_FLAG else ''
        print(f"  {name:12s} ag={agent:3d} src=R{src_reg:<3d} "
              f"dst=R{dst_reg:<3d} imm=0x{imm&0x7FFFFFFF:08x}{mem_flag}")

    def _print_summary(self, ctx):
        print()
        print("[SOMA] Execution complete.")
        used = {k:v for k,v in ctx.R.items() if v}
        if used:
            print("[SOMA] Final register state:")
            for k in sorted(used):
                print(f"  R{k:3d} = 0x{used[k]:08x}  ({used[k]})")
        else:
            print("[SOMA] All registers zero.")

        if self.mem.symbols:
            print("[SOMA] Final data memory:")
            self.mem.dump()

        jit_hits = sum(1 for v in self.jit.hit_count.values() if v >= JITCache.HOT_THRESHOLD)
        if jit_hits:
            print(f"[SOMA] JIT compiled {jit_hits} hot block(s).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    verbose = '--quiet' not in sys.argv
    args    = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        print("Usage: python soma_runtime.py <file.sombin> [--quiet]",
              file=sys.stderr); sys.exit(1)

    try:
        rt = SomaRuntime(args[0], verbose=verbose)
        rt.run()
    except KeyboardInterrupt:
        print("\n[SOMA] Interrupted.")
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(1)
