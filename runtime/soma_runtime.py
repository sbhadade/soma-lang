#!/usr/bin/env python3
"""
SOMA Runtime Interpreter — Production Grade
Executes .sombin files produced by bootstrap_assembler.py

Instruction word layout (64-bit big-endian):
  bits 63-56 : opcode   (8 bits)
  bits 55-48 : agent    (8 bits)
  bits 47-40 : src_reg  (8 bits)
  bits 39-32 : dst_reg  (8 bits)
  bits 31- 0 : imm      (32 bits)
"""

import sys
import struct
import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER_SIZE = 32
MAX_REGISTERS = 256   # R0-R255 (sparse access via dict)
MAX_SOM_REGS  = 16    # S0-S15
MAX_AGENTS    = 256
MAX_CALL_DEPTH = 1024
STREAM_EOF_VALUE = 0  # TRAP 0x30 returns this to signal EOF


# ---------------------------------------------------------------------------
# SOM (Self-Organising Map) — stub implementation
# ---------------------------------------------------------------------------

class SOM:
    def __init__(self, rows=16, cols=16, dim=8):
        self.rows = rows
        self.cols = cols
        self.dim  = dim
        self.lr   = 0.5
        self.nbhd_radius = 3.0
        # weights: rows×cols grid of dim-dimensional vectors, initialised randomly
        import random
        self.weights = [
            [[random.random() for _ in range(dim)] for _ in range(cols)]
            for _ in range(rows)
        ]

    def bmu(self, vec):
        """Find Best Matching Unit. Returns (row, col)."""
        best_dist = float('inf')
        best_r, best_c = 0, 0
        for r in range(self.rows):
            for c in range(self.cols):
                d = sum((a - b) ** 2 for a, b in zip(vec, self.weights[r][c]))
                if d < best_dist:
                    best_dist = d
                    best_r, best_c = r, c
        return best_r, best_c

    def train(self, vec, bmu_r, bmu_c):
        """Update weights around BMU."""
        for r in range(self.rows):
            for c in range(self.cols):
                dist2 = (r - bmu_r) ** 2 + (c - bmu_c) ** 2
                influence = math.exp(-dist2 / (2 * self.nbhd_radius ** 2))
                for k in range(self.dim):
                    inp = vec[k] if k < len(vec) else 0.0
                    self.weights[r][c][k] += self.lr * influence * (inp - self.weights[r][c][k])

    def sense(self, r, c):
        """Return mean activation of node (r,c)."""
        return sum(self.weights[r][c]) / self.dim if self.dim else 0.0

    def dist(self, r1, c1, r2, c2):
        """Euclidean distance between two nodes."""
        return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)


# ---------------------------------------------------------------------------
# Agent (stub)
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, agent_id, entry_pc):
        self.agent_id = agent_id
        self.entry_pc = entry_pc
        self.state    = 'running'
        self.msg_queue = []

    def enqueue(self, msg):
        self.msg_queue.append(msg)

    def dequeue(self):
        return self.msg_queue.pop(0) if self.msg_queue else 0


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class SomaRuntime:

    def __init__(self, binpath: str):
        try:
            data = open(binpath, 'rb').read()
        except OSError as e:
            print(f"ERROR: cannot open '{binpath}': {e}", file=sys.stderr)
            sys.exit(1)

        self._parse_header(data, binpath)

        self.pc      = 0          # offset into self.code (not absolute)
        self.running = True

        # Registers — use dicts for sparse access
        self.R: dict = {}         # int registers  R0-R255
        self.S: list = [0.0] * MAX_SOM_REGS   # SOM float registers

        # SOM
        self.som         = SOM(self.som_rows, self.som_cols)
        self.bmu_pos     = (0, 0)  # current BMU (row, col)
        self.agent_pos   = (0, 0)  # agent's SOM node

        # Agents
        self.agents: dict = {}     # agent_id -> Agent
        self.self_id = 0

        # Messages (stub: main context uses queue 0xFF)
        self.msg_queue: list = []

        # Call stack
        self.call_stack: list = []

        # Merge accumulator
        self.merge_acc = 0

        print(f"[SOMA] Loaded '{binpath}'")
        print(f"[SOMA] Version 0x{self.version:08x}  "
              f"SOM {self.som_rows}×{self.som_cols}  "
              f"MaxAgents={self.max_agents}")
        print(f"[SOMA] Code: {len(self.code)} bytes  "
              f"({len(self.code)//8} instructions)")
        print()

    # ------------------------------------------------------------------ #
    # Header parsing                                                       #
    # ------------------------------------------------------------------ #

    def _parse_header(self, data: bytes, path: str):
        if len(data) < HEADER_SIZE:
            print(f"ERROR: '{path}' too small to be a valid .sombin file", file=sys.stderr)
            sys.exit(1)

        magic = data[0:4]
        if magic != b'SOMA':
            print(f"ERROR: bad magic bytes {magic!r} in '{path}'", file=sys.stderr)
            sys.exit(1)

        (self.version,) = struct.unpack_from('>I', data, 4)
        self.arch       = data[8]
        self.som_rows   = data[9]  if data[9]  > 0 else 16
        self.som_cols   = data[10] if data[10] > 0 else 16
        self.max_agents = data[11] if data[11] > 0 else 64

        (code_offset,) = struct.unpack_from('>I', data, 12)
        (code_size,)   = struct.unpack_from('>I', data, 16)

        if code_offset < HEADER_SIZE:
            print(f"ERROR: invalid code_offset {code_offset}", file=sys.stderr)
            sys.exit(1)

        if code_size == 0 or code_offset + code_size > len(data):
            # Fallback: read everything after header as code
            code_size = len(data) - code_offset
            print(f"[SOMA] WARNING: header code_size invalid; "
                  f"using {code_size} bytes from offset {code_offset}")

        self.code = data[code_offset: code_offset + code_size]

        # Pad to multiple of 8 to avoid short reads
        remainder = len(self.code) % 8
        if remainder:
            self.code = self.code + bytes(8 - remainder)

    # ------------------------------------------------------------------ #
    # Register helpers                                                     #
    # ------------------------------------------------------------------ #

    def rget(self, idx: int) -> int:
        return self.R.get(idx & 0xFF, 0)

    def rset(self, idx: int, val: int):
        self.R[idx & 0xFF] = val & 0xFFFFFFFF

    def sget(self, idx: int) -> float:
        return self.S[idx % MAX_SOM_REGS]

    def sset(self, idx: int, val: float):
        self.S[idx % MAX_SOM_REGS] = val

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self):
        while self.running:
            if self.pc + 8 > len(self.code):
                print(f"[SOMA] PC 0x{self.pc:08x} out of bounds — implicit HALT")
                break

            raw     = struct.unpack_from('>Q', self.code, self.pc)[0]
            opcode  = (raw >> 56) & 0xFF
            agent   = (raw >> 48) & 0xFF
            src_reg = (raw >> 40) & 0xFF
            dst_reg = (raw >> 32) & 0xFF
            imm     = raw & 0xFFFFFFFF

            jumped = self._execute(opcode, agent, src_reg, dst_reg, imm)
            if self.running and not jumped:
                self.pc += 8

        print()
        print("[SOMA] Execution complete.")
        self._print_registers()

    # ------------------------------------------------------------------ #
    # Instruction dispatch                                                 #
    # ------------------------------------------------------------------ #

    def _execute(self, op, agent, src_reg, dst_reg, imm) -> bool:
        """Execute one instruction. Returns True if PC was already updated (jump)."""

        jumped = False

        # ── Agent lifecycle ──────────────────────────────────────── #

        if op == 0x01:    # SPAWN  A<agent>, @entry
            entry = imm
            print(f"  SPAWN    A{agent} → entry=0x{entry:08x}")
            self.agents[agent] = Agent(agent, entry)

        elif op == 0x06:  # SPAWN_MAP  <rows>×<cols>, @entry
            entry = imm
            rows  = (entry >> 24) & 0xFF
            cols  = (entry >> 16) & 0xFF
            print(f"  SPAWN_MAP  {rows}×{cols} agents (stub)")

        elif op == 0x07:  # WAIT  A<agent>
            print(f"  WAIT     A{agent}")

        elif op == 0x02:  # AGENT_KILL
            if imm == 0xFF:
                print(f"  AGENT_KILL  ALL ({len(self.agents)} agents)")
                self.agents.clear()
            elif imm == 0xFE:
                print(f"  AGENT_KILL  SELF")
            else:
                print(f"  AGENT_KILL  A{agent}")
                self.agents.pop(agent, None)

        elif op == 0x03:  # FORK  count, @entry
            count = src_reg
            entry = imm
            print(f"  FORK     {count} agents → entry=0x{entry:08x}")
            for i in range(count):
                self.agents[i] = Agent(i, entry)

        elif op == 0x04:  # MERGE  ALL, R<dst>
            print(f"  MERGE    → R{dst_reg}  (acc={self.merge_acc})")
            self.rset(dst_reg, self.merge_acc)
            self.merge_acc = 0

        elif op == 0x05:  # BARRIER  <count>
            print(f"  BARRIER  {imm}")

        # ── SOM operations ───────────────────────────────────────── #

        elif op == 0x11:  # SOM_BMU  R<src>, R<dst>
            vec = [self.rget(src_reg)]   # stub: treat register as scalar vector
            r, c = self.som.bmu(vec)
            self.bmu_pos = (r, c)
            coord = (r << 8) | c
            self.rset(dst_reg, coord)
            print(f"  SOM_BMU  R{src_reg} → R{dst_reg}=({r},{c})")

        elif op == 0x12:  # SOM_TRAIN  R<src>, S<som>
            val = self.rget(src_reg)
            r, c = self.bmu_pos
            self.som.train([val], r, c)
            print(f"  SOM_TRAIN  R{src_reg}={val} at node ({r},{c})")

        elif op == 0x13:  # SOM_NBHD  R<bmu>, S<out>
            bmu_val = self.rget(src_reg)
            r, c = (bmu_val >> 8) & 0xFF, bmu_val & 0xFF
            nbhd_strength = self.som.sense(r, c)
            self.sset(0, nbhd_strength)
            print(f"  SOM_NBHD  R{src_reg} → S0={nbhd_strength:.4f}")

        elif op == 0x14:  # WGHT_UPD  R<vec>, S<nbhd>
            val = self.rget(src_reg)
            r, c = self.bmu_pos
            self.som.train([val / 1000.0], r, c)
            print(f"  WGHT_UPD  R{src_reg}={val}, S0={self.sget(0):.4f}")

        elif op == 0x19:  # SOM_ELECT  R<dst>
            # elect node with highest activation
            best_val = -1.0
            best_coord = 0
            for r in range(self.som.rows):
                for c in range(self.som.cols):
                    v = self.som.sense(r, c)
                    if v > best_val:
                        best_val = v
                        best_coord = (r << 8) | c
            self.rset(dst_reg, best_coord)
            print(f"  SOM_ELECT  → R{dst_reg}=0x{best_coord:04x}")

        elif op == 0x1A:  # SOM_MAP  A<agent>, (row,col)
            r = (imm >> 8) & 0xFF
            c = imm & 0xFF
            self.agent_pos = (r, c)
            print(f"  SOM_MAP  A{agent} → ({r},{c})")

        elif op == 0x1B:  # SOM_SENSE  R<dst>
            r, c = self.agent_pos
            val = self.som.sense(r, c)
            ival = int(val * 1000) & 0xFFFFFFFF
            self.rset(dst_reg, ival)
            print(f"  SOM_SENSE  → R{dst_reg}={val:.4f} ({ival})")

        elif op == 0x1C:  # SOM_INIT  RANDOM|PCA
            mode = 'PCA' if imm == 0x01 else 'RANDOM'
            print(f"  SOM_INIT  {mode}")
            if mode == 'RANDOM':
                import random
                for r in range(self.som.rows):
                    for c in range(self.som.cols):
                        self.som.weights[r][c] = [random.random() for _ in range(self.som.dim)]
            # PCA init is a stub — would require data matrix

        elif op == 0x1D:  # SOM_WALK  SELF, GRADIENT
            # Move agent toward BMU
            ar, ac = self.agent_pos
            br, bc = self.bmu_pos
            if ar < br: ar += 1
            elif ar > br: ar -= 1
            if ac < bc: ac += 1
            elif ac > bc: ac -= 1
            self.agent_pos = (ar, ac)
            print(f"  SOM_WALK  → ({ar},{ac})")

        elif op == 0x1E:  # SOM_DIST  R<a>, mem, R<dst>
            coord_a = self.rget(src_reg)
            coord_b = self.rget(dst_reg)   # dst_reg holds second coord
            r1, c1 = (coord_a >> 8) & 0xFF, coord_a & 0xFF
            r2, c2 = (coord_b >> 8) & 0xFF, coord_b & 0xFF
            d = self.som.dist(r1, c1, r2, c2)
            ival = int(d * 1000) & 0xFFFFFFFF
            self.rset(dst_reg, ival)
            print(f"  SOM_DIST  ({r1},{c1})↔({r2},{c2}) = {d:.3f}")

        elif op == 0x1F:  # LR_DECAY
            rate = imm / 1000.0
            self.som.lr = max(0.001, self.som.lr * (1.0 - rate))
            print(f"  LR_DECAY  rate={rate:.4f}  new_lr={self.som.lr:.6f}")

        # ── Messaging ────────────────────────────────────────────── #

        elif op == 0x20:  # MSG_SEND
            val = self.rget(src_reg) if src_reg else imm
            target_desc = 'PARENT' if agent == 0xFF else ('SELF' if agent == 0xFE else f'A{agent}')
            print(f"  MSG_SEND  → {target_desc}  val=0x{val:08x}")
            if agent in self.agents:
                self.agents[agent].enqueue(val)
            else:
                self.msg_queue.append(val)

        elif op == 0x21:  # MSG_RECV  R<dst>
            val = self.msg_queue.pop(0) if self.msg_queue else 0
            self.rset(dst_reg, val)
            print(f"  MSG_RECV  → R{dst_reg}=0x{val:08x}")

        elif op == 0x23:  # BROADCAST
            print(f"  BROADCAST  0x{imm:04x}  → {len(self.agents)} agents")
            for ag in self.agents.values():
                ag.enqueue(imm)

        elif op == 0x24:  # ACCUM  R<src>, R<dst>
            sv = self.rget(src_reg)
            dv = self.rget(dst_reg)
            result = (sv + dv) & 0xFFFFFFFF
            self.rset(dst_reg, result)
            self.merge_acc = (self.merge_acc + sv) & 0xFFFFFFFF
            print(f"  ACCUM    R{src_reg}={sv} + R{dst_reg}={dv} → R{dst_reg}={result}")

        # ── Control flow ─────────────────────────────────────────── #

        elif op == 0x30:  # JMP  @addr
            target_pc = imm - HEADER_SIZE
            print(f"  JMP      → 0x{imm:08x}  (code+{target_pc})")
            if 0 <= target_pc <= len(self.code) - 8:
                self.pc = target_pc
                jumped = True
            else:
                print(f"  WARNING: JMP target 0x{imm:08x} out of range — treating as HALT")
                self.running = False

        elif op == 0x31:  # JZ  R<n>, @addr
            val = self.rget(src_reg)
            target_pc = imm - HEADER_SIZE
            taken = (val == 0)
            print(f"  JZ       R{src_reg}={val} → 0x{imm:08x}  ({'TAKEN' if taken else 'skip'})")
            if taken:
                if 0 <= target_pc <= len(self.code) - 8:
                    self.pc = target_pc; jumped = True
                else:
                    print(f"  WARNING: JZ target out of range — halting")
                    self.running = False; jumped = True

        elif op == 0x32:  # JNZ  R<n>, @addr
            val = self.rget(src_reg)
            target_pc = imm - HEADER_SIZE
            taken = (val != 0)
            print(f"  JNZ      R{src_reg}={val} → 0x{imm:08x}  ({'TAKEN' if taken else 'skip'})")
            if taken:
                if 0 <= target_pc <= len(self.code) - 8:
                    self.pc = target_pc; jumped = True
                else:
                    print(f"  WARNING: JNZ target out of range — halting")
                    self.running = False; jumped = True

        elif op == 0x33:  # JEQ  R<a>, R<b>, @addr
            va = self.rget(src_reg)
            vb = self.rget(dst_reg)
            target_pc = imm - HEADER_SIZE
            taken = (va == vb)
            print(f"  JEQ      R{src_reg}={va} == R{dst_reg}={vb}? → "
                  f"0x{imm:08x}  ({'TAKEN' if taken else 'skip'})")
            if taken:
                if 0 <= target_pc <= len(self.code) - 8:
                    self.pc = target_pc; jumped = True

        elif op == 0x34:  # JGT  R<a>, R<b>, @addr
            va = self.rget(src_reg)
            vb = self.rget(dst_reg)
            target_pc = imm - HEADER_SIZE
            taken = (va > vb)
            print(f"  JGT      R{src_reg}={va} > R{dst_reg}={vb}? → "
                  f"0x{imm:08x}  ({'TAKEN' if taken else 'skip'})")
            if taken:
                if 0 <= target_pc <= len(self.code) - 8:
                    self.pc = target_pc; jumped = True

        elif op == 0x35:  # CALL  @addr
            ret_addr = self.pc + 8
            target_pc = imm - HEADER_SIZE
            print(f"  CALL     → 0x{imm:08x}  (ret=0x{ret_addr + HEADER_SIZE:08x})")
            if len(self.call_stack) >= MAX_CALL_DEPTH:
                print("  ERROR: call stack overflow — halting", file=sys.stderr)
                self.running = False; jumped = True
            elif 0 <= target_pc <= len(self.code) - 8:
                self.call_stack.append(ret_addr)
                self.pc = target_pc; jumped = True
            else:
                print(f"  WARNING: CALL target 0x{imm:08x} out of range — skipping")

        elif op == 0x36:  # RET
            if self.call_stack:
                ret_pc = self.call_stack.pop()
                print(f"  RET      → 0x{ret_pc + HEADER_SIZE:08x}")
                self.pc = ret_pc; jumped = True
            else:
                print("  RET      (empty stack — halting)")
                self.running = False; jumped = True

        elif op == 0x37:  # HALT
            print("  HALT")
            self.running = False; jumped = True

        elif op == 0x38:  # NOP
            print("  NOP")

        # ── Data / registers ─────────────────────────────────────── #

        elif op == 0x40:  # MOV  R<dst>, val|R<src>|mem
            if src_reg != 0:
                val = self.rget(src_reg)
                print(f"  MOV      R{dst_reg} ← R{src_reg}={val}")
            else:
                val = imm
                print(f"  MOV      R{dst_reg} ← 0x{val:08x}")
            self.rset(dst_reg, val)

        elif op == 0x41:  # STORE  mem, R<src>
            val = self.rget(src_reg)
            print(f"  STORE    [mem] ← R{src_reg}=0x{val:08x}")

        elif op == 0x42:  # LOAD  R<dst>, mem
            print(f"  LOAD     R{dst_reg} ← [mem] (stub=0)")
            self.rset(dst_reg, 0)

        elif op == 0x43:  # TRAP
            self._trap(imm)

        # ── Arithmetic ───────────────────────────────────────────── #

        elif op == 0x50:  # ADD  R<dst>, R<src>, imm
            sv  = self.rget(src_reg)
            dv  = self.rget(dst_reg)
            res = (dv + sv + imm) & 0xFFFFFFFF
            self.rset(dst_reg, res)
            print(f"  ADD      R{dst_reg}={dv} + R{src_reg}={sv} + {imm} → R{dst_reg}={res}")

        elif op == 0x51:  # SUB
            sv  = self.rget(src_reg)
            dv  = self.rget(dst_reg)
            res = (dv - sv - imm) & 0xFFFFFFFF
            self.rset(dst_reg, res)
            print(f"  SUB      R{dst_reg}={dv} - R{src_reg}={sv} - {imm} → R{dst_reg}={res}")

        elif op == 0x52:  # MUL
            sv  = self.rget(src_reg)
            dv  = self.rget(dst_reg)
            res = (dv * (sv + imm)) & 0xFFFFFFFF
            self.rset(dst_reg, res)
            print(f"  MUL      R{dst_reg}={dv} × R{src_reg}={sv} → R{dst_reg}={res}")

        elif op == 0x54:  # DOT
            print(f"  DOT      R{dst_reg} (stub)")

        elif op == 0x55:  # NORM
            val = self.rget(dst_reg)
            nval = int(math.sqrt(val)) if val >= 0 else 0
            self.rset(dst_reg, nval)
            print(f"  NORM     R{dst_reg}={val} → {nval}")

        else:
            print(f"  [0x{op:02x}] Unimplemented opcode at PC=0x{self.pc + HEADER_SIZE:08x}")

        return jumped

    # ------------------------------------------------------------------ #
    # TRAP handler                                                         #
    # ------------------------------------------------------------------ #

    def _trap(self, code: int):
        if code == 0x01:
            print(f"  TRAP     0x01: read_file  → R0=0 (stub)")
            self.rset(0, 0)
        elif code == 0x02:
            print(f"  TRAP     0x02: write_file (stub)")
        elif code == 0x20:
            print(f"  TRAP     0x20: get_input_vector → R0=0 (stub)")
            self.rset(0, 0)
        elif code == 0x30:
            # read_stream — return 0 (EOF) so JZ → stream_done terminates cleanly
            print(f"  TRAP     0x30: read_stream → R0=0 (EOF stub)")
            self.rset(0, STREAM_EOF_VALUE)
        elif code == 0xFF:
            val = self.rget(0)
            print(f"  TRAP     0xFF: error R0={val}")
            self.running = False
        else:
            print(f"  TRAP     0x{code:02x}: unknown syscall (stub)")

    # ------------------------------------------------------------------ #
    # Debug output                                                         #
    # ------------------------------------------------------------------ #

    def _print_registers(self):
        used = {k: v for k, v in self.R.items() if v != 0}
        if used:
            print("[SOMA] Final register state:")
            for k in sorted(used):
                print(f"  R{k:3d} = 0x{used[k]:08x}  ({used[k]})")
        else:
            print("[SOMA] All registers zero.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python soma_runtime.py <file.sombin>", file=sys.stderr)
        sys.exit(1)

    try:
        rt = SomaRuntime(sys.argv[1])
        rt.run()
    except KeyboardInterrupt:
        print("\n[SOMA] Interrupted.")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n[SOMA] RUNTIME ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
