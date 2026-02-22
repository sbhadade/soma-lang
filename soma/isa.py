"""SOMA ISA v3.0 — Opcode table and register definitions.

Opcode values here are the CANONICAL truth shared by:
  - soma/assembler.py   (encodes .soma source → .sombin)
  - soma/vm.py          (cooperative test VM — uses OPCODES dict, auto-heals)
  - runtime/soma_runtime.py  (real pthreads runtime — _exec dispatch)
  - runtime/soma_emit_c.py   (C transpiler — switch() cases)
  - soma_runtime.h/.c        (C library — no opcodes, pure API)

CHANGE LOG vs v1.0:
  - WAIT      0x35 → 0x07  (moved to agent-lifecycle block)
  - SOM_INIT  0x15 → 0x1C  (moved to SOM block)
  - SOM_WALK  0x16 → 0x1D  (moved to SOM block)
  - SOM_MAP   0x17 → 0x1A  (moved to SOM block)
  - MOV       0x50 → 0x40  (moved to memory block)
  - STORE     0x52 → 0x41
  - LOAD      0x51 → 0x42
  - ADD       0x40 → 0x50  (moved to arithmetic block)
  - SUB       0x41 → 0x51
  - MUL       0x42 → 0x52
  - DIV       0x43 → 0x53
  - CALL      0x33 → 0x35
  - RET       0x34 → 0x36
  - NOP       0x36 → 0x38
  - CMP       0x56 → REMOVED (use JZ/JNZ/JEQ/JGT instead)
  NEW in v3.0: SPAWN_MAP, SOM_SENSE, SOM_DIST, LR_DECAY,
               JEQ, JGT, ACCUM, TRAP
"""

# ── Opcodes ────────────────────────────────────────────────────────────────────
OPCODES = {
    # ── Agent lifecycle ───────────────────────────────────────────────────────
    "SPAWN":       0x01,   # Spawn agent at label
    "AGENT_KILL":  0x02,   # Kill agent (SELF=0xFE, ALL=0xFF)
    "FORK":        0x03,   # Spawn N copies at label
    "MERGE":       0x04,   # Collect accumulated result
    "BARRIER":     0x05,   # Synchronise N agents
    "SPAWN_MAP":   0x06,   # Spawn N×M agents on SOM grid
    "WAIT":        0x07,   # Wait for agent to die (was 0x35 in v1.0)

    # ── SOM operations ────────────────────────────────────────────────────────
    "SOM_BMU":     0x11,   # Find best matching unit
    "SOM_TRAIN":   0x12,   # Kohonen weight update
    "SOM_NBHD":    0x13,   # Compute Gaussian neighbourhood
    "WGHT_UPD":    0x14,   # Weight update (alias for SOM_TRAIN)
    # 0x15 reserved
    # 0x16 reserved
    # 0x17 reserved
    # 0x18 reserved
    "SOM_ELECT":   0x19,   # Democratic leader election
    "SOM_MAP":     0x1A,   # Place agent at SOM coordinate (was 0x17)
    "SOM_SENSE":   0x1B,   # Read activation at agent's current node
    "SOM_INIT":    0x1C,   # Initialise SOM weights (was 0x15)
    "SOM_WALK":    0x1D,   # Move agent along gradient (was 0x16)
    "SOM_DIST":    0x1E,   # Topological distance between two agents
    "LR_DECAY":    0x1F,   # Decay learning rate by factor imm/1000

    # ── Messaging ─────────────────────────────────────────────────────────────
    "MSG_SEND":    0x20,   # Send value to agent
    "MSG_RECV":    0x21,   # Blocking receive into register
    "BROADCAST":   0x23,   # Send to ALL agents
    "ACCUM":       0x24,   # Atomic accumulate + add to register

    # ── Control flow ──────────────────────────────────────────────────────────
    "JMP":         0x30,   # Unconditional jump
    "JZ":          0x31,   # Jump if zero
    "JNZ":         0x32,   # Jump if not zero
    "JEQ":         0x33,   # Jump if reg[src] == reg[dst]
    "JGT":         0x34,   # Jump if reg[src] > reg[dst]
    "CALL":        0x35,   # Call subroutine (was 0x33 in v1.0)
    "RET":         0x36,   # Return from subroutine (was 0x34)
    "HALT":        0x37,   # Terminate program
    "NOP":         0x38,   # No operation (was 0x36)

    # ── Memory ────────────────────────────────────────────────────────────────
    "MOV":         0x40,   # Move immediate/register (was 0x50 in v1.0)
    "STORE":       0x41,   # Store register to data memory (was 0x52)
    "LOAD":        0x42,   # Load from data memory (was 0x51)
    "TRAP":        0x43,   # System trap / EOF stub

    # ── Arithmetic / vector ───────────────────────────────────────────────────
    "ADD":         0x50,   # Vector add (was 0x40 in v1.0)
    "SUB":         0x51,   # Vector subtract (was 0x41)
    "MUL":         0x52,   # Vector multiply (was 0x42)
    "DIV":         0x53,   # Vector divide (was 0x43)
    "DOT":         0x54,   # Dot product
    "NORM":        0x55,   # Normalize vector
    "CMP":         0x56,   # Compare register to immediate (sets zero_flag)

    # ── Phase II: Emotional memory (liveliness) ──────────────────────────────
    "EMOT_TAG":    0x80,   # Attach valence+intensity to current SOM node
    "DECAY_PROTECT": 0x81, # Shield node from weight decay (cycles / permanent)
    "PREDICT_ERR": 0x82,   # Compute prediction error (surprise signal)
    "EMOT_RECALL": 0x83,   # Retrieve emotional tag at coord → reg
    "SURPRISE_CALC": 0x84, # Surprise from two raw vectors

    # ── Phase III: Curiosity (AgentSoul + SomTerrain) ─────────────────────────
    "GOAL_SET":    0x60,   # Set goal vector (target weight-space state)
    "GOAL_CHECK":  0x61,   # Measure distance to goal; updates stall/curiosity
    "SOUL_QUERY":  0x62,   # Query content memory for fingerprint match
    "META_SPAWN":  0x63,   # Spawn N agents with mutated goal vectors
    "EVOLVE":      0x64,   # Select best child by goal proximity; inherit soul
    "INTROSPECT":  0x65,   # Export own soul state as readable data
    "TERRAIN_READ":0x66,   # Read collective terrain memory at (r, c)
    "TERRAIN_MARK":0x67,   # Write emotional data into terrain at (r, c)
    "SOUL_INHERIT":0x68,   # Explicit soul inheritance (agent_id → this agent)
    "GOAL_STALL":  0x69,   # Jump to label if goal is stalled (curiosity trigger)

    # ── Phase IV: CDBG — Context-Discriminated Binary Grammar ────────────────
    "CDBG_EMIT":   0x70,   # Emit one CDBG frame to the message bus
    "CDBG_RECV":   0x71,   # Receive and decode a CDBG frame from inbox
    "CTX_SWITCH":  0x72,   # Set active decode context (CTX nibble)
}

OPCODE_NAMES = {v: k for k, v in OPCODES.items()}

# ── Register encoding ──────────────────────────────────────────────────────────
# R0-R15  → 0x0000-0x000F  (256-bit / 8×f32 weight vectors)
# A0-A63  → 0x0100-0x013F  (64-bit agent handles)
# S0-S15  → 0x0200-0x020F  (64-bit SOM state: S0=lr, S1=sigma, S2=epoch)
# SELF    → 0xFE00
# PARENT  → 0xFF00
# ALL     → 0xFE01

SPECIAL_REGS = {"SELF": 0xFF00, "PARENT": 0xFF01, "ALL": 0xFF02}


def encode_reg(name: str) -> int:
    if name in SPECIAL_REGS:
        return SPECIAL_REGS[name]
    if name.startswith("R") and name[1:].isdigit():
        n = int(name[1:])
        if 0 <= n <= 15:
            return 0x0000 | n
    if name.startswith("A") and name[1:].isdigit():
        n = int(name[1:])
        if 0 <= n <= 63:
            return 0x0100 | n
    if name.startswith("S") and name[1:].isdigit():
        n = int(name[1:])
        if 0 <= n <= 15:
            return 0x0200 | n
    raise ValueError(f"Unknown register: {name}")


def decode_reg(code: int) -> str:
    if code in (0xFF00, 0xFF01, 0xFF02):
        return {0xFF00: "SELF", 0xFF01: "PARENT", 0xFF02: "ALL"}[code]
    hi = (code >> 8) & 0xFF
    lo = code & 0xFF
    if hi == 0x00:
        return f"R{lo}"
    if hi == 0x01:
        return f"A{lo}"
    if hi == 0x02:
        return f"S{lo}"
    return f"#{code:#06x}"


# ── Binary format constants ────────────────────────────────────────────────────
MAGIC = b"SOMA"   # 0x534F4D41
VER_MAJOR = 4
VER_MINOR = 0

ARCH_ANY   = 0
ARCH_X86   = 1
ARCH_ARM   = 2
ARCH_RISCV = 3
ARCH_WASM  = 4

ARCH_NAMES = {ARCH_ANY: "ANY", ARCH_X86: "X86", ARCH_ARM: "ARM64",
              ARCH_RISCV: "RISCV", ARCH_WASM: "WASM"}