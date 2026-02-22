"""SOMA ISA v1.5 — Opcode table and register definitions.

Phases:
  I    (0x01–0x07)  Agent lifecycle
  II   (0x11–0x1F)  SOM operations
  III  (0x20–0x37)  Messaging + Control flow
  IV   (0x40–0x72)  Arithmetic / Vector / Culture / Curiosity
  V    (0x73–0x77)  Collective Intelligence  ← NEW
"""

# ── Opcodes ────────────────────────────────────────────────────────────────────
OPCODES: dict[str, int] = {
    # ── Phase I: Agent lifecycle ────────────────────────────────────────────
    "SPAWN":          0x01,
    "AGENT_KILL":     0x02,
    "FORK":           0x03,
    "MERGE":          0x04,
    "BARRIER":        0x05,
    "SPAWN_MAP":      0x06,
    "WAIT":           0x07,

    # ── Phase II: SOM operations ────────────────────────────────────────────
    "SOM_BMU":        0x11,
    "SOM_TRAIN":      0x12,
    "SOM_NBHD":       0x13,
    "WGHT_UPD":       0x14,
    "SOM_INIT":       0x15,
    "SOM_WALK":       0x16,
    "SOM_MAP":        0x17,
    "SOM_ELECT":      0x19,
    "SOM_SENSE":      0x1B,
    "SOM_DIST":       0x1E,
    "LR_DECAY":       0x1F,

    # ── Phase III: Messaging ────────────────────────────────────────────────
    "MSG_SEND":       0x20,
    "MSG_RECV":       0x21,
    "BROADCAST":      0x23,
    "ACCUM":          0x24,

    # ── Phase III: Control flow ─────────────────────────────────────────────
    "JMP":            0x30,
    "JZ":             0x31,
    "JNZ":            0x32,
    "JEQ":            0x33,
    "JGT":            0x34,
    "CALL":           0x35,
    "RET":            0x36,
    "NOP":            0x36,   # alias
    "HALT":           0x37,

    # ── Phase IV: Arithmetic / vector ───────────────────────────────────────
    "ADD":            0x40,
    "SUB":            0x41,
    "MUL":            0x42,
    "DIV":            0x43,
    "MOV":            0x50,
    "LOAD":           0x51,
    "STORE":          0x52,
    "DOT":            0x54,
    "NORM":           0x55,
    "CMP":            0x56,

    # ── Phase IV: Culture / Curiosity (0x60–0x72) ───────────────────────────
    "SOUL_SAVE":      0x60,   # persist agent's top-K weight vectors to soul store
    "SOUL_LOAD":      0x61,   # restore soul into current agent's registers
    "SOUL_MERGE":     0x62,   # blend two souls (parent + culture pool)
    "CURIOSITY_SET":  0x63,   # write curiosity scalar into S-register
    "CURIOSITY_GET":  0x64,   # read curiosity scalar
    "EXPLORE":        0x65,   # stochastic SOM walk weighted by curiosity
    "EXPLOIT":        0x66,   # greedy BMU descent
    "CULTURE_WRITE":  0x68,   # write pattern to shared culture buffer
    "CULTURE_READ":   0x69,   # read pattern from shared culture buffer
    "CULTURE_BLEND":  0x6A,   # average incoming culture patterns
    "NOVELTY":        0x6B,   # emit novelty score vs. culture buffer
    "CDBG_LOG":       0x70,   # curiosity-debug: log agent state to trace
    "CDBG_ASSERT":    0x71,   # halt with diagnostic if condition fails
    "CDBG_PROBE":     0x72,   # sample SOM activation density

    # ── Phase V: Collective Intelligence (0x73–0x77) ── NEW ─────────────────
    "NICHE_DECLARE":  0x73,   # agent broadcasts specialisation vector
    "SYMBOL_EMERGE":  0x74,   # co-activation binds a symbol ID to SOM region
    "HERITAGE_LOAD":  0x75,   # load parent soul top-K on birth
    "NICHE_QUERY":    0x76,   # returns niche density; migrate if > threshold
    "COLLECTIVE_SYNC":0x77,   # map-wide memory consolidation across all agents
}

# Reverse lookup: opcode int → mnemonic string
OPCODE_NAMES: dict[int, str] = {}
for _k, _v in OPCODES.items():
    if _v not in OPCODE_NAMES:           # first definition wins (NOP/RET alias)
        OPCODE_NAMES[_v] = _k

# ── Phase membership ───────────────────────────────────────────────────────────
PHASE_RANGES: dict[str, tuple[int, int]] = {
    "I":   (0x01, 0x0F),
    "II":  (0x10, 0x1F),
    "III": (0x20, 0x3F),
    "IV":  (0x40, 0x72),
    "V":   (0x73, 0x7F),
}


def opcode_phase(op: int) -> str:
    for name, (lo, hi) in PHASE_RANGES.items():
        if lo <= op <= hi:
            return name
    return "?"


# ── Register encoding ──────────────────────────────────────────────────────────
# R0-R15  → 0x0000-0x000F  (256-bit general purpose / weight vectors)
# A0-A63  → 0x0100-0x013F  (64-bit agent handles)
# S0-S15  → 0x0200-0x020F  (SOM state scalars)
# N0-N63  → 0x0300-0x033F  (niche registers — Phase V)
# SELF    → 0xFF00
# PARENT  → 0xFF01
# ALL     → 0xFF02

SPECIAL_REGS: dict[str, int] = {
    "SELF":   0xFF00,
    "PARENT": 0xFF01,
    "ALL":    0xFF02,
}


def encode_reg(name: str) -> int:
    if name in SPECIAL_REGS:
        return SPECIAL_REGS[name]
    prefix, tail = name[0], name[1:]
    if not tail.isdigit():
        raise ValueError(f"Unknown register: {name!r}")
    n = int(tail)
    mapping = {"R": (0x0000, 15), "A": (0x0100, 63),
               "S": (0x0200, 15), "N": (0x0300, 63)}
    if prefix not in mapping:
        raise ValueError(f"Unknown register prefix: {prefix!r}")
    base, cap = mapping[prefix]
    if not 0 <= n <= cap:
        raise ValueError(f"Register index {n} out of range for {prefix} (0–{cap})")
    return base | n


def decode_reg(code: int) -> str:
    rev = {v: k for k, v in SPECIAL_REGS.items()}
    if code in rev:
        return rev[code]
    hi, lo = (code >> 8) & 0xFF, code & 0xFF
    return {0x00: "R", 0x01: "A", 0x02: "S", 0x03: "N"}.get(hi, "#") + (
        f"{lo}" if hi in (0x00, 0x01, 0x02, 0x03) else f"{code:#06x}"
    )


# ── Binary format constants ────────────────────────────────────────────────────
MAGIC     = b"SOMA"
VER_MAJOR = 1
VER_MINOR = 5          # bumped for Phase V
HEADER_SIZE = 32

ARCH_ANY   = 0
ARCH_X86   = 1
ARCH_ARM   = 2
ARCH_RISCV = 3
ARCH_WASM  = 4
ARCH_NAMES = {ARCH_ANY: "ANY", ARCH_X86: "X86",
              ARCH_ARM: "ARM64", ARCH_RISCV: "RISCV", ARCH_WASM: "WASM"}

# ── Instruction word layout (64-bit, big-endian) ───────────────────────────────
# [63:56] opcode   8 bits
# [55:48] agent_id 8 bits
# [47:40] som_x    8 bits
# [39:32] som_y    8 bits
# [31:16] reg      16 bits
# [15:0]  imm      16 bits

WORD_SIZE = 8  # bytes


def encode_word(opcode: int, agent_id: int = 0, som_x: int = 0,
                som_y: int = 0, reg: int = 0, imm: int = 0) -> int:
    return (
        ((opcode   & 0xFF) << 56) |
        ((agent_id & 0xFF) << 48) |
        ((som_x    & 0xFF) << 40) |
        ((som_y    & 0xFF) << 32) |
        ((reg      & 0xFFFF) << 16) |
        (imm       & 0xFFFF)
    )


def decode_word(word: int) -> dict:
    return {
        "opcode":   (word >> 56) & 0xFF,
        "agent_id": (word >> 48) & 0xFF,
        "som_x":    (word >> 40) & 0xFF,
        "som_y":    (word >> 32) & 0xFF,
        "reg":      (word >> 16) & 0xFFFF,
        "imm":       word        & 0xFFFF,
    }


# ── Phase V semantic constants ─────────────────────────────────────────────────
#   These are embedded as immediates (imm field) in Phase V instructions.

NICHE_CAPACITY      = 64        # max distinct niches (log2 = 6.0)
NICHE_MIGRATE_THRESH = 0.75     # migrate if niche density > 75 %
SYMBOL_BIND_THRESH  = 3         # co-activations needed to bind a symbol
HERITAGE_TOP_K      = 8         # soul vectors copied to child on birth
COLLECTIVE_WINDOW   = 256       # pulses between automatic COLLECTIVE_SYNC

# Immediates for NICHE_DECLARE
NICHE_IMM_DECLARE   = 0x0001
NICHE_IMM_WITHDRAW  = 0x0002

# ── Congruency manifest ────────────────────────────────────────────────────────
#   Every file that deals with opcodes registers itself here.
#   verify_congruency.py checks that each file's opcode set
#   is a subset of OPCODES (no phantoms) and that Phase V is fully covered.

CONGRUENCY_MANIFEST = [
    "soma/isa.py",
    "runtime/bridge.py",
    "runtime/collective.py",
    "runtime/interpreter.py",
]
