"""SOMA ISA v1.0 — Opcode table and register definitions."""

# ── Opcodes ────────────────────────────────────────────────────────────────────
OPCODES = {
    # Agent lifecycle
    "SPAWN":       0x01,
    "AGENT_KILL":  0x02,
    "FORK":        0x03,
    "MERGE":       0x04,
    "BARRIER":     0x05,
    # SOM operations
    "SOM_BMU":     0x11,
    "SOM_TRAIN":   0x12,
    "SOM_NBHD":    0x13,
    "WGHT_UPD":    0x14,
    "SOM_INIT":    0x15,
    "SOM_WALK":    0x16,
    "SOM_MAP":     0x17,
    "SOM_ELECT":   0x19,
    # Messaging
    "MSG_SEND":    0x20,
    "MSG_RECV":    0x21,
    "BROADCAST":   0x23,
    # Control flow
    "JMP":         0x30,
    "JZ":          0x31,
    "JNZ":         0x32,
    "CALL":        0x33,
    "RET":         0x34,
    "WAIT":        0x35,
    "NOP":         0x36,
    "HALT":        0x37,
    # Arithmetic / vector
    "ADD":         0x40,
    "SUB":         0x41,
    "MUL":         0x42,
    "DIV":         0x43,
    "MOV":         0x50,
    "LOAD":        0x51,
    "STORE":       0x52,
    "DOT":         0x54,
    "NORM":        0x55,
    "CMP":         0x56,
}

OPCODE_NAMES = {v: k for k, v in OPCODES.items()}

# ── Register encoding ──────────────────────────────────────────────────────────
# R0-R15  → 0x0000-0x000F  (256-bit general purpose / weight vectors)
# A0-A63  → 0x0100-0x013F  (64-bit agent handles)
# S0-S15  → 0x0200-0x020F  (64-bit SOM state)
# SELF    → 0xFF00
# PARENT  → 0xFF01
# ALL     → 0xFF02

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
MAGIC = b"SOMA"  # 0x534F4D41
VER_MAJOR = 1
VER_MINOR = 0

ARCH_ANY   = 0
ARCH_X86   = 1
ARCH_ARM   = 2
ARCH_RISCV = 3
ARCH_WASM  = 4

ARCH_NAMES = {ARCH_ANY: "ANY", ARCH_X86: "X86", ARCH_ARM: "ARM64",
              ARCH_RISCV: "RISCV", ARCH_WASM: "WASM"}
