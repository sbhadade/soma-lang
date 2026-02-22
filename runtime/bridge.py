"""
runtime/bridge.py — Runtime Bridge between SOMA compiler and execution engine.

This module is the *single source of truth* for the runtime side of the ISA.
It does NOT duplicate the opcode table — it re-exports from soma.isa and adds
the runtime-facing contract types that bind compiler output to interpreter input.

Contract surface:
  ┌──────────────┐  .sombin bytes   ┌─────────────────────┐
  │  soma/       │ ─────────────────▶  runtime/bridge.py  │
  │  assembler   │                  │  (decode + validate) │
  │  soma/isa.py │ ◀── re-export ── │                      │
  └──────────────┘                  └──────────┬──────────┘
                                               │ SomaInstruction objects
                                               ▼
                                     runtime/interpreter.py
                                     runtime/collective.py

Design principles:
  1. Zero duplication — opcodes live in soma/isa.py, bridge imports them.
  2. Decode once — BinaryDecoder converts raw bytes → SomaInstruction.
  3. Validate early — BridgeValidator checks every instruction on load.
  4. Phase-aware — knows which opcodes belong to which phase.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ── Re-export the canonical ISA so runtime code only needs to import bridge ──
from soma.isa import (          # noqa: F401  (all re-exported)
    OPCODES,
    OPCODE_NAMES,
    PHASE_RANGES,
    MAGIC,
    VER_MAJOR,
    HEADER_SIZE,
    WORD_SIZE,
    NICHE_CAPACITY,
    NICHE_MIGRATE_THRESH,
    SYMBOL_BIND_THRESH,
    HERITAGE_TOP_K,
    COLLECTIVE_WINDOW,
    NICHE_IMM_DECLARE,
    NICHE_IMM_WITHDRAW,
    encode_word,
    decode_word,
    encode_reg,
    decode_reg,
    opcode_phase,
)

# ── Phase V opcode constants (frequently referenced in runtime) ───────────────
OP_NICHE_DECLARE   = OPCODES["NICHE_DECLARE"]    # 0x73
OP_SYMBOL_EMERGE   = OPCODES["SYMBOL_EMERGE"]    # 0x74
OP_HERITAGE_LOAD   = OPCODES["HERITAGE_LOAD"]    # 0x75
OP_NICHE_QUERY     = OPCODES["NICHE_QUERY"]      # 0x76
OP_COLLECTIVE_SYNC = OPCODES["COLLECTIVE_SYNC"]  # 0x77

PHASE_V_OPS = frozenset({
    OP_NICHE_DECLARE, OP_SYMBOL_EMERGE, OP_HERITAGE_LOAD,
    OP_NICHE_QUERY, OP_COLLECTIVE_SYNC,
})

# ── Instruction dataclass ─────────────────────────────────────────────────────

@dataclass(slots=True)
class SomaInstruction:
    """Decoded 64-bit SOMA instruction word."""
    opcode:   int   # 8 bits  — ISA opcode
    agent_id: int   # 8 bits  — target/source agent
    som_x:    int   # 8 bits  — SOM grid column
    som_y:    int   # 8 bits  — SOM grid row
    reg:      int   # 16 bits — register specifier
    imm:      int   # 16 bits — immediate value / label offset
    # ── derived ──────────────────────────────────────────────
    mnemonic: str   = field(init=False)
    phase:    str   = field(init=False)
    word:     int   = field(init=False)

    def __post_init__(self) -> None:
        self.mnemonic = OPCODE_NAMES.get(self.opcode, f"UNK_0x{self.opcode:02X}")
        self.phase    = opcode_phase(self.opcode)
        self.word     = encode_word(self.opcode, self.agent_id, self.som_x,
                                    self.som_y, self.reg, self.imm)

    @classmethod
    def from_word(cls, word: int) -> "SomaInstruction":
        d = decode_word(word)
        return cls(**d)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SomaInstruction":
        if len(data) != WORD_SIZE:
            raise ValueError(f"Expected {WORD_SIZE} bytes, got {len(data)}")
        (word,) = struct.unpack(">Q", data)
        return cls.from_word(word)

    def to_bytes(self) -> bytes:
        return struct.pack(">Q", self.word)

    def __repr__(self) -> str:
        return (
            f"<Instr {self.mnemonic}(0x{self.opcode:02X}) "
            f"agent={self.agent_id} som=({self.som_x},{self.som_y}) "
            f"reg=0x{self.reg:04X} imm=0x{self.imm:04X} phase={self.phase}>"
        )


# ── Binary file header ────────────────────────────────────────────────────────

@dataclass
class SomaBinaryHeader:
    """32-byte .sombin file header."""
    magic:      bytes
    ver_major:  int
    ver_minor:  int
    arch:       int
    num_agents: int
    entry_pc:   int
    som_width:  int
    som_height: int
    flags:      int

    _STRUCT = struct.Struct(">4sBBHHIHHH")   # 4+1+1+2+2+4+2+2+2 = 18 → pad to 32

    @classmethod
    def from_bytes(cls, data: bytes) -> "SomaBinaryHeader":
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} < {HEADER_SIZE}")
        fields = cls._STRUCT.unpack_from(data)
        return cls(*fields)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.magic != MAGIC:
            errors.append(f"Bad magic: {self.magic!r} (expected {MAGIC!r})")
        if self.ver_major != VER_MAJOR:
            errors.append(f"Unsupported major version: {self.ver_major}")
        return errors


# ── Binary decoder ────────────────────────────────────────────────────────────

class BinaryDecoder:
    """
    Decodes a .sombin byte stream into a sequence of SomaInstructions.

    Usage:
        dec = BinaryDecoder.from_file("program.sombin")
        for instr in dec.instructions():
            ...
    """

    def __init__(self, data: bytes) -> None:
        self._data = data

    @classmethod
    def from_file(cls, path: str | Path) -> "BinaryDecoder":
        return cls(Path(path).read_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "BinaryDecoder":
        return cls(data)

    @property
    def header(self) -> SomaBinaryHeader:
        return SomaBinaryHeader.from_bytes(self._data)

    def instructions(self) -> Iterator[SomaInstruction]:
        offset = HEADER_SIZE
        while offset + WORD_SIZE <= len(self._data):
            yield SomaInstruction.from_bytes(self._data[offset:offset + WORD_SIZE])
            offset += WORD_SIZE

    def validate_header(self) -> list[str]:
        return self.header.validate()


# ── Bridge validator ──────────────────────────────────────────────────────────

class BridgeValidator:
    """
    Validates a decoded instruction stream for ISA compliance.

    Checks:
      - No unknown opcodes
      - Phase V operands satisfy semantic constraints (imm / reg ranges)
      - agent_id ∈ [0, 63] for Phase V instructions
    """

    def __init__(self, instructions: list[SomaInstruction]) -> None:
        self._instrs = instructions

    def validate(self) -> list[str]:
        errors: list[str] = []
        for i, instr in enumerate(self._instrs):
            errors.extend(self._check(i, instr))
        return errors

    # ─────────────────────────────────────────────────────────────────────────
    def _check(self, idx: int, instr: SomaInstruction) -> list[str]:
        errs: list[str] = []
        loc = f"[PC={idx * WORD_SIZE:#06x}]"

        if instr.opcode not in OPCODE_NAMES:
            errs.append(f"{loc} Unknown opcode 0x{instr.opcode:02X}")
            return errs   # can't do semantic checks on unknowns

        # ── Phase V semantic checks ──────────────────────────────────────────
        if instr.opcode == OP_NICHE_DECLARE:
            if instr.imm not in (NICHE_IMM_DECLARE, NICHE_IMM_WITHDRAW):
                errs.append(
                    f"{loc} NICHE_DECLARE: imm must be 0x0001 (declare) "
                    f"or 0x0002 (withdraw), got 0x{instr.imm:04X}"
                )
            if instr.agent_id >= NICHE_CAPACITY:
                errs.append(
                    f"{loc} NICHE_DECLARE: agent_id {instr.agent_id} >= "
                    f"NICHE_CAPACITY {NICHE_CAPACITY}"
                )

        elif instr.opcode == OP_SYMBOL_EMERGE:
            # imm encodes activation count; must be ≥ SYMBOL_BIND_THRESH
            if instr.imm < SYMBOL_BIND_THRESH:
                errs.append(
                    f"{loc} SYMBOL_EMERGE: imm (activation count) "
                    f"{instr.imm} < SYMBOL_BIND_THRESH {SYMBOL_BIND_THRESH}"
                )

        elif instr.opcode == OP_HERITAGE_LOAD:
            top_k = instr.imm
            if top_k == 0 or top_k > HERITAGE_TOP_K:
                errs.append(
                    f"{loc} HERITAGE_LOAD: imm (top-K) must be 1–{HERITAGE_TOP_K}, "
                    f"got {top_k}"
                )

        elif instr.opcode == OP_COLLECTIVE_SYNC:
            # imm is pulse window; 0 means "sync now" (valid)
            pass   # no constraint

        return errs

    @classmethod
    def from_decoder(cls, dec: BinaryDecoder) -> "BridgeValidator":
        return cls(list(dec.instructions()))


# ── Convenience function: full pipeline ──────────────────────────────────────

def load_and_validate(path: str | Path) -> tuple[SomaBinaryHeader, list[SomaInstruction], list[str]]:
    """
    Load a .sombin file, validate the header, validate all instructions,
    and return (header, instructions, error_list).

    If error_list is empty, the file is ISA-compliant.
    """
    dec    = BinaryDecoder.from_file(path)
    errors = dec.validate_header()
    instrs = list(dec.instructions())
    errors.extend(BridgeValidator(instrs).validate())
    return dec.header, instrs, errors


# ── Bridge self-test ──────────────────────────────────────────────────────────

def _self_test() -> None:
    """Quick smoke-test run when the module is executed directly."""
    print("Bridge self-test…")

    # Encode a NICHE_DECLARE word and round-trip it
    word = encode_word(OP_NICHE_DECLARE, agent_id=7, som_x=3, som_y=5,
                       reg=encode_reg("N7"), imm=NICHE_IMM_DECLARE)
    instr = SomaInstruction.from_word(word)
    assert instr.opcode == OP_NICHE_DECLARE,   f"opcode mismatch: {instr.opcode:#04x}"
    assert instr.agent_id == 7,                f"agent_id mismatch: {instr.agent_id}"
    assert instr.mnemonic == "NICHE_DECLARE",  f"mnemonic: {instr.mnemonic}"
    assert instr.phase == "V",                 f"phase: {instr.phase}"
    print(f"  ✅  NICHE_DECLARE encode/decode: {instr}")

    # Validate instruction semantics
    good = SomaInstruction(OP_HERITAGE_LOAD, 1, 0, 0, 0xFF01, HERITAGE_TOP_K)
    validator = BridgeValidator([good])
    errs = validator.validate()
    assert not errs, f"Unexpected errors: {errs}"
    print(f"  ✅  HERITAGE_LOAD validation: OK")

    # Catch bad SYMBOL_EMERGE (count too low)
    bad = SomaInstruction(OP_SYMBOL_EMERGE, 0, 0, 0, 0, SYMBOL_BIND_THRESH - 1)
    errs2 = BridgeValidator([bad]).validate()
    assert errs2, "Expected validation error for bad SYMBOL_EMERGE"
    print(f"  ✅  SYMBOL_EMERGE bad-input rejected: {errs2[0][:60]}…")

    # Phase V re-exports
    assert OP_NICHE_DECLARE   == 0x73
    assert OP_SYMBOL_EMERGE   == 0x74
    assert OP_HERITAGE_LOAD   == 0x75
    assert OP_NICHE_QUERY     == 0x76
    assert OP_COLLECTIVE_SYNC == 0x77
    print("  ✅  Phase V opcode constants correct")

    print("Bridge self-test PASSED")


if __name__ == "__main__":
    _self_test()
