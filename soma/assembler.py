"""SOMA Assembler — assembles .soma source into .sombin binary."""
from __future__ import annotations
import struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from soma.lexer import tokenize, TT, Token
from soma.isa import (
    OPCODES, encode_reg,
    MAGIC, VER_MAJOR, VER_MINOR,
    ARCH_ANY, ARCH_X86, ARCH_ARM, ARCH_RISCV, ARCH_WASM,
)


# ── AST nodes ─────────────────────────────────────────────────────────────────

@dataclass
class Directive:
    name: str
    args: List[object]
    line: int


@dataclass
class LabelDef:
    name: str
    line: int


@dataclass
class Instruction:
    mnemonic: str
    operands: List[object]
    line: int


@dataclass
class DataDecl:
    name: str
    dtype: str
    value: object
    line: int


# ── Parser ─────────────────────────────────────────────────────────────────────

class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = [t for t in tokens if t.type != TT.NEWLINE or True]
        # Keep newlines for line structure; filter later
        self._pos = 0

    def _peek(self) -> Token:
        while self._pos < len(self.tokens) and self.tokens[self._pos].type == TT.NEWLINE:
            self._pos += 1
        if self._pos >= len(self.tokens):
            return Token(TT.EOF, None, 0, 0)
        return self.tokens[self._pos]

    def _next(self) -> Token:
        t = self._peek()
        self._pos += 1
        return t

    def _expect(self, tt: TT) -> Token:
        t = self._next()
        if t.type != tt:
            raise ParseError(f"Expected {tt.name}, got {t.type.name} ({t.value!r}) at line {t.line}")
        return t

    def _skip_newlines(self):
        while self._pos < len(self.tokens) and self.tokens[self._pos].type == TT.NEWLINE:
            self._pos += 1

    def parse(self):
        """Return (header_directives, data_decls, code_nodes)."""
        nodes = []
        self._pos = 0

        while True:
            self._skip_newlines()
            t = self._peek()
            if t.type == TT.EOF:
                break

            if t.type == TT.DIRECTIVE:
                d = self._parse_directive()
                nodes.append(d)
            elif t.type == TT.LABEL_DEF:
                self._next()
                nodes.append(LabelDef(t.value, t.line))
            elif t.type == TT.MNEMONIC:
                nodes.append(self._parse_instruction())
            elif t.type == TT.IDENT:
                # Could be a data declaration: name : TYPE = value
                nodes.append(self._parse_data_or_ident())
            else:
                self._next()  # skip unknown token

        return nodes

    def _parse_directive(self) -> Directive:
        t = self._next()
        name = t.value
        args = []
        # Consume args until newline or EOF
        while True:
            self._skip_newlines_inline()
            nxt = self.tokens[self._pos] if self._pos < len(self.tokens) else Token(TT.EOF, None, 0, 0)
            if nxt.type in (TT.NEWLINE, TT.EOF):
                break
            if nxt.type == TT.DIRECTIVE:
                break
            args.append(self._next().value)
        return Directive(name, args, t.line)

    def _skip_newlines_inline(self):
        # Don't skip newlines here (they terminate directive args)
        pass

    def _parse_instruction(self) -> Instruction:
        t = self._next()
        mnemonic = t.value
        operands = []
        # Collect operands on the same logical line
        while True:
            if self._pos >= len(self.tokens):
                break
            nxt = self.tokens[self._pos]
            if nxt.type == TT.NEWLINE:
                break
            if nxt.type == TT.EOF:
                break
            if nxt.type == TT.COMMA:
                self._pos += 1
                continue
            op = self._next()
            if op.type == TT.LBRACKET:
                # [ref]
                inner = self._next()
                self._expect(TT.RBRACKET)
                operands.append(("mem", inner.value))
            else:
                operands.append(op.value)
        return Instruction(mnemonic, operands, t.line)

    def _parse_data_or_ident(self) -> DataDecl:
        name_tok = self._next()
        # Expect colon
        if self._pos < len(self.tokens) and self.tokens[self._pos].type == TT.COLON:
            self._pos += 1  # skip ':'
        dtype_tok = self._next()  # MSG, VEC, INT, etc.
        # Expect '='
        # Skip '=' if present as IDENT
        if self._pos < len(self.tokens) and self.tokens[self._pos].value == "=":
            self._pos += 1
        val_tok = self._next()
        return DataDecl(name_tok.value, str(dtype_tok.value), val_tok.value, name_tok.line)


# ── Assembler ──────────────────────────────────────────────────────────────────

class AssemblerError(Exception):
    pass


HEADER_SIZE = 0x20  # 32 bytes


@dataclass
class SomBinHeader:
    arch: int = ARCH_ANY
    som_rows: int = 4
    som_cols: int = 4
    max_agents: int = 16
    flags: int = 0


def _encode_instruction(mnemonic: str, operands: list, labels: Dict[str, int], pc: int) -> bytes:
    """Encode one 64-bit instruction word."""
    opcode = OPCODES.get(mnemonic)
    if opcode is None:
        raise AssemblerError(f"Unknown mnemonic: {mnemonic}")

    agent_id = 0
    som_x = 0
    som_y = 0
    reg = 0
    imm = 0

    def _resolve(val) -> int:
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            # Agent register?
            try:
                return encode_reg(val)
            except ValueError:
                pass
            # Label?
            if val.startswith("@"):
                label = val
                if label in labels:
                    return labels[label]
                raise AssemblerError(f"Undefined label: {label}")
            # Named constants
            if val.upper() == "RANDOM":
                return 0xFFFF
            if val.upper() == "GRADIENT":
                return 0xFFFE
            if val.upper() == "SELF":
                return 0xFF00
            if val.upper() == "PARENT":
                return 0xFF01
            if val.upper() == "ALL":
                return 0xFF02
            # Try int
            try:
                return int(val, 0)
            except (ValueError, TypeError):
                return 0
        if isinstance(val, tuple):
            if val[0] == "mem":
                return _resolve(val[1])
            # coord
            return 0
        return 0

    ops = list(operands)

    if opcode in (OPCODES["HALT"], OPCODES["NOP"]):
        pass
    elif opcode == OPCODES["SPAWN"] and len(ops) >= 2:
        agent_id = _resolve(ops[0]) & 0xFF
        reg = _resolve(ops[0]) & 0xFFFF
        imm = _resolve(ops[1]) & 0xFFFF
    elif opcode == OPCODES["SOM_MAP"] and len(ops) >= 2:
        agent_id = _resolve(ops[0]) & 0xFF
        if isinstance(ops[1], tuple) and len(ops[1]) == 2:
            som_x = ops[1][0] & 0xFF
            som_y = ops[1][1] & 0xFF
        reg = _resolve(ops[0]) & 0xFFFF
    elif opcode == OPCODES["MSG_SEND"] and len(ops) >= 2:
        agent_id = _resolve(ops[0]) & 0xFF
        reg = _resolve(ops[0]) & 0xFFFF
        imm = _resolve(ops[1]) & 0xFFFF
    elif opcode == OPCODES["MSG_RECV"] and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF
    elif opcode == OPCODES["SOM_TRAIN"] and len(ops) >= 2:
        reg = _resolve(ops[0]) & 0xFFFF
        imm = _resolve(ops[1]) & 0xFFFF
    elif opcode == OPCODES["SOM_INIT"] and len(ops) >= 1:
        imm = _resolve(ops[0]) & 0xFFFF
    elif opcode == OPCODES["SOM_WALK"] and len(ops) >= 2:
        agent_id = _resolve(ops[0]) & 0xFF
        imm = _resolve(ops[1]) & 0xFFFF
    elif opcode == OPCODES["SOM_ELECT"] and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF
    elif opcode in (OPCODES["FORK"], OPCODES["BARRIER"], OPCODES["MERGE"]) and len(ops) >= 1:
        # FORK N, @label  ->  agent_id=N, imm=label_byte_offset
        agent_id = int(_resolve(ops[0])) & 0xFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF
    elif opcode == OPCODES["BROADCAST"] and len(ops) >= 1:
        imm = _resolve(ops[0]) & 0xFFFF
    elif opcode in (OPCODES["WAIT"], OPCODES["AGENT_KILL"]) and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF
    elif opcode in (OPCODES["JMP"], OPCODES["CALL"]) and len(ops) >= 1:
        imm = _resolve(ops[0]) & 0xFFFF
    elif opcode in (OPCODES["JZ"], OPCODES["JNZ"]) and len(ops) >= 2:
        reg = _resolve(ops[0]) & 0xFFFF
        imm = _resolve(ops[1]) & 0xFFFF
    elif opcode in (OPCODES["ADD"], OPCODES["SUB"], OPCODES["MUL"], OPCODES["DIV"],
                    OPCODES["MOV"], OPCODES["DOT"], OPCODES["NORM"], OPCODES["CMP"]):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF
    elif opcode in (OPCODES["LOAD"], OPCODES["STORE"]):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF

    # ── Phase II: Emotional memory ─────────────────────────────────────────
    elif opcode == OPCODES.get("EMOT_TAG") and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF
    elif opcode == OPCODES.get("DECAY_PROTECT") and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF
    elif opcode in (OPCODES.get("PREDICT_ERR", -1), OPCODES.get("EMOT_RECALL", -1),
                    OPCODES.get("SURPRISE_CALC", -1)):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF

    # ── Phase III: Curiosity ───────────────────────────────────────────────
    elif opcode == OPCODES.get("GOAL_SET") and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF
    elif opcode == OPCODES.get("GOAL_CHECK") and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF
    elif opcode == OPCODES.get("SOUL_QUERY") and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF
    elif opcode == OPCODES.get("META_SPAWN"):
        if len(ops) >= 1:
            agent_id = _resolve(ops[0]) & 0xFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF
    elif opcode == OPCODES.get("EVOLVE") and len(ops) >= 1:
        agent_id = _resolve(ops[0]) & 0xFF
        reg      = _resolve(ops[0]) & 0xFFFF
    elif opcode == OPCODES.get("INTROSPECT"):
        pass
    elif opcode in (OPCODES.get("TERRAIN_READ", -1), OPCODES.get("TERRAIN_MARK", -1)):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF
    elif opcode == OPCODES.get("SOUL_INHERIT") and len(ops) >= 1:
        agent_id = _resolve(ops[0]) & 0xFF
        reg      = _resolve(ops[0]) & 0xFFFF
    elif opcode == OPCODES.get("GOAL_STALL") and len(ops) >= 1:
        imm = _resolve(ops[0]) & 0xFFFF

    # ── Phase IV: CDBG ─────────────────────────────────────────────────────
    elif opcode == OPCODES.get("CDBG_EMIT"):
        pass
    elif opcode == OPCODES.get("CDBG_RECV") and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF
    elif opcode == OPCODES.get("CTX_SWITCH") and len(ops) >= 1:
        imm = _resolve(ops[0]) & 0xFFFF

    word = (
        (opcode    & 0xFF) << 56 |
        (agent_id  & 0xFF) << 48 |
        (som_x     & 0xFF) << 40 |
        (som_y     & 0xFF) << 32 |
        (reg       & 0xFFFF) << 16 |
        (imm       & 0xFFFF)
    )
    return struct.pack(">Q", word)


def assemble(source: str) -> bytes:
    """Assemble SOMA source → .sombin bytes."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    nodes = parser.parse()

    header = SomBinHeader()
    data_section: List[DataDecl] = []
    code_nodes: List = []
    in_data = False
    in_code = False

    # Separate sections
    for node in nodes:
        if isinstance(node, Directive):
            if node.name == ".ARCH":
                archmap = {"ANY": ARCH_ANY, "X86": ARCH_X86, "ARM64": ARCH_ARM,
                           "ARM": ARCH_ARM, "RISCV": ARCH_RISCV, "WASM": ARCH_WASM}
                header.arch = archmap.get(str(node.args[0]).upper(), ARCH_ANY) if node.args else ARCH_ANY
            elif node.name == ".SOMSIZE" and node.args:
                raw = str(node.args[0])
                if "x" in raw.lower():
                    parts = raw.lower().split("x")
                    header.som_rows = int(parts[0])
                    header.som_cols = int(parts[1])
                elif len(node.args) >= 2:
                    # Split token: e.g. args=['16', 'x16']
                    arg1 = str(node.args[0])
                    arg2 = str(node.args[1])
                    if arg2.startswith('x') or arg2.startswith('X'):
                        header.som_rows = int(arg1)
                        header.som_cols = int(arg2[1:])
                    else:
                        header.som_rows = int(arg1)
                        header.som_cols = int(arg2)
            elif node.name == ".AGENTS" and node.args:
                header.max_agents = int(node.args[0])
            elif node.name == ".DATA":
                in_data = True; in_code = False
            elif node.name == ".CODE":
                in_code = True; in_data = False
            elif node.name == ".SELF_MODIFYING":
                header.flags |= 0x01
        elif isinstance(node, DataDecl):
            data_section.append(node)
        elif isinstance(node, (LabelDef, Instruction)):
            code_nodes.append(node)

    # First pass: collect label addresses (each instruction = 8 bytes)
    labels: Dict[str, int] = {}
    pc = 0
    for node in code_nodes:
        if isinstance(node, LabelDef):
            labels[node.name] = pc
            labels[f"@{node.name.lstrip('@')}"] = pc
        elif isinstance(node, Instruction):
            pc += 8

    # Second pass: emit code
    code_bytes = bytearray()
    for node in code_nodes:
        if isinstance(node, Instruction):
            try:
                word = _encode_instruction(node.mnemonic, node.operands, labels, len(code_bytes))
            except AssemblerError as e:
                raise AssemblerError(f"Line {node.line}: {e}")
            code_bytes.extend(word)

    # Emit data section (simple: each data decl → 8 bytes)
    data_bytes = bytearray()
    data_map: Dict[str, int] = {}
    for decl in data_section:
        data_map[decl.name] = len(data_bytes)
        val = decl.value
        if isinstance(val, int):
            data_bytes.extend(struct.pack(">Q", val & 0xFFFFFFFFFFFFFFFF))
        elif isinstance(val, float):
            data_bytes.extend(struct.pack(">d", val))
        else:
            data_bytes.extend(struct.pack(">Q", 0))

    # Build header (32 bytes)
    code_offset = HEADER_SIZE + len(data_bytes)
    hdr = bytearray()
    hdr.extend(MAGIC)                            # 0x00  4  MAGIC
    hdr.extend(struct.pack(">H", VER_MAJOR))     # 0x04  2  VER_MAJOR
    hdr.extend(struct.pack(">H", VER_MINOR))     # 0x06  2  VER_MINOR
    hdr.append(header.arch)                       # 0x08  1  ARCH_TARGET
    hdr.append(header.som_rows)                   # 0x09  1  SOM_ROWS
    hdr.append(header.som_cols)                   # 0x0A  1  SOM_COLS
    hdr.append(header.max_agents)                 # 0x0B  1  MAX_AGENTS
    hdr.extend(struct.pack(">I", code_offset))   # 0x0C  4  CODE_OFFSET
    hdr.extend(struct.pack(">I", len(code_bytes)))  # 0x10  4  CODE_SIZE
    hdr.extend(struct.pack(">I", HEADER_SIZE))   # 0x14  4  DATA_OFFSET
    hdr.extend(struct.pack(">I", len(data_bytes)))  # 0x18  4  DATA_SIZE
    som_offset_val = (code_offset + len(code_bytes)) & 0xFFFFFFFF
    hdr.extend(struct.pack(">H", som_offset_val & 0xFFFF))  # 0x1C  2  SOM_OFFSET (low 16 bits)

    hdr.extend(struct.pack(">H", header.flags))  # 0x1E  2  FLAGS
    assert len(hdr) == HEADER_SIZE

    return bytes(hdr) + bytes(data_bytes) + bytes(code_bytes)


def disassemble(binary: bytes) -> str:
    """Disassemble .sombin bytes back to human-readable SOMA."""
    from soma.isa import OPCODE_NAMES, decode_reg

    if len(binary) < HEADER_SIZE or binary[:4] != MAGIC:
        raise ValueError("Not a valid .sombin file")

    (ver_major,) = struct.unpack_from(">H", binary, 0x04)
    (ver_minor,) = struct.unpack_from(">H", binary, 0x06)
    arch_target   = binary[0x08]
    som_rows      = binary[0x09]
    som_cols      = binary[0x0A]
    max_agents    = binary[0x0B]
    (code_offset,) = struct.unpack_from(">I", binary, 0x0C)
    (code_size,)   = struct.unpack_from(">I", binary, 0x10)
    (data_offset,) = struct.unpack_from(">I", binary, 0x14)
    (data_size,)   = struct.unpack_from(">I", binary, 0x18)
    (flags,)       = struct.unpack_from(">H", binary, 0x1E)

    from soma.isa import ARCH_NAMES
    lines = [
        f"; SOMA {ver_major}.{ver_minor}  arch={ARCH_NAMES.get(arch_target, arch_target)}",
        f"; SOM {som_rows}x{som_cols}  agents={max_agents}  flags={flags:#06x}",
        "",
        f".SOMA    {ver_major}.{ver_minor}.0",
        f".ARCH    {ARCH_NAMES.get(arch_target, 'ANY')}",
        f".SOMSIZE {som_rows}x{som_cols}",
        f".AGENTS  {max_agents}",
        "",
        ".CODE",
    ]

    pc = 0
    for off in range(code_offset, code_offset + code_size, 8):
        if off + 8 > len(binary):
            break
        (word,) = struct.unpack_from(">Q", binary, off)
        opcode   = (word >> 56) & 0xFF
        agent_id = (word >> 48) & 0xFF
        som_x    = (word >> 40) & 0xFF
        som_y    = (word >> 32) & 0xFF
        reg      = (word >> 16) & 0xFFFF
        imm      = word & 0xFFFF

        mnem = OPCODE_NAMES.get(opcode, f"0x{opcode:02X}")
        reg_name = decode_reg(reg)
        lines.append(
            f"  {pc:04X}:  {word:016X}  {mnem:<12} {reg_name}, #{imm:#06x}"
            f"  ; agent={agent_id} som=({som_x},{som_y})"
        )
        pc += 8

    return "\n".join(lines)
