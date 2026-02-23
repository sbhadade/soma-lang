"""SOMA Assembler — assembles .soma source into .sombin binary.

Fixes vs original:
  - Parser._parse_data_or_ident: proper multiline VEC/WGHT/COORD collection
  - Data section: symbol-table + float32 payload (matches bootstrap_assembler format)
  - MSG_SEND [sym]: data refs encoded as imm = 0x8000 | sym_index
  - _encode_instruction: receives data_sym_names for data-ref resolution
  - Layout: header(32) | data | code  (data_offset=32, code_offset=32+data_size)
"""
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
    name:  str
    dtype: str
    value: object   # float | List[float]
    line:  int


# ── Data section builder ───────────────────────────────────────────────────────

_DTYPE_MAP = {
    'INT': 0, 'IMM': 0,
    'FLOAT': 1,
    'VEC': 2,
    'WGHT': 3,
    'COORD': 4,
    'BYTE': 5, 'BYTES': 5,
}


def _build_data_section(symbols: dict) -> tuple:
    """
    symbols: {name: {'type': int, 'values': [float,...]}}
    Returns (data_bytes, sym_offsets) where
      data_bytes = symbol_table_header + float32_payload
      sym_offsets = {name: byte_offset_into_payload}
    """
    payload_parts = []
    sym_offsets   = {}
    payload_off   = 0

    for name, info in symbols.items():
        sym_offsets[name] = payload_off
        for v in info['values']:
            payload_parts.append(struct.pack('>f', float(v)))
        payload_off += 4 * len(info['values'])

    payload = b''.join(payload_parts)

    hdr = struct.pack('>I', len(symbols))
    for name, info in symbols.items():
        nb = name.encode('utf-8')
        hdr += struct.pack('>H', len(nb)) + nb
        hdr += struct.pack('>BII',
            info['type'],
            sym_offsets[name],
            len(info['values'])
        )

    return hdr + payload, sym_offsets


# ── Parser ─────────────────────────────────────────────────────────────────────

class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self._pos   = 0

    def _peek(self) -> Token:
        # Skip NEWLINE tokens transparently
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
            raise ParseError(
                f"Expected {tt.name}, got {t.type.name} ({t.value!r}) at line {t.line}"
            )
        return t

    def _skip_newlines(self):
        while self._pos < len(self.tokens) and self.tokens[self._pos].type == TT.NEWLINE:
            self._pos += 1

    def parse(self):
        nodes = []
        self._pos = 0

        while True:
            self._skip_newlines()
            t = self._peek()
            if t.type == TT.EOF:
                break
            if t.type == TT.DIRECTIVE:
                nodes.append(self._parse_directive())
            elif t.type == TT.LABEL_DEF:
                self._next()
                nodes.append(LabelDef(t.value, t.line))
            elif t.type == TT.MNEMONIC:
                nodes.append(self._parse_instruction())
            elif t.type == TT.IDENT:
                nodes.append(self._parse_data_or_ident())
            else:
                self._next()

        return nodes

    def _parse_directive(self) -> Directive:
        t    = self._next()
        name = t.value
        args = []
        while self._pos < len(self.tokens):
            nxt = self.tokens[self._pos]
            if nxt.type in (TT.NEWLINE, TT.EOF, TT.DIRECTIVE):
                break
            args.append(self._next().value)
        return Directive(name, args, t.line)

    def _parse_instruction(self) -> Instruction:
        t        = self._next()
        mnemonic = t.value
        operands = []
        while self._pos < len(self.tokens):
            nxt = self.tokens[self._pos]
            if nxt.type in (TT.NEWLINE, TT.EOF):
                break
            if nxt.type == TT.COMMA:
                self._pos += 1
                continue
            op = self._next()
            if op.type == TT.LBRACKET:
                # Collect ALL tokens until ] — supports complex expressions
                # like [patch_locs + R4 * 4] or [R1 + 0]
                inner_tokens = []
                while self._pos < len(self.tokens):
                    t2 = self.tokens[self._pos]
                    if t2.type == TT.RBRACKET or t2.type in (TT.NEWLINE, TT.EOF):
                        break
                    inner_tokens.append(self._next().value)
                # consume ]
                if self._pos < len(self.tokens) and self.tokens[self._pos].type == TT.RBRACKET:
                    self._pos += 1
                if len(inner_tokens) == 1:
                    # Simple: [symbol] or [R1]
                    operands.append(("mem", inner_tokens[0]))
                else:
                    # Complex: [base + reg * scale] — pass as raw expr string
                    operands.append(("mem_expr", " ".join(str(v) for v in inner_tokens)))
            else:
                operands.append(op.value)
        return Instruction(mnemonic, operands, t.line)

    def _parse_data_or_ident(self) -> DataDecl:
        name_tok = self._next()

        # optional colon
        if self._pos < len(self.tokens) and self.tokens[self._pos].type == TT.COLON:
            self._pos += 1

        dtype_tok = self._next()
        dtype_str = str(dtype_tok.value).upper()

        # optional '='
        if self._pos < len(self.tokens) and self.tokens[self._pos].value == "=":
            self._pos += 1

        if dtype_str in ("VEC", "WGHT", "COORD"):
            # Collect all numeric values between [ and ] — multiline safe because
            # _peek() skips NEWLINE tokens, so the closing ] is always reachable.
            values = []
            nxt = self._peek()
            if nxt.type == TT.LBRACKET:
                self._next()                    # consume [
                while True:
                    t = self._peek()
                    if t.type == TT.RBRACKET:
                        self._next()            # consume ]
                        break
                    if t.type == TT.EOF:
                        break
                    if t.type in (TT.FLOAT, TT.INTEGER):
                        self._next()
                        values.append(float(t.value))
                    elif t.type == TT.COMMA:
                        self._next()
                    else:
                        break
            elif nxt.type in (TT.FLOAT, TT.INTEGER):
                self._next()
                values.append(float(nxt.value))
            return DataDecl(name_tok.value, dtype_str, values or [0.0], name_tok.line)

        else:
            # Scalar or array size: INT/FLOAT/IMM/BYTE[expr] — one token or expression
            # Handle BYTE[256 * 64] style array size declarations
            nxt = self._peek()
            if nxt.type == TT.LBRACKET:
                # Array size expression: consume [expr] and evaluate it
                self._next()  # consume [
                expr_tokens = []
                while True:
                    t = self._peek()
                    if t.type == TT.RBRACKET or t.type == TT.EOF:
                        self._next()  # consume ]
                        break
                    expr_tokens.append(self._next().value)
                # Evaluate simple constant expression (supports +, -, *)
                try:
                    size = int(eval(" ".join(str(v) for v in expr_tokens)))
                except Exception:
                    size = 0
                return DataDecl(name_tok.value, dtype_str, float(size), name_tok.line)

            val_tok = self._next()
            val = val_tok.value
            if isinstance(val, (int, float)):
                return DataDecl(name_tok.value, dtype_str, float(val), name_tok.line)
            try:
                return DataDecl(name_tok.value, dtype_str, float(int(str(val), 0)), name_tok.line)
            except (ValueError, TypeError):
                return DataDecl(name_tok.value, dtype_str, 0.0, name_tok.line)


# ── Assembler ──────────────────────────────────────────────────────────────────

class AssemblerError(Exception):
    pass


HEADER_SIZE = 0x20  # 32 bytes


@dataclass
class SomBinHeader:
    arch:       int = ARCH_ANY
    som_rows:   int = 4
    som_cols:   int = 4
    max_agents: int = 16
    flags:      int = 0


def _encode_instruction(
    mnemonic:       str,
    operands:       list,
    labels:         Dict[str, int],
    pc:             int,
    data_sym_offsets: Dict[str, int] = None,
    data_sym_names:   List[str]      = None,
) -> bytes:
    """Encode one 64-bit instruction word."""
    opcode = OPCODES.get(mnemonic)
    if opcode is None:
        raise AssemblerError(f"Unknown mnemonic: {mnemonic}")

    agent_id = som_x = som_y = reg = imm = 0
    data_sym_offsets = data_sym_offsets or {}
    data_sym_names   = data_sym_names   or []

    def _resolve(val) -> int:
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            try:
                return encode_reg(val)
            except ValueError:
                pass
            if val.startswith("@"):
                if val in labels:
                    return labels[val]
                raise AssemblerError(f"Undefined label: {val}")
            if val.upper() == "RANDOM":   return 0xFFFF
            if val.upper() == "GRADIENT": return 0xFFFE
            if val.upper() == "SELF":     return 0xFF00
            if val.upper() == "PARENT":   return 0xFF01
            if val.upper() == "ALL":      return 0xFF02
            try:
                return int(val, 0)
            except (ValueError, TypeError):
                return 0
        if isinstance(val, tuple) and val[0] == "mem":
            # Data-symbol reference → 0x8000 | sym_index  (vm.py convention)
            sym_name = val[1]
            if sym_name in data_sym_names:
                return 0x8000 | (data_sym_names.index(sym_name) & 0x7FFF)
            return _resolve(sym_name)
        if isinstance(val, tuple) and val[0] == "mem_expr":
            # Complex address expression [base + R4 * 4] — resolve base symbol only
            # The Python VM doesn't support full address arithmetic; treat as base ref
            expr = val[1]
            first_token = expr.split()[0]
            return _resolve(first_token)
        return 0

    ops = list(operands)

    # ── Agent lifecycle ───────────────────────────────────────────────────────
    if opcode in (OPCODES["HALT"], OPCODES["NOP"]):
        pass

    elif opcode == OPCODES["SPAWN"] and len(ops) >= 2:
        agent_id = _resolve(ops[0]) & 0xFF
        reg      = _resolve(ops[0]) & 0xFFFF
        imm      = _resolve(ops[1]) & 0xFFFF

    elif opcode == OPCODES["SOM_MAP"] and len(ops) >= 2:
        agent_id = _resolve(ops[0]) & 0xFF
        if isinstance(ops[1], tuple) and ops[1][0] != "mem" and len(ops[1]) == 2:
            som_x = ops[1][0] & 0xFF
            som_y = ops[1][1] & 0xFF
        reg = _resolve(ops[0]) & 0xFFFF

    elif opcode == OPCODES["MSG_SEND"] and len(ops) >= 2:
        agent_id = _resolve(ops[0]) & 0xFF
        reg      = _resolve(ops[0]) & 0xFFFF
        op1      = ops[1]
        # ("mem", sym_name)  →  data ref
        if isinstance(op1, tuple) and op1[0] == "mem":
            imm = _resolve(op1) & 0xFFFF          # 0x8000 | sym_index
        else:
            imm = _resolve(op1) & 0xFFFF

    elif opcode == OPCODES["MSG_RECV"] and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF

    elif opcode == OPCODES["SOM_TRAIN"] and len(ops) >= 2:
        reg = _resolve(ops[0]) & 0xFFFF
        imm = _resolve(ops[1]) & 0xFFFF

    elif opcode == OPCODES["SOM_INIT"] and len(ops) >= 1:
        imm = _resolve(ops[0]) & 0xFFFF

    elif opcode == OPCODES["SOM_WALK"] and len(ops) >= 2:
        agent_id = _resolve(ops[0]) & 0xFF
        imm      = _resolve(ops[1]) & 0xFFFF

    elif opcode == OPCODES["SOM_ELECT"] and len(ops) >= 1:
        reg = _resolve(ops[0]) & 0xFFFF

    elif opcode in (OPCODES["FORK"], OPCODES["BARRIER"], OPCODES["MERGE"]) and len(ops) >= 1:
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

    elif opcode in (OPCODES["JEQ"], OPCODES["JGT"]) and len(ops) >= 3:
        reg   = _resolve(ops[0]) & 0xFFFF
        som_x = _resolve(ops[1]) & 0xFF
        imm   = _resolve(ops[2]) & 0xFFFF

    elif opcode in (OPCODES["ADD"], OPCODES["SUB"], OPCODES["MUL"], OPCODES["DIV"],
                    OPCODES["MOV"], OPCODES["DOT"], OPCODES["NORM"],
                    OPCODES.get("CMP", -1)):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF

    elif opcode in (OPCODES["LOAD"], OPCODES["STORE"]):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF

    # ── Phase II: Emotional memory ────────────────────────────────────────────
    elif opcode == OPCODES.get("EMOT_TAG"):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF

    elif opcode == OPCODES.get("DECAY_PROTECT"):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF

    elif opcode in (OPCODES.get("PREDICT_ERR", -1),
                    OPCODES.get("EMOT_RECALL",  -1),
                    OPCODES.get("SURPRISE_CALC", -1)):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF

    # ── Phase III: Curiosity ──────────────────────────────────────────────────
    elif opcode in (OPCODES.get("GOAL_SET",     -1),
                    OPCODES.get("GOAL_CHECK",    -1),
                    OPCODES.get("SOUL_QUERY",    -1),
                    OPCODES.get("TERRAIN_READ",  -1),
                    OPCODES.get("TERRAIN_MARK",  -1)):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF

    elif opcode == OPCODES.get("META_SPAWN"):
        if len(ops) >= 1:
            op0 = ops[0]
            if isinstance(op0, tuple) and op0[0] == "mem":
                agent_id = 4   # default count; runtime reads actual value
            else:
                agent_id = _resolve(op0) & 0xFF
        if len(ops) >= 2:
            imm = _resolve(ops[1]) & 0xFFFF

    elif opcode == OPCODES.get("EVOLVE"):
        if len(ops) >= 1:
            agent_id = _resolve(ops[0]) & 0xFF
            reg      = _resolve(ops[0]) & 0xFFFF

    elif opcode == OPCODES.get("INTROSPECT"):
        pass

    elif opcode == OPCODES.get("SOUL_INHERIT"):
        if len(ops) >= 1:
            agent_id = _resolve(ops[0]) & 0xFF
            reg      = _resolve(ops[0]) & 0xFFFF

    elif opcode == OPCODES.get("GOAL_STALL"):
        if len(ops) >= 1:
            imm = _resolve(ops[0]) & 0xFFFF

    # ── Phase IV: CDBG ────────────────────────────────────────────────────────
    elif opcode == OPCODES.get("CDBG_EMIT"):
        pass

    elif opcode == OPCODES.get("CDBG_RECV"):
        if len(ops) >= 1:
            reg = _resolve(ops[0]) & 0xFFFF

    elif opcode == OPCODES.get("CTX_SWITCH"):
        if len(ops) >= 1:
            imm = _resolve(ops[0]) & 0xFFFF

    word = (
        (opcode   & 0xFF)   << 56 |
        (agent_id & 0xFF)   << 48 |
        (som_x    & 0xFF)   << 40 |
        (som_y    & 0xFF)   << 32 |
        (reg      & 0xFFFF) << 16 |
        (imm      & 0xFFFF)
    )
    return struct.pack(">Q", word)


def assemble(source: str) -> bytes:
    """Assemble SOMA source → .sombin bytes."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    nodes  = parser.parse()

    header       = SomBinHeader()
    data_section: List[DataDecl] = []
    code_nodes:   List           = []

    for node in nodes:
        if isinstance(node, Directive):
            if node.name == ".ARCH":
                archmap = {"ANY": ARCH_ANY, "X86": ARCH_X86, "ARM64": ARCH_ARM,
                           "ARM": ARCH_ARM, "RISCV": ARCH_RISCV, "WASM": ARCH_WASM}
                header.arch = archmap.get(
                    str(node.args[0]).upper(), ARCH_ANY) if node.args else ARCH_ANY
            elif node.name == ".SOMSIZE" and node.args:
                raw = str(node.args[0])
                if "x" in raw.lower():
                    parts = raw.lower().split("x")
                    header.som_rows = int(parts[0])
                    header.som_cols = int(parts[1])
                elif len(node.args) >= 2:
                    arg1, arg2 = str(node.args[0]), str(node.args[1])
                    header.som_rows = int(arg1)
                    header.som_cols = int(arg2.lstrip('xX'))
            elif node.name == ".AGENTS" and node.args:
                header.max_agents = int(node.args[0])
            elif node.name == ".SELF_MODIFYING":
                header.flags |= 0x01
        elif isinstance(node, DataDecl):
            data_section.append(node)
        elif isinstance(node, (LabelDef, Instruction)):
            code_nodes.append(node)

    # ── Build data section ────────────────────────────────────────────────────
    data_syms: Dict[str, dict] = {}
    for decl in data_section:
        dtype_int = _DTYPE_MAP.get(decl.dtype.upper(), 1)
        if isinstance(decl.value, list):
            values = [float(v) for v in decl.value]
        elif isinstance(decl.value, (int, float)):
            values = [float(decl.value)]
        else:
            try:
                values = [float(int(str(decl.value), 0))]
            except (ValueError, TypeError):
                values = [0.0]
        data_syms[decl.name] = {'type': dtype_int, 'values': values}

    if data_syms:
        data_bytes, data_sym_offsets = _build_data_section(data_syms)
    else:
        data_bytes, data_sym_offsets = b'', {}

    data_sym_names = list(data_syms.keys())

    # ── Pass 1: collect labels (byte offset from code start = 0) ─────────────
    labels: Dict[str, int] = {}
    pc = 0
    for node in code_nodes:
        if isinstance(node, LabelDef):
            labels[node.name] = pc
            labels[f"@{node.name.lstrip('@')}"] = pc
        elif isinstance(node, Instruction):
            pc += 8

    # ── Pass 2: emit code ─────────────────────────────────────────────────────
    code_bytes = bytearray()
    for node in code_nodes:
        if isinstance(node, Instruction):
            try:
                word = _encode_instruction(
                    node.mnemonic, node.operands, labels,
                    len(code_bytes), data_sym_offsets, data_sym_names,
                )
            except AssemblerError as e:
                raise AssemblerError(f"Line {node.line}: {e}")
            code_bytes.extend(word)

    # ── Build 32-byte header ──────────────────────────────────────────────────
    # Binary layout: header(32) | data_section | code_section
    data_offset = HEADER_SIZE
    code_offset = HEADER_SIZE + len(data_bytes)
    som_offset  = (code_offset + len(code_bytes)) & 0xFFFFFFFF

    hdr = bytearray()
    hdr.extend(MAGIC)
    hdr.extend(struct.pack(">H", VER_MAJOR))
    hdr.extend(struct.pack(">H", VER_MINOR))
    hdr.append(header.arch)
    hdr.append(header.som_rows  & 0xFF)
    hdr.append(header.som_cols  & 0xFF)
    hdr.append(header.max_agents & 0xFF)
    hdr.extend(struct.pack(">I", code_offset))
    hdr.extend(struct.pack(">I", len(code_bytes)))
    hdr.extend(struct.pack(">I", data_offset))
    hdr.extend(struct.pack(">I", len(data_bytes)))
    hdr.extend(struct.pack(">H", som_offset & 0xFFFF))
    hdr.extend(struct.pack(">H", header.flags))
    assert len(hdr) == HEADER_SIZE

    return bytes(hdr) + bytes(data_bytes) + bytes(code_bytes)


def disassemble(binary: bytes) -> str:
    """Disassemble .sombin bytes back to human-readable SOMA."""
    from soma.isa import OPCODE_NAMES, decode_reg

    if len(binary) < HEADER_SIZE or binary[:4] != MAGIC:
        raise ValueError("Not a valid .sombin file")

    (ver_major,)  = struct.unpack_from(">H", binary, 0x04)
    (ver_minor,)  = struct.unpack_from(">H", binary, 0x06)
    arch_target    = binary[0x08]
    som_rows       = binary[0x09]
    som_cols       = binary[0x0A]
    max_agents     = binary[0x0B]
    (code_offset,) = struct.unpack_from(">I", binary, 0x0C)
    (code_size,)   = struct.unpack_from(">I", binary, 0x10)
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

        mnem     = OPCODE_NAMES.get(opcode, f"0x{opcode:02X}")
        reg_name = decode_reg(reg)
        imm_str  = f"[sym#{imm & 0x7FFF}]" if (imm & 0x8000) else f"#{imm:#06x}"
        lines.append(
            f"  {pc:04X}:  {word:016X}  {mnem:<14} {reg_name}, {imm_str}"
            f"  ; agent={agent_id} som=({som_x},{som_y})"
        )
        pc += 8

    return "\n".join(lines)
