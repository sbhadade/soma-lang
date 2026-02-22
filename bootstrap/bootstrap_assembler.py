#!/usr/bin/env python3
"""
SOMA Bootstrap Assembler — v2.0 Production
Phase 1: Real DATA section parsing and binary serialisation.

Binary layout:
  [0:32]              Header (32 bytes)
  [32:32+code_size]   Code section
  [data_offset:...]   Data section (symbol table + float32 payload)

Data section wire format:
  uint32  num_symbols
  per symbol:
    uint16  name_len
    bytes   name (utf-8)
    uint8   dtype  (0=INT 1=FLOAT 2=VEC 3=WGHT 4=COORD 5=BYTES)
    uint32  payload_offset  (byte offset into float32 payload)
    uint32  count           (number of float32 elements)
  [float32 payload — IEEE 754 big-endian, 4 bytes each]

Memory reference encoding in instruction imm field:
  bit 31 set  →  data-section reference; bits 30-0 = payload byte offset
  bit 31 clear → code label address or literal immediate
"""

import sys
import struct
import re

# ---------------------------------------------------------------------------
# Opcode table
# ---------------------------------------------------------------------------
OPCODES = {
    'SPAWN':0x01,'AGENT_KILL':0x02,'FORK':0x03,'MERGE':0x04,'BARRIER':0x05,
    'SPAWN_MAP':0x06,'WAIT':0x07,
    'SOM_BMU':0x11,'SOM_TRAIN':0x12,'SOM_NBHD':0x13,'WGHT_UPD':0x14,
    'SOM_ELECT':0x19,'SOM_MAP':0x1A,'SOM_SENSE':0x1B,'SOM_INIT':0x1C,
    'SOM_WALK':0x1D,'SOM_DIST':0x1E,'LR_DECAY':0x1F,
    'MSG_SEND':0x20,'MSG_RECV':0x21,'BROADCAST':0x23,'ACCUM':0x24,
    'JMP':0x30,'JZ':0x31,'JNZ':0x32,'JEQ':0x33,'JGT':0x34,
    'CALL':0x35,'RET':0x36,'HALT':0x37,'NOP':0x38,
    'MOV':0x40,'STORE':0x41,'LOAD':0x42,'TRAP':0x43,
    'ADD':0x50,'SUB':0x51,'MUL':0x52,'DIV':0x53,'DOT':0x54,'NORM':0x55,
    # Phase II — Emotional memory
    'EMOT_TAG':0x80,'DECAY_PROTECT':0x81,'PREDICT_ERR':0x82,
    'EMOT_RECALL':0x83,'SURPRISE_CALC':0x84,
    # Phase III — Curiosity (AgentSoul + SomTerrain)
    'GOAL_SET':0x60,'GOAL_CHECK':0x61,'SOUL_QUERY':0x62,
    'META_SPAWN':0x63,'EVOLVE':0x64,'INTROSPECT':0x65,
    'TERRAIN_READ':0x66,'TERRAIN_MARK':0x67,
    'SOUL_INHERIT':0x68,'GOAL_STALL':0x69,
    # Phase IV — CDBG
    'CDBG_EMIT':0x70,'CDBG_RECV':0x71,'CTX_SWITCH':0x72,
}
OPCODE_SET = set(OPCODES.keys())

DIRECTIVES = {
    '.SOMA','.ARCH','.SOMSIZE','.AGENTS','.LEARNRATE',
    '.EPOCHS','.NBHD','.IMPORT','.DATA','.CODE','.DECAY','.ENTRY',
}
SPECIAL = {'SELF','PARENT','ALL','RANDOM','PCA','GRADIENT','GAUSSIAN'}

DTYPE_INT=0; DTYPE_FLOAT=1; DTYPE_VEC=2; DTYPE_WGHT=3; DTYPE_COORD=4; DTYPE_BYTES=5
DTYPE_MAP = {'INT':DTYPE_INT,'FLOAT':DTYPE_FLOAT,'VEC':DTYPE_VEC,
             'WGHT':DTYPE_WGHT,'COORD':DTYPE_COORD,'BYTE':DTYPE_BYTES}


# ---------------------------------------------------------------------------
# Phase 1 — DATA section parser
# ---------------------------------------------------------------------------

def parse_data_section(source: str) -> 'dict[str, dict]':
    """
    Returns: {name: {'type': DTYPE_*, 'values': [float, ...]}}
    Supports multiline VEC = [v0, v1, ...,
                               vN] definitions.
    """
    symbols = {}
    in_data = False
    pending_name = None   # for multiline VEC accumulation
    pending_dtype = None
    pending_accum = ''    # accumulated raw value string across lines

    lines = source.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]; i += 1
        line = re.sub(r';;;.*', '', raw)
        line = re.sub(r';.*',   '', line).strip()
        if not line:
            # still accumulate if in a multiline VEC
            if pending_name:
                pending_accum += ' '
            continue

        if line.startswith('.DATA'):
            in_data = True;  continue
        if line.startswith('.CODE') or line.startswith('.SOMA') or line.startswith('@'):
            in_data = False
            # close any open multiline
            if pending_name:
                _flush_pending(symbols, pending_name, pending_dtype, pending_accum)
                pending_name = pending_dtype = None; pending_accum = ''
            continue
        if not in_data or line.startswith('.'):
            continue

        # Are we mid-multiline accumulation?
        if pending_name is not None:
            pending_accum += ' ' + line
            if ']' in pending_accum or '>' in pending_accum:
                _flush_pending(symbols, pending_name, pending_dtype, pending_accum)
                pending_name = pending_dtype = None; pending_accum = ''
            continue

        # name : TYPE[N] = value_expression
        m = re.match(
            r'^([A-Za-z_]\w*)\s*:\s*([A-Z]+)(\[\d+\])?\s*(?:=\s*(.+))?$',
            line
        )
        if not m:
            continue

        name, type_str = m.group(1), m.group(2)
        arr_suffix = m.group(3)
        val_str    = (m.group(4) or '').strip()
        dtype      = DTYPE_MAP.get(type_str, DTYPE_FLOAT)

        if dtype == DTYPE_VEC:
            # Check if the bracket is closed on this line
            has_open  = '[' in val_str or '<' in val_str
            has_close = ']' in val_str or '>' in val_str
            if has_open and not has_close:
                # multiline — start accumulation
                pending_name  = name
                pending_dtype = (dtype, arr_suffix)
                pending_accum = val_str
                continue
            # single-line VEC — flush immediately
            _flush_pending(symbols, name, (dtype, arr_suffix), val_str)
        else:
            # non-VEC types — handle inline
            values = []
            if dtype == DTYPE_COORD:
                if '(' in val_str and ')' in val_str:
                    inner = val_str[val_str.index('(')+1 : val_str.index(')')]
                    values = [float(x.strip()) for x in inner.split(',') if x.strip()]
                else:
                    values = [0.0, 0.0]
            elif dtype == DTYPE_INT:
                try:    values = [float(int(val_str, 0))]
                except: values = [0.0]
            elif dtype in (DTYPE_FLOAT, DTYPE_WGHT):
                try:    values = [float(val_str)]
                except: values = [0.0]
            elif dtype == DTYPE_BYTES:
                n = int(arr_suffix[1:-1]) if arr_suffix else 1
                values = [0.0] * n
            if not values:
                values = [0.0]
            symbols[name] = {'type': dtype, 'values': values}

    # flush any trailing multiline
    if pending_name:
        _flush_pending(symbols, pending_name, pending_dtype, pending_accum)

    return symbols


def _flush_pending(symbols, name, dtype_info, val_str):
    """Parse a (possibly multiline-joined) VEC value string and store it."""
    dtype, arr_suffix = dtype_info if isinstance(dtype_info, tuple) else (dtype_info, None)
    values = []
    if dtype == DTYPE_VEC:
        for open_b, close_b in (('[', ']'), ('<', '>')):
            if open_b in val_str and close_b in val_str:
                inner = val_str[val_str.index(open_b)+1 : val_str.index(close_b)]
                values = [float(x.strip()) for x in inner.split(',') if x.strip()]
                break
        if not values and arr_suffix:
            n = int(arr_suffix[1:-1])
            values = [0.0] * n
    if not values:
        values = [0.0]
    symbols[name] = {'type': dtype if isinstance(dtype, int) else DTYPE_VEC, 'values': values}



def build_data_section(symbols: dict) -> 'tuple[bytes, dict]':
    """
    Returns (data_bytes, sym_offsets) where
      data_bytes  = symbol_table_header + float32_payload
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

    # Symbol table header
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


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def tokenize(source: str) -> 'list[tuple[int,str]]':
    tokens = []
    for lineno, raw in enumerate(source.splitlines(), 1):
        line = re.sub(r';;;.*', '', raw)
        line = re.sub(r';.*',   '', line).strip()
        if not line:
            continue
        parts = re.split(r'[\s,()]+', line)
        parts = [p for p in parts if p]
        if not parts:
            continue
        first = parts[0]
        if first in DIRECTIVES:
            continue
        if len(parts) >= 2 and parts[1] == ':':
            continue
        for p in parts:
            tok = p.replace('[','').replace(']','').replace('<','').replace('>','').strip()
            if tok:
                tokens.append((lineno, tok))
    return tokens


# ---------------------------------------------------------------------------
# Immediate / register helpers
# ---------------------------------------------------------------------------

def parse_imm(tok):
    if tok is None: return None
    try:
        if tok.lower().startswith('0x'): return int(tok, 16)
        if re.match(r'^-?\d+$', tok):    return int(tok)
        if re.match(r'^-?\d*\.\d+$', tok):
            return int(float(tok) * 1000) & 0x000FFFFF
    except (ValueError, OverflowError):
        pass
    return None

def reg_idx(tok, prefix):
    if tok and tok.startswith(prefix):
        try: return int(tok[len(prefix):]) & 0xFF
        except ValueError: pass
    return 0

def is_ldef(t): return t.startswith('@') and t.endswith(':')
def is_lref(t): return t.startswith('@') and not t.endswith(':')
def lname(t):   return t[1:-1] if t.endswith(':') else t[1:]


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class Assembler:
    HEADER_SIZE = 32

    def __init__(self):
        self.sym        = {}   # code labels → abs address
        self.data_sym   = {}   # data symbols → payload byte offset
        self.patches    = []   # (output_byte_offset, label_name)
        self.output     = bytearray()
        self.data_bytes = b''
        self.som_rows   = 16
        self.som_cols   = 16
        self.max_agents = 64

    # ── public ──────────────────────────────────────────────────────── #

    def assemble(self, source: str) -> bytearray:
        self._parse_directives(source)

        data_syms = parse_data_section(source)
        if data_syms:
            self.data_bytes, self.data_sym = build_data_section(data_syms)
            count = len(data_syms)
        else:
            self.data_bytes = b''
            self.data_sym   = {}
            count = 0

        tokens = tokenize(source)
        self._pass_one(tokens)

        self.output = bytearray(self.HEADER_SIZE)
        self._write_header_placeholder()
        self._pass_two(tokens)
        self._patch_labels()
        self._finalise_header()
        return self.output, count

    # ── directive parsing ────────────────────────────────────────────── #

    def _parse_directives(self, source: str):
        for line in source.splitlines():
            line = re.sub(r'[;]+.*', '', line).strip()
            m = re.match(r'^\.SOMSIZE\s+(\d+)x(\d+)', line)
            if m: self.som_rows, self.som_cols = int(m.group(1)), int(m.group(2))
            m = re.match(r'^\.AGENTS\s+(\d+)', line)
            if m: self.max_agents = min(int(m.group(1)), 255)

    # ── pass 1 ──────────────────────────────────────────────────────── #

    def _pass_one(self, tokens):
        ptr = self.HEADER_SIZE
        for _, tok in tokens:
            if is_ldef(tok):
                name = lname(tok)
                self.sym[name] = ptr
            elif tok in OPCODE_SET:
                ptr += 8

    # ── pass 2 ──────────────────────────────────────────────────────── #

    def _pass_two(self, tokens):
        toks = [t for _, t in tokens]
        n = len(toks)
        i = 0

        while i < n:
            tok = toks[i]
            if is_ldef(tok): i += 1; continue
            if tok not in OPCODE_SET: i += 1; continue

            op = OPCODES[tok]; i += 1
            agent = src_reg = dst_reg = imm = 0

            def peek(off=0):
                idx = i + off
                return toks[idx] if idx < n else None

            def consume():
                nonlocal i; v = peek(); i += 1; return v

            def skip_special():
                if peek() in SPECIAL: consume()

            def creg(prefix):
                t = peek()
                if t and t.startswith(prefix): return reg_idx(consume(), prefix)
                return 0

            def cimm():
                v = parse_imm(peek())
                if v is not None: consume(); return v
                return 0

            def clabel():
                t = peek()
                if t and is_lref(t):
                    consume(); name = lname(t)
                    if name in self.sym: return self.sym[name]
                    self.patches.append((len(self.output)+4, name)); return 0
                return 0

            def cmem():
                """Consume a memory symbol name; returns data-ref imm."""
                t = peek()
                if t and is_lref(t): return clabel()
                # plain symbol name (brackets stripped by tokeniser)
                if t and re.match(r'^[A-Za-z_]\w*$', t) and t not in SPECIAL and t not in OPCODE_SET:
                    consume()
                    off = self.data_sym.get(t, 0)
                    return 0x80000000 | off
                v = parse_imm(t)
                if v is not None: consume(); return v
                return 0

            # ── operand dispatch ─────────────────────────────────── #
            if   op==0x01: agent=creg('A'); imm=clabel()
            elif op==0x06:
                r=cimm(); c=cimm()
                t=peek()
                if t and is_lref(t):
                    consume(); nm=lname(t)
                    if nm in self.sym: imm=self.sym[nm]
                    else: self.patches.append((len(self.output)+4,nm)); imm=0
                else: imm=(r<<8)|c
            elif op==0x07: agent=creg('A')
            elif op==0x02:
                t=peek()
                if t=='ALL':  consume(); imm=0xFF
                elif t=='SELF': consume(); imm=0xFE
                elif t and t.startswith('A'): agent=creg('A')
            elif op==0x03:
                v=parse_imm(peek())
                if v is not None: consume(); src_reg=v&0xFF
                imm=clabel()
            elif op==0x04: skip_special(); dst_reg=creg('R')
            elif op==0x05: imm=cimm()
            elif op==0x1A: agent=creg('A'); r=cimm(); c=cimm(); imm=(r<<8)|c
            elif op==0x1C:
                t=peek()
                if t in SPECIAL: consume(); imm=0x01 if t=='PCA' else 0x00
            elif op==0x1D: skip_special(); skip_special()
            elif op==0x11: src_reg=creg('R'); dst_reg=creg('R')
            elif op==0x12:
                src_reg=creg('R')
                if peek() and peek().startswith('S'): consume()
            elif op==0x1B: dst_reg=creg('R')
            elif op==0x13:
                src_reg=creg('R')
                if peek() and peek().startswith('S'): consume()
            elif op==0x14:
                src_reg=creg('R')
                if peek() and peek().startswith('S'): consume()
            elif op==0x19: dst_reg=creg('R')
            elif op==0x1E:
                src_reg=creg('R')
                t=peek()
                if t and not t.startswith('R') and not is_lref(t): consume()
                dst_reg=creg('R')
            elif op==0x1F: imm=cimm()
            elif op==0x20:
                t=peek()
                if t in SPECIAL: consume(); agent=0xFF if t=='PARENT' else 0xFE
                elif t and t.startswith('A'): agent=creg('A')
                t2=peek()
                if t2 and t2.startswith('R'): src_reg=creg('R')
                else: imm=cmem()  # handles data refs like [goal_template] and literals
            elif op==0x21: dst_reg=creg('R')
            elif op==0x23: imm=cimm()
            elif op==0x24: src_reg=creg('R'); dst_reg=creg('R')
            elif op==0x30: imm=clabel()
            elif op in(0x31,0x32): src_reg=creg('R'); imm=clabel()
            elif op in(0x33,0x34):
                src_reg=creg('R')
                t=peek()
                if t and t.startswith('R'): dst_reg=creg('R')
                else:
                    v=parse_imm(t)
                    if v is not None: consume()
                imm=clabel()
            elif op==0x35: imm=clabel()
            elif op in(0x36,0x37,0x38): pass
            elif op==0x40:
                t=peek()
                if t and t.startswith('R'):   dst_reg=creg('R')
                elif t and t.startswith('S'): consume(); dst_reg=0xFF
                t2=peek()
                if t2 and t2.startswith('R'): src_reg=creg('R')
                else: imm=cmem()
            elif op==0x41:
                # STORE [mem], R<src>
                t=peek()
                if t and re.match(r'^[A-Za-z_]\w*$',t) and t not in SPECIAL and t not in OPCODE_SET:
                    consume(); imm=0x80000000|self.data_sym.get(t,0)
                src_reg=creg('R')
            elif op==0x42: dst_reg=creg('R'); consume()  # skip mem name
            elif op==0x43: imm=cimm()
            elif op==0x50:
                dst_reg=creg('R')
                t=peek()
                if t and t.startswith('R'): src_reg=creg('R')
                else: src_reg=dst_reg
                t2=peek()
                v=parse_imm(t2)
                if v is not None: consume(); imm=v
                elif t2 and t2.startswith('R'): imm=creg('R')
            elif op in(0x51,0x52,0x53):
                dst_reg=creg('R'); src_reg=creg('R')
                v=parse_imm(peek())
                if v is not None: consume(); imm=v
            elif op in(0x54,0x55): dst_reg=creg('R')

            # ── Phase II: Emotional memory ────────────────────────────── #
            elif op==0x80:  # EMOT_TAG  reg, imm_intensity
                t=peek()
                if t and t.startswith('S'): consume()       # accept S-regs (treat as R0)
                elif t and t.startswith('R'): dst_reg=creg('R')
                t2=peek()
                v=parse_imm(t2)
                if v is not None: consume(); imm=v
                elif t2 and t2.startswith('R'): src_reg=creg('R')
            elif op==0x81:  # DECAY_PROTECT  imm_cycles
                imm=cimm()
            elif op==0x82:  # PREDICT_ERR  dst, src
                dst_reg=creg('R'); src_reg=creg('R')
            elif op==0x83:  # EMOT_RECALL  dst
                dst_reg=creg('R')
            elif op==0x84:  # SURPRISE_CALC  dst, src
                dst_reg=creg('R'); src_reg=creg('R')

            # ── Phase III: Curiosity ─────────────────────────────────── #
            elif op==0x60:  # GOAL_SET  reg
                dst_reg=creg('R')
            elif op==0x61:  # GOAL_CHECK  reg
                dst_reg=creg('R')
            elif op==0x62:  # SOUL_QUERY  reg
                dst_reg=creg('R')
            elif op==0x63:  # META_SPAWN  [count_sym], @entry
                # count goes in agent byte (resolve data sym value at asm time)
                t=peek()
                count_val=0
                if t and re.match(r'^[A-Za-z_]\w*$',t) and t not in OPCODE_SET:
                    consume()
                    off=self.data_sym.get(t,None)
                    if off is not None:
                        # read the float value back from data_bytes and cast to int
                        import struct as _s
                        try: count_val=int(_s.unpack_from('>f',self.data_bytes,
                                self._data_payload_start()+off)[0])
                        except: count_val=4
                    else: count_val=4
                elif t: v=parse_imm(t); count_val=v if v is not None else 0; consume()
                agent=count_val & 0xFF
                imm=clabel()
            elif op==0x64:  # EVOLVE  A<winner>
                agent=creg('A')
                t=peek()
                if t and t.startswith('R'): dst_reg=creg('R')
            elif op==0x65:  # INTROSPECT  (no operands)
                pass
            elif op==0x66:  # TERRAIN_READ  dst
                dst_reg=creg('R')
            elif op==0x67:  # TERRAIN_MARK  reg
                dst_reg=creg('R')
            elif op==0x68:  # SOUL_INHERIT  A<src>
                agent=creg('A')
            elif op==0x69:  # GOAL_STALL  @label
                imm=clabel()

            # ── Phase IV: CDBG ───────────────────────────────────────── #
            elif op==0x70:  # CDBG_EMIT  (no operands)
                pass
            elif op==0x71:  # CDBG_RECV  dst
                dst_reg=creg('R')
            elif op==0x72:  # CTX_SWITCH  imm_ctx
                imm=cimm()

            word = (
                (op      & 0xFF) << 56 |
                (agent   & 0xFF) << 48 |
                (src_reg & 0xFF) << 40 |
                (dst_reg & 0xFF) << 32 |
                (imm     & 0xFFFFFFFF)
            )
            self.output += struct.pack('>Q', word)

    def _data_payload_start(self) -> int:
        """Return byte offset within data_bytes where the float32 payload begins
        (i.e., after the symbol table header)."""
        if not self.data_bytes:
            return 0
        off = 0
        import struct as _s
        (n,) = _s.unpack_from('>I', self.data_bytes, off); off += 4
        for _ in range(n):
            (nl,) = _s.unpack_from('>H', self.data_bytes, off); off += 2 + nl
            off += 9  # dtype(1) + payload_offset(4) + count(4)
        return off

    # ── label patching ───────────────────────────────────────────────── #

    def _patch_labels(self):
        for offset, name in self.patches:
            if name not in self.sym:
                print(f"ERROR: undefined label '{name}'", file=sys.stderr); sys.exit(1)
            self.output[offset:offset+4] = struct.pack('>I', self.sym[name])

    # ── header ──────────────────────────────────────────────────────── #

    def _write_header_placeholder(self):
        hdr = struct.pack('>4sIBBBBIIIII',
            b'SOMA', 0x00010000,
            0x00, self.som_rows&0xFF, self.som_cols&0xFF, self.max_agents&0xFF,
            self.HEADER_SIZE, 0, 0, 0, 0)
        self.output[:self.HEADER_SIZE] = hdr

    def _finalise_header(self):
        code_size   = len(self.output) - self.HEADER_SIZE
        data_offset = len(self.output)
        data_size   = len(self.data_bytes)
        self.output += bytearray(self.data_bytes)
        struct.pack_into('>I', self.output, 16, code_size)
        struct.pack_into('>I', self.output, 20, data_offset)
        struct.pack_into('>I', self.output, 24, data_size)
        struct.pack_into('>I', self.output, 28, data_offset)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python bootstrap_assembler.py <input.soma> <output.sombin>",
              file=sys.stderr); sys.exit(1)

    try:
        source = open(sys.argv[1], 'r', encoding='utf-8').read()
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

    asm = Assembler()
    try:
        binary, dsym_count = asm.assemble(source)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"ASSEMBLY FAILED: {e}", file=sys.stderr); sys.exit(1)

    try:
        open(sys.argv[2], 'wb').write(binary)
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

    code_instrs = (len(binary) - 32 - len(asm.data_bytes)) // 8
    print(f"✅ Assembled {sys.argv[1]} → {sys.argv[2]}  "
          f"({code_instrs} instructions, {len(binary)} bytes, "
          f"{dsym_count} data symbols)")
