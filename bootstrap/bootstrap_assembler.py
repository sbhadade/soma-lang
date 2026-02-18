#!/usr/bin/env python3
"""
SOMA Bootstrap Assembler — Production Grade
Assembles SOMA assembly source (.soma) into SOMA binary (.sombin).

Instruction word layout (64-bit big-endian):
  bits 63-56 : opcode   (8 bits)
  bits 55-48 : agent    (8 bits)
  bits 47-40 : src_reg  (8 bits)
  bits 39-32 : dst_reg  (8 bits)
  bits 31- 0 : imm      (32 bits)

Header layout (32 bytes):
   0- 3  magic        b'SOMA'
   4- 7  version      uint32 BE
   8      arch
   9      som_rows
  10      som_cols
  11      max_agents
  12-15  code_offset  uint32 BE  (always 32)
  16-19  code_size    uint32 BE
  20-23  data_offset  uint32 BE
  24-27  data_size    uint32 BE
  28-31  som_offset   uint32 BE
"""

import sys
import struct
import re

# ---------------------------------------------------------------------------
# Opcode table
# ---------------------------------------------------------------------------
OPCODES = {
    # Agent lifecycle
    'SPAWN':      0x01,
    'AGENT_KILL': 0x02,
    'FORK':       0x03,
    'MERGE':      0x04,
    'BARRIER':    0x05,
    'SPAWN_MAP':  0x06,
    'WAIT':       0x07,
    # SOM ops
    'SOM_BMU':    0x11,
    'SOM_TRAIN':  0x12,
    'SOM_NBHD':   0x13,
    'WGHT_UPD':   0x14,
    'SOM_ELECT':  0x19,
    'SOM_MAP':    0x1A,
    'SOM_SENSE':  0x1B,
    'SOM_INIT':   0x1C,
    'SOM_WALK':   0x1D,
    'SOM_DIST':   0x1E,
    'LR_DECAY':   0x1F,
    # Messaging
    'MSG_SEND':   0x20,
    'MSG_RECV':   0x21,
    'BROADCAST':  0x23,
    'ACCUM':      0x24,
    # Control flow
    'JMP':        0x30,
    'JZ':         0x31,
    'JNZ':        0x32,
    'JEQ':        0x33,
    'JGT':        0x34,
    'CALL':       0x35,
    'RET':        0x36,
    'HALT':       0x37,
    'NOP':        0x38,
    # Data
    'MOV':        0x40,
    'STORE':      0x41,
    'LOAD':       0x42,
    'TRAP':       0x43,
    # Arithmetic
    'ADD':        0x50,
    'SUB':        0x51,
    'MUL':        0x52,
    'DIV':        0x53,
    'DOT':        0x54,
    'NORM':       0x55,
}

OPCODE_SET = set(OPCODES.keys())

# Directives — lines starting with these are metadata only
DIRECTIVES = {
    '.SOMA', '.ARCH', '.SOMSIZE', '.AGENTS', '.LEARNRATE',
    '.EPOCHS', '.NBHD', '.IMPORT', '.DATA', '.CODE',
    '.DECAY', '.ENTRY',
}

# Special pseudo-register/value tokens
SPECIAL = {'SELF', 'PARENT', 'ALL', 'RANDOM', 'PCA', 'GRADIENT', 'GAUSSIAN'}


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list:
    """
    Returns a list of (line_no, token_string) tuples.
    - Strips ;;; and ; comments
    - Skips blank lines and directive lines
    - Skips data-declaration lines (identifier : TYPE = ...)
    - Strips commas and parentheses (they're syntactic sugar only)
    - Keeps: labels (@foo:), opcodes, registers (R0 A0 S0), immediates,
             special words, memory refs (strips brackets but keeps the name)
    """
    tokens = []
    for lineno, raw_line in enumerate(source.splitlines(), 1):
        # Strip comments
        line = re.sub(r';;;.*', '', raw_line)
        line = re.sub(r';.*', '', line).strip()
        if not line:
            continue

        # Tokenise the line by splitting on whitespace and punctuation,
        # keeping meaningful pieces.
        # First pull out raw words (split on whitespace, commas, parens)
        parts = re.split(r'[\s,()]+', line)
        parts = [p for p in parts if p]

        if not parts:
            continue

        first = parts[0]

        # Skip directive lines
        if first in DIRECTIVES:
            continue

        # Skip data declaration lines: identifier : TYPE ...
        if len(parts) >= 2 and parts[1] == ':':
            continue

        # Now process each part
        for p in parts:
            # Strip square brackets (memory references)
            tok = p.replace('[', '').replace(']', '').strip()
            if not tok:
                continue
            tokens.append((lineno, tok))

    return tokens


# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------

def parse_imm(tok: str):
    """Parse immediate integer from token. Returns int or None."""
    if tok is None:
        return None
    try:
        if tok.lower().startswith('0x'):
            return int(tok, 16)
        if re.match(r'^-?\d+$', tok):
            return int(tok)
        if re.match(r'^-?\d*\.\d+$', tok):
            # encode float as fixed-point * 1000 truncated to 20 bits
            return int(float(tok) * 1000) & 0x000FFFFF
    except (ValueError, OverflowError):
        pass
    return None

def reg_idx(tok: str, prefix: str) -> int:
    """Return register index from token like R3, A0, S1. Returns 0 on failure."""
    if tok and tok.startswith(prefix):
        try:
            return int(tok[len(prefix):]) & 0xFF
        except ValueError:
            pass
    return 0

def is_label_def(tok: str) -> bool:
    return tok.startswith('@') and tok.endswith(':')

def is_label_ref(tok: str) -> bool:
    return tok.startswith('@') and not tok.endswith(':')

def label_name_from_def(tok: str) -> str:
    return tok[1:-1]   # strip @ and :

def label_name_from_ref(tok: str) -> str:
    return tok[1:]     # strip @


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class Assembler:
    HEADER_SIZE = 32

    def __init__(self):
        self.sym: dict   = {}       # label_name -> absolute byte address
        self.patches: list = []     # (bin_data_offset, label_name) for forward refs
        self.output: bytearray = bytearray()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def assemble(self, source: str) -> bytearray:
        tokens = tokenize(source)

        # Pass 1: collect label addresses
        self._pass_one(tokens)

        # Initialise output with header placeholder
        self.output = bytearray(self.HEADER_SIZE)
        self._write_header_magic()

        # Pass 2: emit instructions
        self._pass_two(tokens)

        # Patch forward references
        self._patch_labels()

        # Finalise header
        self._finalise_header()

        return self.output

    # ------------------------------------------------------------------ #
    # Pass 1 — label collection                                           #
    # ------------------------------------------------------------------ #

    def _pass_one(self, tokens: list):
        ptr = self.HEADER_SIZE
        for _lineno, tok in tokens:
            if is_label_def(tok):
                name = label_name_from_def(tok)
                if name in self.sym:
                    print(f"WARNING: duplicate label '{name}' — overwriting", file=sys.stderr)
                self.sym[name] = ptr
            elif tok in OPCODE_SET:
                ptr += 8   # every instruction is 8 bytes

    # ------------------------------------------------------------------ #
    # Pass 2 — instruction emission                                       #
    # ------------------------------------------------------------------ #

    def _pass_two(self, tokens: list):
        toks = [t for _, t in tokens]  # flat list for indexed access
        n = len(toks)
        i = 0

        while i < n:
            tok = toks[i]

            # Labels are address markers only
            if is_label_def(tok):
                i += 1
                continue

            # Unknown tokens that aren't opcodes are skipped
            if tok not in OPCODE_SET:
                i += 1
                continue

            op = OPCODES[tok]
            i += 1

            # Fields
            agent   = 0
            src_reg = 0
            dst_reg = 0
            imm     = 0

            def peek(offset=0):
                idx = i + offset
                return toks[idx] if idx < n else None

            def consume():
                nonlocal i
                v = peek()
                i += 1
                return v

            def consume_imm():
                """Consume next token as immediate. Returns int."""
                t = peek()
                v = parse_imm(t)
                if v is not None:
                    consume()
                    return v
                return 0

            def consume_reg(prefix):
                """Consume next token as register with given prefix. Returns index."""
                t = peek()
                if t and t.startswith(prefix):
                    consume()
                    return reg_idx(t, prefix)
                return 0

            def consume_label():
                """Consume a label ref token and return its address (or queue patch)."""
                t = peek()
                if t and is_label_ref(t):
                    consume()
                    name = label_name_from_ref(t)
                    if name in self.sym:
                        return self.sym[name]
                    else:
                        # Record patch: imm field is bytes [4:8] of the word
                        self.patches.append((len(self.output) + 4, name))
                        return 0
                return 0

            def consume_any_imm_or_label():
                """Consume either a label ref (returning address) or a numeric imm."""
                t = peek()
                if t and is_label_ref(t):
                    return consume_label()
                return consume_imm()

            def skip_special():
                """Skip a SPECIAL token if present."""
                t = peek()
                if t in SPECIAL:
                    consume()

            # ── Operand parsing per opcode ────────────────────────── #

            if op == 0x01:    # SPAWN  A<n>, @label
                agent = consume_reg('A')
                imm   = consume_label()

            elif op == 0x06:  # SPAWN_MAP  <rows>, <cols>, @label
                r   = consume_imm()
                c   = consume_imm()
                imm = (r << 8) | c
                if peek() and is_label_ref(peek()):
                    # store label address in upper 24 bits is awkward;
                    # just record for patching
                    t = consume()
                    name = label_name_from_ref(t)
                    if name in self.sym:
                        imm = self.sym[name]
                    else:
                        self.patches.append((len(self.output) + 4, name))
                        imm = 0

            elif op == 0x07:  # WAIT  A<n>
                agent = consume_reg('A')

            elif op == 0x02:  # AGENT_KILL  SELF | ALL | A<n>
                t = peek()
                if t == 'ALL':
                    consume()
                    imm = 0xFF
                elif t == 'SELF':
                    consume()
                    imm = 0xFE
                elif t and t.startswith('A'):
                    agent = consume_reg('A')

            elif op == 0x03:  # FORK  <count>, @label
                v = parse_imm(peek())
                if v is not None:
                    consume()
                    src_reg = v & 0xFF   # store count in src_reg
                imm = consume_label()

            elif op == 0x04:  # MERGE  ALL, R<n>
                skip_special()   # skip ALL / SELF
                dst_reg = consume_reg('R')

            elif op == 0x05:  # BARRIER  <count>
                imm = consume_imm()

            elif op == 0x1A:  # SOM_MAP  A<n>, (row, col)
                agent = consume_reg('A')
                r     = consume_imm()
                c     = consume_imm()
                imm   = (r << 8) | c

            elif op == 0x1C:  # SOM_INIT  RANDOM | PCA
                t = peek()
                if t in SPECIAL:
                    consume()
                    imm = 0x01 if t == 'PCA' else 0x00

            elif op == 0x1D:  # SOM_WALK  SELF, GRADIENT
                skip_special()
                skip_special()

            elif op == 0x11:  # SOM_BMU  R<src>, R<dst>
                src_reg = consume_reg('R')
                dst_reg = consume_reg('R')

            elif op == 0x12:  # SOM_TRAIN  R<src>, S<som>
                src_reg = consume_reg('R')
                t = peek()
                if t and t.startswith('S'):
                    consume()

            elif op == 0x1B:  # SOM_SENSE  R<dst>
                dst_reg = consume_reg('R')

            elif op == 0x13:  # SOM_NBHD  R<bmu>, S<out>
                src_reg = consume_reg('R')
                t = peek()
                if t and t.startswith('S'):
                    consume()

            elif op == 0x14:  # WGHT_UPD  R<vec>, S<nbhd>
                src_reg = consume_reg('R')
                t = peek()
                if t and t.startswith('S'):
                    consume()

            elif op == 0x19:  # SOM_ELECT  R<n>
                dst_reg = consume_reg('R')

            elif op == 0x1E:  # SOM_DIST  R<a>, mem_label, R<dst>
                src_reg = consume_reg('R')
                # middle token is a memory label (stripped of brackets)
                t = peek()
                if t and not t.startswith('R') and not is_label_ref(t):
                    consume()   # skip memory label token
                dst_reg = consume_reg('R')

            elif op == 0x1F:  # LR_DECAY  <float|imm>
                imm = consume_imm()

            elif op == 0x20:  # MSG_SEND  (A<n>|PARENT|SELF), (R<n>|<imm>)
                t = peek()
                if t in SPECIAL:
                    consume()
                    agent = 0xFF if t == 'PARENT' else 0xFE
                elif t and t.startswith('A'):
                    agent = consume_reg('A')
                t2 = peek()
                if t2 and t2.startswith('R'):
                    src_reg = consume_reg('R')
                else:
                    imm = consume_imm()

            elif op == 0x21:  # MSG_RECV  R<dst>
                dst_reg = consume_reg('R')

            elif op == 0x23:  # BROADCAST  <imm>
                imm = consume_imm()

            elif op == 0x24:  # ACCUM  R<src>, R<dst>
                src_reg = consume_reg('R')
                dst_reg = consume_reg('R')

            elif op == 0x30:  # JMP  @label
                imm = consume_label()

            elif op in (0x31, 0x32):  # JZ / JNZ  R<n>, @label
                src_reg = consume_reg('R')
                imm     = consume_label()

            elif op in (0x33, 0x34):  # JEQ / JGT  R<a>, R<b>|<imm>, @label
                src_reg = consume_reg('R')
                t = peek()
                if t and t.startswith('R'):
                    dst_reg = consume_reg('R')
                else:
                    imm = consume_imm()
                    # now consume label
                    label_addr = consume_label()
                    imm = label_addr   # label overwrites numeric — that's the branch target

            elif op == 0x35:  # CALL  @label
                imm = consume_label()

            elif op == 0x36:  # RET
                pass

            elif op == 0x37:  # HALT
                pass

            elif op == 0x38:  # NOP
                pass

            elif op == 0x40:  # MOV  R<dst>|S<n>, val|R<src>|mem
                t = peek()
                if t and t.startswith('R'):
                    dst_reg = consume_reg('R')
                elif t and t.startswith('S'):
                    consume()  # S register — encode as dst_reg=0xFF stub
                    dst_reg = 0xFF
                # source
                t2 = peek()
                if t2 and t2.startswith('R'):
                    src_reg = consume_reg('R')
                    imm = 0
                else:
                    v = parse_imm(t2)
                    if v is not None:
                        consume()
                        imm = v
                    elif t2 and not is_label_ref(t2):
                        consume()   # memory label — skip, imm=0 stub

            elif op == 0x41:  # STORE  mem_label, R<src>
                t = peek()
                if t and not t.startswith('R') and not is_label_ref(t):
                    consume()   # skip memory label
                src_reg = consume_reg('R')

            elif op == 0x42:  # LOAD  R<dst>, mem_label
                dst_reg = consume_reg('R')
                t = peek()
                if t and not t.startswith('R'):
                    consume()   # skip memory label

            elif op == 0x43:  # TRAP  <imm>
                imm = consume_any_imm_or_label()

            elif op == 0x50:  # ADD  R<dst>, R<src>, imm|R<n>
                dst_reg = consume_reg('R')
                t = peek()
                if t and t.startswith('R'):
                    src_reg = consume_reg('R')
                else:
                    src_reg = dst_reg  # ADD Rdst, Rdst, imm shorthand
                t2 = peek()
                v = parse_imm(t2)
                if v is not None:
                    consume()
                    imm = v
                elif t2 and t2.startswith('R'):
                    # ADD Rdst, Rsrc, Rsrc2 — encode src2 in imm
                    imm = consume_reg('R')

            elif op in (0x51, 0x52, 0x53):  # SUB / MUL / DIV
                dst_reg = consume_reg('R')
                src_reg = consume_reg('R')
                t = peek()
                v = parse_imm(t)
                if v is not None:
                    consume(); imm = v
                elif t and t.startswith('R'):
                    imm = consume_reg('R')

            elif op in (0x54, 0x55):  # DOT / NORM
                dst_reg = consume_reg('R')

            else:
                pass  # no operands

            # ── Encode and emit the 8-byte instruction word ──────── #
            word = (
                (op      & 0xFF) << 56 |
                (agent   & 0xFF) << 48 |
                (src_reg & 0xFF) << 40 |
                (dst_reg & 0xFF) << 32 |
                (imm     & 0xFFFFFFFF)
            )
            self.output += struct.pack('>Q', word)

    # ------------------------------------------------------------------ #
    # Label patching                                                       #
    # ------------------------------------------------------------------ #

    def _patch_labels(self):
        for offset, name in self.patches:
            if name not in self.sym:
                print(f"ERROR: undefined label '{name}'", file=sys.stderr)
                sys.exit(1)
            addr = self.sym[name]
            self.output[offset:offset + 4] = struct.pack('>I', addr)

    # ------------------------------------------------------------------ #
    # Header                                                               #
    # ------------------------------------------------------------------ #

    def _write_header_magic(self):
        """Write initial header with placeholders."""
        hdr = struct.pack('>4sIBBBBIIIII',
            b'SOMA',
            0x00010000,    # version 1.0.0
            0x00,          # arch = ANY
            0x10,          # som_rows = 16
            0x10,          # som_cols = 16
            0x40,          # max_agents = 64
            self.HEADER_SIZE,  # code_offset = 32
            0,             # code_size   (filled later)
            0,             # data_offset (filled later)
            0,             # data_size
            0,             # som_offset
        )
        self.output[:self.HEADER_SIZE] = hdr

    def _finalise_header(self):
        code_size   = len(self.output) - self.HEADER_SIZE
        data_offset = len(self.output)   # data section would follow
        struct.pack_into('>I', self.output, 16, code_size)
        struct.pack_into('>I', self.output, 20, data_offset)
        struct.pack_into('>I', self.output, 24, 0)            # data_size
        struct.pack_into('>I', self.output, 28, data_offset)  # som_offset


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python bootstrap_assembler.py <input.soma> <output.sombin>",
              file=sys.stderr)
        sys.exit(1)

    src_path, out_path = sys.argv[1], sys.argv[2]

    try:
        with open(src_path, 'r', encoding='utf-8') as f:
            source = f.read()
    except OSError as e:
        print(f"ERROR: cannot read '{src_path}': {e}", file=sys.stderr)
        sys.exit(1)

    asm = Assembler()
    try:
        binary = asm.assemble(source)
    except Exception as e:
        print(f"ASSEMBLY ERROR: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

    try:
        with open(out_path, 'wb') as f:
            f.write(binary)
    except OSError as e:
        print(f"ERROR: cannot write '{out_path}': {e}", file=sys.stderr)
        sys.exit(1)

    instr_count = (len(binary) - 32) // 8
    print(f"✅ Assembled {src_path} → {out_path}  "
          f"({instr_count} instructions, {len(binary)} bytes)")
