"""SOMA Lexer — tokenizes .soma assembly source."""
from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class TT(Enum):
    """Token types."""
    DIRECTIVE   = auto()   # .SOMA  .ARCH  .CODE  .DATA  .SOMSIZE  .AGENTS
    LABEL_DEF   = auto()   # @label:
    LABEL_REF   = auto()   # @label
    MNEMONIC    = auto()   # SPAWN  MSG_SEND  ...
    REG         = auto()   # R0  A3  S1  SELF  PARENT  ALL
    INTEGER     = auto()   # 42  0xFF  0b1010
    FLOAT       = auto()   # 3.14
    STRING      = auto()   # "hello"
    COORD       = auto()   # (x,y)
    COMMA       = auto()
    COLON       = auto()
    LBRACKET    = auto()
    RBRACKET    = auto()
    NEWLINE     = auto()
    EOF         = auto()
    COMMENT     = auto()   # ; ...
    IDENT       = auto()   # bare identifiers (RANDOM, GRADIENT, etc.)


@dataclass
class Token:
    type: TT
    value: object
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


_DIRECTIVES = {
    ".SOMA", ".ARCH", ".SOMSIZE", ".AGENTS", ".CODE", ".DATA",
    ".SELF_MODIFYING", ".FLAGS",
}

_REG_RE = re.compile(
    r'\b(R(?:1[0-5]|[0-9])|A(?:[1-5][0-9]|6[0-3]|[0-9])|S(?:1[0-5]|[0-9])|SELF|PARENT|ALL)\b'
)

_TOKEN_SPEC = [
    ("COMMENT",   r';[^\n]*'),
    ("VERSION",   r'\d+\.\d+\.\d+'),          # version strings like 1.0.0
    ("FLOAT",     r'[+-]?\d+\.\d+(?:[eE][+-]?\d+)?'),
    ("HEX",       r'0[xX][0-9A-Fa-f]+'),
    ("BIN",       r'0[bB][01]+'),
    ("INT",       r'[+-]?\d+'),
    ("STRING",    r'"[^"]*"'),
    ("COORD",     r'\(\s*\d+\s*,\s*\d+\s*\)'),
    ("LABEL_DEF", r'@[A-Za-z_][A-Za-z0-9_]*\s*:'),
    ("LABEL_REF", r'@[A-Za-z_][A-Za-z0-9_]*'),
    ("DIRECTIVE", r'\.[A-Z_][A-Z0-9_]*'),
    ("WORD",      r'[A-Za-z_][A-Za-z0-9_]*'),
    ("LBRACKET",  r'\['),
    ("RBRACKET",  r'\]'),
    ("COMMA",     r','),
    ("COLON",     r':'),
    ("EQUALS",    r'='),
    ("NEWLINE",   r'\n'),
    ("SKIP",      r'[ \t\r]+'),
    ("MISMATCH",  r'.'),
]

_MASTER_RE = re.compile(
    '|'.join(f'(?P<{name}>{pat})' for name, pat in _TOKEN_SPEC)
)

_REGISTERS = {
    "SELF", "PARENT", "ALL",
    *[f"R{i}" for i in range(16)],
    *[f"A{i}" for i in range(64)],
    *[f"S{i}" for i in range(16)],
}

from soma.isa import OPCODES
_MNEMONICS = set(OPCODES.keys())


def tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    line = 1
    line_start = 0

    for mo in _MASTER_RE.finditer(source):
        kind = mo.lastgroup
        val  = mo.group()
        col  = mo.start() - line_start + 1

        if kind == "SKIP":
            continue
        elif kind == "VERSION":
            tokens.append(Token(TT.IDENT, val, line, col))
        elif kind == "COMMENT":
            # skip comments entirely
            continue
        elif kind == "NEWLINE":
            tokens.append(Token(TT.NEWLINE, "\n", line, col))
            line += 1
            line_start = mo.end()
        elif kind == "DIRECTIVE":
            if val.upper() in _DIRECTIVES:
                tokens.append(Token(TT.DIRECTIVE, val.upper(), line, col))
            else:
                tokens.append(Token(TT.IDENT, val, line, col))
        elif kind == "LABEL_DEF":
            name = val.rstrip(":").strip()
            tokens.append(Token(TT.LABEL_DEF, name, line, col))
        elif kind == "LABEL_REF":
            tokens.append(Token(TT.LABEL_REF, val, line, col))
        elif kind == "HEX":
            tokens.append(Token(TT.INTEGER, int(val, 16), line, col))
        elif kind == "BIN":
            tokens.append(Token(TT.INTEGER, int(val, 2), line, col))
        elif kind == "INT":
            tokens.append(Token(TT.INTEGER, int(val, 10), line, col))
        elif kind == "FLOAT":
            tokens.append(Token(TT.FLOAT, float(val), line, col))
        elif kind == "STRING":
            tokens.append(Token(TT.STRING, val[1:-1], line, col))
        elif kind == "COORD":
            inner = val[1:-1].split(",")
            tokens.append(Token(TT.COORD, (int(inner[0].strip()), int(inner[1].strip())), line, col))
        elif kind == "WORD":
            up = val.upper()
            if up in _MNEMONICS:
                tokens.append(Token(TT.MNEMONIC, up, line, col))
            elif val in _REGISTERS or up in _REGISTERS:
                tokens.append(Token(TT.REG, val.upper(), line, col))
            else:
                tokens.append(Token(TT.IDENT, val, line, col))
        elif kind == "LBRACKET":
            tokens.append(Token(TT.LBRACKET, "[", line, col))
        elif kind == "RBRACKET":
            tokens.append(Token(TT.RBRACKET, "]", line, col))
        elif kind == "COMMA":
            tokens.append(Token(TT.COMMA, ",", line, col))
        elif kind == "COLON":
            tokens.append(Token(TT.COLON, ":", line, col))
        elif kind == "EQUALS":
            tokens.append(Token(TT.IDENT, "=", line, col))
        elif kind == "MISMATCH":
            raise SyntaxError(f"Unexpected character {val!r} at line {line}, col {col}")

    tokens.append(Token(TT.EOF, None, line, 0))
    return tokens
