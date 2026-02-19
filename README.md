<div align="center">
<img src="soma-logo.svg" alt="SOMA Language" width="800"/>
<br/>



# SOMA 🧠 Implementation

> Full working implementation of the SOMA Self-Organizing Multi-Agent Binary Language

## What's in this repository

```
soma-lang/
├── soma/                    # Python package (pure stdlib, no deps)
│   ├── __init__.py          # version = "1.0.0"
│   ├── isa.py               # Opcode table, register encoding, binary constants
│   ├── lexer.py             # Tokenizer for .soma source files
│   ├── assembler.py         # .soma → .sombin assembler + disassembler
│   ├── vm.py                # SomMap, AgentState, SomaVM interpreter
│   └── cli.py               # `soma` command-line tool
├── tests/
│   └── test_soma.py         # 97 automated tests (stdlib unittest)
├── playground/
│   └── index.html           # Single-file browser playground (pure JS)
├── pyproject.toml           # pip install soma-lang
├── setup.py
└── .github/workflows/ci.yml # GitHub Actions CI (3 OS × 4 Python versions)
```

---

## Quick Start

### Install

```bash
# From PyPI (once published)
pip install soma-lang

# Or from source
git clone https://github.com/sbhadade/soma-lang
cd soma-lang
pip install -e .
```

### CLI Usage

```bash
# Assemble a .soma source file
soma asm hello.soma                     # → hello.sombin
soma asm hello.soma -o out.sombin

# Run a .sombin binary
soma run hello.sombin
soma run hello.sombin --trace           # step-by-step trace
soma run hello.sombin --max-steps 1000000

# Assemble + run in one shot
soma exec hello.soma

# Disassemble a binary
soma disasm hello.sombin

# Show version
soma version
```

### Python API

```python
from soma.assembler import assemble, disassemble
from soma.vm import SomaVM

src = """
.SOMA    1.0.0
.ARCH    ANY
.SOMSIZE 4x4
.AGENTS  2

.CODE
@_start:
  SPAWN A0, @worker
  MSG_SEND A0, 0x42
  WAIT A0
  HALT

@worker:
  MSG_RECV R0
  AGENT_KILL SELF
"""

binary = assemble(src)          # → bytes (.sombin format)
print(f"Binary: {len(binary)} bytes")

vm = SomaVM(binary, max_steps=100_000)
vm.run()

child = vm.agents[1]
print(f"Worker received: R0 = {child.R[0]:#x}")   # 0x42

# Inspect SOM map
som = vm.som
bmu_r, bmu_c = som.bmu([0.5] * 16)
print(f"BMU for [0.5…]: ({bmu_r}, {bmu_c})")

# Disassemble
print(disassemble(binary))
```

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│    .soma source  →  lexer  →  parser    │
│                  assembler              │
│                  .sombin binary         │
├─────────────────────────────────────────┤
│              SomaVM                     │
│   ┌──────────┬──────────┬────────────┐  │
│   │ Agents   │ SomMap   │ Msg buses  │  │
│   │ R/A/S    │ BMU      │ inbox[]    │  │
│   │ registers│ train    │ broadcast  │  │
│   └──────────┴──────────┴────────────┘  │
├──────────┬──────────┬────────┬──────────┤
│  x86-64  │  ARM64   │ RISCV  │  WASM    │
│ (header  │ (header  │ (hdr)  │ (hdr)   │
│  byte=1) │  byte=2) │ b=3    │ b=4     │
└──────────┴──────────┴────────┴──────────┘
```

---

## .sombin Binary Format

```
Offset  Size  Field
0x00    4     MAGIC       "SOMA" (0x534F4D41)
0x04    2     VER_MAJOR
0x06    2     VER_MINOR
0x08    1     ARCH_TARGET  0=ANY 1=X86 2=ARM64 3=RISCV 4=WASM
0x09    1     SOM_ROWS
0x0A    1     SOM_COLS
0x0B    1     MAX_AGENTS
0x0C    4     CODE_OFFSET
0x10    4     CODE_SIZE
0x14    4     DATA_OFFSET
0x18    4     DATA_SIZE
0x1C    2     SOM_OFFSET
0x1E    2     FLAGS        bit0=SELF_MODIFYING
```

Each instruction is a 64-bit big-endian word:

```
63      56 55     48 47     40 39     32 31      16 15       0
┌─────────┬─────────┬─────────┬─────────┬──────────┬─────────┐
│ OPCODE  │ AGENT-ID│  SOM-X  │  SOM-Y  │   REG    │   IMM   │
│  8 bits │  8 bits │  8 bits │  8 bits │  16 bits │ 16 bits │
└─────────┴─────────┴─────────┴─────────┴──────────┴─────────┘
```

---

## Register Architecture

| Register | Count | Encoding  | Purpose                        |
|----------|-------|-----------|--------------------------------|
| R0–R15   | 16    | 0x0000–F  | General purpose / weight vectors |
| A0–A63   | 64    | 0x0100–3F | Agent handles                  |
| S0–S15   | 16    | 0x0200–F  | SOM state (lr, sigma, epoch…)  |
| SELF     | —     | 0xFF00    | This agent's ID                |
| PARENT   | —     | 0xFF01    | Parent agent's ID              |
| ALL      | —     | 0xFF02    | All agents (broadcast target)  |

---

## ARM64 / WASM Backend

The `.sombin` format's `ARCH_TARGET` byte selects the native code generation
target when a JIT or AOT backend is present. The current Python runtime is
target-agnostic (it interprets the binary regardless of the header byte).

To add a true native backend:

```python
from soma.vm import SomaVM

class ARM64Backend(SomaVM):
    """Override _step to emit AArch64 machine code via LLVM/ctypes."""
    def _step(self, agent):
        # Emit AArch64 instructions for each SOMA opcode
        ...
```

For WebAssembly: the `playground/index.html` is already a pure-JS implementation
that runs in any browser with zero installation.

---

## Running Tests

```bash
# stdlib unittest (zero dependencies)
python -m unittest discover -s tests -v

# With pytest (if installed)
pytest tests/ -v

# Specific test class
python -m unittest tests.test_soma.TestVMSemantics -v
```

Test coverage:
- ISA opcode table completeness (31+ mnemonics)
- Register encoding/decoding (R/A/S/SELF/PARENT/ALL)
- Lexer correctness (all token types)
- Assembler binary format (magic, version, header fields, offsets)
- Instruction encoding (64-bit layout, operand placement)
- VM semantics (MOV, ADD, SUB, CMP, JMP, CALL/RET)
- SOM operations (BMU exact match, training convergence, neighbourhood)
- Multi-agent (SPAWN, FORK, MSG_SEND/RECV, BROADCAST, SOM_ELECT)
- Full programs (Hello Agent, Swarm Clustering)
- Disassembler round-trip
- Architecture targets (ANY, X86, ARM64, RISCV, WASM header bytes)
- Error handling (invalid binary, corrupted PC, inbox blocking)

---

## Web Playground

Open `playground/index.html` in any modern browser — no server, no install.

Features:
- Full SOMA assembler (JS port of `soma/assembler.py`)
- Complete VM interpreter with agent scheduler and SOM map
- Live hex dump of `.sombin` output
- Disassembly view
- VM state panel (agents, registers, SOM position)
- SOM map visualization (color-coded node activation)
- 4 built-in examples
- Ctrl+Enter to run

---

## License

MIT
