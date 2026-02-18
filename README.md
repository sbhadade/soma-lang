# SOMA 🧠
**Self-Organizing Multi-Agent Binary Language**

> *A binary programming language built from the ground up for SOM neural topology, multi-agent coordination, and autonomous execution — on any substrate.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Status](https://img.shields.io/badge/status-working-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

---

## What is SOMA?

SOMA is not a framework. It is not a library. It is a **new programming language** — with its own binary format, grammar, instruction set, assembler, and runtime — designed to treat agents and self-organizing maps as **hardware primitives**, not abstractions.

Where other languages bolt agent systems on top as libraries, SOMA builds them into the **instruction word itself**. Every 64-bit instruction encodes who executes it, where on the SOM topology, and what to do.

---

## Quick Start

```bash
git clone https://github.com/sbhadade/soma-lang.git
cd soma-lang
chmod +x build.sh
./build.sh
```

That's it. No dependencies beyond Python 3.8+.

**Expected output:**
```
=== SOMA Build v2.0 ===
[1/5] Assembling SOMA self-assembler...
✅ Assembled assembler/somasc.soma → bin/somasc.sombin
[2/5] Assembling examples...
✅ Assembled examples/hello_agent.soma → bin/hello_agent.sombin  (13 instructions, 2 data symbols)
✅ Assembled examples/swarm_cluster.soma → bin/swarm_cluster.sombin  (23 instructions, 3 data symbols)
✅ Assembled examples/online_learner.soma → bin/online_learner.sombin  (37 instructions, 4 data symbols)
[3/5] Running hello_agent...   ✅
[4/5] Running swarm_cluster... ✅  (16 concurrent agents, JIT compiled 10 hot blocks)
[5/5] Running online_learner...✅
✅ Build & test successful!
```

---

## Repository Structure

```
soma-lang/
├── build.sh                        # One-command build & test
├── assembler/
│   └── somasc.soma                 # Self-hosting assembler (written in SOMA)
├── bootstrap/
│   └── bootstrap_assembler.py      # Python bootstrap assembler
├── runtime/
│   └── soma_runtime.py             # Python runtime interpreter
├── examples/
│   ├── hello_agent.soma            # Spawn agent, train SOM, receive result
│   ├── swarm_cluster.soma          # 16 concurrent agents, SOM leader election
│   └── online_learner.soma         # Streaming input, concept drift detection
├── spec/
│   └── SOMA.grammar                # Full EBNF grammar
├── bin/
│   └── SOMBIN.spec                 # Binary format specification
├── stdlib/
│   └── soma.stdlib                 # Standard library
└── docs/
    └── RATIONALE.md                # Design decisions & philosophy
```

---

## The Language

### Hello Agent

```soma
.SOMA    1.0.0
.ARCH    ANY
.SOMSIZE 4x4
.AGENTS  2
.LEARNRATE 0.5

.DATA
  input_vec : VEC  = <0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7, 0.3>
  result    : WGHT = 0.0

.CODE

@_start:
  SPAWN     A0, @worker_agent
  SOM_MAP   A0, (0,0)
  MOV       R0, [input_vec]
  MSG_SEND  A0, R0
  WAIT      A0
  MSG_RECV  R1
  STORE     [result], R1
  HALT

@worker_agent:
  MSG_RECV  R0
  SOM_TRAIN R0, S0
  SOM_SENSE R1
  MSG_SEND  PARENT, R1
  AGENT_KILL SELF
```

### Swarm Clustering

```soma
.SOMA    1.0.0
.SOMSIZE 16x16
.AGENTS  256

.CODE

@_start:
  SOM_INIT  RANDOM
  FORK      16, @explorer
  BROADCAST 0xBEEF
  BARRIER   16
  MERGE     ALL, R0
  SOM_ELECT R0
  HALT

@explorer:
  MSG_RECV  R0
  SOM_BMU   R0, R3
  SOM_WALK  SELF, GRADIENT
  SOM_TRAIN R0, S0
  AGENT_KILL SELF
```

---

## The 64-bit Instruction Word

```
 63      56 55     48 47     40 39     32 31                   0
┌──────────┬─────────┬─────────┬─────────┬──────────────────────┐
│  OPCODE  │ AGENT   │ SRC_REG │ DST_REG │         IMM          │
│  8 bits  │  8 bits │  8 bits │  8 bits │        32 bits       │
└──────────┴─────────┴─────────┴─────────┴──────────────────────┘
```

---

## Instruction Set

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| `0x01` | `SPAWN` | Create a new agent at entry point |
| `0x02` | `AGENT_KILL` | Terminate agent (SELF / ALL / A\<n\>) |
| `0x03` | `FORK` | Spawn N copies of agent |
| `0x04` | `MERGE` | Collect all agent results |
| `0x05` | `BARRIER` | Synchronise N agents |
| `0x07` | `WAIT` | Block until agent completes |
| `0x11` | `SOM_BMU` | Find Best Matching Unit |
| `0x12` | `SOM_TRAIN` | Train SOM node toward input vector |
| `0x13` | `SOM_NBHD` | Compute Gaussian neighbourhood |
| `0x14` | `WGHT_UPD` | Update weights in neighbourhood |
| `0x19` | `SOM_ELECT` | Elect leader node by activation |
| `0x1A` | `SOM_MAP` | Place agent at SOM coordinate |
| `0x1B` | `SOM_SENSE` | Read post-train node activation |
| `0x1C` | `SOM_INIT` | Initialise SOM (RANDOM / PCA) |
| `0x1F` | `LR_DECAY` | Decay learning rate |
| `0x20` | `MSG_SEND` | Send message to agent |
| `0x21` | `MSG_RECV` | Blocking receive |
| `0x23` | `BROADCAST` | Send to all agents |
| `0x30` | `JMP` | Unconditional jump |
| `0x31` | `JZ` | Jump if zero |
| `0x33` | `JEQ` | Jump if equal |
| `0x34` | `JGT` | Jump if greater |
| `0x37` | `HALT` | Terminate program |
| `0x40` | `MOV` | Load register or memory |
| `0x41` | `STORE` | Write to data memory |
| `0x43` | `TRAP` | System call |
| `0x50` | `ADD` | Integer addition |

Full ISA: [`bin/SOMBIN.spec`](bin/SOMBIN.spec)

---

## Runtime Features (v2.0)

### Phase 1 — Real DATA Section
`.DATA` declarations are parsed and binary-serialised as IEEE 754 float32. `MOV R0, [input_vec]` and `STORE [result], R1` perform actual memory reads/writes at runtime.

```
[DATA] Loaded symbols:
  input_vec   VEC   count=8   [0.8000, 0.2000, 0.6000, 0.4000...]
  result      WGHT  count=1   [0.0000]
```

### Phase 2 — Real Multi-Agent Threading
`FORK 16, @explorer` spawns 16 actual `threading.Thread` instances. Each has its own registers, call stack, and message inbox. `BROADCAST`, `BARRIER`, `WAIT`, and `MSG_SEND`/`MSG_RECV` are all thread-safe.

### Phase 3 — JIT Compiler
Hot basic blocks (executed 10+ times) are compiled from bytecode into native Python functions via `compile()`. The swarm_cluster example reports `JIT compiled 10 hot block(s)` — those blocks skip the interpreter loop entirely.

---

## .sombin Binary Format

```
Offset  Size  Field
0x00    4     MAGIC        "SOMA"
0x04    4     VERSION      0x00010000 = v1.0
0x08    1     ARCH         0=ANY 1=X86 2=ARM 3=RISCV 4=WASM
0x09    1     SOM_ROWS
0x0A    1     SOM_COLS
0x0B    1     MAX_AGENTS
0x0C    4     CODE_OFFSET  (always 32)
0x10    4     CODE_SIZE
0x14    4     DATA_OFFSET
0x18    4     DATA_SIZE
0x1C    4     SOM_OFFSET
```

Followed by:
- **Code section** — 8-byte instruction words (big-endian)
- **Data section** — symbol table + IEEE 754 float32 payload

---

## Status

| Component | Status |
|-----------|--------|
| Grammar spec | ✅ Complete |
| Binary format (.sombin) | ✅ Complete |
| ISA (54 opcodes) | ✅ v1.0 |
| Bootstrap assembler | ✅ Working — all 54 opcodes |
| DATA section (VEC/WGHT/COORD/INT) | ✅ Real binary serialisation |
| Runtime interpreter | ✅ Working — all opcodes |
| Multi-agent threading | ✅ Real concurrent threads |
| JIT compiler | ✅ Hot-block compilation |
| Standard library | ✅ Core done |
| Examples (3 programs) | ✅ All assembling & running |
| Self-hosting assembler | 🔧 In progress (somasc.soma) |
| x86-64 native backend | 📋 Planned |
| ARM64 native backend | 📋 Planned |

---

## Design Philosophy

**1. Agents are not threads.**
SOMA agents are first-class architectural objects with identity, SOM position, message queue, and lifecycle — encoded in the instruction set itself.

**2. The SOM is the scheduler.**
Agents migrate toward high-activation regions, creating emergent adaptive scheduling without a traditional OS scheduler.

**3. One binary. Any substrate.**
`.sombin` files target x86-64, ARM64, RISC-V, WASM, or bare metal. The runtime adapts to the host.

**4. The language bootstraps itself.**
`somasc.soma` — the SOMA assembler — is written in SOMA. The bootstrap Python assembler exists only to compile it.

---

## Author

**Swapnil Bhadade** — *Architect of SOMA*

> *"Most languages run on operating systems. SOMA is the operating system."*

---

## License

MIT — see [LICENSE](LICENSE)

---

*SOMA v2.0.0 — The language that thinks in maps. 🗺️*
