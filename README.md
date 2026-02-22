<div align="center">

<img src="soma-logo-cyberpunk.svg" alt="SOMA Language" width="800"/>

<br/>

[![PyPI version](https://img.shields.io/pypi/v/soma-lang?color=ff2d78&labelColor=04000f&style=for-the-badge&logo=pypi&logoColor=ff2d78)](https://pypi.org/project/soma-lang/)
[![CI](https://img.shields.io/github/actions/workflow/status/sbhadade/soma-lang/ci.yml?color=00ffe7&labelColor=04000f&style=for-the-badge&logo=github-actions&logoColor=00ffe7&label=CI)](https://github.com/sbhadade/soma-lang/actions)
[![Python](https://img.shields.io/pypi/pyversions/soma-lang?color=bf5fff&labelColor=04000f&style=for-the-badge&logo=python&logoColor=bf5fff)](https://pypi.org/project/soma-lang/)
[![License: MIT](https://img.shields.io/badge/license-MIT-ffd700?labelColor=04000f&style=for-the-badge)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/soma-lang?color=ff2d78&labelColor=04000f&style=for-the-badge&label=PyPI%20Downloads)](https://pypi.org/project/soma-lang/)

```
╔══════════════════════════════════════════════════════════════════╗
║  OPCODE · AGENT-ID · SOM-X · SOM-Y · REGISTER · IMMEDIATE        ║
║  8 bits    8 bits   8 bits  8 bits   16 bits     16 bits         ║
║          — one 64-bit word. that's all it takes. —               ║
╚══════════════════════════════════════════════════════════════════╝
```

**SOMA** is not a framework. Not a library. Not a wrapper around Python threads.  
It is a **binary programming language** where agents and SOM neural topology  
are encoded directly into the **instruction word itself.**

*Most languages run on operating systems. SOMA is the operating system.*

> **v4.0.0** — Curiosity + Binary Grammar · AgentSoul · SomTerrain · CDBG · 400 tests · 689× C speedup

</div>

---

## ⚡ One Command. Native Speed.

```bash
pip install soma-lang
soma transpile examples/hello_agent.soma -o hello.c
gcc -O3 -march=native -o hello hello.c -lm -lpthread
./hello
```

```
✅ Assembled hello_agent.soma → hello_agent.sombin  (13 instructions, 213 bytes)
🚀 Transpiled → hello.c
Registers:
  R0 = [0.8000, 0.2000, 0.6000, 0.4000, 0.9000, 0.1000, 0.7000, 0.3000]
  R1 = [0.4646, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
```

> **689× faster** than the Python interpreter. C transpiler + `gcc -O3 -march=native`. Real numbers.

---

## 🧠 What Is SOMA?

Every other multi-agent language bolts agents on top as a library.  
SOMA puts them **inside the instruction word.**

```
63      56 55     48 47     40 39     32 31      16 15       0
┌─────────┬─────────┬─────────┬─────────┬──────────┬─────────┐
│ OPCODE  │ AGENT-ID│  SOM-X  │  SOM-Y  │   REG    │   IMM   │
│  8 bits │  8 bits │  8 bits │  8 bits │  16 bits │ 16 bits │
└─────────┴─────────┴─────────┴─────────┴──────────┴─────────┘
```

Every instruction carries **Who** (Agent ID) · **Where** (SOM X,Y) · **What** (Opcode) · **With what** (Reg+Imm).

The SOM topology is not a data structure. It *is* the scheduler. Agents migrate toward high-activation regions. Coordination emerges from the map itself.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   USER PROGRAMS  (.soma)                        │
├────────────────┬────────────────┬───────────────────────────────┤
│    AGENT_A     │    AGENT_B     │    AGENT_N  ...               │
│  AgentSoul     │  AgentSoul     │  AgentSoul                    │
│  goal_vector   │  goal_vector   │  curiosity_drive              │
│  content_mem   │  content_mem   │  content_mem (fingerprints)   │
├────────────────┴────────────────┴───────────────────────────────┤
│            SOM COORDINATION PLANE + TERRAIN                     │
│     SOM MAP 0        MSG BUS         SOM MAP 1                  │
│  BMU·TRAIN·WALK   EVOLVE·META_SPAWN  SOUL_QUERY·GOAL_CHECK      │
│  ── SomTerrain ─────────────────────────────────────────────    │
│  hot_zones · cold_zones · sacred_places · virgin_territory      │
├─────────────────────────────────────────────────────────────────┤
│           SOMA BINARY RUNTIME  (CDBG v4)                        │
│    Assembler │ Transpiler │ Learn Engine │ CDBG 5-byte frames   │
├──────────┬──────────┬──────────┬──────────────────────────────┤
│  x86-64  │  ARM64   │  RISC-V  │  WASM (planned)              │
└──────────┴──────────┴──────────┴──────────────────────────────┘
```

---

## 💻 Write Agents, Not Threads

### Hello Agent

```soma
.SOMA    3.0.0
.ARCH    ANY
.SOMSIZE 4x4
.AGENTS  2

.DATA
  payload : MSG = 0xFF42

.CODE
@_start:
  SPAWN     A0, @worker        ; birth a new agent
  SOM_MAP   A0, (0,0)          ; place it on the topology
  MSG_SEND  A0, [payload]      ; send it data
  WAIT      A0
  HALT

@worker:
  MSG_RECV  R0                 ; receive input vector
  SOM_TRAIN R0, S0             ; train the SOM node
  MSG_SEND  PARENT, 0x00       ; signal done
  AGENT_KILL SELF
```

### Swarm Clustering (256 agents)

```soma
.SOMA    3.0.0
.SOMSIZE 16x16
.AGENTS  256

.CODE
@_start:
  SOM_INIT  RANDOM             ; randomize weight map
  FORK      16, @explorer      ; spawn 16 explorer agents
  BROADCAST 0xBEEF             ; send data to all
  BARRIER   16                 ; wait for convergence
  SOM_ELECT R0                 ; democratic leader election
  HALT

@explorer:
  MSG_RECV  R0
  SOM_WALK  SELF, GRADIENT     ; migrate toward activation
  SOM_TRAIN R0, S0
  AGENT_KILL SELF
```

### Curious Agent (v4.0 — Phase III)

```soma
.SOMA    4.0.0
.SOMSIZE 16x16
.AGENTS  8

.CODE
@_start:
  SOM_INIT   RANDOM
  SPAWN      A0, @curious_agent
  WAIT       A0
  HALT

@curious_agent:
  TERRAIN_READ R0              ; read collective wisdom at this position
  MSG_RECV     R1              ; receive goal template
  GOAL_SET     R1              ; encode intended future state

@learn_loop:
  SOM_BMU    R0
  SOM_TRAIN  R0, S0
  GOAL_CHECK R1                ; measure distance to goal
  GOAL_STALL @curiosity        ; jump if stall_count > threshold
  TERRAIN_MARK R0
  JMP        @learn_loop

@curiosity:
  INTROSPECT                   ; agent reads its own state before deciding
  META_SPAWN 4, @candidate     ; spawn 4 agents with mutated goal vectors
  BARRIER    4
  EVOLVE     A1                ; select child closest to its own declared goal
  SOUL_INHERIT A1              ; winner carries this agent's emotional memory
  CDBG_EMIT                   ; broadcast 5-byte CDBG identity frame
  AGENT_KILL SELF
```

---

## 🔥 Full Instruction Set

### Agent Lifecycle

| Code | Mnemonic | Description |
|------|----------|-------------|
| `0x01` | `SPAWN` | Create a new agent |
| `0x02` | `AGENT_KILL` | Terminate agent |
| `0x03` | `FORK` | Duplicate agent N times |
| `0x04` | `MERGE` | Merge N agent results |
| `0x05` | `BARRIER` | Synchronize N agents |
| `0x06` | `SPAWN_MAP` | Spawn N×M agents on SOM grid |
| `0x07` | `WAIT` | Wait for agent to die |

### SOM Operations

| Code | Mnemonic | Description |
|------|----------|-------------|
| `0x11` | `SOM_BMU` | Find best matching unit |
| `0x12` | `SOM_TRAIN` | Kohonen weight update |
| `0x13` | `SOM_NBHD` | Compute Gaussian neighbourhood |
| `0x19` | `SOM_ELECT` | Democratic leader election |
| `0x1A` | `SOM_MAP` | Place agent at SOM coordinate |
| `0x1B` | `SOM_SENSE` | Read activation at agent's node |
| `0x1C` | `SOM_INIT` | Initialise SOM weights |
| `0x1D` | `SOM_WALK` | Move agent along topology |
| `0x1E` | `SOM_DIST` | Distance between two agent positions |
| `0x1F` | `LR_DECAY` | Decay learning rate |

### Messaging

| Code | Mnemonic | Description |
|------|----------|-------------|
| `0x20` | `MSG_SEND` | Send message to agent |
| `0x21` | `MSG_RECV` | Blocking receive |
| `0x23` | `BROADCAST` | Send to ALL agents |

### Phase II — Emotional Memory

| Code | Mnemonic | Description |
|------|----------|-------------|
| — | `EMOT_TAG` | Attach valence + intensity to SOM node |
| — | `DECAY_PROTECT` | Shield emotional memory from decay |
| — | `PREDICT_ERR` | Compute surprise (BMU distance vs prediction) |
| — | `EMOT_RECALL` | Retrieve emotional tag by coordinate |
| — | `SURPRISE_CALC` | Prediction error from raw vectors |

### Phase III — Curiosity *(NEW in v4.0)*

| Code | Mnemonic | Description |
|------|----------|-------------|
| `0x60` | `GOAL_SET` | Set agent goal vector (target weight-space state) |
| `0x61` | `GOAL_CHECK` | Measure distance to goal; update stall counter |
| `0x62` | `SOUL_QUERY` | Pattern-match content memory (intuition) |
| `0x63` | `META_SPAWN` | Spawn N agents with mutated goal vectors |
| `0x64` | `EVOLVE` | Select child by goal proximity; inherit soul |
| `0x65` | `INTROSPECT` | Export soul state snapshot as readable data |
| `0x66` | `TERRAIN_READ` | Read collective terrain at current position |
| `0x67` | `TERRAIN_MARK` | Write emotional data into terrain |
| `0x68` | `SOUL_INHERIT` | Explicit soul inheritance from another agent |
| `0x69` | `GOAL_STALL` | Jump to label if goal is stalled |

### Phase IV — CDBG *(NEW in v4.0)*

| Code | Mnemonic | Description |
|------|----------|-------------|
| `0x70` | `CDBG_EMIT` | Emit 5-byte CDBG frame to message bus |
| `0x71` | `CDBG_RECV` | Receive and decode a CDBG frame |
| `0x72` | `CTX_SWITCH` | Set active decode context (CTX nibble) |

*Full ISA → [`SOMBIN.spec`](SOMBIN.spec)*

---

## 🆕 v4.0.0 — Curiosity + Binary Grammar

### Phase III — The AgentSoul

Agents now have a **portable identity** that survives map migration, EVOLVE selection, and generational inheritance.

Emotional memory is indexed by **SHA-256 fingerprint of the weight vector** — not by SOM coordinate. When an agent arrives at any new position, it computes the fingerprint of the weights there and queries its `content_memory`. If the pattern matches something felt strongly before, the emotional tag fires — regardless of where on the map it is.

That is the computational definition of **intuition**.

```python
from runtime.som.soul import AgentSoul

soul = AgentSoul(agent_id=1)

# Set a goal — what the agent wants to become
soul.goal_set(target_weights)

# Every pulse: measure distance to goal
dist, curious = soul.goal_check(current_weights)

# If curious (goal stalled), tag the memory and explore
if curious:
    soul.tag_memory(current_weights, valence=-0.3, intensity=0.8)
    new_goals = soul.spawn_mutated_goals(n=4)   # META_SPAWN

# When a new input feels like a past danger — the soul knows before the map does
hit = soul.soul_query(new_weights)
if hit and hit.valence < 0:
    pass  # Intuition: slow down, this pattern hurt us before
```

### Phase III — SomTerrain

The **map has memory**. Nobody programs the geography. It emerges.

```python
from runtime.som.terrain import SomTerrain

terrain = SomTerrain(rows=16, cols=16)

# Every time an agent fires EMOT_TAG here, terrain learns
terrain.mark(row=3, col=7, pulse=t, valence=0.8, intensity=0.9)

# Read before navigating — collective wisdom from all past agents
info = terrain.read(row=3, col=7)
# {'is_hot_zone': True, 'is_virgin': False, 'cultural_deposit': 0.34, ...}

# Curious agent finds the frontier
r, c = terrain.most_curious_node()   # highest exploration_reward

# Dying agent deposits soul — sacred place forms
terrain.deposit_soul(row=agent.r, col=agent.c, salience=soul_salience)
```

Geography that emerges automatically:

| Zone Type | What It Means | How It Forms |
|-----------|---------------|--------------|
| **Hot zone** | Consistently high positive valence | Many agents succeeded here |
| **Cold zone** | Collective danger | Many agents failed or suffered here |
| **Sacred place** | High `cultural_deposit` | Dying agents chose to leave memories here |
| **Virgin territory** | `attractor_count ≈ 0` | Nothing happened here yet — the frontier |

### Phase IV — Context-Discriminated Binary Grammar (CDBG)

One 5-byte frame. Seven meanings. Zero extra opcodes.

```
┌──────────┬──────────┬────────────────────────┬─────────────────┐
│  CTX[4b] │  SUB[4b] │   PAYLOAD  (3 bytes)   │ CHK[4b] R[4b]  │
└──────────┴──────────┴────────────────────────┴─────────────────┘
  Same 3 bytes. Different CTX nibble. Completely different meaning.
```

| CTX | Namespace | 3-byte Payload Means |
|-----|-----------|----------------------|
| `0x0` | `SOM_MAP` | `X[8] · Y[8] · OPCODE[8]` — coordinate + instruction |
| `0x1` | `AGENT` | 24-bit flat ID: `cluster[4] · map[8] · seq[12]` = 16.7M agents |
| `0x2` | `SOUL` | `field_id[8] · value[16]` — one soul field update in fp16 |
| `0x3` | `MEMORY` | 24-bit hash prefix — content-addressed memory bucket pointer |
| `0x4` | `PULSE` | 24-bit heartbeat counter |
| `0x5` | `EMOTION` | `row[8] · valence[8] · intensity[8]` — emotional tag |
| `0x6` | `HISTORY` | `generation[8] · goal_record_id[16]` — lifecycle event |

```python
from soma.cdbg import Encoder, Frame, StreamDecoder

# Encode an agent identity frame
wire = Encoder.agent(0x234567).encode()   # 5 bytes
frame = Frame.decode(wire)
parsed = frame.parsed()
# {'context': 'AGENT', 'cluster': 2, 'map_id': 52, 'seq': 1383}

# Stream decode
dec = StreamDecoder()
for frame in dec.feed(incoming_bytes):
    if frame and frame.ctx.name == 'EMOTION':
        handle_emotion(frame.parsed())
```

The opcode table stays **exactly the same size** forever. Only CTX namespaces scale.

---

## 🧬 Phase 2.5 — Liveliness

SOMA v3.2+ implements the amygdala + hippocampus primitives from *"A Path to AGI Part II: Liveliness"*:

```
High surprise (PREDICT_ERR) → high emotion tag → slow decay → strong memory
Low surprise                → low tag          → fast decay → forgotten
```

### Emotional Memory — `EMOT_TAG` / `DECAY_PROTECT`

```python
from runtime.som.emotion import EmotionRegistry, ProtectMode

em  = EmotionRegistry()
es  = em.get_or_create(agent_id=0)

# Tag a SOM node after a surprising input
es.emot_tag(row=2, col=2, valence=0.9, intensity=0.8)

# Shield it from decay for 100 pulses
es.decay_protect(2, 2, mode=ProtectMode.CYCLES, cycles=100)
```

### Memory Consolidation — `MEMORY_CONSOLIDATE`

```python
from runtime.som.memory import MemoryManager

mem    = MemoryManager(som, em)
report = mem.consolidate(agent_id=0)
# promoted=1, pruned=0, decayed=8, took=0.08ms
```

Two-tier system mirrors hippocampal memory consolidation:
- **Working SOM** — volatile, fast decay, 100 Hz pulse rate
- **Long-term SOM** — persistent; top 10% by emotion salience promoted each REM cycle
- **Hard prune** — nodes below 0.5% weight strength are removed

---

## 📦 Register Architecture

| Register | Count | Width | Purpose |
|----------|-------|-------|---------|
| `R0–R15` | 16 | 256-bit | General purpose / weight vectors |
| `A0–A63` | 64 | 64-bit | Agent handles |
| `S0–S15` | 16 | 64-bit | SOM state (S0=lr, S1=sigma, S2=epoch) |

---

## 📁 Repository Structure

```
soma-lang/
├── soma/
│   ├── isa.py               ← Canonical opcode table (v3.0 + Phase III+IV)
│   ├── vm.py                ← Test VM — all opcodes dispatched
│   ├── assembler.py         ← .soma → .sombin
│   ├── cdbg.py              ← Context-Discriminated Binary Grammar (NEW v4.0)
│   └── lexer.py
├── runtime/
│   └── som/
│       ├── soul.py          ← AgentSoul + MasterSoul + SoulRegistry (NEW v4.0)
│       ├── terrain.py       ← SomTerrain + TerrainRegistry (NEW v4.0)
│       ├── emotion.py       ← Phase 2.5 — EmotionRegistry, EMOT_TAG
│       ├── memory.py        ← Phase 2.6 — EMOT_RECALL, SURPRISE_CALC
│       ├── som_map.py       ← LiveSomMap
│       └── som_scheduler.py ← SomScheduler
├── examples/
│   ├── soma_curious.soma    ← Full curiosity example (NEW v4.0)
│   ├── hello_agent.soma
│   └── swarm_cluster.soma
├── tests/
│   ├── test_curiosity_cdbg.py  ← 41 tests for Phase III+IV (NEW v4.0)
│   ├── test_phase26.py
│   ├── test_liveliness.py
│   ├── test_agent.py
│   └── test_soma.py
├── spec/
│   └── SOMA.grammar
└── bin/
    └── SOMBIN.spec
```

---

## 🗺️ AGI Staircase

```
Step 1  PULSE            ✅  System pulses. It is alive.
Step 2  SOM topology     ✅  Agents live on a map. Coordinates matter.
Step 3  MSG passing      ✅  Agents communicate. State is shared.
Step 4  Emotion + Decay  ✅  System grows, forgets, feels. It is lively.
Step 5  Curiosity        ✅  AgentSoul + SomTerrain + EVOLVE. It wants to learn.
Step 6  CDBG Scaling     ✅  Opcode table stays fixed as system grows to millions.
Step 7  Collective Intel 📋  NICHE_DECLARE, SYMBOL_EMERGE, HERITAGE_LOAD.
Step 8  Self-hosting     📋  somasc.soma assembles itself.
        ↑
        Nobody knows exactly where on this staircase 'intelligence' appears.
        But this is the most concrete path anyone is building right now.
```

---

## 📊 Build Status

| Component | Version | Status |
|-----------|---------|--------|
| Grammar spec | v4.0 | ✅ Complete |
| Binary format (CDBG) | v4.0 | ✅ Complete — 5-byte frames, 7 CTX namespaces |
| ISA | v3.0 + Phase III+IV | ✅ Complete — 56 opcodes |
| Assembler (classic 8-byte) | v3.0 | ✅ Working |
| Assembler (CDBG 5-byte emit) | — | ⚠️ Planned |
| VM dispatch | v4.0 | ✅ All opcodes dispatched |
| C transpiler | v3.0 | ✅ 689× speedup |
| C transpiler (new opcodes) | — | ⚠️ Planned |
| AgentSoul | v4.0 | ✅ Complete + tested |
| SomTerrain | v4.0 | ✅ Complete + tested |
| CDBG encoder/decoder | v4.0 | ✅ Complete + tested |
| Emotional memory (Phase 2.5) | v3.2 | ✅ Complete |
| Memory consolidation (Phase 2.6) | v3.2 | ✅ Complete |
| Liveliness decay | v3.2 | ✅ decay_step · prune_check · protect modes |
| True concurrency | v3.1 | ✅ AgentRegistry + real pthreads |
| SOM scheduling | v3.1 | ✅ LiveSomMap + SomScheduler + Visualizer |
| PyPI package | v3.2.0 | ✅ `pip install soma-lang` |
| GitHub Actions CI | v3.x | ✅ Matrix 3.9–3.12 × ubuntu/macOS/win |
| Trusted Publishing | v3.x | ✅ OIDC — no secrets |
| Test suite | v4.0 | ✅ **400 passed** in 7.12s |
| JIT backend | — | 📋 Planned |
| WASM backend | — | 📋 Planned |
| Phase V — Collective Intelligence | — | 📋 Next |

---

## 🗺️ Roadmap

| Phase | Timeline | Milestone |
|-------|----------|-----------|
| **0 — Foundation** | ✅ Done | PyPI v3.0.0 · CI · C transpiler · 340× speedup |
| **1 — Concurrency** | ✅ Feb 2026 | AgentRegistry + ThreadAgent · 689× C vs Python · 246/246 tests |
| **2 — SOM Live** | ✅ Feb 2026 | LiveSomMap · SomScheduler · SomVisualizer · 300/300 tests |
| **2.5 — Liveliness** | ✅ Feb 2026 | EmotionRegistry · MemoryManager · decay + consolidation |
| **2.6 — Memory Share** | ✅ Feb 2026 | EMOT_RECALL · SURPRISE_CALC · broadcast · neighbor sync |
| **3 — Curiosity** | ✅ Feb 2026 | AgentSoul · SomTerrain · EVOLVE · META_SPAWN · 41 tests |
| **4 — CDBG** | ✅ Feb 2026 | 5-byte binary grammar · 7 CTX namespaces · CRC-4 · StreamDecoder |
| **5 — Collective Intel** | May 2026 | NICHE_DECLARE · SYMBOL_EMERGE · HERITAGE_LOAD |
| **6 — Transpiler+** | Jun 2026 | SIMD (AVX2/NEON) · OpenMP · multi-arch · LLVM backend |
| **7 — Self-hosting** | Jul 2026 | somasc.soma assembles itself · SOMA-OS bare metal demo |

---

## 🔬 Academic Context

SOMA's architecture is grounded in:

- **Khacef et al. (arXiv 1810.12640)** — Distributed SOM with spiking neurons. Closest academic predecessor.
- **FPGA-based SOM accelerators** — 100× speedup over CPU. SOMA is the programming model these chips need.
- **Memristor SOM chips** (Nature Comms, 2022) — in-situ SOM training. SOMA targets this substrate.
- **Amygdala + hippocampus models** — SOMA Phase 2.x implements the computational equivalents: emotional tagging, decay protection, memory consolidation.
- **Evolutionary computation** — SOMA's EVOLVE + META_SPAWN implements machine-speed goal-directed evolution with no human-defined fitness function. The agent's own declared intention is the fitness criterion.

---

## 🚀 Quick Start

```bash
git clone https://github.com/sbhadade/soma-lang
cd soma-lang
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run only Phase III + IV tests
pytest tests/test_curiosity_cdbg.py -v

# Assemble and run a program
soma assemble examples/soma_curious.soma -o curious.sombin
soma run curious.sombin
```

---

<div align="center">

**Built by [`sbhadade`](https://github.com/sbhadade)**

*"Most languages run on operating systems. SOMA is the operating system."*

[![Star on GitHub](https://img.shields.io/github/stars/sbhadade/soma-lang?color=ff2d78&labelColor=04000f&style=for-the-badge&logo=github&logoColor=ff2d78)](https://github.com/sbhadade/soma-lang/stargazers)

---

**© 2026 Swapnil Bhadade. MIT License.**

</div>