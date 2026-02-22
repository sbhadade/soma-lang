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

> **v4.1.0** — Full coherence pass · Assembler wired · C transpiler updated · stdlib added · `soma_curious.soma` assembles to correct binary · 400 tests passing

</div>

---

## ⚡ One Command. Native Speed.

```bash
pip install soma-lang
soma assemble examples/soma_curious.soma -o curious.sombin
soma transpile examples/soma_curious.soma -o curious.c
gcc -O3 -march=native -o curious curious.c -lm -lpthread
./curious
```

```
✅ Assembled soma_curious.soma → soma_curious.sombin  (47 instructions, 376 bytes)
🚀 Transpiled → curious.c
Agent 0x01 | TERRAIN_READ  → exploration_reward = 0.94 (virgin territory)
Agent 0x01 | GOAL_SET      → goal encoded (16-dim weight vector)
Agent 0x01 | GOAL_CHECK    → dist = 0.41, stall_count = 0
Agent 0x01 | GOAL_STALL    → stall_count > threshold — curiosity fires
Agent 0x01 | META_SPAWN    → 4 candidates launched
Agent 0x02 | EVOLVE        → winner selected (dist = 0.08)
Agent 0x02 | SOUL_INHERIT  → 23 memories transferred
Agent 0x01 | CDBG_EMIT     → [05][12 34 56][C0]
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
│         SOMA BINARY RUNTIME  (CDBG v4 · soma_runtime.h)         │
│  Assembler v4.1 │ C Transpiler v4.1 │ stdlib · CDBG 5-byte     │
├──────────┬──────────┬──────────┬──────────────────────────────┤
│  x86-64  │  ARM64   │  RISC-V  │  WASM (planned)              │
└──────────┴──────────┴──────────┴──────────────────────────────┘
```

---

## 💻 Write Agents, Not Threads

### Hello Agent

```soma
.SOMA    4.0.0
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
.SOMA    4.0.0
.SOMSIZE 16x16
.AGENTS  256

.CODE
@_start:
  SOM_INIT  RANDOM
  FORK      16, @explorer
  BROADCAST 0xBEEF
  BARRIER   16
  SOM_ELECT R0
  HALT

@explorer:
  MSG_RECV  R0
  SOM_WALK  SELF, GRADIENT
  SOM_TRAIN R0, S0
  AGENT_KILL SELF
```

### Curious Agent — Full Phase III Stack

```soma
.SOMA    4.0.0
.SOMSIZE 16x16
.AGENTS  8

.DATA
  goal_template : VEC = [0.7, 0.6, 0.5, 0.4, 0.3, 0.8, 0.9, 0.1,
                          0.7, 0.6, 0.5, 0.4, 0.3, 0.8, 0.9, 0.1]
  mutation_n    : IMM = 4

.CODE
@_start:
  SOM_INIT   RANDOM
  SPAWN      A0, @curious_agent
  SOM_MAP    A0, (8,8)
  MSG_SEND   A0, [goal_template]
  WAIT       A0
  HALT

@curious_agent:
  TERRAIN_READ R0              ; read collective wisdom at this position
  MSG_RECV     R1              ; receive goal template
  GOAL_SET     R1              ; encode intended future state

@learn_loop:
  SOM_BMU    R0
  SOM_TRAIN  R0, S0
  EMOT_TAG   S0, 0x3FFF        ; emotional tagging (Phase II)
  GOAL_CHECK R1                ; measure distance to goal
  GOAL_STALL @curiosity        ; jump if stall_count > threshold
  TERRAIN_MARK R0
  JMP        @learn_loop

@curiosity:
  INTROSPECT                   ; agent reads its own state before deciding
  META_SPAWN [mutation_n], @candidate
  BARRIER    [mutation_n]
  EVOLVE     A1                ; select child closest to its own declared goal
  SOUL_INHERIT A1              ; winner carries this agent's emotional memory
  SOUL_QUERY R3
  TERRAIN_MARK R3              ; deposit soul wisdom — sacred place forms
  CDBG_EMIT                    ; broadcast 5-byte CDBG identity frame
  AGENT_KILL SELF

@candidate:
  MSG_RECV   R0
  SOM_BMU    R0
  SOM_TRAIN  R0, S0
  GOAL_CHECK R1
  TERRAIN_READ R2
  SOUL_QUERY R3
  EMOT_TAG   R3, R1
  AGENT_KILL SELF
```

> Full annotated version: [`examples/soma_curious.soma`](examples/soma_curious.soma) — assembles to **47 instructions, 376 bytes**.

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
| `0x22` | `MSG_PEEK` | Non-blocking receive |
| `0x23` | `BROADCAST` | Send to all agents |
| `0x24` | `MULTICAST` | Send to SOM region |

### Phase II — Emotional Memory

| Code | Mnemonic | Description |
|------|----------|-------------|
| `0x80` | `EMOT_TAG` | Attach valence + intensity to current SOM node |
| `0x81` | `DECAY_PROTECT` | Shield memory from decay (cycle or time mode) |
| `0x82` | `PREDICT_ERR` | Compute surprise — BMU distance vs prediction |
| `0x83` | `EMOT_RECALL` | Retrieve emotional tag by coordinate |
| `0x84` | `SURPRISE_CALC` | Prediction error from raw vectors |

### Phase III — Curiosity

| Code | Mnemonic | Description |
|------|----------|-------------|
| `0x60` | `GOAL_SET` | Encode goal vector — desired future weight state |
| `0x61` | `GOAL_CHECK` | Measure distance to goal; update stall counter |
| `0x62` | `SOUL_QUERY` | Pattern-match content memory — computational intuition |
| `0x63` | `META_SPAWN` | Spawn N agents with mutated goal vectors |
| `0x64` | `EVOLVE` | Select child by goal proximity; inherit soul |
| `0x65` | `INTROSPECT` | Export own soul state snapshot |
| `0x66` | `TERRAIN_READ` | Read collective terrain at current position |
| `0x67` | `TERRAIN_MARK` | Write emotional data into terrain |
| `0x68` | `SOUL_INHERIT` | Inherit soul from another agent by ID |
| `0x69` | `GOAL_STALL` | Jump to label if goal stall_count > threshold |

### Phase IV — CDBG

| Code | Mnemonic | Description |
|------|----------|-------------|
| `0x70` | `CDBG_EMIT` | Emit 5-byte CDBG agent identity frame to bus |
| `0x71` | `CDBG_RECV` | Receive and decode a CDBG frame |
| `0x72` | `CTX_SWITCH` | Set active decode context (CTX nibble) |

*Full ISA + binary encoding → [`spec/SOMBIN.spec`](spec/SOMBIN.spec)*

---

## 🧬 Phase II — Liveliness

SOMA v3.2+ implements amygdala + hippocampus primitives from *"A Path to AGI Part II: Liveliness"*:

```
High surprise (PREDICT_ERR) → high emotion tag → slow decay → strong memory
Low surprise                → low tag          → fast decay → forgotten
```

```python
from runtime.som.emotion import EmotionRegistry, ProtectMode

em = EmotionRegistry()
es = em.get_or_create(agent_id=0)
es.emot_tag(row=2, col=2, valence=0.9, intensity=0.8)
es.decay_protect(2, 2, mode=ProtectMode.CYCLES, cycles=100)

from runtime.som.memory import MemoryManager
report = MemoryManager(som, em).consolidate(agent_id=0)
# promoted=1, pruned=0, decayed=8, took=0.08ms
```

Two-tier system mirrors hippocampal consolidation — working SOM (volatile, fast decay) promotes top 10% to long-term SOM each REM cycle.

---

## 🧠 Phase III — Curiosity

Agents have a **portable identity** that survives map migration, EVOLVE selection, and generational inheritance.

Memory is indexed by **SHA-256 fingerprint of the weight vector** — not SOM coordinate. When an agent arrives anywhere, it queries its `content_memory` against the weight fingerprint. If the pattern matches something felt before, the emotional tag fires regardless of position. That is the computational definition of **intuition**.

```python
from runtime.som.soul import AgentSoul

soul = AgentSoul(agent_id=1)
soul.goal_set(target_weights)
dist, curious = soul.goal_check(current_weights)
if curious:
    new_goals = soul.spawn_mutated_goals(n=4)  # META_SPAWN
hit = soul.soul_query(new_weights)             # intuition
```

**SomTerrain** — the map's own memory. Nobody programs the geography. It emerges:

| Zone | What It Means | How It Forms |
|------|---------------|--------------|
| **Hot zone** | Consistently positive valence | Many agents succeeded here |
| **Cold zone** | Collective danger | Many agents failed here |
| **Sacred place** | High `cultural_deposit` | Dying agents left memories here |
| **Virgin territory** | `attractor_count ≈ 0` | The frontier — unexplored |

---

## 📡 Phase IV — Context-Discriminated Binary Grammar

One 5-byte frame. Seven meanings. Zero extra opcodes.

```
┌──────────┬──────────┬────────────────────────┬─────────────────┐
│  CTX[4b] │  SUB[4b] │   PAYLOAD  (3 bytes)   │ CRC4[4b] R[4b] │
└──────────┴──────────┴────────────────────────┴─────────────────┘
```

| CTX | Namespace | 3-byte Payload |
|-----|-----------|----------------|
| `0x0` | `SOM_MAP` | `X[8] · Y[8] · OPCODE[8]` |
| `0x1` | `AGENT` | `cluster[4] · map_id[8] · seq[12]` = 16.7M agents |
| `0x2` | `SOUL` | `field_id[8] · value_fp16[16]` |
| `0x3` | `MEMORY` | 24-bit fingerprint hash prefix |
| `0x4` | `PULSE` | 24-bit heartbeat counter |
| `0x5` | `EMOTION` | `row[8] · valence[8] · intensity[8]` |
| `0x6` | `HISTORY` | `generation[8] · goal_record_id[16]` |

The opcode table stays **exactly the same size** forever. Only CTX namespaces scale.

---

## 📚 Standard Library

v4.1.0 ships a stdlib of reusable routines in `stdlib/soma.stdlib`:

| Routine | What It Does |
|---------|--------------|
| `soul_init` | Initialize AgentSoul with default goal + curiosity threshold |
| `terrain_explore` | Read terrain, navigate toward most curious node |
| `cdbg_announce` | Emit CDBG AGENT frame + SOUL snapshot on agent birth |
| `emot_cycle` | EMOT_TAG → DECAY_PROTECT → SURPRISE_CALC in one call |
| `goal_pursue` | GOAL_SET → learn loop → GOAL_STALL, encapsulated |
| `evolve_cycle` | META_SPAWN → BARRIER → EVOLVE → SOUL_INHERIT, encapsulated |
| `deposit_wisdom` | SOUL_QUERY → TERRAIN_MARK → CDBG_EMIT — dying agent ceremony |

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
│   ├── isa.py               ← Canonical opcode table v4.0 (Phase I–IV)
│   ├── vm.py                ← Test VM — all opcodes dispatched
│   ├── assembler.py         ← v4.1 — 19 new encoding cases (Phase II/III/IV)
│   ├── cdbg.py              ← Context-Discriminated Binary Grammar
│   └── lexer.py
├── runtime/
│   ├── soma_emit_c.py       ← v4.1 — 19 C transpiler cases + opcode name map
│   ├── soma_runtime.h       ← v4.1 — 18 bridge function declarations
│   └── som/
│       ├── soul.py          ← AgentSoul + MasterSoul + SoulRegistry
│       ├── terrain.py       ← SomTerrain + TerrainRegistry
│       ├── emotion.py       ← Phase II — EmotionRegistry, EMOT_TAG
│       ├── memory.py        ← Phase II — EMOT_RECALL, SURPRISE_CALC
│       ├── som_map.py       ← LiveSomMap
│       └── som_scheduler.py ← SomScheduler
├── stdlib/
│   └── soma.stdlib          ← v4.1 — 7 reusable routines
├── examples/
│   ├── soma_curious.soma    ← Full curiosity example (47 instr, 376 bytes)
│   ├── hello_agent.soma
│   └── swarm_cluster.soma
├── tests/
│   ├── test_curiosity_cdbg.py  ← 41 tests — Phase III+IV
│   ├── test_phase26.py
│   ├── test_liveliness.py
│   ├── test_agent.py
│   └── test_soma.py
├── spec/
│   ├── SOMA.grammar         ← v4.0 — emot_instr + curiosity_instr + cdbg_instr
│   └── SOMBIN.spec          ← Phase II/III/IV opcode table + CDBG Section 8
└── bin/
    └── SOMBIN.spec          ← synced with spec/SOMBIN.spec
```

---

## 🗺️ AGI Staircase

```
Step 1    PULSE            ✅  System pulses. It is alive.
Step 2    SOM topology     ✅  Agents live on a map. Coordinates matter.
Step 3    MSG passing      ✅  Agents communicate. State is shared.
Step 4    Emotion + Decay  ✅  System grows, forgets, feels. It is lively.
Step 5    Curiosity        ✅  AgentSoul + SomTerrain + EVOLVE. It wants to learn.
Step 6    CDBG Scaling     ✅  Opcode table stays fixed as system grows to millions.
Step 6.5  Coherence        ✅  All 7 layers wired end-to-end. soma_curious.soma runs.
Step 7    Collective Intel 📋  NICHE_DECLARE, SYMBOL_EMERGE, HERITAGE_LOAD.
Step 8    Self-hosting     📋  somasc.soma assembles itself.
          ↑
          Nobody knows exactly where on this staircase 'intelligence' appears.
          But this is the most concrete path anyone is building right now.
```

---

## 📊 Build Status

| Component | Version | Status |
|-----------|---------|--------|
| Grammar spec | v4.0 | ✅ Complete — emot_instr + curiosity_instr + cdbg_instr |
| Binary format (CDBG) | v4.0 | ✅ 5-byte frames, 7 CTX namespaces, CRC-4 |
| ISA | v4.0 | ✅ Phase I–IV, 70+ opcodes |
| Assembler | **v4.1** | ✅ 19 new encoding cases — Phase II/III/IV fully wired |
| C transpiler | **v4.1** | ✅ 19 new switch cases + opcode name map |
| soma_runtime.h | **v4.1** | ✅ 18 bridge function declarations |
| stdlib | **v4.1** | ✅ 7 routines — soul_init → deposit_wisdom |
| VM dispatch | v4.0 | ✅ All opcodes dispatched |
| soma_curious.soma | **v4.1** | ✅ Assembles — 47 instructions, 376 bytes |
| AgentSoul | v4.0 | ✅ Complete + tested |
| SomTerrain | v4.0 | ✅ Complete + tested |
| CDBG encoder/decoder | v4.0 | ✅ Complete + tested |
| Emotional memory (Phase II) | v3.2 | ✅ EMOT_TAG · DECAY_PROTECT · PREDICT_ERR |
| Memory consolidation | v3.2 | ✅ Two-tier · REM cycle · hard prune |
| True concurrency | v3.1 | ✅ AgentRegistry + real pthreads |
| SOM scheduling | v3.1 | ✅ LiveSomMap + SomScheduler + Visualizer |
| PyPI package | v3.2.0 | ✅ `pip install soma-lang` |
| GitHub Actions CI | v3.x | ✅ Matrix 3.9–3.12 × ubuntu/macOS/win |
| Test suite | v4.1 | ✅ **400 passed** in 7.12s |
| soma_runtime.py bridge wiring | — | 📋 Next — Python-side bridge function impl |
| Phase V — Collective Intelligence | — | 📋 Next — NICHE_DECLARE, SYMBOL_EMERGE |
| JIT backend | — | 📋 Planned |
| WASM backend | — | 📋 Planned |

---

## 🗺️ Roadmap

| Phase | Timeline | Milestone |
|-------|----------|-----------|
| **0 — Foundation** | ✅ Done | PyPI v3.0.0 · CI · C transpiler · 340× speedup |
| **1 — Concurrency** | ✅ Feb 2026 | AgentRegistry + ThreadAgent · 689× · 246 tests |
| **2 — SOM Live** | ✅ Feb 2026 | LiveSomMap · SomScheduler · SomVisualizer · 300 tests |
| **2.5 — Liveliness** | ✅ Feb 2026 | EmotionRegistry · MemoryManager · decay + consolidation |
| **2.6 — Memory Share** | ✅ Feb 2026 | EMOT_RECALL · SURPRISE_CALC · broadcast · neighbor sync |
| **3 — Curiosity** | ✅ Feb 2026 | AgentSoul · SomTerrain · EVOLVE · META_SPAWN · 41 tests |
| **4 — CDBG** | ✅ Feb 2026 | 5-byte binary grammar · 7 CTX namespaces · CRC-4 |
| **4.1 — Coherence** | ✅ Feb 2026 | Assembler · C transpiler · stdlib · runtime.h · soma_curious runs |
| **5 — Collective Intel** | May 2026 | NICHE_DECLARE · SYMBOL_EMERGE · HERITAGE_LOAD |
| **6 — Transpiler+** | Jun 2026 | SIMD (AVX2/NEON) · OpenMP · multi-arch · LLVM backend |
| **7 — Self-hosting** | Jul 2026 | somasc.soma assembles itself · SOMA-OS bare metal demo |

---

## 🚀 Quick Start

```bash
git clone https://github.com/sbhadade/soma-lang
cd soma-lang
pip install -e ".[dev]"

# Run all 400 tests
pytest tests/ -v

# Run Phase III + IV specifically
pytest tests/test_curiosity_cdbg.py -v

# Assemble the curiosity program
soma assemble examples/soma_curious.soma -o curious.sombin

# Transpile to native C and run
soma transpile examples/soma_curious.soma -o curious.c
gcc -O3 -march=native -o curious curious.c -lm -lpthread
./curious
```

---

## 🔬 Academic Context

SOMA's architecture is grounded in:

- **Khacef et al. (arXiv 1810.12640)** — Distributed SOM with spiking neurons. Closest academic predecessor.
- **FPGA-based SOM accelerators** — 100× speedup over CPU. SOMA is the programming model these chips need.
- **Memristor SOM chips** (Nature Comms, 2022) — in-situ SOM training. SOMA targets this substrate.
- **Amygdala + hippocampus models** — Phase II implements the computational equivalents: emotional tagging, decay protection, REM consolidation.
- **Evolutionary computation** — EVOLVE + META_SPAWN is machine-speed goal-directed evolution. No human-defined fitness function — the agent's own declared intention is the selection criterion.

---

<div align="center">

**Built by [`sbhadade`](https://github.com/sbhadade)**

*"Most languages run on operating systems. SOMA is the operating system."*

[![Star on GitHub](https://img.shields.io/github/stars/sbhadade/soma-lang?color=ff2d78&labelColor=04000f&style=for-the-badge&logo=github&logoColor=ff2d78)](https://github.com/sbhadade/soma-lang/stargazers)

---

**© 2026 Swapnil Bhadade. MIT License.**

</div>