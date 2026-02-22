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

> **v5.0.0** — Phase V Collective Intelligence · 60 opcodes · NicheMap + SymbolTable + SoulStore · H=5.06 bits emergence · 140 tests passing

</div>

---

## ⚡ One Command. Native Speed.

```bash
pip install soma-lang
soma assemble examples/soma_curious.soma -o curious.sombin
soma transpile examples/soma_curious.soma -o curious.c
gcc -O3 -march=native -o curious curious.c runtime/soma_bridge.c -lm -lpthread
./curious
```

```
✅ Assembled soma_curious.soma → soma_curious.sombin  (44 instructions, 526 bytes)
🚀 Transpiled → curious.c
[GOAL_SET]    agent=0 goal[0]=0.700
[EMOT_TAG]    agent=0 (8,9) valence=1.000 intensity=0.500
[INTROSPECT]  agent=0 stall=2 has_goal=1 goal_dist=0.5489
[CDBG_EMIT]   agent=0 CTX=0x1 frame=1000000000
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
│  niche_id      │  niche_id      │  heritage_stack               │
├────────────────┴────────────────┴───────────────────────────────┤
│       SOM COORDINATION PLANE + TERRAIN + COLLECTIVE LAYER       │
│     SOM MAP 0        MSG BUS         SOM MAP 1                  │
│  BMU·TRAIN·WALK   EVOLVE·META_SPAWN  SOUL_QUERY·GOAL_CHECK      │
│  ── SomTerrain ─────────────────────────────────────────────    │
│  hot_zones · cold_zones · sacred_places · virgin_territory      │
│  ── Phase V ────────────────────────────────────────────────    │
│  NicheMap · SymbolTable · SoulStore · CollectiveMemory          │
│  H=5.06 bits · 256 symbols · 64 niches · HERITAGE_LOAD          │
├─────────────────────────────────────────────────────────────────┤
│         SOMA BINARY RUNTIME  (CDBG v4 · soma_runtime.h)         │
│  Assembler v5.0 │ C Transpiler v5.0 │ stdlib · CDBG 5-byte     │
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

### Collective Intelligence — Phase V

```soma
.SOMA    5.0.0
.SOMSIZE 16x16
.AGENTS  64

.CODE
@_start:
  SOM_INIT   RANDOM
  FORK       64, @collective_agent
  BARRIER    64
  COLLECTIVE_SYNC              ; map-wide memory consolidation
  HALT

@collective_agent:
  HERITAGE_LOAD  8             ; inherit top-8 soul vectors from parent lineage
  NICHE_DECLARE  0x0001        ; broadcast specialisation — "I own this region"
  SOM_BMU    R0
  SOM_TRAIN  R0, S0
  SYMBOL_EMERGE                ; co-activation binds a symbol to this SOM region
  NICHE_QUERY R1               ; am I overcrowded? migrate if density > 75%
  JNZ        R1, @migrate
  AGENT_KILL SELF

@migrate:
  SOM_WALK   SELF, GRADIENT    ; move toward lower-density region
  NICHE_DECLARE 0x0002         ; withdraw from current niche
  JMP        @collective_agent
```

> Full annotated version: [`examples/soma_curious.soma`](examples/soma_curious.soma) — assembles to **44 instructions, 526 bytes**.

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

### Phase V — Collective Intelligence ← NEW

| Code | Mnemonic | Description |
|------|----------|-------------|
| `0x73` | `NICHE_DECLARE` | Agent broadcasts specialisation vector; claim or withdraw niche |
| `0x74` | `SYMBOL_EMERGE` | Co-activation binds a symbol ID to SOM region (threshold=3) |
| `0x75` | `HERITAGE_LOAD` | Load parent soul top-K vectors on agent birth |
| `0x76` | `NICHE_QUERY` | Return niche density; triggers migration if > 75% |
| `0x77` | `COLLECTIVE_SYNC` | Map-wide memory consolidation across all agents |

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

## 🌐 Phase V — Collective Intelligence

64 agents. No programmer. 256 symbols self-organized from raw co-activation.

```
Pulse 0       : 64 agents spawn, inherit parent heritage, declare niches
Pulse 1,000   : First symbols bind — repeated co-activation crystallizes meaning
Pulse 10,000  : NicheMap stabilises — agents migrate away from overcrowded regions
Pulse 100,000 : H = 5.06 bits (84.4% of log2(64)) — near-maximum niche diversity
               256 symbols emerged · 18/18 congruency checks pass
```

```python
from runtime.collective import CollectiveMemory, NicheMap

cm = CollectiveMemory(agents=64, som_rows=16, som_cols=16)
cm.run(pulses=100_000)

print(cm.niche_map.entropy())      # 5.06 bits
print(cm.symbol_table.count())     # 256 symbols
print(cm.soul_store.generations)   # cross-generation wisdom accumulated
```

**What emerges (nobody programmed this):**

| Concept | How It Forms |
|---------|--------------|
| **Niche specialisation** | Agents that repeatedly activate the same SOM region claim it via `NICHE_DECLARE` |
| **Symbol binding** | When 3+ agents co-activate the same region, `SYMBOL_EMERGE` crystallises a symbol ID |
| **Heritage** | `HERITAGE_LOAD` copies parent soul top-K on birth — wisdom crosses the generation gap |
| **Migration** | `NICHE_QUERY` triggers relocation when niche density > 75% — prevents crowding |
| **Consolidation** | `COLLECTIVE_SYNC` runs map-wide memory merge — the collective remembers what individuals forget |

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

v5.0.0 ships a stdlib of reusable routines in `stdlib/soma.stdlib`:

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
| `N0–N63` | 64 | 64-bit | Niche registers — Phase V |

---

## 📁 Repository Structure

```
soma-lang/
├── soma/
│   ├── isa.py               ← Canonical opcode table v5.0 (Phase I–V, 60 opcodes)
│   ├── vm.py                ← Test VM — all opcodes dispatched
│   ├── assembler.py         ← v5.0 — Phase V encoding
│   ├── cdbg.py              ← Context-Discriminated Binary Grammar
│   └── lexer.py
├── runtime/
│   ├── bridge.py            ← Single import point · BinaryDecoder · BridgeValidator
│   ├── collective.py        ← Phase V · NicheMap · SymbolTable · SoulStore
│   ├── soma_emit_c.py       ← v5.0 — fixed GOAL_CHECK + META_SPAWN thread spawn
│   ├── soma_runtime.h       ← v5.0 — bridge function declarations
│   ├── soma_bridge.c        ← v5.0 — fixed goal_dist [0,1] · EMOT_TAG clamp
│   └── som/
│       ├── soul.py          ← AgentSoul + MasterSoul + SoulRegistry
│       ├── terrain.py       ← SomTerrain + TerrainRegistry
│       ├── emotion.py       ← Phase II — EmotionRegistry, EMOT_TAG
│       ├── memory.py        ← Phase II — EMOT_RECALL, SURPRISE_CALC
│       ├── som_map.py       ← LiveSomMap
│       └── som_scheduler.py ← SomScheduler
├── stdlib/
│   └── soma.stdlib          ← v5.0 — 7 reusable routines
├── examples/
│   ├── soma_curious.soma    ← Full curiosity example (44 instr, 526 bytes)
│   ├── phase_v_emergence.py ← 64-agent demo · 256 symbols · H=5.06 bits
│   ├── hello_agent.soma
│   └── swarm_cluster.soma
├── tests/
│   ├── test_collective.py   ← 83 tests — Phase V
│   ├── test_bridge.py       ← 57 tests — bridge + ISA validation
│   ├── test_curiosity_cdbg.py  ← 41 tests — Phase III+IV
│   ├── test_phase26.py
│   ├── test_liveliness.py
│   ├── test_agent.py
│   └── test_soma.py
├── verify_congruency.py     ← Triple-layer ISA congruency checker (18/18)
├── spec/
│   ├── SOMA.grammar         ← v5.0 — Phase V opcodes added
│   └── SOMBIN.spec          ← Phase I–V opcode table + CDBG Section 8
└── bin/
    └── SOMBIN.spec          ← synced with spec/SOMBIN.spec
```

---

## 🗺️ AGI Staircase

```
Step 1    PULSE               ✅  System pulses. It is alive.
Step 2    SOM topology        ✅  Agents live on a map. Coordinates matter.
Step 3    MSG passing         ✅  Agents communicate. State is shared.
Step 4    Emotion + Decay     ✅  System grows, forgets, feels. It is lively.
Step 5    Curiosity           ✅  AgentSoul + SomTerrain + EVOLVE. It wants to learn.
Step 6    CDBG Scaling        ✅  Opcode table stays fixed as system grows to millions.
Step 6.5  Coherence           ✅  All 7 layers wired end-to-end. soma_curious.soma runs.
Step 7    Collective Intel    ✅  NICHE_DECLARE · SYMBOL_EMERGE · HERITAGE_LOAD.
                                  256 symbols self-organised. H=5.06 bits. Nobody programmed this.
Step 8    Self-hosting        📋  somasc.soma assembles itself.
          ↑
          Nobody knows exactly where on this staircase 'intelligence' appears.
          But this is the most concrete path anyone is building right now.
```

---

## 📊 Build Status

| Component | Version | Status |
|-----------|---------|--------|
| Grammar spec | v5.0 | ✅ Complete — Phase V opcodes added |
| Binary format (CDBG) | v4.0 | ✅ 5-byte frames, 7 CTX namespaces, CRC-4 |
| ISA | **v5.0** | ✅ Phase I–V, 60 opcodes |
| Assembler | **v5.0** | ✅ Phase V encoding wired |
| C transpiler | **v5.0** | ✅ Fixed GOAL_CHECK scale + META_SPAWN thread spawn |
| soma_bridge.c | **v5.0** | ✅ Fixed goal_dist normalisation + EMOT_TAG valence clamp |
| soma_runtime.h | **v5.0** | ✅ Phase V bridge declarations |
| runtime/bridge.py | **v5.0** | ✅ BinaryDecoder + BridgeValidator + load_and_validate() |
| runtime/collective.py | **v5.0** | ✅ NicheMap + SymbolTable + SoulStore + CollectiveMemory |
| stdlib | **v4.1** | ✅ 7 routines — soul_init → deposit_wisdom |
| VM dispatch | v5.0 | ✅ All 60 opcodes dispatched |
| soma_curious.soma | **v5.0** | ✅ Assembles — 44 instructions, 526 bytes · clean exit |
| AgentSoul | v4.0 | ✅ Complete + tested |
| SomTerrain | v4.0 | ✅ Complete + tested |
| CDBG encoder/decoder | v4.0 | ✅ Complete + tested |
| Phase V — Collective Intel | **v5.0** | ✅ H=5.06 bits · 256 symbols · 18/18 congruency |
| Emotional memory (Phase II) | v3.2 | ✅ EMOT_TAG · DECAY_PROTECT · PREDICT_ERR |
| Memory consolidation | v3.2 | ✅ Two-tier · REM cycle · hard prune |
| True concurrency | v3.1 | ✅ AgentRegistry + real pthreads |
| SOM scheduling | v3.1 | ✅ LiveSomMap + SomScheduler + Visualizer |
| PyPI package | **v5.0.0** | ✅ `pip install soma-lang` |
| GitHub Actions CI | v3.x | ✅ Matrix 3.9–3.12 × ubuntu/macOS/win |
| Test suite | **v5.0** | ✅ **140 passed** in 13.27s |
| verify_congruency.py | **v5.0** | ✅ 18/18 — triple-layer ISA check |
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
| **5 — Collective Intel** | ✅ Feb 2026 | NICHE_DECLARE · SYMBOL_EMERGE · HERITAGE_LOAD · 140 tests · H=5.06 bits |
| **6 — Transpiler+** | Jun 2026 | SIMD (AVX2/NEON) · OpenMP · multi-arch · LLVM backend |
| **7 — Self-hosting** | Jul 2026 | somasc.soma assembles itself · SOMA-OS bare metal demo |

---

## 🚀 Quick Start

```bash
git clone https://github.com/sbhadade/soma-lang
cd soma-lang
pip install -e ".[dev]"

# Run all 140 tests
pytest tests/ -v

# Run Phase V specifically
pytest tests/test_collective.py tests/test_bridge.py -v

# Run Phase III + IV
pytest tests/test_curiosity_cdbg.py -v

# Assemble the curiosity program
soma assemble examples/soma_curious.soma -o curious.sombin

# Transpile to native C and run
soma transpile examples/soma_curious.soma -o curious.c
gcc -O3 -march=native -o curious curious.c runtime/soma_bridge.c -lm -lpthread
./curious

# Run the Phase V emergence demo
python examples/phase_v_emergence.py

# Verify ISA congruency (18/18)
python verify_congruency.py
```

---

## 🔬 Academic Context

SOMA's architecture is grounded in:

- **Khacef et al. (arXiv 1810.12640)** — Distributed SOM with spiking neurons. Closest academic predecessor.
- **FPGA-based SOM accelerators** — 100× speedup over CPU. SOMA is the programming model these chips need.
- **Memristor SOM chips** (Nature Comms, 2022) — in-situ SOM training. SOMA targets this substrate.
- **Amygdala + hippocampus models** — Phase II implements the computational equivalents: emotional tagging, decay protection, REM consolidation.
- **Evolutionary computation** — EVOLVE + META_SPAWN is machine-speed goal-directed evolution. No human-defined fitness function — the agent's own declared intention is the selection criterion.
- **Niche theory (ecology)** — Phase V NicheMap mirrors competitive exclusion: agents specialise or migrate. Diversity emerges from pressure, not design.
- **Symbol grounding** — SYMBOL_EMERGE is a computational implementation of Harnad's symbol grounding problem: symbols bind to SOM regions through repeated co-activation, not programmer assignment.

---

<div align="center">

**Built by [`sbhadade`](https://github.com/sbhadade)**

*"Most languages run on operating systems. SOMA is the operating system."*

[![Star on GitHub](https://img.shields.io/github/stars/sbhadade/soma-lang?color=ff2d78&labelColor=04000f&style=for-the-badge&logo=github&logoColor=ff2d78)](https://github.com/sbhadade/soma-lang/stargazers)

---

**© 2026 Swapnil Bhadade. MIT License.**

</div>