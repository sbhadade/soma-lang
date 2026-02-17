# SOMA — Design Rationale & Philosophy

## Why a New Binary Language?

Every existing agent language (AgentSpeak, SARL, Jason, JADE) is a
high-level framework that runs *on top of* an existing OS and runtime.
They borrow threads from the OS, memory from a heap, scheduling from a
kernel. The agents are second-class citizens in their own programs.

SOMA inverts this. Agents and SOM topology are encoded directly at the
binary instruction level. There is no OS below SOMA — SOMA *is* the OS
model.

---

## Why Self-Organizing Maps?

Most multi-agent systems use flat address spaces or tree hierarchies
for agent coordination. SOM topology gives us something richer:

- **Spatial locality.** Agents close together on the SOM share similar
  activation profiles, enabling natural clustering without explicit
  coordination code.

- **Emergent scheduling.** The SOM's neighborhood function creates a
  gradient — agents naturally flow toward high-activation regions.
  This replaces the traditional OS scheduler with a learned, adaptive
  one.

- **Online learning.** The SOM trains continuously during execution.
  A SOMA program that runs longer becomes *smarter* — its internal
  topology adapts to the workload.

---

## Why a New Binary Format?

Existing bytecodes (JVM, LLVM IR, WASM) were designed for single-
threaded or coarse-grained concurrent programs. Their instruction words
encode *what* to do, but not *who* does it or *where* in a topology.

SOMA's 64-bit instruction word encodes all of these simultaneously:

```
[OPCODE:8][AGENT-ID:8][SOM-X:8][SOM-Y:8][REG:16][IMM:16]
```

This means the hardware (or JIT) knows the full execution context
from a single 64-bit fetch — no context table lookups, no thread-local
storage overhead.

---

## Why Self-Hosting?

`somasc.soma` — the SOMA assembler — is written in SOMA itself.

This is not a gimmick. It is a correctness proof. If the language
cannot express its own assembler, the language is not expressive
enough. The bootstrap process is:

1. A minimal *host bootstrap* (< 500 lines, platform-specific) handles
   the 20 TRAP syscalls that `somasc.soma` requires.
2. `somasc.soma` compiles itself, producing `somasc.sombin`.
3. From that point forward, the host bootstrap is never needed again.

---

## Design Decisions

### Decision: 64-bit fixed-width instructions

Pros: Simple decoder, cache-friendly, branchless dispatch.
Cons: Some instructions waste bits (HALT uses 56 zero bits).
Verdict: Predictability outweighs compactness. SOMA is not embedded-first.

### Decision: 256-bit general registers

Each R-register holds an 8×f32 weight vector natively. This matches
the SOM node weight dimension for the default configuration and maps
directly to AVX-256 on x86, NEON on ARM, and V extension on RISC-V.

### Decision: Message-passing, no shared memory

Agents do not share memory directly. All coordination is via MSG_SEND /
MSG_RECV / BROADCAST. This eliminates lock complexity and makes agent
behavior deterministic from the message trace alone.

### Decision: SOM state in S-registers

S0–S15 are the SOM's live parameters. Any agent can read them.
Only the runtime's learning engine writes S0 (learning rate) and
S1 (neighborhood radius). Agents can read these to adapt their own
behavior, but cannot corrupt the training loop.

### Decision: TRAP for I/O

SOMA has no built-in I/O instructions. All I/O goes through TRAP,
which the host runtime maps to platform syscalls. This keeps the core
ISA portable and substrate-agnostic.

---

## Roadmap

### v1.1
- Type-checked .soma source (static agent signature verification)
- Compressed .sombin with LZ4 (FLAG bit 3)
- Debug symbol table format

### v1.2
- SOMA → LLVM IR transpiler (for existing toolchain integration)
- SOM visualization protocol (real-time topology inspection)
- Distributed SOMA: agents across networked nodes

### v2.0
- Hardware SOMA accelerator ISA extension proposal
- Encrypted agent execution (FLAG bit 2, homomorphic weight updates)
- Formal verification of agent message invariants

---

*"The map is not the territory — but in SOMA, the map is the machine."*
