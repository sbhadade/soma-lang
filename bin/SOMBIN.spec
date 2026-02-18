;;; ═══════════════════════════════════════════════════════════════
;;; SOMA BINARY FORMAT SPECIFICATION — .sombin v1.0
;;; The wire format for compiled SOMA programs
;;; Every number is big-endian unless noted
;;; ═══════════════════════════════════════════════════════════════

;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;; SECTION 1: FILE HEADER  (32 bytes)
;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;;
;;;  Offset  Size  Field
;;;  ──────  ────  ─────────────────────────────────
;;;  0x00    4     MAGIC       = 0x534F4D41  ("SOMA")
;;;  0x04    2     VER_MAJOR   = 0x0001
;;;  0x06    2     VER_MINOR   = 0x0000
;;;  0x08    1     ARCH_TARGET (see codes below)
;;;  0x09    1     SOM_ROWS    (0–255)
;;;  0x0A    1     SOM_COLS    (0–255)
;;;  0x0B    1     MAX_AGENTS  (0–255)
;;;  0x0C    4     CODE_OFFSET (byte offset to code section)
;;;  0x10    4     CODE_SIZE   (bytes)
;;;  0x14    4     DATA_OFFSET
;;;  0x18    4     DATA_SIZE
;;;  0x1C    4     SOM_OFFSET  (initial SOM weight table)
;;;  0x1E    2     FLAGS       (see flag bits below)
;;;
;;; ARCH_TARGET codes:
;;;   0x00 = ANY (runtime decides)
;;;   0x01 = X86_64
;;;   0x02 = ARM64
;;;   0x03 = RISCV64
;;;   0x04 = WASM32
;;;   0x05 = BARE_METAL
;;;
;;; FLAG bits:
;;;   Bit 0 = SELF_MODIFYING  (code can rewrite itself)
;;;   Bit 1 = ONLINE_LEARN    (SOM trains during execution)
;;;   Bit 2 = ENCRYPTED       (payload is AES-256 encrypted)
;;;   Bit 3 = COMPRESSED      (payload is LZ4 compressed)
;;;   Bit 4 = DEBUG_SYMS      (debug symbol table present)
;;;   Bit 5–15 = reserved

;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;; SECTION 2: THE 64-BIT INSTRUCTION WORD
;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;;
;;;  63      56 55     48 47    40 39    32 31     16 15      0
;;;  ┌─────────┬─────────┬────────┬────────┬─────────┬────────┐
;;;  │ OPCODE  │ AGENT-ID│ SOM-X  │ SOM-Y  │  REG    │  IMM   │
;;;  │  8 bits │  8 bits │ 8 bits │ 8 bits │ 16 bits │16 bits │
;;;  └─────────┴─────────┴────────┴────────┴─────────┴────────┘
;;;
;;;  OPCODE   — instruction (see table below)
;;;  AGENT-ID — which agent executes this (0xFF = broadcast)
;;;  SOM-X    — SOM grid X coordinate context
;;;  SOM-Y    — SOM grid Y coordinate context
;;;  REG      — register fields: [DST:4][SRC1:4][SRC2:4][FLAGS:4]
;;;  IMM      — 16-bit immediate / payload / address offset

;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;; SECTION 3: OPCODE TABLE
;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;;
;;;  Code   Mnemonic     Operands              Description
;;;  ─────  ───────────  ────────────────────  ─────────────────────────────
;;;  0x01   SPAWN        AREG, @label          Spawn agent, entry = label
;;;  0x02   AGENT_KILL   AREG/SELF/ALL         Terminate agent(s)
;;;  0x03   FORK         imm8, @label          Fork N copies at label
;;;  0x04   MERGE        AREG_list, RREG       Merge results into RREG
;;;  0x05   BARRIER      imm8                  Sync N agents
;;;  0x06   WAIT         AREG                  Block until AREG replies
;;;  0x07   SELF         RREG                  Write own agent ID → RREG
;;;  0x08   PARENT       RREG                  Write parent ID → RREG
;;;  0x09   CHILDREN     RREG                  Write child count → RREG
;;;
;;;  0x10   SOM_INIT     mode                  Init SOM weights
;;;  0x11   SOM_BMU      RREG, RREG            Find BMU for input vector
;;;  0x12   SOM_TRAIN    RREG, RREG            Train node toward input
;;;  0x13   SOM_NBHD     RREG, RREG            Compute neighborhood σ
;;;  0x14   WGHT_UPD     RREG, RREG            Apply weight update
;;;  0x15   LR_DECAY     float16               Decay learning rate
;;;  0x16   SOM_WALK     RREG, mode            Move agent on SOM
;;;  0x17   SOM_SENSE    RREG                  Read local node activation
;;;  0x18   SOM_MAP      AREG, coord           Place agent at coord
;;;  0x19   SOM_ELECT    RREG                  Leader election → RREG
;;;  0x1A   SOM_DIST     RREG, RREG, RREG      Euclidean dist on SOM
;;;  0x1B   SPAWN_MAP    rows, cols, @label    Spawn agent per node
;;;
;;;  0x20   MSG_SEND     AREG, operand         Send message to agent
;;;  0x21   MSG_RECV     RREG                  Blocking receive
;;;  0x22   MSG_PEEK     RREG                  Non-blocking receive
;;;  0x23   BROADCAST    operand               Send to all agents
;;;  0x24   MULTICAST    RREG, operand         Send to SOM region
;;;  0x25   MSG_FLUSH                          Clear message queue
;;;  0x26   MSG_COUNT    RREG                  Queue depth → RREG
;;;
;;;  0x30   JMP          @label                Unconditional jump
;;;  0x31   JZ           RREG, @label          Jump if zero
;;;  0x32   JNZ          RREG, @label          Jump if non-zero
;;;  0x33   JEQ          RREG, RREG, @label    Jump if equal
;;;  0x34   JGT          RREG, RREG, @label    Jump if greater
;;;  0x35   CALL         @label                Call subroutine
;;;  0x36   RET                               Return from subroutine
;;;  0x37   HALT                              Terminate program
;;;  0x38   NOP                               No operation
;;;  0x39   TRAP         imm8                  OS syscall
;;;
;;;  0x40   MOV          RREG, operand         Move/load immediate
;;;  0x41   LOAD         RREG, [addr]          Load from memory
;;;  0x42   STORE        [addr], RREG          Store to memory
;;;  0x43   PUSH         RREG                  Push to stack
;;;  0x44   POP          RREG                  Pop from stack
;;;  0x45   ZERO         RREG                  Zero a register
;;;  0x46   COPY         RREG, RREG            Register copy
;;;
;;;  0x50   ADD          RREG, RREG, RREG      Integer/vector add
;;;  0x51   SUB          RREG, RREG, RREG      Subtract
;;;  0x52   MUL          RREG, RREG, RREG      Multiply
;;;  0x53   DIV          RREG, RREG, RREG      Divide
;;;  0x54   DOT          RREG, RREG, RREG      Vector dot product
;;;  0x55   NORM         RREG, RREG            Normalize vector
;;;  0x56   DIST         RREG, RREG, RREG      Euclidean distance
;;;  0x57   ACCUM        RREG, RREG            Accumulate (+=)
;;;  0x58   SCALE        RREG, RREG, float16   Scalar multiply
;;;  0x59   AND          RREG, RREG, RREG      Bitwise AND
;;;  0x5A   OR           RREG, RREG, RREG      Bitwise OR
;;;  0x5B   XOR          RREG, RREG, RREG      Bitwise XOR
;;;  0x5C   NOT          RREG, RREG            Bitwise NOT
;;;  0x5D   SHL          RREG, RREG, imm4      Shift left
;;;  0x5E   SHR          RREG, RREG, imm4      Shift right
;;;
;;;  0xFE   DBG          imm16                 Debug breakpoint (stripped in release)
;;;  0xFF   EXT          imm16                 Extension opcode (future use)

;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;; SECTION 4: REGISTER ENCODING
;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;;
;;;  4-bit register field encoding:
;;;
;;;  0x0–0xF = R0–R15   General purpose (256-bit vector)
;;;  ── agent registers encoded in AGENT-ID field directly ──
;;;  ── SOM state registers accessed via SOM_* instructions ──
;;;
;;;  Special pseudo-register codes in AGENT-ID field:
;;;  0xFE = SELF    (current agent)
;;;  0xFD = PARENT  (spawner agent)
;;;  0xFC = ALL     (broadcast / kill all)
;;;  0xFB = ANY     (any available agent)

;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;; SECTION 5: SOM WEIGHT TABLE FORMAT
;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;;
;;;  Laid out in row-major order, one entry per node:
;;;
;;;  Per-node entry (40 bytes):
;;;   Offset  Size  Field
;;;   0x00    4     NODE_X       (uint32)
;;;   0x04    4     NODE_Y       (uint32)
;;;   0x08    32    WEIGHT_VEC   (8× f32, 256-bit vector)
;;;
;;;  Total table size = SOM_ROWS × SOM_COLS × 40 bytes
;;;  Example: 16×16 SOM = 256 nodes × 40 = 10,240 bytes

;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;; SECTION 6: MESSAGE FORMAT
;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;;
;;;  64-bit message word:
;;;
;;;  63     56 55     48 47     32 31      0
;;;  ┌────────┬─────────┬─────────┬────────┐
;;;  │ FLAGS  │ SENDER  │  TYPE   │ PAYLOAD│
;;;  │ 8 bits │  8 bits │ 16 bits │32 bits │
;;;  └────────┴─────────┴─────────┴────────┘
;;;
;;;  FLAGS:  Bit 0 = ACK_REQUIRED
;;;          Bit 1 = PRIORITY
;;;          Bit 2 = ENCRYPTED
;;;          Bit 3 = VECTOR_PAYLOAD (payload is a VEC pointer)
;;;  SENDER: Source agent ID
;;;  TYPE:   Application-defined message type
;;;  PAYLOAD: 32-bit value, or pointer to VEC in shared memory

;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;; SECTION 7: EXAMPLE BINARY (hex dump, annotated)
;;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;;;
;;;  A minimal SOMA binary that spawns one agent and halts:
;;;
;;;  00000000: 534F 4D41  ← MAGIC "SOMA"
;;;  00000004: 0001 0000  ← v1.0
;;;  00000008: 00         ← ARCH: ANY
;;;  00000009: 08         ← SOM_ROWS: 8
;;;  0000000A: 08         ← SOM_COLS: 8
;;;  0000000B: 04         ← MAX_AGENTS: 4
;;;  0000000C: 00000020   ← CODE_OFFSET: 32
;;;  00000010: 00000010   ← CODE_SIZE: 16 bytes (2 instructions)
;;;  00000014: 00000030   ← DATA_OFFSET: 48
;;;  00000018: 00000000   ← DATA_SIZE: 0
;;;  0000001C: 00000030   ← SOM_OFFSET: 48
;;;  0000001E: 0001       ← FLAGS: ONLINE_LEARN
;;;
;;;  ── CODE (2 × 8 bytes = 16 bytes) ──
;;;  00000020: 01 00 00 00 0010 0008  ← SPAWN A0, @worker (coord 0,0, entry 0x0008)
;;;  00000028: 37 00 00 00 0000 0000  ← HALT
