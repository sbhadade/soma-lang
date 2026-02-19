"""SOMA Automated Test Suite — stdlib unittest (also runnable with pytest).

Coverage:
  1.  ISA / opcode table
  2.  Register encoding + decoding
  3.  Lexer tokens
  4.  Assembler binary format
  5.  Instruction encoding
  6.  VM semantics (MOV, ADD, SUB, CMP, JMP, CALL/RET)
  7.  SOM map (BMU, training, neighbourhood)
  8.  Multi-agent (SPAWN, FORK, MSG_SEND/RECV, BROADCAST, ELECT)
  9.  Full programs (Hello Agent, Swarm)
  10. Disassembler round-trip
  11. Architecture-target metadata
  12. Error handling / edge cases
  13. Packaging hygiene
"""
import struct
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from soma.isa import (
    OPCODES, OPCODE_NAMES, encode_reg, decode_reg,
    MAGIC, VER_MAJOR, VER_MINOR, ARCH_ANY, ARCH_ARM, ARCH_WASM,
)
from soma.lexer import tokenize, TT
from soma.assembler import assemble, disassemble, AssemblerError, HEADER_SIZE
from soma.vm import SomaVM, VMError, SomMap, AgentState


MINIMAL_SRC = """
.SOMA    1.0.0
.ARCH    ANY
.SOMSIZE 4x4
.AGENTS  2

.CODE
@_start:
  HALT
"""


def _make_vm(src, **kwargs):
    return SomaVM(assemble(src), **kwargs)


def _toks(src):
    return [t for t in tokenize(src) if t.type not in (TT.NEWLINE, TT.EOF)]


# ──────────────────────────────────────────────────────────────────────────────
class TestISA(unittest.TestCase):
    REQUIRED = [
        "SPAWN","AGENT_KILL","FORK","MERGE","BARRIER",
        "SOM_BMU","SOM_TRAIN","SOM_NBHD","WGHT_UPD","SOM_WALK","SOM_ELECT",
        "MSG_SEND","MSG_RECV","BROADCAST",
        "HALT","NOP","WAIT","JMP","JZ","JNZ","CALL","RET",
        "ADD","SUB","MUL","DOT","NORM","MOV","LOAD","STORE","CMP",
    ]

    def test_all_required_opcodes(self):
        for m in self.REQUIRED:
            self.assertIn(m, OPCODES, f"Missing opcode: {m}")

    def test_opcode_values_unique(self):
        vals = list(OPCODES.values())
        self.assertEqual(len(vals), len(set(vals)))

    def test_opcode_roundtrip(self):
        for name, code in OPCODES.items():
            self.assertEqual(OPCODE_NAMES[code], name)

    def test_halt_0x37(self):   self.assertEqual(OPCODES["HALT"],     0x37)
    def test_spawn_0x01(self):  self.assertEqual(OPCODES["SPAWN"],    0x01)
    def test_send_0x20(self):   self.assertEqual(OPCODES["MSG_SEND"], 0x20)
    def test_recv_0x21(self):   self.assertEqual(OPCODES["MSG_RECV"], 0x21)
    def test_som_bmu_0x11(self):self.assertEqual(OPCODES["SOM_BMU"],  0x11)


# ──────────────────────────────────────────────────────────────────────────────
class TestRegisters(unittest.TestCase):
    def test_R_registers(self):
        for i in range(16):
            enc = encode_reg(f"R{i}")
            self.assertEqual(enc, i)
            self.assertEqual(decode_reg(enc), f"R{i}")

    def test_A_registers(self):
        for i in range(64):
            enc = encode_reg(f"A{i}")
            self.assertEqual(enc, 0x0100 | i)
            self.assertEqual(decode_reg(enc), f"A{i}")

    def test_S_registers(self):
        for i in range(16):
            enc = encode_reg(f"S{i}")
            self.assertEqual(enc, 0x0200 | i)
            self.assertEqual(decode_reg(enc), f"S{i}")

    def test_SELF(self):
        self.assertEqual(encode_reg("SELF"),   0xFF00)
        self.assertEqual(decode_reg(0xFF00), "SELF")

    def test_PARENT(self):
        self.assertEqual(encode_reg("PARENT"), 0xFF01)
        self.assertEqual(decode_reg(0xFF01), "PARENT")

    def test_ALL(self):
        self.assertEqual(encode_reg("ALL"),    0xFF02)
        self.assertEqual(decode_reg(0xFF02), "ALL")

    def test_invalid_raises(self):
        self.assertRaises(ValueError, encode_reg, "X99")


# ──────────────────────────────────────────────────────────────────────────────
class TestLexer(unittest.TestCase):
    def test_directive(self):
        t = _toks(".SOMA 1.0.0")[0]
        self.assertEqual(t.type, TT.DIRECTIVE)
        self.assertEqual(t.value, ".SOMA")

    def test_mnemonic_HALT(self):
        self.assertEqual(_toks("HALT")[0].type, TT.MNEMONIC)

    def test_hex_int(self):
        t = _toks("0xFF42")[0]
        self.assertEqual(t.type, TT.INTEGER)
        self.assertEqual(t.value, 0xFF42)

    def test_decimal_int(self):
        t = _toks("256")[0]
        self.assertEqual(t.type, TT.INTEGER)
        self.assertEqual(t.value, 256)

    def test_binary_int(self):
        t = _toks("0b1010")[0]
        self.assertEqual(t.type, TT.INTEGER)
        self.assertEqual(t.value, 0b1010)

    def test_label_def(self):
        self.assertEqual(_toks("@_start:")[0].type, TT.LABEL_DEF)

    def test_label_ref(self):
        t = _toks("@worker")[0]
        self.assertEqual(t.type, TT.LABEL_REF)
        self.assertEqual(t.value, "@worker")

    def test_register_R0(self):
        t = _toks("R0")[0]
        self.assertEqual(t.type, TT.REG)
        self.assertEqual(t.value, "R0")

    def test_register_SELF(self):
        self.assertEqual(_toks("SELF")[0].type, TT.REG)

    def test_coord(self):
        t = _toks("(3,7)")[0]
        self.assertEqual(t.type, TT.COORD)
        self.assertEqual(t.value, (3, 7))

    def test_comment_stripped(self):
        toks = [t for t in tokenize("; comment\nHALT") if t.type not in (TT.NEWLINE, TT.EOF)]
        self.assertEqual(len(toks), 1)
        self.assertEqual(toks[0].type, TT.MNEMONIC)

    def test_all_mnemonics_lex_correctly(self):
        for mnem in OPCODES:
            t = _toks(mnem)[0]
            self.assertEqual(t.type, TT.MNEMONIC, f"{mnem} didn't lex as MNEMONIC")

    def test_brackets(self):
        toks = _toks("[foo]")
        self.assertEqual(toks[0].type, TT.LBRACKET)
        self.assertEqual(toks[2].type, TT.RBRACKET)

    def test_string_literal(self):
        t = _toks('"hello"')[0]
        self.assertEqual(t.type, TT.STRING)
        self.assertEqual(t.value, "hello")


# ──────────────────────────────────────────────────────────────────────────────
class TestAssembler(unittest.TestCase):
    def _b(self, src=None):
        return assemble(src or MINIMAL_SRC)

    def test_produces_bytes(self):     self.assertIsInstance(self._b(), bytes)
    def test_magic(self):              self.assertEqual(self._b()[:4], MAGIC)
    def test_arch_any(self):           self.assertEqual(self._b()[0x08], 0)
    def test_somsize_rows(self):       self.assertEqual(self._b()[0x09], 4)
    def test_somsize_cols(self):       self.assertEqual(self._b()[0x0A], 4)
    def test_agents(self):             self.assertEqual(self._b()[0x0B], 2)

    def test_version(self):
        b = self._b()
        self.assertEqual(struct.unpack_from(">H", b, 0x04)[0], VER_MAJOR)
        self.assertEqual(struct.unpack_from(">H", b, 0x06)[0], VER_MINOR)

    def test_code_offset_gte_header(self):
        b = self._b()
        co = struct.unpack_from(">I", b, 0x0C)[0]
        self.assertGreaterEqual(co, HEADER_SIZE)

    def test_one_instruction_8_bytes(self):
        b = self._b()
        cs = struct.unpack_from(">I", b, 0x10)[0]
        self.assertEqual(cs, 8)

    def test_halt_opcode_in_binary(self):
        b = self._b()
        co = struct.unpack_from(">I", b, 0x0C)[0]
        (word,) = struct.unpack_from(">Q", b, co)
        self.assertEqual((word >> 56) & 0xFF, OPCODES["HALT"])

    def test_three_instructions_24_bytes(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 1
.CODE
@_start:
  NOP
  NOP
  HALT
"""
        b = assemble(src)
        self.assertEqual(struct.unpack_from(">I", b, 0x10)[0], 24)

    def test_arch_arm64(self):
        src = MINIMAL_SRC.replace(".ARCH    ANY", ".ARCH    ARM64")
        self.assertEqual(assemble(src)[0x08], 2)

    def test_arch_wasm(self):
        src = MINIMAL_SRC.replace(".ARCH    ANY", ".ARCH    WASM")
        self.assertEqual(assemble(src)[0x08], 4)

    def test_somsize_16x16(self):
        src = MINIMAL_SRC.replace(".SOMSIZE 4x4", ".SOMSIZE 16x16")
        b = assemble(src)
        self.assertEqual(b[0x09], 16)
        self.assertEqual(b[0x0A], 16)

    def test_data_section(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 1
.DATA
  payload : MSG = 0xFF42
.CODE
@_start:
  HALT
"""
        b = assemble(src)
        ds = struct.unpack_from(">I", b, 0x18)[0]
        self.assertGreater(ds, 0)

    def test_self_modifying_flag(self):
        src = MINIMAL_SRC + "\n.SELF_MODIFYING\n"
        b = assemble(src)
        flags = struct.unpack_from(">H", b, 0x1E)[0]
        self.assertTrue(flags & 0x01)

    def test_opcode_in_top_8_bits(self):
        b = self._b()
        co = struct.unpack_from(">I", b, 0x0C)[0]
        (word,) = struct.unpack_from(">Q", b, co)
        self.assertEqual((word >> 56) & 0xFF, OPCODES["HALT"])


# ──────────────────────────────────────────────────────────────────────────────
class TestVMSemantics(unittest.TestCase):
    def test_halt_kills_all(self):
        vm = _make_vm(MINIMAL_SRC)
        vm.run()
        self.assertTrue(all(a.state == AgentState.DEAD for a in vm.agents.values()))

    def _run(self, body):
        src = f"""
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 1
.CODE
@_start:
{body}
  HALT
"""
        vm = _make_vm(src)
        vm.run()
        return vm.agents[0]

    def test_mov(self):
        ag = self._run("  MOV R0, 42")
        self.assertEqual(ag.R[0], 42)

    def test_add(self):
        ag = self._run("  MOV R0, 10\n  ADD R0, 5")
        self.assertEqual(ag.R[0], 15)

    def test_sub(self):
        ag = self._run("  MOV R0, 20\n  SUB R0, 7")
        self.assertEqual(ag.R[0], 13)

    def test_cmp_equal(self):
        ag = self._run("  MOV R0, 5\n  CMP R0, 5")
        self.assertTrue(ag.zero_flag)

    def test_cmp_not_equal(self):
        ag = self._run("  MOV R0, 5\n  CMP R0, 6")
        self.assertFalse(ag.zero_flag)

    def test_jmp_skips(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 1
.CODE
@_start:
  JMP @end
  MOV R0, 99
@end:
  HALT
"""
        vm = _make_vm(src); vm.run()
        self.assertEqual(vm.agents[0].R[0], 0)

    def test_call_ret(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 1
.CODE
@_start:
  CALL @func
  HALT
@func:
  MOV R0, 7
  RET
"""
        vm = _make_vm(src); vm.run()
        self.assertEqual(vm.agents[0].R[0], 7)

    def test_infinite_loop_terminates(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 1
.CODE
@_start:
  JMP @_start
"""
        _make_vm(src, max_steps=100).run()

    def test_som_init_random(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 1
.CODE
@_start:
  SOM_INIT RANDOM
  HALT
"""
        vm = _make_vm(src); vm.run()
        for r in range(4):
            for c in range(4):
                for w in vm.som.nodes[r][c].weights:
                    self.assertGreaterEqual(w, 0.0)
                    self.assertLessEqual(w, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
class TestSomMap(unittest.TestCase):
    def test_bmu_valid(self):
        som = SomMap(4, 4)
        r, c = som.bmu([0.5] * 16)
        self.assertIn(r, range(4))
        self.assertIn(c, range(4))

    def test_bmu_exact(self):
        som = SomMap(2, 2, dims=4)
        som.nodes[0][0].weights = [1,0,0,0]
        som.nodes[0][1].weights = [0,1,0,0]
        som.nodes[1][0].weights = [0,0,1,0]
        som.nodes[1][1].weights = [0,0,0,1]
        self.assertEqual(som.bmu([1,0,0,0]), (0,0))

    def test_train_moves_closer(self):
        som = SomMap(2, 2, dims=4)
        for r in range(2):
            for c in range(2):
                som.nodes[r][c].weights = [0.0]*4
        vec = [1.0]*4
        br, bc = som.bmu(vec)
        before = som.nodes[br][bc].weights[0]
        som.train(vec, br, bc, lr=0.5, sigma=1.0)
        self.assertGreater(som.nodes[br][bc].weights[0], before)

    def test_train_neighbours(self):
        som = SomMap(3, 3, dims=2)
        for r in range(3):
            for c in range(3):
                som.nodes[r][c].weights = [0.0, 0.0]
        som.train([1.0, 1.0], 1, 1, lr=0.5, sigma=2.0)
        for r in range(3):
            for c in range(3):
                self.assertGreater(som.nodes[r][c].weights[0], 0.0)

    def test_dims(self):
        som = SomMap(8, 8, dims=32)
        self.assertEqual(len(som.nodes[0][0].weights), 32)

    def test_shape(self):
        som = SomMap(16, 8)
        self.assertEqual(som.rows, 16)
        self.assertEqual(som.cols, 8)


# ──────────────────────────────────────────────────────────────────────────────
class TestMultiAgent(unittest.TestCase):
    def test_spawn_creates_child(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 4
.CODE
@_start:
  SPAWN A0, @worker
  WAIT A0
  HALT
@worker:
  MOV R0, 1
  AGENT_KILL SELF
"""
        vm = _make_vm(src, max_steps=50000); vm.run()
        self.assertGreaterEqual(len(vm.agents), 2)

    def test_fork_creates_multiple(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 16
.CODE
@_start:
  FORK 4, @worker
  HALT
@worker:
  AGENT_KILL SELF
"""
        vm = _make_vm(src, max_steps=50000); vm.run()
        self.assertGreaterEqual(len(vm.agents), 5)

    def test_msg_send_recv(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 4
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
        vm = _make_vm(src, max_steps=50000); vm.run()
        child = vm.agents.get(1)
        if child:
            self.assertEqual(child.R[0], 0x42)

    def test_som_elect(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 4
.CODE
@_start:
  SPAWN A0, @worker
  SOM_ELECT R0
  HALT
@worker:
  AGENT_KILL SELF
"""
        vm = _make_vm(src, max_steps=50000); vm.run()
        self.assertEqual(vm.agents[0].R[0], 0)

    def test_child_has_parent_id(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 4
.CODE
@_start:
  SPAWN A0, @worker
  HALT
@worker:
  AGENT_KILL SELF
"""
        vm = _make_vm(src, max_steps=10000); vm.run()
        child = vm.agents.get(1)
        if child:
            self.assertEqual(child.parent_id, 0)


# ──────────────────────────────────────────────────────────────────────────────
HELLO_AGENT_SRC = """
.SOMA    1.0.0
.ARCH    ANY
.SOMSIZE 4x4
.AGENTS  2
.DATA
  payload : MSG = 0xFF42
.CODE
@_start:
  SPAWN     A0, @worker
  SOM_MAP   A0, (0,0)
  MSG_SEND  A0, 0xFF42
  WAIT      A0
  HALT
@worker:
  MSG_RECV  R0
  SOM_TRAIN R0, S0
  MSG_SEND  PARENT, 0x00
  AGENT_KILL SELF
"""

SWARM_SRC = """
.SOMA    1.0.0
.SOMSIZE 8x8
.AGENTS  64
.CODE
@_start:
  SOM_INIT  RANDOM
  FORK      4, @explorer
  HALT
@explorer:
  MOV R0, 1
  SOM_WALK  SELF, GRADIENT
  SOM_TRAIN R0, S0
  AGENT_KILL SELF
"""


class TestFullPrograms(unittest.TestCase):
    def test_hello_assembles(self):
        self.assertEqual(assemble(HELLO_AGENT_SRC)[:4], MAGIC)

    def test_hello_runs(self):
        _make_vm(HELLO_AGENT_SRC, max_steps=50000).run()

    def test_hello_spawns_worker(self):
        vm = _make_vm(HELLO_AGENT_SRC, max_steps=50000); vm.run()
        self.assertGreaterEqual(len(vm.agents), 2)

    def test_swarm_assembles(self):
        self.assertEqual(assemble(SWARM_SRC)[:4], MAGIC)

    def test_swarm_runs(self):
        _make_vm(SWARM_SRC, max_steps=100000).run()

    def test_swarm_spawns_explorers(self):
        vm = _make_vm(SWARM_SRC, max_steps=100000); vm.run()
        self.assertGreaterEqual(len(vm.agents), 5)


# ──────────────────────────────────────────────────────────────────────────────
class TestDisassembler(unittest.TestCase):
    def test_produces_string(self):
        self.assertIsInstance(disassemble(assemble(MINIMAL_SRC)), str)

    def test_contains_halt(self):
        self.assertIn("HALT", disassemble(assemble(MINIMAL_SRC)))

    def test_contains_soma(self):
        self.assertIn("SOMA", disassemble(assemble(MINIMAL_SRC)))

    def test_invalid_magic_raises(self):
        self.assertRaises(ValueError, disassemble, b"BADM" + b"\x00"*28)

    def test_multiple_mnemonics(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 1
.CODE
@_start:
  NOP
  MOV R0, 42
  HALT
"""
        text = disassemble(assemble(src))
        for m in ("NOP", "MOV", "HALT"):
            self.assertIn(m, text)

    def test_arch_in_disasm(self):
        src = MINIMAL_SRC.replace(".ARCH    ANY", ".ARCH    ARM64")
        self.assertIn("ARM64", disassemble(assemble(src)))

    def test_pc_counter(self):
        self.assertIn("0000", disassemble(assemble(MINIMAL_SRC)))


# ──────────────────────────────────────────────────────────────────────────────
class TestArchTargets(unittest.TestCase):
    def _arch_byte(self, arch):
        src = MINIMAL_SRC.replace(".ARCH    ANY", f".ARCH    {arch}")
        return assemble(src)[0x08]

    def test_any(self):    self.assertEqual(self._arch_byte("ANY"),   0)
    def test_x86(self):    self.assertEqual(self._arch_byte("X86"),   1)
    def test_arm64(self):  self.assertEqual(self._arch_byte("ARM64"), 2)
    def test_arm(self):    self.assertEqual(self._arch_byte("ARM"),   2)
    def test_riscv(self):  self.assertEqual(self._arch_byte("RISCV"), 3)
    def test_wasm(self):   self.assertEqual(self._arch_byte("WASM"),  4)

    def test_wasm_runs(self):
        _make_vm(MINIMAL_SRC.replace(".ARCH    ANY", ".ARCH    WASM")).run()

    def test_arm64_runs(self):
        _make_vm(MINIMAL_SRC.replace(".ARCH    ANY", ".ARCH    ARM64")).run()


# ──────────────────────────────────────────────────────────────────────────────
class TestEdgeCases(unittest.TestCase):
    def test_invalid_binary_raises(self):
        self.assertRaises(VMError, SomaVM, b"JUNK" + b"\x00"*28)

    def test_empty_raises(self):
        self.assertRaises(VMError, SomaVM, b"")

    def test_corrupted_pc_graceful(self):
        vm = _make_vm(MINIMAL_SRC)
        vm.agents[0].pc = 999999
        vm.run()

    def test_som_walk_in_bounds(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 1
.CODE
@_start:
  SOM_WALK SELF, GRADIENT
  SOM_WALK SELF, GRADIENT
  SOM_WALK SELF, GRADIENT
  SOM_WALK SELF, GRADIENT
  SOM_WALK SELF, GRADIENT
  HALT
"""
        vm = _make_vm(src); vm.run()
        root = vm.agents[0]
        self.assertGreaterEqual(root.som_x, 0)
        self.assertLess(root.som_x, vm.som.rows)
        self.assertGreaterEqual(root.som_y, 0)
        self.assertLess(root.som_y, vm.som.cols)

    def test_large_som(self):
        src = MINIMAL_SRC.replace(".SOMSIZE 4x4", ".SOMSIZE 16x16")
        vm = _make_vm(src); vm.run()
        self.assertEqual(vm.som.rows, 16)
        self.assertEqual(vm.som.cols, 16)

    def test_blocked_recv_no_crash(self):
        src = """
.SOMA 1.0.0
.ARCH ANY
.SOMSIZE 4x4
.AGENTS 1
.CODE
@_start:
  MSG_RECV R0
  HALT
"""
        vm = _make_vm(src, max_steps=200); vm.run()
        root = vm.agents[0]
        self.assertIn(root.state, (AgentState.BLOCKED, AgentState.RUNNING, AgentState.DEAD))


# ──────────────────────────────────────────────────────────────────────────────
class TestPackaging(unittest.TestCase):
    def test_version(self):
        import soma
        self.assertEqual(soma.__version__, "1.0.0")

    def test_cli_callable(self):
        from soma.cli import main
        self.assertTrue(callable(main))

    def test_all_modules(self):
        import soma.isa, soma.lexer, soma.assembler, soma.vm, soma.cli


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
