"""
tests/test_bridge.py — runtime/bridge.py unit tests
====================================================

Test categories
---------------
Unit  — SomaInstruction   (from_word, from_bytes, to_bytes, repr, mnemonic, phase)
Unit  — SomaBinaryHeader  (from_bytes, validate: good magic, bad magic, bad version)
Unit  — BinaryDecoder     (from_bytes, instructions iterator, validate_header)
Unit  — BridgeValidator   (good instructions pass, all bad-operand cases rejected)
Unit  — load_and_validate (smoke: minimal .sombin payload)
Unit  — Phase V re-exports (opcode constants, PHASE_V_OPS set)
Unit  — encode/decode roundtrip for all 5 Phase V opcodes

Run
---
    pytest tests/test_bridge.py -v
    pytest tests/test_bridge.py -v -k "roundtrip"

Paper: "A Path to AGI Part V: Collective Intelligence", Swapnil Bhadade, 2026
"""

from __future__ import annotations

import struct
import pytest

from runtime.bridge import (
    # Constants re-exported from soma.isa
    OPCODES,
    OPCODE_NAMES,
    MAGIC,
    VER_MAJOR,
    HEADER_SIZE,
    WORD_SIZE,
    NICHE_CAPACITY,
    NICHE_MIGRATE_THRESH,
    SYMBOL_BIND_THRESH,
    HERITAGE_TOP_K,
    NICHE_IMM_DECLARE,
    NICHE_IMM_WITHDRAW,
    encode_word,
    decode_word,
    encode_reg,
    decode_reg,
    opcode_phase,
    # Bridge-specific
    OP_NICHE_DECLARE,
    OP_SYMBOL_EMERGE,
    OP_HERITAGE_LOAD,
    OP_NICHE_QUERY,
    OP_COLLECTIVE_SYNC,
    PHASE_V_OPS,
    SomaInstruction,
    SomaBinaryHeader,
    BinaryDecoder,
    BridgeValidator,
    load_and_validate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_instr(opcode, agent_id=0, som_x=0, som_y=0, reg=0, imm=0) -> SomaInstruction:
    word = encode_word(opcode, agent_id, som_x, som_y, reg, imm)
    return SomaInstruction.from_word(word)


def _minimal_sombin(instructions: list[SomaInstruction] | None = None) -> bytes:
    """Build a minimal valid .sombin payload (header + optional instructions)."""
    _HDR_STRUCT = struct.Struct(">4sBBHHIHHH")
    # Pack to 18 bytes then pad to HEADER_SIZE (32)
    raw = _HDR_STRUCT.pack(
        MAGIC, VER_MAJOR, 0,   # magic, major, minor
        0,                      # arch
        1,                      # num_agents
        HEADER_SIZE,            # entry_pc (points right after header)
        16, 16,                 # som_width, som_height
        0,                      # flags
    )
    header_bytes = raw + b"\x00" * (HEADER_SIZE - len(raw))
    body = b"".join(i.to_bytes() for i in (instructions or []))
    return header_bytes + body


# ══════════════════════════════════════════════════════════════════════════════
# Phase V opcode constants — sanity checks
# ══════════════════════════════════════════════════════════════════════════════

class TestPhaseVConstants:

    def test_opcode_values(self):
        assert OP_NICHE_DECLARE   == 0x73
        assert OP_SYMBOL_EMERGE   == 0x74
        assert OP_HERITAGE_LOAD   == 0x75
        assert OP_NICHE_QUERY     == 0x76
        assert OP_COLLECTIVE_SYNC == 0x77

    def test_phase_v_ops_set_contains_all_five(self):
        assert PHASE_V_OPS == {
            OP_NICHE_DECLARE, OP_SYMBOL_EMERGE, OP_HERITAGE_LOAD,
            OP_NICHE_QUERY, OP_COLLECTIVE_SYNC,
        }

    def test_opcodes_present_in_global_table(self):
        for op in PHASE_V_OPS:
            assert op in OPCODE_NAMES, f"0x{op:02X} missing from OPCODE_NAMES"

    def test_niche_imm_constants(self):
        assert NICHE_IMM_DECLARE  == 0x0001
        assert NICHE_IMM_WITHDRAW == 0x0002

    def test_tuning_constants_positive(self):
        assert NICHE_CAPACITY        >  0
        assert 0 < NICHE_MIGRATE_THRESH < 1.0
        assert SYMBOL_BIND_THRESH    >= 1
        assert HERITAGE_TOP_K        >= 1


# ══════════════════════════════════════════════════════════════════════════════
# SomaInstruction — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSomaInstruction:

    def test_from_word_roundtrip(self):
        word = encode_word(OP_NICHE_DECLARE, 5, 3, 7, 0xFF01, NICHE_IMM_DECLARE)
        instr = SomaInstruction.from_word(word)
        assert instr.opcode   == OP_NICHE_DECLARE
        assert instr.agent_id == 5
        assert instr.som_x    == 3
        assert instr.som_y    == 7
        assert instr.imm      == NICHE_IMM_DECLARE

    def test_to_bytes_and_back(self):
        instr = _make_instr(OP_COLLECTIVE_SYNC, 0, 1, 2, 0, 0)
        raw   = instr.to_bytes()
        assert len(raw) == WORD_SIZE
        back  = SomaInstruction.from_bytes(raw)
        assert back.opcode   == instr.opcode
        assert back.agent_id == instr.agent_id
        assert back.som_x    == instr.som_x
        assert back.som_y    == instr.som_y

    def test_from_bytes_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            SomaInstruction.from_bytes(b"\x00" * 3)

    def test_mnemonic_assigned(self):
        instr = _make_instr(OP_NICHE_DECLARE)
        assert instr.mnemonic == "NICHE_DECLARE"

    def test_mnemonic_unknown_opcode(self):
        word  = encode_word(0xFE, 0, 0, 0, 0, 0)   # 0xFE not in ISA
        instr = SomaInstruction.from_word(word)
        assert "UNK" in instr.mnemonic or "0xFE" in instr.mnemonic.upper()

    def test_phase_v_instructions_return_phase_v(self):
        for op in PHASE_V_OPS:
            instr = _make_instr(op)
            assert instr.phase == "V", f"0x{op:02X} phase={instr.phase!r}"

    def test_phase_i_instruction_returns_phase_i(self):
        spawn_op = OPCODES["SPAWN"]
        instr = _make_instr(spawn_op)
        assert instr.phase == "I"

    def test_repr_contains_mnemonic(self):
        instr = _make_instr(OP_NICHE_DECLARE, imm=NICHE_IMM_DECLARE)
        r = repr(instr)
        assert "NICHE_DECLARE" in r

    def test_word_field_consistent(self):
        instr = _make_instr(OP_SYMBOL_EMERGE, 2, 3, 4, 0x10, SYMBOL_BIND_THRESH)
        decoded = decode_word(instr.word)
        assert decoded["opcode"]   == OP_SYMBOL_EMERGE
        assert decoded["agent_id"] == 2
        assert decoded["som_x"]    == 3
        assert decoded["som_y"]    == 4

    @pytest.mark.parametrize("op", list(PHASE_V_OPS))
    def test_all_phase_v_opcodes_roundtrip(self, op):
        word  = encode_word(op, 1, 2, 3, 0x0010, 0x0001)
        instr = SomaInstruction.from_word(word)
        assert instr.opcode   == op
        assert instr.agent_id == 1
        assert instr.som_x    == 2
        assert instr.som_y    == 3


# ══════════════════════════════════════════════════════════════════════════════
# SomaBinaryHeader — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSomaBinaryHeader:

    def test_validate_good_header(self):
        payload = _minimal_sombin()
        hdr = SomaBinaryHeader.from_bytes(payload)
        errors = hdr.validate()
        assert errors == []

    def test_validate_bad_magic(self):
        payload  = bytearray(_minimal_sombin())
        payload[:4] = b"XXXX"
        hdr = SomaBinaryHeader.from_bytes(bytes(payload))
        errors = hdr.validate()
        assert any("magic" in e.lower() for e in errors)

    def test_validate_bad_version(self):
        payload = bytearray(_minimal_sombin())
        # Major version byte is at offset 4
        payload[4] = 0xFF
        hdr = SomaBinaryHeader.from_bytes(bytes(payload))
        errors = hdr.validate()
        assert any("version" in e.lower() for e in errors)

    def test_from_bytes_too_short_raises(self):
        with pytest.raises(ValueError):
            SomaBinaryHeader.from_bytes(b"\x00" * 4)

    def test_num_agents_field(self):
        payload = _minimal_sombin()
        hdr = SomaBinaryHeader.from_bytes(payload)
        assert hdr.num_agents == 1

    def test_som_dimensions(self):
        payload = _minimal_sombin()
        hdr = SomaBinaryHeader.from_bytes(payload)
        assert hdr.som_width  == 16
        assert hdr.som_height == 16


# ══════════════════════════════════════════════════════════════════════════════
# BinaryDecoder — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBinaryDecoder:

    def test_empty_body_yields_no_instructions(self):
        payload = _minimal_sombin([])
        dec = BinaryDecoder.from_bytes(payload)
        assert list(dec.instructions()) == []

    def test_single_instruction_decoded(self):
        instr = _make_instr(OP_NICHE_DECLARE, 0, 1, 2, 0, NICHE_IMM_DECLARE)
        payload = _minimal_sombin([instr])
        dec = BinaryDecoder.from_bytes(payload)
        decoded = list(dec.instructions())
        assert len(decoded) == 1
        assert decoded[0].opcode   == OP_NICHE_DECLARE
        assert decoded[0].agent_id == 0
        assert decoded[0].som_x    == 1
        assert decoded[0].som_y    == 2

    def test_multiple_instructions_in_order(self):
        ops = [OP_NICHE_DECLARE, OP_SYMBOL_EMERGE, OP_COLLECTIVE_SYNC]
        instrs = [_make_instr(op) for op in ops]
        payload = _minimal_sombin(instrs)
        dec  = BinaryDecoder.from_bytes(payload)
        decoded = list(dec.instructions())
        assert [d.opcode for d in decoded] == ops

    def test_header_accessible_from_decoder(self):
        payload = _minimal_sombin()
        dec = BinaryDecoder.from_bytes(payload)
        assert dec.header.magic == MAGIC

    def test_validate_header_good(self):
        payload = _minimal_sombin()
        dec = BinaryDecoder.from_bytes(payload)
        assert dec.validate_header() == []

    def test_validate_header_bad(self):
        payload = bytearray(_minimal_sombin())
        payload[:4] = b"FAIL"
        dec = BinaryDecoder.from_bytes(bytes(payload))
        assert len(dec.validate_header()) > 0

    def test_from_bytes_is_idempotent(self):
        """Decoding the same bytes twice gives the same instructions."""
        instr = _make_instr(OP_NICHE_QUERY, 3, 4, 5)
        payload = _minimal_sombin([instr])
        dec1 = list(BinaryDecoder.from_bytes(payload).instructions())
        dec2 = list(BinaryDecoder.from_bytes(payload).instructions())
        assert dec1[0].opcode == dec2[0].opcode

    def test_from_decoder_factory(self):
        instr = _make_instr(OP_HERITAGE_LOAD, imm=HERITAGE_TOP_K)
        payload = _minimal_sombin([instr])
        dec = BinaryDecoder.from_bytes(payload)
        validator = BridgeValidator.from_decoder(dec)
        assert isinstance(validator, BridgeValidator)


# ══════════════════════════════════════════════════════════════════════════════
# BridgeValidator — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBridgeValidator:

    # ── Good instructions ─────────────────────────────────────────────────────

    def test_niche_declare_valid_declare(self):
        instr = _make_instr(OP_NICHE_DECLARE, 0, 0, 0, 0, NICHE_IMM_DECLARE)
        errs  = BridgeValidator([instr]).validate()
        assert errs == []

    def test_niche_declare_valid_withdraw(self):
        instr = _make_instr(OP_NICHE_DECLARE, 0, 0, 0, 0, NICHE_IMM_WITHDRAW)
        errs  = BridgeValidator([instr]).validate()
        assert errs == []

    def test_symbol_emerge_valid(self):
        instr = _make_instr(OP_SYMBOL_EMERGE, 0, 0, 0, 0, SYMBOL_BIND_THRESH)
        errs  = BridgeValidator([instr]).validate()
        assert errs == []

    def test_heritage_load_valid(self):
        instr = _make_instr(OP_HERITAGE_LOAD, 0, 0, 0, 0, HERITAGE_TOP_K)
        errs  = BridgeValidator([instr]).validate()
        assert errs == []

    def test_collective_sync_valid(self):
        instr = _make_instr(OP_COLLECTIVE_SYNC)
        errs  = BridgeValidator([instr]).validate()
        assert errs == []

    def test_niche_query_valid(self):
        instr = _make_instr(OP_NICHE_QUERY, 0, 5, 5)
        errs  = BridgeValidator([instr]).validate()
        assert errs == []

    def test_empty_instruction_list(self):
        assert BridgeValidator([]).validate() == []

    # ── Bad instructions ──────────────────────────────────────────────────────

    def test_niche_declare_bad_imm(self):
        instr = _make_instr(OP_NICHE_DECLARE, 0, 0, 0, 0, 0x0099)   # not 1 or 2
        errs  = BridgeValidator([instr]).validate()
        assert any("imm" in e.lower() or "NICHE_DECLARE" in e for e in errs)

    def test_niche_declare_agent_id_too_large(self):
        instr = _make_instr(OP_NICHE_DECLARE, NICHE_CAPACITY, 0, 0, 0, NICHE_IMM_DECLARE)
        errs  = BridgeValidator([instr]).validate()
        assert any("agent_id" in e.lower() or "NICHE_DECLARE" in e for e in errs)

    def test_symbol_emerge_count_below_threshold(self):
        instr = _make_instr(OP_SYMBOL_EMERGE, 0, 0, 0, 0, SYMBOL_BIND_THRESH - 1)
        errs  = BridgeValidator([instr]).validate()
        assert any("SYMBOL_EMERGE" in e or "threshold" in e.lower() for e in errs)

    def test_heritage_load_top_k_zero(self):
        instr = _make_instr(OP_HERITAGE_LOAD, 0, 0, 0, 0, 0)   # top-K = 0
        errs  = BridgeValidator([instr]).validate()
        assert any("HERITAGE_LOAD" in e or "top" in e.lower() for e in errs)

    def test_heritage_load_top_k_too_large(self):
        instr = _make_instr(OP_HERITAGE_LOAD, 0, 0, 0, 0, HERITAGE_TOP_K + 100)
        errs  = BridgeValidator([instr]).validate()
        assert any("HERITAGE_LOAD" in e for e in errs)

    def test_unknown_opcode_flagged(self):
        # 0xFE is not in the ISA
        word  = encode_word(0xFE, 0, 0, 0, 0, 0)
        instr = SomaInstruction.from_word(word)
        errs  = BridgeValidator([instr]).validate()
        assert any("unknown" in e.lower() or "0xFE" in e.upper() for e in errs)

    def test_multiple_bad_instructions_all_reported(self):
        bad1 = _make_instr(OP_NICHE_DECLARE, 0, 0, 0, 0, 0x0099)
        bad2 = _make_instr(OP_SYMBOL_EMERGE, 0, 0, 0, 0, 0)
        errs = BridgeValidator([bad1, bad2]).validate()
        assert len(errs) >= 2

    def test_mixed_good_and_bad(self):
        good = _make_instr(OP_COLLECTIVE_SYNC)
        bad  = _make_instr(OP_NICHE_DECLARE, 0, 0, 0, 0, 0x0099)
        errs = BridgeValidator([good, bad]).validate()
        assert len(errs) == 1   # only bad instruction errors


# ══════════════════════════════════════════════════════════════════════════════
# load_and_validate — integration smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadAndValidate:

    def test_valid_payload_no_errors(self, tmp_path):
        instr   = _make_instr(OP_NICHE_DECLARE, 0, 0, 0, 0, NICHE_IMM_DECLARE)
        payload = _minimal_sombin([instr])
        path    = tmp_path / "test.sombin"
        path.write_bytes(payload)

        hdr, instrs, errors = load_and_validate(path)
        assert errors == []
        assert len(instrs) == 1
        assert instrs[0].opcode == OP_NICHE_DECLARE
        assert hdr.magic == MAGIC

    def test_invalid_payload_returns_errors(self, tmp_path):
        payload = bytearray(_minimal_sombin())
        payload[:4] = b"XXXX"   # corrupt magic
        path = tmp_path / "bad.sombin"
        path.write_bytes(bytes(payload))

        _, _, errors = load_and_validate(path)
        assert len(errors) > 0

    def test_empty_body_valid_header(self, tmp_path):
        payload = _minimal_sombin([])
        path    = tmp_path / "empty.sombin"
        path.write_bytes(payload)

        hdr, instrs, errors = load_and_validate(path)
        assert errors == []
        assert instrs == []

    def test_multiple_instructions_all_decoded(self, tmp_path):
        ops     = [OP_NICHE_DECLARE, OP_SYMBOL_EMERGE, OP_HERITAGE_LOAD,
                   OP_NICHE_QUERY, OP_COLLECTIVE_SYNC]
        instrs  = [
            _make_instr(OP_NICHE_DECLARE,   0, 0, 0, 0, NICHE_IMM_DECLARE),
            _make_instr(OP_SYMBOL_EMERGE,   0, 0, 0, 0, SYMBOL_BIND_THRESH),
            _make_instr(OP_HERITAGE_LOAD,   0, 0, 0, 0, HERITAGE_TOP_K),
            _make_instr(OP_NICHE_QUERY,     0, 1, 1),
            _make_instr(OP_COLLECTIVE_SYNC, 0, 0, 0),
        ]
        payload = _minimal_sombin(instrs)
        path    = tmp_path / "all5.sombin"
        path.write_bytes(payload)

        _, decoded, errors = load_and_validate(path)
        assert errors == []
        assert [d.opcode for d in decoded] == ops


# ══════════════════════════════════════════════════════════════════════════════
# encode / decode helpers re-exported through bridge
# ══════════════════════════════════════════════════════════════════════════════

class TestEncodeDecodeFunctions:

    def test_encode_word_decode_word_roundtrip(self):
        fields = dict(opcode=OP_NICHE_DECLARE, agent_id=7, som_x=3,
                      som_y=12, reg=0x00FF, imm=0x0001)
        word   = encode_word(**fields)
        back   = decode_word(word)
        for k, v in fields.items():
            assert back[k] == v, f"{k}: {back[k]!r} != {v!r}"

    def test_encode_reg_named_register(self):
        """encode_reg should handle register name strings without raising."""
        r = encode_reg("R0")
        assert isinstance(r, int)

    def test_opcode_phase_phase_v(self):
        for op in PHASE_V_OPS:
            assert opcode_phase(op) == "V"

    def test_opcode_phase_phase_i(self):
        assert opcode_phase(OPCODES["SPAWN"]) == "I"
