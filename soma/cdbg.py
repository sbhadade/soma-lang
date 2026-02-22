"""
soma/cdbg.py — Context-Discriminated Binary Grammar (CDBG)
===========================================================

SOMA v4 Part IV: How the same 3 payload bytes mean different things
depending on the 4-bit CTX nibble in the frame header.

Frame format (5 bytes total):
    ┌──────────┬──────────┬─────────────────────┬─────────────────┐
    │  CTX[4b] │  SUB[4b] │   PAYLOAD (3 bytes) │ CHK[4b] R[4b]  │
    └──────────┴──────────┴─────────────────────┴─────────────────┘
    Byte 0  = (CTX << 4) | SUB
    Bytes 1-3 = 3-byte payload (varies by CTX)
    Byte 4  = (CRC4 << 4) | 0x0  (reserved nibble)

CTX Namespaces:
    0x0  SOM_MAP   — coordinate + opcode
    0x1  AGENT     — 24-bit agent identity (cluster/map/seq)
    0x2  SOUL      — soul field update (field_id + fp16 value)
    0x3  MEMORY    — content-addressed memory reference (hash slice)
    0x4  PULSE     — heartbeat / timing
    0x5  EMOTION   — EMOT_TAG / SURPRISE_CALC payload
    0x6  HISTORY   — lifecycle event records

CRC-4 polynomial: 0x13 (standard ITU-T CRC-4)
No external dependencies — CRC-4 is computed inline.

Design principle (Part IV §3.3):
    Decode is a single table jump on the CTX nibble — O(1), fits in 6
    instructions, branch predictor learns it after the first few hundred
    frames.  Zero dynamic dispatch overhead.

Paper: "A Path to AGI Part IV: Binary Grammar", Swapnil Bhadade, Feb 2026
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional


# ── CRC-4 (ITU-T polynomial 0x13) ────────────────────────────────────────────

_CRC4_TABLE = [0] * 16

def _build_crc4_table() -> None:
    poly = 0x13   # x^4 + x + 1, standard CRC-4/ITU
    for i in range(16):
        crc = i
        for _ in range(4):
            if crc & 0x8:
                crc = ((crc << 1) ^ poly) & 0xF
            else:
                crc = (crc << 1) & 0xF
        _CRC4_TABLE[i] = crc

_build_crc4_table()


def crc4(data: bytes) -> int:
    """Compute CRC-4/ITU of `data`.  Returns 4-bit value (0–15)."""
    crc = 0
    for byte in data:
        crc = _CRC4_TABLE[crc ^ (byte >> 4)]
        crc = _CRC4_TABLE[crc ^ (byte & 0xF)]
    return crc & 0xF


# ── Context namespaces ────────────────────────────────────────────────────────

class CTX(IntEnum):
    SOM_MAP  = 0x0   # SOM coordinate + opcode
    AGENT    = 0x1   # 24-bit agent identity
    SOUL     = 0x2   # soul field update
    MEMORY   = 0x3   # content-addressed memory ref
    PULSE    = 0x4   # heartbeat / timing
    EMOTION  = 0x5   # emotional tag
    HISTORY  = 0x6   # lifecycle event
    # 0x7–0xF reserved for user extensions


CTX_NAMES = {v: v.name for v in CTX}


# ── Frame dataclass ────────────────────────────────────────────────────────────

@dataclass
class Frame:
    """
    One CDBG binary frame (5 bytes on the wire).

    ctx     : CTX — which grammar tree applies
    sub     : int 0–15 — schema version / subtype
    payload : bytes — exactly 3 bytes; meaning depends on ctx
    """
    ctx:     CTX
    sub:     int
    payload: bytes

    def __post_init__(self) -> None:
        if len(self.payload) != 3:
            raise ValueError(
                f"CDBG payload must be exactly 3 bytes, got {len(self.payload)}"
            )
        if not 0 <= self.sub <= 15:
            raise ValueError(f"CDBG SUB nibble must be 0–15, got {self.sub}")

    def encode(self) -> bytes:
        """Encode to 5-byte wire format."""
        header = ((int(self.ctx) & 0xF) << 4) | (self.sub & 0xF)
        body   = bytes([header]) + self.payload
        chk    = crc4(body)
        footer = (chk << 4) & 0xFF   # lower nibble = reserved 0x0
        return body + bytes([footer])

    @classmethod
    def decode(cls, raw: bytes) -> "Frame":
        """Decode from 5-byte wire format.  Raises ValueError on CRC mismatch."""
        if len(raw) != 5:
            raise ValueError(f"CDBG frame must be 5 bytes, got {len(raw)}")
        header    = raw[0]
        ctx_nibble = (header >> 4) & 0xF
        sub        = header & 0xF
        payload    = raw[1:4]
        footer     = raw[4]
        chk_recv   = (footer >> 4) & 0xF
        chk_calc   = crc4(raw[:4])
        if chk_recv != chk_calc:
            raise ValueError(
                f"CDBG CRC-4 mismatch: got {chk_recv:#x}, expected {chk_calc:#x}"
            )
        try:
            ctx = CTX(ctx_nibble)
        except ValueError:
            raise ValueError(f"Unknown CTX nibble: {ctx_nibble:#x}")
        return cls(ctx=ctx, sub=sub, payload=payload)

    def parsed(self) -> Dict[str, Any]:
        """Grammar-aware parse of payload — delegates to ctx-specific parser."""
        return _PARSERS[self.ctx](self)

    def __repr__(self) -> str:
        p = self.payload.hex()
        return (f"Frame(ctx={self.ctx.name} sub={self.sub} "
                f"payload=0x{p} chk={crc4(bytes([((int(self.ctx)&0xF)<<4)|self.sub]) + self.payload):#x})")


# ── Grammar parsers (one per CTX) ────────────────────────────────────────────

def _parse_som_map(f: Frame) -> Dict[str, Any]:
    """
    CTX=0x0 — SOM_MAP
    SUB=0x3: X[8] · Y[8] · OPCODE[8]   (max 256×256 map)
    SUB=0x4: XY_PACKED[24] 12-bit X + 12-bit Y (max 4096×4096 map)
    """
    if f.sub == 0x4:
        packed = int.from_bytes(f.payload, "big")
        x = (packed >> 12) & 0xFFF
        y = packed & 0xFFF
        return {"context": "SOM_MAP_XL", "x": x, "y": y}
    x, y, opcode = f.payload
    return {"context": "SOM_MAP", "x": x, "y": y, "opcode": opcode}


def _parse_agent(f: Frame) -> Dict[str, Any]:
    """
    CTX=0x1 — AGENT
    SUB=0x0: flat 24-bit agent ID with embedded cluster/map/seq
    SUB=0x1: SPAWN record — parent[8] parent[8] generation[8]
    """
    if f.sub == 0x1:
        p_hi, p_lo, gen = f.payload
        parent = (p_hi << 8) | p_lo
        return {"context": "AGENT_SPAWN", "parent_id": parent, "generation": gen}

    agent_id = int.from_bytes(f.payload, "big")
    cluster  = (agent_id >> 20) & 0xF
    map_id   = (agent_id >> 12) & 0xFF
    seq      = agent_id & 0xFFF
    return {
        "context":  "AGENT",
        "agent_id": agent_id,
        "cluster":  cluster,
        "map_id":   map_id,
        "seq":      seq,
    }


# Soul field registry (Part IV §2.3)
SOUL_FIELDS = {
    0x00: "valence_mean",
    0x01: "surprise_sensitivity",
    0x02: "decay_signature",
    0x03: "curiosity_drive",
    0x04: "goal_stall_count",
    0x05: "inheritance_depth",
    0x10: "birth_pulse_hi",
    0x11: "birth_pulse_lo",
    0xFF: "soul_checksum",
}


def _parse_soul(f: Frame) -> Dict[str, Any]:
    """
    CTX=0x2 — SOUL
    SUB=0x0: SOUL_FIELD[8] · VALUE[16] (fp16)
    """
    field_id = f.payload[0]
    value    = struct.unpack(">e", f.payload[1:3])[0]   # fp16 big-endian
    name     = SOUL_FIELDS.get(field_id, f"field_{field_id:#04x}")
    return {"context": "SOUL", "field": name, "field_id": field_id, "value": float(value)}


def _parse_memory(f: Frame) -> Dict[str, Any]:
    """
    CTX=0x3 — MEMORY
    SUB=0x0: REF — 24-bit hash prefix (bucket lookup)
    SUB=0x1: TAG — hash_anchor[8] · valence[8] · intensity[8]
    """
    if f.sub == 0x1:
        anchor, valence_raw, intensity_raw = f.payload
        valence   = (valence_raw - 128) / 128.0   # s8 → [-1, +1]
        intensity = intensity_raw / 255.0
        return {
            "context":   "MEMORY_TAG",
            "hash_anchor": anchor,
            "valence":   round(valence, 4),
            "intensity": round(intensity, 4),
        }
    hash_prefix = int.from_bytes(f.payload, "big")
    return {"context": "MEMORY_REF", "hash_prefix": hash_prefix}


def _parse_pulse(f: Frame) -> Dict[str, Any]:
    """
    CTX=0x4 — PULSE
    SUB=0x0: pulse_counter[24]
    """
    pulse = int.from_bytes(f.payload, "big")
    return {"context": "PULSE", "pulse": pulse}


def _parse_emotion(f: Frame) -> Dict[str, Any]:
    """
    CTX=0x5 — EMOTION
    SUB=0x0: EMOT_TAG    — coord_r[8] · valence[8] · intensity[8]
    SUB=0x1: SURPRISE    — bmu_r[8] · bmu_c[8] · err[8] (fixed-point 0–255 = 0–1)
    """
    if f.sub == 0x1:
        r, c, err_raw = f.payload
        return {
            "context":   "SURPRISE",
            "bmu_r": r, "bmu_c": c,
            "error": round(err_raw / 255.0, 4),
        }
    r, valence_raw, intensity_raw = f.payload
    valence   = (valence_raw - 128) / 128.0
    intensity = intensity_raw / 255.0
    return {
        "context":   "EMOT_TAG",
        "row":       r,
        "valence":   round(valence, 4),
        "intensity": round(intensity, 4),
    }


def _parse_history(f: Frame) -> Dict[str, Any]:
    """
    CTX=0x6 — HISTORY
    SUB=0x0: GOAL record — generation[8] · goal_record_id[16]
    SUB=0x1: MAP_RESIDENCY — som_id[8] · pulses_spent[16]
    """
    if f.sub == 0x1:
        som_id, p_hi, p_lo = f.payload
        pulses = (p_hi << 8) | p_lo
        return {"context": "MAP_RESIDENCY", "som_id": som_id, "pulses_spent": pulses}
    gen = f.payload[0]
    goal_id = (f.payload[1] << 8) | f.payload[2]
    return {"context": "HISTORY_GOAL", "generation": gen, "goal_record_id": goal_id}


_PARSERS = {
    CTX.SOM_MAP:  _parse_som_map,
    CTX.AGENT:    _parse_agent,
    CTX.SOUL:     _parse_soul,
    CTX.MEMORY:   _parse_memory,
    CTX.PULSE:    _parse_pulse,
    CTX.EMOTION:  _parse_emotion,
    CTX.HISTORY:  _parse_history,
}


# ── Encoder helpers ────────────────────────────────────────────────────────────

class Encoder:
    """Convenience factory for all CDBG frame types."""

    @staticmethod
    def som_map(x: int, y: int, opcode: int, sub: int = 3) -> Frame:
        if sub == 4:
            packed = ((x & 0xFFF) << 12) | (y & 0xFFF)
            payload = packed.to_bytes(3, "big")
        else:
            payload = bytes([x & 0xFF, y & 0xFF, opcode & 0xFF])
        return Frame(CTX.SOM_MAP, sub, payload)

    @staticmethod
    def agent(agent_id: int) -> Frame:
        payload = (agent_id & 0xFFFFFF).to_bytes(3, "big")
        return Frame(CTX.AGENT, 0, payload)

    @staticmethod
    def agent_spawn(parent_id: int, generation: int) -> Frame:
        p_hi = (parent_id >> 8) & 0xFF
        p_lo = parent_id & 0xFF
        payload = bytes([p_hi, p_lo, generation & 0xFF])
        return Frame(CTX.AGENT, 1, payload)

    @staticmethod
    def soul_field(field_id: int, value: float) -> Frame:
        fp16 = struct.pack(">e", value)
        payload = bytes([field_id & 0xFF]) + fp16
        return Frame(CTX.SOUL, 0, payload)

    @staticmethod
    def memory_ref(hash_prefix: int) -> Frame:
        payload = (hash_prefix & 0xFFFFFF).to_bytes(3, "big")
        return Frame(CTX.MEMORY, 0, payload)

    @staticmethod
    def memory_tag(hash_anchor: int, valence: float, intensity: float) -> Frame:
        v8  = max(0, min(255, int((valence + 1.0) / 2.0 * 255)))
        i8  = max(0, min(255, int(intensity * 255)))
        payload = bytes([hash_anchor & 0xFF, v8, i8])
        return Frame(CTX.MEMORY, 1, payload)

    @staticmethod
    def pulse(pulse_count: int) -> Frame:
        payload = (pulse_count & 0xFFFFFF).to_bytes(3, "big")
        return Frame(CTX.PULSE, 0, payload)

    @staticmethod
    def emot_tag(row: int, valence: float, intensity: float) -> Frame:
        v8  = max(0, min(255, int((valence + 1.0) / 2.0 * 255)))
        i8  = max(0, min(255, int(intensity * 255)))
        payload = bytes([row & 0xFF, v8, i8])
        return Frame(CTX.EMOTION, 0, payload)

    @staticmethod
    def surprise(bmu_r: int, bmu_c: int, error: float) -> Frame:
        err8 = max(0, min(255, int(error * 255)))
        payload = bytes([bmu_r & 0xFF, bmu_c & 0xFF, err8])
        return Frame(CTX.EMOTION, 1, payload)

    @staticmethod
    def history_goal(generation: int, goal_record_id: int) -> Frame:
        payload = bytes([generation & 0xFF,
                         (goal_record_id >> 8) & 0xFF,
                         goal_record_id & 0xFF])
        return Frame(CTX.HISTORY, 0, payload)

    @staticmethod
    def history_map(som_id: int, pulses_spent: int) -> Frame:
        payload = bytes([som_id & 0xFF,
                         (pulses_spent >> 8) & 0xFF,
                         pulses_spent & 0xFF])
        return Frame(CTX.HISTORY, 1, payload)


# ── Stream decoder ─────────────────────────────────────────────────────────────

class StreamDecoder:
    """
    Stateful decoder for a raw byte stream of concatenated CDBG frames.
    Accumulates bytes and emits Frame objects as each 5-byte boundary is reached.
    """

    FRAME_SIZE = 5

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes):
        """Feed bytes; yields complete Frame objects."""
        self._buf.extend(data)
        while len(self._buf) >= self.FRAME_SIZE:
            raw = bytes(self._buf[:self.FRAME_SIZE])
            self._buf = self._buf[self.FRAME_SIZE:]
            try:
                yield Frame.decode(raw)
            except ValueError as e:
                # Bad frame — skip and re-sync
                yield None   # caller can handle or ignore

    def __repr__(self) -> str:
        return f"StreamDecoder(buffered={len(self._buf)} bytes)"


# ── Agent ID utils ────────────────────────────────────────────────────────────

def make_agent_id(cluster: int, map_id: int, seq: int) -> int:
    """
    Build a 24-bit agent ID from cluster, map, and sequence number.
    cluster : 0–15   (4 bits)
    map_id  : 0–255  (8 bits)
    seq     : 0–4095 (12 bits)
    Total   : 16.7M unique agents.
    """
    return ((cluster & 0xF) << 20) | ((map_id & 0xFF) << 12) | (seq & 0xFFF)


def parse_agent_id(agent_id: int) -> Dict[str, int]:
    return {
        "cluster": (agent_id >> 20) & 0xF,
        "map_id":  (agent_id >> 12) & 0xFF,
        "seq":     agent_id & 0xFFF,
    }


# ── Soul streaming ────────────────────────────────────────────────────────────

def encode_soul_snapshot(agent_id: int, soul_fields: Dict[str, float]) -> bytes:
    """
    Encode a soul snapshot as a sequence of CDBG frames.
    Returns concatenated 5-byte frames ready for wire transmission.

    soul_fields: dict mapping SOUL_FIELDS names → float values.
    """
    _REV_FIELDS = {v: k for k, v in SOUL_FIELDS.items()}
    frames = b""
    # Agent header frame first
    frames += Encoder.agent(agent_id).encode()
    for name, value in soul_fields.items():
        fid = _REV_FIELDS.get(name)
        if fid is None:
            continue
        frames += Encoder.soul_field(fid, float(value)).encode()
    return frames
