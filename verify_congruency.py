#!/usr/bin/env python3
"""
verify_congruency.py — Triple-layer ISA congruency checker for SOMA.

Three layers verified:
  1. CANONICAL  — soma/isa.py defines the ground-truth opcode table.
  2. BRIDGE     — runtime/bridge.py re-exports the table; must be identical.
  3. RUNTIME    — runtime/collective.py + runtime/interpreter.py must declare
                  handlers for every Phase-V opcode (0x73–0x77).

Exit codes:
  0  — all checks pass (green)
  1  — one or more congruency failures (red, with diff)

Usage:
  python verify_congruency.py            # run checks, print report
  python verify_congruency.py --strict   # additionally fail on any warning
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
STRICT = "--strict" in sys.argv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_module(rel_path: str) -> Any:
    """Import a module by path relative to ROOT without running __main__."""
    abs_path = ROOT / rel_path
    if not abs_path.exists():
        raise FileNotFoundError(f"Module not found: {abs_path}")
    spec = importlib.util.spec_from_file_location(
        rel_path.replace("/", ".").removesuffix(".py"), abs_path
    )
    mod = importlib.util.module_from_spec(spec)      # type: ignore[arg-type]
    spec.loader.exec_module(mod)                     # type: ignore[union-attr]
    return mod


def _extract_opcode_dict_from_ast(source: str) -> dict[str, int] | None:
    """
    Parse the Python source and return the first dict literal assigned to a
    variable named OPCODES, without executing the module.  Returns None if the
    pattern is not found.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t for t in node.targets if isinstance(t, ast.Name)]
        if any(t.id == "OPCODES" for t in targets) and isinstance(node.value, ast.Dict):
            result: dict[str, int] = {}
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    # value may be ast.Constant(int) or ast.Constant(hex)
                    if isinstance(v, ast.Constant) and isinstance(v.value, int):
                        result[k.value] = v.value
            return result
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Check results collector
# ─────────────────────────────────────────────────────────────────────────────

class Report:
    def __init__(self) -> None:
        self.errors:   list[str] = []
        self.warnings: list[str] = []
        self.oks:      list[str] = []

    def ok(self, msg: str) -> None:
        self.oks.append(msg)
        print(f"  ✅  {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  ⚠️   {msg}")

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"  ❌  {msg}")

    @property
    def passed(self) -> bool:
        if self.errors:
            return False
        if STRICT and self.warnings:
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Canonical ISA
# ─────────────────────────────────────────────────────────────────────────────

def check_canonical(r: Report) -> dict[str, int]:
    print("\n── Layer 1: Canonical ISA (soma/isa.py) ──────────────────────────────────")
    try:
        isa = _load_module("soma/isa.py")
    except Exception as exc:
        r.error(f"Cannot load soma/isa.py: {exc}")
        return {}

    opcodes: dict[str, int] = getattr(isa, "OPCODES", {})
    names:   dict[int, str] = getattr(isa, "OPCODE_NAMES", {})

    if not opcodes:
        r.error("OPCODES dict is empty or missing in soma/isa.py")
        return {}

    r.ok(f"OPCODES table present — {len(opcodes)} entries")

    # Check reverse table completeness (ignoring NOP/RET collision)
    missing_reverse = [
        (mn, op) for mn, op in opcodes.items()
        if names.get(op) not in (mn, "NOP", "RET", "CALL")
        and names.get(op) != mn
    ]
    if missing_reverse:
        r.warn(
            f"OPCODE_NAMES missing reverse entries for: "
            + ", ".join(f"{mn}=0x{op:02X}" for mn, op in missing_reverse[:5])
        )
    else:
        r.ok("OPCODE_NAMES reverse table is consistent with OPCODES")

    # Phase V opcodes must all be present
    phase_v = {"NICHE_DECLARE": 0x73, "SYMBOL_EMERGE": 0x74,
               "HERITAGE_LOAD": 0x75, "NICHE_QUERY": 0x76,
               "COLLECTIVE_SYNC": 0x77}
    for mn, expected_op in phase_v.items():
        actual = opcodes.get(mn)
        if actual is None:
            r.error(f"Phase V opcode {mn!r} missing from OPCODES")
        elif actual != expected_op:
            r.error(f"Phase V opcode {mn!r}: expected 0x{expected_op:02X}, got 0x{actual:02X}")
        else:
            r.ok(f"Phase V  0x{actual:02X}  {mn}")

    # Encode/decode roundtrip for every opcode
    encode_word = getattr(isa, "encode_word", None)
    decode_word = getattr(isa, "decode_word", None)
    if encode_word and decode_word:
        failures = []
        for mn, op in opcodes.items():
            word = encode_word(op, agent_id=0x0A, som_x=0x1B, som_y=0x2C,
                               reg=0x0102, imm=0xBEEF)
            dec = decode_word(word)
            if dec["opcode"] != op:
                failures.append(f"{mn}(0x{op:02X}): decoded opcode 0x{dec['opcode']:02X}")
        if failures:
            r.error("encode/decode roundtrip failures:\n" +
                    "\n".join(f"    {f}" for f in failures))
        else:
            r.ok(f"encode_word/decode_word roundtrip: all {len(opcodes)} opcodes pass")
    else:
        r.warn("encode_word or decode_word not found in soma/isa.py — skipping roundtrip")

    return opcodes


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Bridge congruency
# ─────────────────────────────────────────────────────────────────────────────

def check_bridge(r: Report, canonical: dict[str, int]) -> None:
    print("\n── Layer 2: Runtime Bridge (runtime/bridge.py) ───────────────────────────")
    bridge_path = ROOT / "runtime" / "bridge.py"
    if not bridge_path.exists():
        r.error("runtime/bridge.py not found — run `python verify_congruency.py` after creating it")
        return

    source = bridge_path.read_text()

    # Static AST check — no module execution needed
    bridge_opcodes = _extract_opcode_dict_from_ast(source)
    if bridge_opcodes is None:
        # Bridge may import from isa rather than define its own — that's fine
        if "from soma.isa import" in source or "import soma.isa" in source:
            r.ok("Bridge imports OPCODES from soma.isa (re-export pattern) — OK")
        else:
            r.warn("Cannot statically verify bridge OPCODES — no dict literal and no import found")
        return

    # Bridge has its own copy: verify it matches canonical exactly
    extra   = {mn: op for mn, op in bridge_opcodes.items() if mn not in canonical}
    missing = {mn: op for mn, op in canonical.items()    if mn not in bridge_opcodes}
    clashes = {
        mn: (bridge_opcodes[mn], canonical[mn])
        for mn in bridge_opcodes
        if mn in canonical and bridge_opcodes[mn] != canonical[mn]
    }

    if extra:
        r.error(f"Bridge defines phantom opcodes not in ISA: {list(extra)[:5]}")
    else:
        r.ok("No phantom opcodes in bridge")

    if missing:
        r.error(f"Bridge missing opcodes from ISA: {list(missing)[:5]}")
    else:
        r.ok("Bridge covers all ISA opcodes")

    if clashes:
        for mn, (got, want) in clashes.items():
            r.error(f"Bridge opcode mismatch: {mn} = 0x{got:02X} (bridge) vs 0x{want:02X} (ISA)")
    else:
        r.ok("All shared opcodes have identical values in bridge and ISA")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Runtime handler coverage
# ─────────────────────────────────────────────────────────────────────────────

PHASE_V_OPCODES = {0x73, 0x74, 0x75, 0x76, 0x77}
PHASE_V_NAMES   = {
    0x73: "NICHE_DECLARE",
    0x74: "SYMBOL_EMERGE",
    0x75: "HERITAGE_LOAD",
    0x76: "NICHE_QUERY",
    0x77: "COLLECTIVE_SYNC",
}

def _find_handler_coverage(source: str) -> set[int]:
    """
    Heuristic: scan source for any mention of '0x73', '0x74', … '0x77'
    or the mnemonic strings.  Returns the set of opcodes that appear to
    be handled.
    """
    covered: set[int] = set()
    for op in PHASE_V_OPCODES:
        hex_str  = f"0x{op:02X}"
        hex_str2 = f"0x{op:02x}"
        mnemonic = PHASE_V_NAMES[op]
        if any(s in source for s in (hex_str, hex_str2, mnemonic)):
            covered.add(op)
    return covered


def check_runtime(r: Report) -> None:
    print("\n── Layer 3: Runtime handler coverage ────────────────────────────────────")

    targets = {
        "runtime/collective.py": ROOT / "runtime" / "collective.py",
        "runtime/interpreter.py": ROOT / "runtime" / "interpreter.py",
    }

    union_covered: set[int] = set()

    for label, path in targets.items():
        if not path.exists():
            r.warn(f"{label} not found — skipping handler scan")
            continue
        source = path.read_text()
        covered = _find_handler_coverage(source)
        for op in sorted(covered):
            r.ok(f"{label}: handles 0x{op:02X} {PHASE_V_NAMES[op]}")
        union_covered |= covered

    uncovered = PHASE_V_OPCODES - union_covered
    if uncovered:
        for op in sorted(uncovered):
            r.error(
                f"Phase V opcode 0x{op:02X} ({PHASE_V_NAMES[op]}) has NO handler "
                "in collective.py or interpreter.py"
            )
    else:
        r.ok("All 5 Phase V opcodes (0x73–0x77) have runtime handlers")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: word encoding sanity
# ─────────────────────────────────────────────────────────────────────────────

def check_encoding(r: Report) -> None:
    print("\n── Bonus: 64-bit word encoding sanity ───────────────────────────────────")
    try:
        isa = _load_module("soma/isa.py")
        enc = isa.encode_word
        dec = isa.decode_word
    except Exception:
        r.warn("Cannot load encode/decode from soma/isa.py — skipping")
        return

    test_cases = [
        dict(opcode=0x73, agent_id=0x05, som_x=0x12, som_y=0x34,
             reg=0x0300, imm=0x0001),  # NICHE_DECLARE N0
        dict(opcode=0x77, agent_id=0xFF, som_x=0x00, som_y=0x00,
             reg=0x0000,  imm=0x0100),  # COLLECTIVE_SYNC, 256-pulse window
        dict(opcode=0x75, agent_id=0x01, som_x=0x08, som_y=0x08,
             reg=0xFF01,  imm=0x0008),  # HERITAGE_LOAD PARENT, top-K=8
    ]
    for tc in test_cases:
        word = enc(**tc)
        got  = dec(word)
        fails = [k for k in tc if got[k] != tc[k]]
        name = PHASE_V_NAMES.get(tc["opcode"], f"0x{tc['opcode']:02X}")
        if fails:
            r.error(f"Word roundtrip FAILED for {name}: fields {fails} mismatch")
        else:
            r.ok(f"Word roundtrip OK for {name}  word=0x{word:016X}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 72)
    print("  SOMA Congruency Verifier  —  Triple-layer ISA consistency check")
    print("=" * 72)

    r = Report()
    canonical = check_canonical(r)
    check_bridge(r, canonical)
    check_runtime(r)
    check_encoding(r)

    print()
    print("=" * 72)
    total = len(r.oks) + len(r.errors) + len(r.warnings)
    print(f"  Results: {len(r.oks)} ✅  {len(r.warnings)} ⚠️   {len(r.errors)} ❌  "
          f"({total} checks)")
    if r.passed:
        print("  STATUS: CONGRUENT — all layers consistent")
    else:
        print("  STATUS: INCONGRUENT — fix errors before proceeding")
    print("=" * 72)
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
