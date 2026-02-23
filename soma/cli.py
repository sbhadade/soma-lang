#!/usr/bin/env python3
"""
SOMA CLI v5.0 — Unified toolchain for Self-Organizing Multi-Agent Binary Language
Commands: asm · run · exec · disasm · transpile · build · version
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _assemble_src(src_path: str) -> bytes:
    """Read a .soma file, assemble and return raw binary bytes."""
    from soma.assembler import assemble, AssemblerError
    src = open(src_path).read()
    try:
        return assemble(src)
    except (AssemblerError, SyntaxError) as e:
        print(f"❌ Assembly error: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# sub-command handlers
# ---------------------------------------------------------------------------

def cmd_assemble(args):
    """soma asm program.soma [-o program.sombin]"""
    binary = _assemble_src(args.input)
    out = args.output or args.input.replace(".soma", ".sombin")
    with open(out, "wb") as f:
        f.write(binary)
    print(f"✅ Assembled → {out}  ({len(binary)} bytes)")


def cmd_run(args):
    """soma run program.sombin [--trace] [-v] [--max-steps N]"""
    from soma.vm import SomaVM, VMError
    with open(args.input, "rb") as f:
        binary = f.read()
    try:
        vm = SomaVM(binary, max_steps=args.max_steps, trace=args.trace)
        out = vm.run()
    except VMError as e:
        print(f"❌ Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    for line in out:
        print(line)
    if args.verbose:
        living = sum(1 for ag in vm.agents.values() if ag.state != "dead")
        print(f"\n[VM] Agents total={len(vm.agents)} alive={living}")
        print(f"[VM] SOM size={vm.som.rows}x{vm.som.cols}")


def cmd_exec(args):
    """soma exec program.soma  — assemble + run in one shot"""
    from soma.vm import SomaVM, VMError
    binary = _assemble_src(args.input)
    try:
        vm = SomaVM(binary, max_steps=args.max_steps, trace=args.trace)
        out = vm.run()
    except VMError as e:
        print(f"❌ Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    for line in out:
        print(line)
    if args.verbose:
        living = sum(1 for ag in vm.agents.values() if ag.state != "dead")
        print(f"\n[VM] Agents total={len(vm.agents)} alive={living}")
        print(f"[VM] SOM size={vm.som.rows}x{vm.som.cols}")


def cmd_disasm(args):
    """soma disasm program.sombin"""
    from soma.assembler import disassemble
    with open(args.input, "rb") as f:
        binary = f.read()
    print(disassemble(binary))


def cmd_transpile(args):
    """soma transpile program.soma -o program.c  (auto-assembles if .soma given)"""
    input_path = Path(args.input)
    output = Path(args.output) if args.output else input_path.with_suffix(".c")

    # Auto-assemble if the user handed us a .soma source
    if input_path.suffix == ".soma":
        temp_bin = input_path.with_suffix(".sombin")
        print(f"   Auto-assembling {input_path} → {temp_bin}")
        binary = _assemble_src(str(input_path))
        with open(temp_bin, "wb") as f:
            f.write(binary)
        print(f"✅ Assembled → {temp_bin}  ({len(binary)} bytes)")
        input_path = temp_bin

    # Call the C emitter
    print(f"🚀 Transpiling {input_path} → {output}  (native C)")
    emit_script = Path(__file__).parent.parent / "runtime" / "soma_emit_c.py"
    with open(output, "w") as f:
        result = subprocess.run(
            [sys.executable, str(emit_script), str(input_path)],
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode == 0:
        print(f"✅ Transpiled → {output}")
        print(f"   compile: gcc -O3 -march=native -o myprog {output} -lm -lpthread")
    else:
        print("❌ Transpile failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        # Remove empty/broken output file
        if output.exists():
            output.unlink()
        sys.exit(1)


def cmd_build(args):
    """soma build  — full pipeline via build.sh (asm + transpile + gcc)"""
    build_sh = Path(__file__).parent.parent / "build.sh"
    if not build_sh.exists():
        print(f"❌ build.sh not found at {build_sh}", file=sys.stderr)
        sys.exit(1)
    print("🚀 Running full native build (asm + transpile + gcc)…")
    subprocess.run(["bash", str(build_sh)], check=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="soma",
        description=(
            "SOMA 🧠 v5.0 — Self-Organizing Multi-Agent Binary Language\n"
            "SOM topology + native multi-agent execution\n\n"
            "  asm        Assemble .soma → .sombin\n"
            "  run        Run a .sombin binary (Python VM)\n"
            "  exec       Assemble + run .soma in one shot\n"
            "  disasm     Disassemble .sombin → human-readable\n"
            "  transpile  Compile .soma/.sombin → high-performance C\n"
            "  build      Full pipeline: asm + transpile + gcc\n"
            "  version    Show version info\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version="SOMA 5.0.0"
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── asm ────────────────────────────────────────────────────────────────
    p_asm = sub.add_parser("asm", help="Assemble .soma source → .sombin")
    p_asm.add_argument("input", help=".soma source file")
    p_asm.add_argument("-o", "--output", help="Output .sombin path")
    p_asm.set_defaults(func=cmd_assemble)

    # ── run ────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run a .sombin binary")
    p_run.add_argument("input", help=".sombin binary file")
    p_run.add_argument("--max-steps", type=int, default=100_000, metavar="N")
    p_run.add_argument("--trace", action="store_true", help="Trace execution")
    p_run.add_argument("-v", "--verbose", action="store_true", help="Show agent/SOM summary")
    p_run.set_defaults(func=cmd_run)

    # ── exec ───────────────────────────────────────────────────────────────
    p_exec = sub.add_parser("exec", help="Assemble + run .soma in one shot")
    p_exec.add_argument("input", help=".soma source file")
    p_exec.add_argument("--max-steps", type=int, default=100_000, metavar="N")
    p_exec.add_argument("--trace", action="store_true", help="Trace execution")
    p_exec.add_argument("-v", "--verbose", action="store_true", help="Show agent/SOM summary")
    p_exec.set_defaults(func=cmd_exec)

    # ── disasm ─────────────────────────────────────────────────────────────
    p_dis = sub.add_parser("disasm", help="Disassemble a .sombin binary")
    p_dis.add_argument("input", help=".sombin binary file")
    p_dis.set_defaults(func=cmd_disasm)

    # ── transpile ──────────────────────────────────────────────────────────
    p_trans = sub.add_parser(
        "transpile", help="Transpile .soma/.sombin → high-performance C"
    )
    p_trans.add_argument("input", help=".soma source or .sombin binary")
    p_trans.add_argument("-o", "--output", help="Output .c path (default: <name>.c)")
    p_trans.set_defaults(func=cmd_transpile)

    # ── build ──────────────────────────────────────────────────────────────
    p_build = sub.add_parser(
        "build", help="Full pipeline: asm + transpile + gcc  (uses build.sh)"
    )
    p_build.set_defaults(func=cmd_build)

    # ── version ────────────────────────────────────────────────────────────
    p_ver = sub.add_parser("version", help="Show version info")
    p_ver.set_defaults(func=lambda _: print("SOMA 5.0.0"))

    # ── dispatch ───────────────────────────────────────────────────────────
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()