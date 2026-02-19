#!/usr/bin/env python3
"""
SOMA CLI v3.0 — Unified toolchain for Self-Organizing Multi-Agent Binary Language
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        prog="soma",
        description="SOMA v3.0 — Self-Organizing Multi-Agent Binary Language\n"
                    "SOM topology + native multi-agent execution (340× faster in C)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="SOMA 3.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # soma run
    run_p = subparsers.add_parser("run", help="Run a compiled .sombin (Python interpreter)")
    run_p.add_argument("binary", type=Path, help="Path to .sombin file")
    run_p.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    run_p.add_argument("-v", "--verbose", action="store_true", help="Verbose VM debug")

    # soma asm
    asm_p = subparsers.add_parser("asm", help="Assemble .soma → .sombin")
    asm_p.add_argument("source", type=Path, help=".soma source file")
    asm_p.add_argument("-o", "--output", type=Path, help="Output .sombin (default: same name)")

    # soma transpile  ← FIXED & IMPROVED
    tp_p = subparsers.add_parser("transpile", help="Transpile .sombin → high-performance C source")
    tp_p.add_argument("binary", type=Path, help=".sombin file (or .soma — will auto-assemble)")
    tp_p.add_argument("-o", "--output", type=Path, help="Output .c file (default: <name>.c)")

    # soma build
    build_p = subparsers.add_parser("build", help="Full pipeline: asm + transpile + gcc (uses build.sh)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        cmd = [sys.executable, "-m", "runtime.soma_runtime", str(args.binary)]
        if args.quiet:
            cmd.append("--quiet")
        if args.verbose:
            cmd.append("-v")
        subprocess.run(cmd, check=True)

    elif args.command == "asm":
        output = args.output or args.source.with_suffix(".sombin")
        subprocess.run([
            sys.executable,
            "bootstrap/bootstrap_assembler.py",
            str(args.source),
            str(output)
        ], check=True)
        print(f"✅ Assembled → {output}")

    elif args.command == "transpile":
        input_path = args.binary
        output = args.output or input_path.with_suffix(".c")

        # If user gave .soma, auto-assemble first (super convenient)
        if str(input_path).endswith(".soma"):
            temp_bin = input_path.with_suffix(".sombin")
            print(f"   Auto-assembling {input_path} → {temp_bin}")
            subprocess.run([
                sys.executable,
                "bootstrap/bootstrap_assembler.py",
                str(input_path),
                str(temp_bin)
            ], check=True)
            input_path = temp_bin

        # Now transpile (your real C emitter)
        print(f"🚀 Transpiling {input_path} → {output} (native C)")
        with open(output, "w") as f:
            result = subprocess.run([
                sys.executable,
                "runtime/soma_emit_c.py",
                str(input_path)
            ], stdout=f, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            print(f"✅ Transpiled → {output}")
            print(f"   (compile with: gcc -O3 -march=native -o myprog {output} -lm -lpthread)")
        else:
            print("❌ Transpile failed")
            print(result.stderr)

    elif args.command == "build":
        print("🚀 Running full native build (asm + transpile + gcc)...")
        subprocess.run(["bash", "./build.sh"], check=True)

    else:
        parser.error(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()