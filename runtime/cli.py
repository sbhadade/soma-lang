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
                    "SOM topology + native multi-agent execution on any substrate",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="SOMA 3.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # soma run
    run_p = subparsers.add_parser("run", help="Run a compiled .sombin (Python or native)")
    run_p.add_argument("binary", type=Path, help="Path to .sombin file")
    run_p.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    run_p.add_argument("-v", "--verbose", action="store_true", help="Verbose VM debug")

    # soma asm
    asm_p = subparsers.add_parser("asm", help="Assemble .soma → .sombin")
    asm_p.add_argument("source", type=Path, help=".soma source file")
    asm_p.add_argument("-o", "--output", type=Path, help="Output .sombin (default: same name)")

    # soma transpile
    tp_p = subparsers.add_parser("transpile", help="Transpile .sombin → C source")
    tp_p.add_argument("binary", type=Path, help=".sombin file")
    tp_p.add_argument("-o", "--output", type=Path, help="Output .c file (default: stdout)")

    # soma build (full pipeline — just calls your existing build.sh for now)
    build_p = subparsers.add_parser("build", help="Full build: asm + transpile + gcc")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        cmd = [sys.executable, "-m", "runtime.soma_runtime", str(args.binary)]
        if args.quiet:
            cmd.append("--quiet")
        if args.verbose:
            cmd.append("-v")  # if your runtime supports it
        subprocess.run(cmd)

    elif args.command == "asm":
        output = args.output or args.source.with_suffix(".sombin")
        cmd = [sys.executable, "bootstrap/bootstrap_assembler.py", str(args.source), str(output)]
        subprocess.run(cmd)
        print(f"✅ Assembled → {output}")

    elif args.command == "transpile":
        output = args.output or Path(args.binary).with_suffix(".c")
        cmd = [sys.executable, "-m", "runtime.soma_emit_c", str(args.binary)]
        with open(output, "w") as f:
            subprocess.run(cmd, stdout=f)
        print(f"✅ Transpiled → {output}")

    elif args.command == "build":
        print("🚀 Running full native build...")
        subprocess.run(["bash", "./build.sh"])

    else:
        parser.error(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()