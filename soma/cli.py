"""SOMA command-line interface."""
import argparse
import sys
import os


def cmd_assemble(args):
    from soma.assembler import assemble, AssemblerError
    src = open(args.input).read()
    try:
        binary = assemble(src)
    except (AssemblerError, SyntaxError) as e:
        print(f"Assembly error: {e}", file=sys.stderr)
        sys.exit(1)
    out = args.output or args.input.replace(".soma", ".sombin")
    with open(out, "wb") as f:
        f.write(binary)
    print(f"Assembled → {out}  ({len(binary)} bytes)")


def cmd_run(args):
    from soma.vm import SomaVM, VMError
    with open(args.input, "rb") as f:
        binary = f.read()
    try:
        vm = SomaVM(binary, max_steps=args.max_steps, trace=args.trace)
        out = vm.run()
    except VMError as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    for line in out:
        print(line)
    # Print final agent states if verbose
    if args.verbose:
        living = sum(1 for ag in vm.agents.values() if ag.state != "dead")
        print(f"\n[VM] Agents total={len(vm.agents)} alive={living}")
        print(f"[VM] SOM size={vm.som.rows}x{vm.som.cols}")


def cmd_disasm(args):
    from soma.assembler import disassemble
    with open(args.input, "rb") as f:
        binary = f.read()
    print(disassemble(binary))


def cmd_run_soma(args):
    """Assemble then run a .soma file directly."""
    from soma.assembler import assemble, AssemblerError
    from soma.vm import SomaVM, VMError
    import tempfile
    src = open(args.input).read()
    try:
        binary = assemble(src)
    except (AssemblerError, SyntaxError) as e:
        print(f"Assembly error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        vm = SomaVM(binary, max_steps=args.max_steps, trace=args.trace)
        out = vm.run()
    except VMError as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    for line in out:
        print(line)


def main():
    parser = argparse.ArgumentParser(
        prog="soma",
        description="SOMA 🧠 Self-Organizing Multi-Agent Binary Language",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # soma asm input.soma [-o output.sombin]
    p_asm = sub.add_parser("asm", help="Assemble .soma source → .sombin")
    p_asm.add_argument("input", help=".soma source file")
    p_asm.add_argument("-o", "--output", help="output .sombin path")
    p_asm.set_defaults(func=cmd_assemble)

    # soma run input.sombin
    p_run = sub.add_parser("run", help="Run a .sombin binary")
    p_run.add_argument("input", help=".sombin binary file")
    p_run.add_argument("--max-steps", type=int, default=100_000)
    p_run.add_argument("--trace", action="store_true")
    p_run.add_argument("-v", "--verbose", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # soma exec input.soma  (assemble + run in one shot)
    p_exec = sub.add_parser("exec", help="Assemble and run a .soma source file")
    p_exec.add_argument("input", help=".soma source file")
    p_exec.add_argument("--max-steps", type=int, default=100_000)
    p_exec.add_argument("--trace", action="store_true")
    p_exec.set_defaults(func=cmd_run_soma)

    # soma disasm input.sombin
    p_dis = sub.add_parser("disasm", help="Disassemble a .sombin binary")
    p_dis.add_argument("input", help=".sombin binary file")
    p_dis.set_defaults(func=cmd_disasm)

    # soma version
    p_ver = sub.add_parser("version", help="Show version")
    p_ver.set_defaults(func=lambda a: print("SOMA 1.0.0"))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
