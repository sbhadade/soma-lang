#!/bin/bash
set -e

echo "=== SOMA Build ==="
echo ""

# ── Step 1: Bootstrap assembler ───────────────────────────────────────────
echo "[1/5] Assembling SOMA self-assembler (somasc.soma)..."
python bootstrap/bootstrap_assembler.py assembler/somasc.soma bin/somasc.sombin

# ── Step 2: Assemble each example ─────────────────────────────────────────
echo ""
echo "[2/5] Assembling examples..."
python bootstrap/bootstrap_assembler.py examples/hello_agent.soma    bin/hello_agent.sombin
python bootstrap/bootstrap_assembler.py examples/swarm_cluster.soma  bin/swarm_cluster.sombin
python bootstrap/bootstrap_assembler.py examples/online_learner.soma bin/online_learner.sombin

# ── Step 3–5: Run each example ────────────────────────────────────────────
echo ""
echo "[3/5] Running hello_agent..."
echo "─────────────────────────────────────────────────────"
python runtime/soma_runtime.py bin/hello_agent.sombin

echo ""
echo "[4/5] Running swarm_cluster..."
echo "─────────────────────────────────────────────────────"
python runtime/soma_runtime.py bin/swarm_cluster.sombin

echo ""
echo "[5/5] Running online_learner..."
echo "─────────────────────────────────────────────────────"
python runtime/soma_runtime.py bin/online_learner.sombin

echo ""
echo "✅ Build & test successful! All examples assembled and executed."
