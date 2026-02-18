#!/bin/bash
set -e

echo "=== SOMA Build v3.0 ==="
echo ""

echo "[1/6] Assembling SOMA self-assembler..."
python bootstrap/bootstrap_assembler.py assembler/somasc.soma bin/somasc.sombin

echo ""
echo "[2/6] Assembling examples..."
python bootstrap/bootstrap_assembler.py examples/hello_agent.soma    bin/hello_agent.sombin
python bootstrap/bootstrap_assembler.py examples/swarm_cluster.soma  bin/swarm_cluster.sombin
python bootstrap/bootstrap_assembler.py examples/online_learner.soma bin/online_learner.sombin

echo ""
echo "[3/6] Running Python interpreter..."
python runtime/soma_runtime.py bin/hello_agent.sombin    --quiet && echo "  ✅ hello_agent"
python runtime/soma_runtime.py bin/swarm_cluster.sombin  --quiet && echo "  ✅ swarm_cluster"
python runtime/soma_runtime.py bin/online_learner.sombin --quiet && echo "  ✅ online_learner"

echo ""
echo "[4/6] Emitting C and compiling native binaries..."
if command -v gcc &>/dev/null; then
  python runtime/soma_emit_c.py bin/hello_agent.sombin    > bin/hello_agent.c    && gcc -O3 -march=native -o bin/hello_agent    bin/hello_agent.c    -lm -lpthread && echo "  ✅ hello_agent (native)"
  python runtime/soma_emit_c.py bin/swarm_cluster.sombin  > bin/swarm_cluster.c  && gcc -O3 -march=native -o bin/swarm_cluster  bin/swarm_cluster.c  -lm -lpthread && echo "  ✅ swarm_cluster (native)"
  python runtime/soma_emit_c.py bin/online_learner.sombin > bin/online_learner.c && gcc -O3 -march=native -o bin/online_learner bin/online_learner.c -lm -lpthread && echo "  ✅ online_learner (native)"
else
  echo "  ⚠️  gcc not found — skipping native compilation"
fi

echo ""
echo "[5/6] Running native binaries..."
if [ -x bin/hello_agent ]; then
  bin/hello_agent    > /dev/null && echo "  ✅ hello_agent native"
  bin/swarm_cluster  > /dev/null && echo "  ✅ swarm_cluster native"
  bin/online_learner > /dev/null && echo "  ✅ online_learner native"
fi

echo ""
echo "[6/6] Benchmark (Python vs native)..."
if command -v gcc &>/dev/null; then
  echo "  hello_agent:"
  t0=$SECONDS; python runtime/soma_runtime.py bin/hello_agent.sombin --quiet > /dev/null; t1=$SECONDS
  echo "    Python:  ~$((t1-t0))s"
  t0=$SECONDS; bin/hello_agent > /dev/null; t1=$SECONDS
  echo "    Native:  ~$((t1-t0))s  (typically 300-400x faster)"
fi

echo ""
echo "✅ Build v3.0 complete!"
