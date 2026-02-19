#!/bin/bash
# SOMA repo cleanup — run from /mnt/c/Users/swapn/downloads/soma_1
set -e
cd /mnt/c/Users/swapn/downloads/soma_1

echo "=== SOMA Repo Cleanup ==="
echo ""

# ── 1. Remove compiled artifacts from git tracking ────────────────
echo "[1/5] Removing compiled artifacts from git..."
git rm --cached bin/hello_agent      2>/dev/null || true
git rm --cached bin/hello_agent.c    2>/dev/null || true
git rm --cached bin/hello_agent.sombin 2>/dev/null || true
git rm --cached bin/swarm_cluster    2>/dev/null || true
git rm --cached bin/swarm_cluster.c  2>/dev/null || true
git rm --cached bin/swarm_cluster.sombin 2>/dev/null || true
git rm --cached bin/online_learner   2>/dev/null || true
git rm --cached bin/online_learner.c 2>/dev/null || true
git rm --cached bin/online_learner.sombin 2>/dev/null || true
git rm --cached bin/somasc.sombin    2>/dev/null || true
git rm --cached bin/examples.sombin  2>/dev/null || true
git rm --cached bin/SOMBIN.spec      2>/dev/null || true
echo "  ✅ Compiled artifacts untracked"

# ── 2. Move SOMBIN.spec to spec/ where it belongs ─────────────────
echo ""
echo "[2/5] Fixing file locations..."
if [ -f bin/SOMBIN.spec ]; then
  cp bin/SOMBIN.spec spec/SOMBIN.spec
  echo "  ✅ bin/SOMBIN.spec → spec/SOMBIN.spec"
fi

# Remove examples/examples.soma (it's the old unsplit version, superseded)
git rm --cached examples/examples.soma 2>/dev/null || true
rm -f examples/examples.soma
echo "  ✅ Removed examples/examples.soma (superseded by split files)"

# ── 3. Write clean .gitignore ─────────────────────────────────────
echo ""
echo "[3/5] Writing .gitignore..."
cat > .gitignore << 'GITIGNORE'
# Compiled binaries
bin/*.sombin
bin/*.c
bin/hello_agent
bin/swarm_cluster
bin/online_learner
bin/somasc
bin/somasc.sombin

# Keep spec and gitkeep
!bin/.gitkeep
!bin/SOMBIN.spec

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
*.egg-info/
dist/
build/

# OS
.DS_Store
Thumbs.db
*.swp
*.swo

# Editor
.vscode/
.idea/
*.iml
GITIGNORE

echo "  ✅ .gitignore written"

# ── 4. Create bin/.gitkeep so empty bin/ is tracked ───────────────
touch bin/.gitkeep
echo "  ✅ bin/.gitkeep created"

# ── 5. Stage and commit ───────────────────────────────────────────
echo ""
echo "[4/5] Staging changes..."
git add .gitignore
git add spec/SOMBIN.spec 2>/dev/null || true
git add bin/.gitkeep
git add -A
git status --short

echo ""
echo "[5/5] Committing..."
git commit -m "🧹 Clean repo — .gitignore, remove compiled artifacts, fix file locations"

echo ""
echo "=== Final structure ==="
find . -not -path "./.git/*" -not -path "./.git" \
       -not -name "*.sombin" -not -name "*.c" \
       -not -path "./bin/hello_agent" \
       -not -path "./bin/swarm_cluster" \
       -not -path "./bin/online_learner" \
       | sort | sed 's|^./||'

echo ""
echo "Push with: git push origin main"