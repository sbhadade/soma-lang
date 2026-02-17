#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SOMA — GitHub Push Script
# Run this once to initialize and push the repo to GitHub
# ═══════════════════════════════════════════════════════════════

echo "🧠 Initializing SOMA language repository..."

# Initialize git
git init
git branch -M main

# Stage everything
git add .

# First commit
git commit -m "🧠 SOMA v1.0.0 — Initial release

Self-Organizing Multi-Agent Binary Language

- Full language grammar spec (EBNF)
- 64-bit binary instruction format (.sombin)
- Complete ISA with 54 opcodes across 6 groups
- Self-hosting assembler written in SOMA itself
- Standard library (SOM, agent pool, vector math, messaging)
- 3 example programs (hello agent, swarm cluster, online learner)
- Design rationale & philosophy docs

The language that thinks in maps. 🗺️"

# Add GitHub remote
git remote add origin https://github.com/sbhadade/soma-lang.git

# Push
echo ""
echo "📡 Pushing to GitHub..."
echo ""
echo "Run this command next:"
echo ""
echo "  git push -u origin main"
echo ""
echo "Or if repo doesn't exist yet, create it first at:"
echo "  https://github.com/new"
echo "  → Name: soma-lang"
echo "  → Description: Self-Organizing Multi-Agent Binary Language"
echo "  → Public ✓"
echo "  → DON'T initialize with README (we have our own)"
echo ""
echo "Then run: git push -u origin main"
echo ""
echo "✅ Done! Your repo will be live at:"
echo "   https://github.com/sbhadade/soma-lang"
