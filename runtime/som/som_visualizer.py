"""
runtime/som/som_visualizer.py — Phase 2: Live SOM topology viewer
==================================================================

ASCII terminal visualizer for the live SOM map.
Shows agent positions, activation heatmap, and hit counts.

Usage
-----
    vis = SomVisualizer(scheduler)
    vis.render()          # print once
    vis.start(fps=2)      # live refresh in terminal
    vis.stop()
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from runtime.som.som_scheduler import SomScheduler

# Heat map characters — 9 levels from cold to hot
_HEAT = " .,:;+*#@"

# ANSI colours (disabled on Windows / non-TTY)
_USE_COLOUR = sys.stdout.isatty() and os.name != "nt"
_RESET  = "\033[0m"   if _USE_COLOUR else ""
_CYAN   = "\033[96m"  if _USE_COLOUR else ""
_YELLOW = "\033[93m"  if _USE_COLOUR else ""
_RED    = "\033[91m"  if _USE_COLOUR else ""
_BOLD   = "\033[1m"   if _USE_COLOUR else ""


class SomVisualizer:
    """
    Renders the SOM topology to the terminal.

    Each cell shows:
      - Activation level (heat character)
      - Agent marker if an agent is sitting on the node
    """

    def __init__(self, scheduler: "SomScheduler"):
        self.scheduler = scheduler
        self._thread:  Optional[threading.Thread] = None
        self._stop_ev = threading.Event()

    # ── Single frame ────────────────────────────────────────────────────────

    def render(self) -> str:
        """Return a multi-line string of the current SOM state."""
        snap  = self.scheduler.snapshot()
        som   = snap["som"]
        rows  = som["rows"]
        cols  = som["cols"]
        epoch = som["epoch"]
        lr    = som["lr"]
        sigma = som["sigma"]

        # Build activation grid
        grid = [[0.0] * cols for _ in range(rows)]
        hits = [[0]   * cols for _ in range(rows)]
        for nd in som["nodes"]:
            r, c = nd["row"], nd["col"]
            grid[r][c] = nd["activation"]
            hits[r][c] = nd["hit_count"]

        # Agent positions  →  set of (r, c)
        agent_cells: dict[tuple, list[int]] = {}
        for ag in snap["agents"]:
            key = (ag["som_r"], ag["som_c"])
            agent_cells.setdefault(key, []).append(ag["id"])

        # Max hit for normalisation
        max_hit = max((hits[r][c] for r in range(rows) for c in range(cols)), default=1)
        max_hit = max(max_hit, 1)

        lines = []
        lines.append(
            f"{_BOLD}SOMA SOM  {rows}×{cols}  "
            f"epoch={epoch}  lr={lr:.4f}  σ={sigma:.4f}{_RESET}"
        )
        lines.append("┌" + "─" * (cols * 2 - 1) + "┐")

        for r in range(rows):
            row_chars = []
            for c in range(cols):
                act   = grid[r][c]
                idx   = min(int(act * 8), 8)
                ch    = _HEAT[idx]

                if (r, c) in agent_cells:
                    aids = agent_cells[(r, c)]
                    # Show agent count or ID
                    label = str(aids[0]) if len(aids) == 1 else str(len(aids))
                    label = label[-1]    # single char
                    ch    = f"{_YELLOW}{label}{_RESET}" if _USE_COLOUR else label
                elif act > 0.7:
                    ch = f"{_RED}{ch}{_RESET}" if _USE_COLOUR else ch
                elif act > 0.3:
                    ch = f"{_CYAN}{ch}{_RESET}" if _USE_COLOUR else ch

                row_chars.append(ch)

            lines.append("│" + " ".join(row_chars) + "│")

        lines.append("└" + "─" * (cols * 2 - 1) + "┘")

        # Legend
        alive = len(snap["agents"])
        lines.append(
            f"Agents alive: {_BOLD}{alive}{_RESET}  "
            f"Heat: {_HEAT}  (cold→hot)  "
            f"Agents: {_YELLOW}0-9{_RESET}"
        )
        return "\n".join(lines)

    def print(self):
        """Print current frame to stdout."""
        print(self.render())

    # ── Live refresh ────────────────────────────────────────────────────────

    def start(self, fps: float = 2.0):
        """
        Start a background thread that refreshes the terminal at `fps`.
        Clears screen between frames.
        """
        self._stop_ev.clear()
        interval = 1.0 / max(fps, 0.1)

        def _loop():
            while not self._stop_ev.wait(timeout=interval):
                if _USE_COLOUR:
                    print("\033[2J\033[H", end="")   # clear screen + home cursor
                else:
                    print("\n" + "=" * 60)
                self.print()

        self._thread = threading.Thread(
            target=_loop, daemon=True, name="soma-som-vis"
        )
        self._thread.start()

    def stop(self):
        """Stop the live refresh thread."""
        self._stop_ev.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ── Hit map (ASCII only, no colour) ─────────────────────────────────────

    def hit_map(self) -> str:
        """Return hit-count ASCII map (like soma_som_print_activations in C)."""
        snap = self.scheduler.snapshot()
        som  = snap["som"]
        rows, cols = som["rows"], som["cols"]

        hits = [[0] * cols for _ in range(rows)]
        for nd in som["nodes"]:
            hits[nd["row"]][nd["col"]] = nd["hit_count"]

        max_hit = max(
            (hits[r][c] for r in range(rows) for c in range(cols)),
            default=1
        )
        max_hit = max(max_hit, 1)

        lines = [f"SOM Hit Map ({rows}×{cols})  [max hits = {max_hit}]"]
        for r in range(rows):
            row = ""
            for c in range(cols):
                ratio = hits[r][c] / max_hit
                idx   = min(int(ratio * 8), 8)
                row  += _HEAT[idx]
            lines.append(row)
        return "\n".join(lines)
