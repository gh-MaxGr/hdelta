"""Draw the two figures in WRITEUP.md. Numbers are copied from RESULTS.md (F.2 and F.5).

    python figures/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

OUT = Path(__file__).parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LORA, FAST, REPLAY = "#2a78d6", "#eb6834", "#1baf7a"
BASE_PPL = 9.34

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def title(ax, head, sub):
    ax.text(0, 1.13, head, transform=ax.transAxes, fontsize=12.5, weight="bold", color=INK)
    ax.text(0, 1.05, sub, transform=ax.transAxes, fontsize=9.5, color=INK2)


def snippets_frontier():
    # F.2, seed 0: (perplexity change %, accuracy, label)
    fast = [(0.9, 0.738, "gentle recipe"), (4.7, 0.825, "decay .001, budget .06"),
            (16.0, 0.859, "no decay, budget .06"), (130.0, 0.884, "no decay, budget .12")]
    replay = (11.5, 0.864, "no decay, budget .06, + replay")
    lora = (1.6, 0.883, "LoRA r=16")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_xscale("log")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    xs, ys = [p[0] for p in fast], [p[1] for p in fast]
    ax.plot(xs, ys, color=FAST, linewidth=2, zorder=2)
    ax.scatter(xs, ys, s=64, color=FAST, edgecolor=SURFACE, linewidth=2, zorder=3, label="fast weights")
    ax.scatter([replay[0]], [replay[1]], s=64, facecolor=SURFACE, edgecolor=FAST, linewidth=2, zorder=3,
               label="fast weights + replay")
    ax.scatter([lora[0]], [lora[1]], s=110, color=LORA, edgecolor=SURFACE, linewidth=2, zorder=4, label="LoRA")

    offsets = {"gentle recipe": (8, -4), "decay .001, budget .06": (8, -12),
               "no decay, budget .06": (8, -14), "no decay, budget .12": (-4, 10)}
    for x, y, lab in fast:
        dx, dy = offsets[lab]
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                    color=INK2, ha="right" if dx < 0 else "left")
    ax.annotate("+ replay", replay[:2], xytext=(0, 10), textcoords="offset points", fontsize=8.5,
                color=INK2, ha="center")
    ax.annotate("LoRA r=16: 0.883 at +1.6 %", lora[:2], xytext=(10, 6), textcoords="offset points",
                fontsize=9.5, color=INK, weight="bold")

    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"+{v:g} %"))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(0.6, 250)
    ax.set_ylim(0.70, 0.91)
    ax.set_xlabel("Base-model perplexity increase (log scale)")
    ax.set_ylabel("Held-out accuracy")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    title(ax, "Every fast-weight setting sits below LoRA's trade-off",
          "\"Does this Python snippet raise?\"  Qwen2.5-3B, online, one pass, seed 0 (3-seed means in the text)")
    fig.tight_layout()
    fig.savefig(OUT / "snippets_frontier.png", dpi=200)
    plt.close(fig)


def sequential_retention():
    # F.5, mean of seeds 0 and 1: skill-1 accuracy after learning it, then after learning skill 2
    rows = [("Fast weights (no decay, budget .12)", FAST, (0.845, 0.881), (0.809, 0.827), (38.4, 11.84)),
            ("LoRA r=16", LORA, (0.865, 0.860), (0.752, 0.745), (9.63, 9.56)),
            ("LoRA r=16 + 1 replay step", REPLAY, (0.873, 0.895), (0.860, 0.895), (9.76, 9.75))]

    fig, ax = plt.subplots(figsize=(8, 3.9))
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, (name, color, learned, kept, ppl) in enumerate(rows):
        y = len(rows) - 1 - i
        a, b = sum(learned) / 2, sum(kept) / 2
        ax.plot([a, b], [y, y], color=color, linewidth=2, zorder=2)
        ax.scatter([a], [y], s=70, facecolor=SURFACE, edgecolor=color, linewidth=2, zorder=3)
        ax.scatter([b], [y], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        pct = sorted((p / BASE_PPL - 1) * 100 for p in ppl)
        fmt = [f"+{v:.0f} %" if v >= 10 else f"+{v:.1f} %" for v in pct]
        ax.text(0.935, y, f"perplexity {fmt[0]} / {fmt[1]}", va="center", fontsize=9, color=INK2)
        ax.text(min(a, b) - 0.004, y + 0.22, f"{name}:  kept {b:.3f}", ha="left", fontsize=9.5,
                color=INK, weight="bold")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlim(0.70, 1.02)
    ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90])
    ax.set_xlabel("Skill 1 accuracy (snippets)    ○ after learning it    ● after then learning skill 2 (mutants)")
    title(ax, "With one replay step, LoRA keeps the first skill best",
          "Snippets, then mutants. Mean of 2 seeds; perplexity change per seed (base 9.34)")
    fig.tight_layout()
    fig.savefig(OUT / "sequential_retention.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    snippets_frontier()
    sequential_retention()
