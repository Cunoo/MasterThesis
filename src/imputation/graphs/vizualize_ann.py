import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_gru_architecture():
    fig, ax = plt.subplots(figsize=(6, 14))
    ax.axis("off")

    blocks = [
        ("Input\n(B × seq_len × F)", 0.92),
        ("Concatenate\nvalues + mask\n(B × seq_len × 2F)", 0.78),
        ("GRU\n(2 layers, hidden=256)", 0.64),
        ("Temporal\nAttention", 0.50),
        ("Context vektor\n(B × H)", 0.37),
        ("LayerNorm + Dropout", 0.24),
        ("Linear\n(256 → 79)", 0.12),
        ("Output\n(B × 79)", 0.00),
    ]

    for label, y in blocks:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.2, y - 0.04), 0.6, 0.1,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="steelblue",
            facecolor="lightsteelblue",
            transform=ax.transAxes,
            clip_on=False
        ))
        ax.text(0.5, y + 0.01, label,
                ha="center", va="center",
                fontsize=9, transform=ax.transAxes)

    # Arrows top to bottom
    for i in range(len(blocks) - 1):
        ax.annotate("",
                    xy=(0.5, blocks[i+1][1] + 0.06),
                    xytext=(0.5, blocks[i][1] - 0.04),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax.set_title("GRU Imputation Model Architecture", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig("src/imputation/graphs/gru_architecture.png", dpi=150, bbox_inches="tight")
    print("Saved: src/imputation/graphs/gru_architecture.png")


if __name__ == "__main__":
    draw_gru_architecture()