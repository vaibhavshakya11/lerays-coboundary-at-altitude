"""
plot_style.py: shared figure aesthetics.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt


# Color palette (colorblind-friendly, distinct print and on-screen)
COLORS = {
    'sheaf':    '#0F766E',  # teal
    'tmr':      '#C2410C',  # coral / burnt orange
    'secded':   '#1E3A8A',  # dark blue
    'swift':    '#6B21A8',  # purple
    'abft':     '#92400E',  # brown
    'baseline': '#737373',  # gray
    'highlight':'#DC2626',  # red for emphasis
    'success':  '#16A34A',  # green for positive
}


def apply_style():
    """Apply consistent matplotlib style for all paper figures."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 13,

        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#404040',
        'axes.labelcolor': '#202020',

        'xtick.color': '#404040',
        'ytick.color': '#404040',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,

        'grid.color': '#D4D4D4',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.7,

        'legend.frameon': True,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '#D4D4D4',
        'legend.fancybox': False,

        'figure.dpi': 110,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,

        'lines.linewidth': 1.6,
        'lines.markersize': 5.5,

        'patch.linewidth': 0.6,
    })


def add_grid(ax, axis='y', alpha=0.5):
    """Add a subtle grid."""
    ax.grid(True, axis=axis, alpha=alpha, linestyle='-', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    apply_style()
    import numpy as np
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 10, 100)
    for name, color in [('sheaf', COLORS['sheaf']),
                        ('tmr', COLORS['tmr']),
                        ('secded', COLORS['secded'])]:
        ax.plot(x, np.sin(x + hash(name) % 5), label=name, color=color)
    ax.legend()
    add_grid(ax)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.savefig('/tmp/style_test.png')
    print("style_test.png written")
