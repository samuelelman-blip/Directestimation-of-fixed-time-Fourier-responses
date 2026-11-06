"""Figures comparing this package (the Qualtran 0.4 port) with the reference implementation
(written for Qualtran 0.7; not included, its counts are recorded in data/).

    python tools/plot_versions.py        # writes figures/*.png and figures/*.csv

Reads data/stream_counts.json (this package, written by tools/count_cells.py) and
data/reference_cells.json and data/reference_counts.json (the reference implementation, written by
tools/reference_counts.py), and streams the 43 small reference cases again (a few seconds).

    fig1_counts_parity.png      every gate count of the two versions, one against the other
    fig2_iterate_gate_mix.png   one Grover iterate by gate type, for each cell
    fig3_t_by_part.png          the T count of each part, for each cell
    fig4_qubit_peaks.png        peak qubits of each part and cell: the reference, the port, and the
                                port with each decomposition in Qualtran 0.4's own order
    gate_counts.csv, gate_counts_small_cases.csv, qubit_peaks.csv    the numbers plotted
"""
import csv
import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects, transforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'figures')

# colours: categorical slots 1-3 of the reference palette, checked for colour-vision deficiency
# against a white surface; text in ink tones, never in a series colour
PORT = '#2a78d6'          # this package, Qualtran 0.4
REF = '#eb6834'           # the reference implementation, Qualtran 0.7
ORDER04 = '#1baf7a'       # this package, decompositions in Qualtran 0.4's own order
SURFACE = '#ffffff'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
NEUTRAL = '#f0efec'       # diverging midpoint (a difference of 0)
UP = '#e34948'            # diverging warm pole (one more qubit)

CELL_ORDER = ['dense eps=0.01', 'dense eps=1e-06', 'PD-F eps=0.01', 'PD-F eps=1e-06',
              'PD-T eps=0.01', 'PD-T eps=1e-06', 'sparse eps=1e-06', 'PS-T eps=1e-06',
              'PS-F eps=1e-06']
PARTS = [('P', 'P'), ('Pdg', 'P†'), ('X', 'C-X'), ('W', 'W (block)'),
         ('refl', 'S$_0$ (reflection)'), ('pg', 'phase-gradient state'), ('Q', 'Q (iterate)')]
PART_LABEL = dict(PARTS)
CSV_PART = {'P': 'P', 'Pdg': 'P-dagger', 'X': 'C-X', 'W': 'W', 'refl': 'S0', 'pg': 'phase gradient',
            'Q': 'Q'}
GATE_GROUPS = [('AND', ('AND',)), ('AND† (measured)', ('ANDDG',)), ('T, T†', ('T', 'TDG')),
               ('Rz, Ry', ('RZ', 'RY')), ('CNOT', ('CNOT',)), ('H', ('H',)), ('X', ('X',)),
               ('S, S†', ('S', 'SDG')), ('Z', ('Z',)), ('CZ', ('CZ',)),
               ('X measurement', ('MX',)), ('CZ, conditioned', ('CZ_IF',))]


def cell_label(label):
    family, eps = label.split(' eps=')
    exp = {'0.01': '⁻²', '1e-06': '⁻⁶'}[eps]
    return f'{family}, ε = 10{exp}'


def setup():
    plt.rcParams.update({
        'font.family': ['Inter', 'DejaVu Sans'], 'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': BASELINE,
        'axes.linewidth': 0.8, 'axes.titlesize': 10, 'axes.titleweight': 'semibold',
        'axes.titlecolor': INK, 'axes.titlelocation': 'left', 'axes.titlepad': 8,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2,
        'ytick.labelcolor': INK2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'savefig.dpi': 200, 'legend.frameon': False, 'legend.fontsize': 8.5,
        'mathtext.fontset': 'dejavusans', 'mathtext.default': 'regular',
    })


def clean(ax, grid_axis='x'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, which='major')
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def panel_title(ax, title, subtitle=None):
    """A panel title, and an optional line under it, at fixed offsets above the axes."""
    ax.annotate(title, xy=(0, 1), xycoords='axes fraction', xytext=(0, 17 if subtitle else 7),
                textcoords='offset points', ha='left', va='bottom', fontsize=10,
                fontweight='semibold', color=INK)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords='axes fraction', xytext=(0, 5),
                    textcoords='offset points', ha='left', va='bottom', fontsize=8, color=INK2)


def figure_title(fig, title, subtitle=None, y=0.985):
    """The figure's title at the top left, and a line under it at a fixed distance (in points, so
    the same in every figure whatever its height)."""
    fig.text(0.06, y, title, ha='left', va='top', fontsize=12, fontweight='semibold', color=INK)
    if subtitle:
        below = fig.transFigure + transforms.ScaledTranslation(0, -21 / 72, fig.dpi_scale_trans)
        fig.text(0.06, y, subtitle, transform=below, ha='left', va='top', fontsize=9, color=INK2)


def load():
    port = json.load(open(os.path.join(DATA, 'stream_counts.json')))
    ref = json.load(open(os.path.join(DATA, 'reference_cells.json')))
    assert sorted(port) == sorted(ref) == sorted(CELL_ORDER)
    return port, ref


def small_cases():
    """(case, gate, port count, reference count) for the 43 small reference cases."""
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'tests'))
    from reference_cases import CASES
    from ftfr_q04.stream import BloqStream, paper_counts
    ref = json.load(open(os.path.join(DATA, 'reference_counts.json')))
    rows = []
    for name in sorted(CASES):
        c = paper_counts(BloqStream(CASES[name]()))
        got = {k: v for k, v in c.items() if k not in ('ALLOC', 'FREE')}
        want = ref[name]['counts']
        for g in sorted(set(got) | set(want)):
            rows.append((name, g, got.get(g, 0), want.get(g, 0)))
    return rows


def full_rows(port, ref):
    """(cell, part, gate, port count, reference count) at the nine cells."""
    rows = []
    for cell in CELL_ORDER:
        for key, _ in PARTS:
            a, b = port[cell]['parts'][key]['counts'], ref[cell]['parts'][key]['counts']
            for g in sorted(set(a) | set(b)):
                rows.append((cell, key, g, a.get(g, 0), b.get(g, 0)))
    return rows


def ref_tick(ax, x, y, half):
    """The reference value: a short vertical stroke across the bar end, ringed in the surface
    colour so it stays visible where it meets the bar."""
    ax.plot([x, x], [y - half, y + half], color=REF, linewidth=2.2, solid_capstyle='butt',
            zorder=4, path_effects=[patheffects.Stroke(linewidth=4.2, foreground=SURFACE),
                                   patheffects.Normal()])


def hbars(ax, ys, values, xmin, height=0.56):
    """Horizontal bars on a log axis, starting at the axis minimum; a surface edge keeps
    neighbouring bars apart."""
    for y, v in zip(ys, values):
        if v > 0:
            ax.barh(y, v - xmin, left=xmin, height=height, color=PORT, edgecolor=SURFACE,
                    linewidth=1.0, zorder=3)


def legend_handles(include_order=False):
    h = [Patch(facecolor=PORT, edgecolor=PORT, label='Qualtran 0.4 port (this package)'
               + (', greedy order' if include_order else '')),
         Line2D([], [], color=REF, linewidth=2.2, marker='|', markersize=0,
                label='reference implementation (Qualtran 0.7)')]
    if include_order:
        h.append(Line2D([], [], color=ORDER04, marker='D', linestyle='none', markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.0,
                        label="port, decompositions in Qualtran 0.4's own order"))
    return h


# ------------------------------------------------------------------------------------------------
# figure 1: every count against every count
# ------------------------------------------------------------------------------------------------

def fig_parity(rows, small):
    fig = plt.figure(figsize=(11.0, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.0], hspace=0.10, wspace=0.16,
                          left=0.08, right=0.98, top=0.80, bottom=0.10)
    sets = [('Nine cells at n = 2$^{20}$',
             'P, P†, C-X, W, S$_0$, phase-gradient state and Q, by gate type',
             [r[4] for r in rows], [r[3] for r in rows], (0.6, 4e8)),
            ('43 small test cases', 'the primitives and the selector, by gate type',
             [r[3] for r in small], [r[2] for r in small], (0.6, 600))]
    for col, (title, subtitle, xs, ys, (lo, hi)) in enumerate(sets):
        ax = fig.add_subplot(gs[0, col])
        axr = fig.add_subplot(gs[1, col], sharex=ax)
        ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1.0, zorder=1)
        ax.scatter(xs, ys, s=34, color=PORT, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        clean(ax, 'both')
        plt.setp(ax.get_xticklabels(), visible=False)
        panel_title(ax, f'{title}: {len(xs)} counts', subtitle)
        if col == 0:
            ax.set_ylabel('count in the Qualtran 0.4 port')
        d = [y - x for x, y in zip(xs, ys)]
        axr.axhline(0, color=BASELINE, linewidth=1.0, zorder=1)
        axr.scatter(xs, d, s=34, color=PORT, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        axr.set_ylim(-1.6, 1.6)
        axr.set_yticks([-1, 0, 1])
        clean(axr, 'x')
        axr.set_xlabel('count in the reference implementation (Qualtran 0.7)')
        if col == 0:
            axr.set_ylabel('port − reference')
        n_diff = sum(1 for v in d if v != 0)
        axr.text(0.99, 0.95, f'{n_diff} of {len(d)} differ', transform=axr.transAxes,
                 ha='right', va='top', fontsize=8.5, color=INK2)
    figure_title(fig, 'Every gate count of the Qualtran 0.4 port equals the reference '
                 'implementation’s',
                 'One point per gate type and part. Every point lies on the line y = x, and every '
                 'difference (lower strips) is 0.')
    fig.savefig(os.path.join(OUT, 'fig1_counts_parity.png'))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# figure 2: one iterate by gate type
# ------------------------------------------------------------------------------------------------

def grouped(counts):
    return [sum(counts.get(g, 0) for g in names) for _, names in GATE_GROUPS]


def fig_gate_mix(port, ref):
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.0), sharex=True, sharey=True)
    xmin, xmax = 0.8, 4e8
    ys = list(range(len(GATE_GROUPS)))
    for ax, cell in zip(axes.flat, CELL_ORDER):
        a = grouped(port[cell]['parts']['Q']['counts'])
        b = grouped(ref[cell]['parts']['Q']['counts'])
        hbars(ax, ys, a, xmin)
        for y, v in zip(ys, b):
            if v > 0:
                ref_tick(ax, v, y, 0.4)
        ax.set_xscale('log')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(len(ys) - 0.4, -0.6)
        clean(ax)
        total = sum(port[cell]['parts']['Q']['counts'].values())
        panel_title(ax, cell_label(cell), f'{total:,} gates, {port[cell]["T_Q"]:,} T')
        for y, v, w in zip(ys, a, b):
            if v == 0 and w == 0:
                ax.text(xmin * 1.25, y, 'none', va='center', ha='left', fontsize=7.5,
                        color=MUTED)
    for ax in axes[:, 0]:
        ax.set_yticks(ys)
        ax.set_yticklabels([name for name, _ in GATE_GROUPS])
    for ax in axes[-1, :]:
        ax.set_xlabel('gates in one Grover iterate Q')
    figure_title(fig, 'One Grover iterate by gate type, for each cell',
                 'Log scale, shared by all nine panels. In every row the reference stroke sits at '
                 'the end of the port’s bar.', y=0.99)
    fig.legend(handles=legend_handles(), loc='upper left', bbox_to_anchor=(0.055, 0.945), ncol=2)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.875, bottom=0.05, hspace=0.36, wspace=0.08)
    fig.savefig(os.path.join(OUT, 'fig2_iterate_gate_mix.png'))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# figure 3: T by part
# ------------------------------------------------------------------------------------------------

def t_count(counts, ts):
    return (counts.get('T', 0) + counts.get('TDG', 0) + 4 * counts.get('AND', 0)
            + 4 * counts.get('TOFFOLI', 0) + ts * (counts.get('RZ', 0) + counts.get('RY', 0)))


def fig_t_by_part(port, ref):
    keys = ['P', 'Pdg', 'X', 'W', 'refl', 'Q']
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), sharey=True)
    xmin, xmax = 30, 3e6
    ys = list(range(len(CELL_ORDER)))
    for ax, key in zip(axes.flat, keys):
        a = [port[c]['parts'][key]['T'] for c in CELL_ORDER]
        b = [t_count(ref[c]['parts'][key]['counts'], ref[c]['ts_per_rot']) for c in CELL_ORDER]
        assert b == [ref[c]['parts'][key]['T'] for c in CELL_ORDER]
        hbars(ax, ys, a, xmin)
        for y, v in zip(ys, b):
            ref_tick(ax, v, y, 0.4)
        if key == 'Q':                                 # the headline number: T per iterate
            for y, v in zip(ys, a):
                ax.text(v * 1.35, y, f'{v:,}', va='center', ha='left', fontsize=7.5, color=INK2)
        ax.set_xscale('log')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(len(ys) - 0.4, -0.6)
        clean(ax)
        panel_title(ax, PART_LABEL[key])
        ax.set_xlabel('T count')
    for ax in axes[:, 0]:
        ax.set_yticks(ys)
        ax.set_yticklabels([cell_label(c) for c in CELL_ORDER])
    figure_title(fig, 'T count of each part, for each cell',
                 'T + T† + 4 per AND, and each rotation at the cell’s T per rotation. '
                 'Q is two blocks W and two reflections S$_0$.', y=0.99)
    fig.legend(handles=legend_handles(), loc='upper left', bbox_to_anchor=(0.055, 0.93), ncol=2)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.83, bottom=0.07, hspace=0.45, wspace=0.08)
    fig.savefig(os.path.join(OUT, 'fig3_t_by_part.png'))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# figure 4: qubit peaks in both orders
# ------------------------------------------------------------------------------------------------

def fig_peaks(port, ref):
    keys = ['P', 'X', 'W', 'Q']
    fig = plt.figure(figsize=(11.5, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.8], hspace=0.42, wspace=0.08,
                          left=0.13, right=0.98, top=0.84, bottom=0.04)
    ys = list(range(len(CELL_ORDER)))
    xmin, xmax = 20, 1.2e5
    top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for i, (ax, key) in enumerate(zip(top, keys)):
        greedy = [port[c]['parts'][key]['peak'] for c in CELL_ORDER]
        order04 = [port[c]['parts'][key]['peak_q04_order'] for c in CELL_ORDER]
        reference = [ref[c]['parts'][key]['peak'] for c in CELL_ORDER]
        hbars(ax, ys, greedy, xmin, height=0.5)
        for y, v in zip(ys, reference):
            ref_tick(ax, v, y, 0.36)
        ax.scatter(order04, [y - 0.42 for y in ys], marker='D', s=30, color=ORDER04,
                   edgecolor=SURFACE, linewidth=1.0, zorder=5)
        if key == 'Q':                                 # the qubit count of the run
            for y, v, w in zip(ys, greedy, order04):
                label = f'{v:,}' if v == w else f'{v:,} / {w:,}'
                ax.text(v * 1.3, y, label, va='center', ha='left', fontsize=7.5, color=INK2)
        ax.set_xscale('log')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(len(ys) - 0.4, -0.9)
        clean(ax)
        panel_title(ax, PART_LABEL[key])
        ax.set_xlabel('peak live qubits')
        if i == 0:
            ax.set_yticks(ys)
            ax.set_yticklabels([cell_label(c) for c in CELL_ORDER])
        else:
            ax.set_yticks(ys)
            ax.set_yticklabels([])

    # the differences, as two small matrices with every value written in
    left = fig.add_subplot(gs[1, 0:2])
    right = fig.add_subplot(gs[1, 2:4])
    all_keys = [k for k, _ in PARTS]
    d_ref = [[port[c]['parts'][k]['peak'] - ref[c]['parts'][k]['peak'] for k in all_keys]
             for c in CELL_ORDER]
    d_ord = [[port[c]['parts'][k]['peak_q04_order'] - port[c]['parts'][k]['peak'] for k in keys]
             for c in CELL_ORDER]
    short = {'P': 'P', 'Pdg': 'P†', 'X': 'C-X', 'W': 'W', 'refl': 'S$_0$', 'pg': 'pg',
             'Q': 'Q'}
    matrix(left, d_ref, [short[k] for k in all_keys], [cell_label(c) for c in CELL_ORDER],
           'Port minus reference, both in the greedy order', show_rows=True)
    matrix(right, d_ord, [short[k] for k in keys], [cell_label(c) for c in CELL_ORDER],
           "Port in Qualtran 0.4's order minus port in the greedy order", show_rows=False)
    figure_title(fig, 'Peak live qubits of each part, for each cell',
                 'Greedy order: each qubit allocated at its first use and released after its last '
                 '(the paper’s convention). Qualtran 0.4’s order allocates at the start of '
                 'each decomposition.', y=0.99)
    fig.legend(handles=legend_handles(include_order=True), loc='upper left',
               bbox_to_anchor=(0.055, 0.935), ncol=3)
    fig.savefig(os.path.join(OUT, 'fig4_qubit_peaks.png'))
    plt.close(fig)


def matrix(ax, values, cols, rows, title, show_rows):
    """A grid of small integers: 0 on the neutral grey, +1 on the warm pole, each value written in
    the cell (white on the warm pole, ink on grey)."""
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            face = NEUTRAL if v == 0 else (UP if v > 0 else PORT)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=face, edgecolor=SURFACE,
                                       linewidth=2.0))
            text = '0' if v == 0 else f'{v:+d}'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=8.5,
                    color=INK if v == 0 else SURFACE, fontweight='semibold' if v else 'normal')
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.5 for j in range(len(cols))])
    ax.set_xticklabels(cols)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels(rows if show_rows else [])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.annotate(title, xy=(0, 1), xycoords='axes fraction', xytext=(0, 24),
                textcoords='offset points', ha='left', va='bottom', fontsize=10,
                fontweight='semibold', color=INK)


# ------------------------------------------------------------------------------------------------
# the numbers, as tables
# ------------------------------------------------------------------------------------------------

def write_tables(rows, small, port, ref):
    with open(os.path.join(OUT, 'gate_counts.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell', 'part', 'gate', 'port (Qualtran 0.4)', 'reference (Qualtran 0.7)',
                    'difference'])
        for cell, key, g, a, b in rows:
            w.writerow([cell, CSV_PART[key], g, a, b, a - b])
    with open(os.path.join(OUT, 'gate_counts_small_cases.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'gate', 'port (Qualtran 0.4)', 'reference (Qualtran 0.7)',
                    'difference'])
        for name, g, a, b in small:
            w.writerow([name, g, a, b, a - b])
    with open(os.path.join(OUT, 'qubit_peaks.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell', 'part', 'reference (Qualtran 0.7)', 'port, greedy order',
                    "port, Qualtran 0.4's order"])
        for cell in CELL_ORDER:
            for key, _ in PARTS:
                p = port[cell]['parts'][key]
                w.writerow([cell, CSV_PART[key], ref[cell]['parts'][key]['peak'], p['peak'],
                            p.get('peak_q04_order', '')])


def main():
    os.makedirs(OUT, exist_ok=True)
    setup()
    port, ref = load()
    rows = full_rows(port, ref)
    small = small_cases()
    write_tables(rows, small, port, ref)
    fig_parity(rows, small)
    fig_gate_mix(port, ref)
    fig_t_by_part(port, ref)
    fig_peaks(port, ref)
    n = Counter(a == b for *_, a, b in rows) + Counter(a == b for *_, a, b in small)
    print(f'{n[True]} counts equal, {n[False]} differ; figures in {os.path.abspath(OUT)}')


if __name__ == '__main__':
    main()
