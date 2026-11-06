"""Figures of the estimator's errors, from data/error_checks.json (tools/error_checks.py).

    python tools/plot_errors.py         # writes figures/fig5..fig7 and figures/errors_*.csv

    fig5_small_scale_errors.png   small instances: the gate-level circuit against the reference,
                                  measured against calculated error, and each source of error
                                  against its parameter (time grid, truncation order, selector)
    fig6_error_vs_n.png           the nine cells' circuit parameters at n = 4 .. 2^20: measured error
                                  of the estimate, its calculated bound, and the target eps
    fig7_error_sources_vs_n.png   the same, source by source, for four cells

The estimate is Xi_M <0|W|0>, the scalar the circuit's amplitude encodes; amplitude estimation
reads it to within the readout allowance and a compiler adds the synthesis error of the
rotations, both of which are allowances of the budget and not simulated here.
"""
import csv
import json
import math
import os

import numpy as np

from plot_versions import (BASELINE, INK, INK2, MUTED, ORDER04, OUT, PORT, REF, SURFACE,
                           cell_label, clean, figure_title, panel_title, plt, setup)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('ERROR_CHECKS', os.path.join(HERE, '..', 'data', 'error_checks.json'))
CELL_ORDER = ['dense eps=0.01', 'dense eps=1e-06', 'PD-F eps=0.01', 'PD-F eps=1e-06',
              'PD-T eps=0.01', 'PD-T eps=1e-06', 'sparse eps=1e-06', 'PS-T eps=1e-06',
              'PS-F eps=1e-06']
FAMILY_NAME = {('dense', 'flat'): 'dense', ('PD', 'F'): 'PD-F', ('PD', 'T'): 'PD-T',
               ('sparse', 'flat'): 'sparse', ('PS', 'F'): 'PS-F', ('PS', 'T'): 'PS-T'}


def log2_ticks(ax, sizes):
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'2$^{{{b}}}$' if b > 0 else '1' for b in sizes])


def handles(entries):
    """Legend entries (label, colour, marker or None, line or not)."""
    from matplotlib.lines import Line2D
    out = []
    for label, color, marker, line in entries:
        out.append(Line2D([], [], color=color, linewidth=2 if line else 0, marker=marker,
                          markersize=6 if marker else 0, markeredgecolor=SURFACE,
                          linestyle='-' if line else 'none', label=label))
    return out


def spaced_labels(ax, x, items, gap=0.055):
    """Direct labels at the right end of lines, (y, text, colour), on a log axis: labels closer
    than `gap` of the axis height are gathered into a cluster centred on their lines' mean
    position, so each label stays as near its line as the spacing allows."""
    lo, hi = (math.log10(v) for v in ax.get_ylim())
    step = gap * (hi - lo)
    order = sorted(items, key=lambda it: it[0])
    clusters = [[math.log10(it[0])] for it in order]       # each: the targets of its labels
    pos = [c[0] for c in clusters]                          # each: its lowest label's position

    def place(c):
        return sum(c) / len(c) - step * (len(c) - 1) / 2

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters) - 1):
            top_i = pos[i] + step * (len(clusters[i]) - 1)
            if pos[i + 1] - top_i < step:
                clusters[i] = clusters[i] + clusters.pop(i + 1)
                pos.pop(i + 1)
                pos[i] = place(clusters[i])
                merged = True
                break
    ys = [p + step * k for p, c in zip(pos, clusters) for k in range(len(c))]
    for (y, text, color), ly in zip(order, ys):
        ax.text(x, 10 ** ly, text, fontsize=7.5, color=color, va='center', ha='left',
                clip_on=False)


def dot(ax, x, y, color, size=30, label=None, zorder=4, marker='o'):
    return ax.scatter(x, y, s=size, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=zorder,
                      label=label, marker=marker)


# ------------------------------------------------------------------------------------------------
# figure 5: small scale
# ------------------------------------------------------------------------------------------------

def fig_small(d):
    fig = plt.figure(figsize=(12.0, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.30,
                          left=0.10, right=0.98, top=0.86, bottom=0.07)

    # (a) the gate-level circuit against the reference simulator
    ax = fig.add_subplot(gs[0, 0:2])
    gate = d['gate']
    groups = [('ANDs as leaves', [r for r in gate if not r['clifford_t']]),
              ('ANDs in Clifford+T', [r for r in gate if r['clifford_t']])]
    ax.axvspan(1e-17, 1e-14, color='#f0efec', zorder=0, linewidth=0)
    ax.text(1.3e-17, 1.62, 'double-precision rounding', fontsize=7.5, color=MUTED, va='center')
    for k, (name, rows) in enumerate(groups):
        x = [max(r['amp_difference'], 1.2e-17) for r in rows]
        rng = np.random.default_rng(k)
        y = k + rng.uniform(-0.18, 0.18, len(x))
        dot(ax, x, y, PORT, size=34)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f'{g[0]}\n({len(g[1])} runs)' for g in groups])
    ax.set_ylim(1.8, -0.6)
    ax.set_xscale('log')
    ax.set_xlim(1e-17, 1e-9)
    clean(ax)
    ax.set_xlabel('|amplitude of the streamed circuit − reference simulator|')
    worst = max(r['amp_difference'] for r in gate)
    ns = sorted({r['n'] for r in gate})
    panel_title(ax, 'The streamed circuit gives the reference block amplitude',
                f'{len(gate)} gate-level runs at n = {", ".join(map(str, ns))}, M = 1 to 3 (7 with '
                f'Clifford+T ANDs and uncomputes measured at random); largest difference '
                f'{worst:.1e}')

    # (b) measured against calculated error
    ax = fig.add_subplot(gs[0, 2])
    sc = d['scatter']
    lo, hi = 1e-11, 1.0
    ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1.0, zorder=1)
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color='#fbe9e2', zorder=0, linewidth=0)
    ax.text(2e-11, 0.2, 'measured > calculated', fontsize=7.5, color=MUTED)
    dot(ax, [r['calculated_total'] for r in sc], [r['error'] for r in sc], PORT, size=14,
        label=f'reference engine ({len(sc)} configurations)')
    gl = [r for r in gate]
    dot(ax, [r['calculated_total'] for r in gl], [r['error'] for r in gl], REF, size=30,
        label=f'gate-level circuit ({len(gl)} runs)', marker='D')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    clean(ax, 'both')
    ax.set_xlabel('calculated error (truncation + time grid + selector)')
    ax.set_ylabel('measured |estimate − g|')
    ax.legend(loc='lower right', fontsize=7.5, handletextpad=0.3, borderaxespad=0.3)
    ratio = max(r['error'] / r['calculated_total'] for r in sc + gl)
    panel_title(ax, 'Measured against calculated error',
                f'n = 4 to 16, all six families; largest ratio {ratio:.2f}')

    # (c) the time grid
    ax = fig.add_subplot(gs[1, 0])
    rows = [r for r in d['time'] if r['family'] == 'dense' and r['b'] == 2 and r['M'] == 3]
    bt = [r['b_t'] for r in rows]
    ax.plot(bt, [r['bound_worst'] for r in rows], color=MUTED, linewidth=1.5, zorder=2)
    ax.plot(bt, [r['bound_mean'] for r in rows], color=REF, linewidth=2, zorder=2)
    dot(ax, bt, [r['error'] for r in rows], PORT, size=28)
    ax.set_yscale('log')
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.legend(handles=handles([('measured', PORT, 'o', False),
                               ('Appendix A bound, mean order (cost model)', REF, None, True),
                               ('Appendix A bound, every block at order M', MUTED, None, True)]),
              loc='lower left', fontsize=7.5, handlelength=1.6)
    clean(ax, 'both')
    ax.set_xlabel('time-register width b$_t$ (bits)')
    ax.set_ylabel('|estimate − h$_M$| from the time grid')
    panel_title(ax, 'Time grid: second order, under its bound',
                'dense, n = 4, M = 3; error falls 4\u00d7 per bit, the bound 2\u00d7')

    # (d) truncation
    ax = fig.add_subplot(gs[1, 1])
    tr = [r for r in d['truncation'] if r['family'] == 'dense']
    for b, color, mk in ((2, PORT, 'o'), (20, REF, 'D')):
        rr = [r for r in tr if r['b'] == b]
        M = [r['M'] for r in rr]
        ax.plot(M, [min(r['bound_22'], r['bound_inf1']) for r in rr], color=color, linewidth=2,
                zorder=2)
        dot(ax, M, [max(r['error'], 1e-15) for r in rr], color, size=28, marker=mk)
    ax.set_yscale('log')
    ax.set_xlim(-0.3, 6.3)
    ax.set_xticks(range(7))
    ax.legend(handles=handles([('n = 4', PORT, 'o', True), ('n = 2$^{20}$', REF, 'D', True)]),
              loc='lower left', fontsize=7.5, handlelength=1.6)
    clean(ax, 'both')
    ax.set_xlabel('interaction order M')
    ax.set_ylabel('|g − g$_M$|')
    panel_title(ax, 'Truncation: under the certified bound',
                'dense family; points measured, lines the better certificate')

    # (e) selector
    ax = fig.add_subplot(gs[1, 2])
    se = d['selector']
    for M, color, mk in ((1, PORT, 'o'), (4, REF, 'D')):
        rr = [r for r in se if r['M'] == M]
        p = [r['precision'] for r in rr]
        ax.plot(p, [r['bound'] for r in rr], color=color, linewidth=2, zorder=2)
        dot(ax, p, [max(r['error'], 1e-12) for r in rr], color, size=28, marker=mk)
    ax.legend(handles=handles([('M = 1', PORT, 'o', True), ('M = 4', REF, 'D', True)]),
              loc='lower left', fontsize=7.5, handlelength=1.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    clean(ax, 'both')
    ax.set_xlabel('selector precision')
    ax.set_ylabel('|estimate − h$_M$| from the selector')
    panel_title(ax, 'Selector: under its bound',
                'dense, n = 4; points measured, lines \u039e$_M$(M+1) \u00d7 precision')

    figure_title(fig, 'Small instances: the circuit computes the intended amplitude, and every '
                 'error is under its calculated bound',
                 r'Estimate = $\Xi_M\langle 0|W|0\rangle$, the scalar amplitude estimation reads; '
                 'g = the exact response of the full quadratic equation (integrated directly).')
    fig.savefig(os.path.join(OUT, 'fig5_small_scale_errors.png'))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# figure 6: the nine cells against n
# ------------------------------------------------------------------------------------------------

def calc_total(r):
    """The calculated error of what is measured: truncation, time grid and selector (the
    measurement has exact rotations, so their synthesis allowance is left out)."""
    c = r['calculated']
    return c['truncation'] + c['time'] + c['selector']


def fig_cells(d):
    cells = d['cells']
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.6), sharex=True)
    sizes = [r['b'] for r in cells[CELL_ORDER[0]]['rows']]
    for ax, label in zip(axes.flat, CELL_ORDER):
        c = cells[label]
        rows = c['rows']
        b = [r['b'] for r in rows]
        eps = c['eps']
        ax.axhline(eps, color=INK, linewidth=1.0, zorder=1)
        ax.text(b[-1] + 0.4, eps, 'ε', fontsize=9, color=INK, va='center')
        ax.plot(b, [calc_total(r) for r in rows], color=REF, linewidth=2, zorder=2)
        dot(ax, b, [r['error'] for r in rows], PORT, size=30)
        ax.set_yscale('log')
        lo = min(min(r['error'] for r in rows), 1e-3 * eps)
        hi = max(eps, max(r['error'] for r in rows), max(calc_total(r) for r in rows))
        ax.set_ylim(lo / 3, hi * 4)
        ax.set_xlim(1, 21.8)
        log2_ticks(ax, [2, 6, 10, 14, 20])
        clean(ax, 'both')
        r20 = rows[-1]
        panel_title(ax, cell_label(label),
                    f'M = {c["M"]}, b$_t$ = {c["b_t"]}; at n = 2$^{{20}}$ measured / calculated '
                    f'= {r20["error"] / calc_total(r20):.2f}')
    for ax in axes[-1, :]:
        ax.set_xlabel('n')
    for ax in axes[:, 0]:
        ax.set_ylabel('absolute error')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=PORT, marker='o', linestyle='none', markersize=6,
                      markeredgecolor=SURFACE, label='measured error of the estimate'),
               Line2D([], [], color=REF, linewidth=2,
                      label='calculated: truncation + time grid + selector'),
               Line2D([], [], color=INK, linewidth=1.0, label='target ε')]
    figure_title(fig, 'Errors against n, with each cell’s circuit parameters held fixed',
                 'M, b$_t$ and selector precision as costed at n = 2$^{20}$. Dense skeleton: the '
                 'calculated error changes by at most 1.3× for n ≥ 2$^6$.\n'
                 'Sparse skeleton: error and bound both fall as 1/n, and the costed M = 1 relies '
                 'on that, so below n ≈ 2$^{10}$ both are above ε.\n'
                 'Not shown: the rotation and readout allowances, which use the rest of ε at '
                 'n = 2$^{20}$.', y=0.99)
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.055, 0.912), ncol=3)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.815, bottom=0.06, hspace=0.42, wspace=0.22)
    fig.savefig(os.path.join(OUT, 'fig6_error_vs_n.png'))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# figure 7: source by source, four cells
# ------------------------------------------------------------------------------------------------

def fig_sources(d):
    cells = d['cells']
    picks = ['dense eps=0.01', 'dense eps=1e-06', 'PD-T eps=1e-06', 'PS-T eps=1e-06']
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8), sharex=True)
    for ax, label in zip(axes.flat, picks):
        c = cells[label]
        rows = c['rows']
        b = [r['b'] for r in rows]
        eps = c['eps']
        series = [('truncation', PORT, 'o'), ('selector', REF, 'D'), ('time', ORDER04, 's')]
        labels = []
        shown_time = False
        for key, color, mk in series:
            calc = [r['calculated'][key] for r in rows]
            ax.plot(b, calc, color=color, linewidth=2, zorder=2)
            meas = [r[key] for r in rows]
            if all(v is not None for v in meas) and max(meas) > 1e-13:
                dot(ax, b, meas, color, size=28, marker=mk)
                shown_time = shown_time or key == 'time'
            name = {'truncation': 'truncation', 'selector': 'selector', 'time': 'time grid'}[key]
            labels.append((calc[-1], name, INK2))
        rot = [r['calculated']['rotations'] for r in rows]
        ax.plot(b, rot, color=MUTED, linewidth=1.2, zorder=2)
        labels.append((rot[-1], 'rotations', MUTED))
        read = c['budget']['readout']
        ax.axhline(read, color=MUTED, linewidth=1.2, zorder=1)
        labels.append((read, 'readout', MUTED))
        ax.axhline(eps, color=INK, linewidth=1.0, zorder=1)
        labels.append((eps, '\u03b5 (total)', INK))
        ax.set_yscale('log')
        ys = [v for r in rows for v in (r['truncation'], r['selector']) if v and v > 0]
        lo = min(min(ys), min(r['calculated']['selector'] for r in rows)) / 4
        if shown_time:
            lo = min(lo, min(r['time'] for r in rows) / 4)
        top = max(eps, max(r['calculated']['truncation'] for r in rows)) * 3
        ax.set_ylim(lo, top)
        ax.set_xlim(1, 21)
        spaced_labels(ax, 21.3, labels)
        log2_ticks(ax, [2, 6, 10, 14, 20])
        clean(ax, 'both')
        if c['M'] == 1 and shown_time:
            note = 'time-grid error computed exactly (M = 1)'
        elif c['M'] == 1:
            note = 'time-grid error below 1e-13 (not shown)'
        else:
            note = 'time-grid error below 1e-13 at this b$_t$ (second order; not shown)'
        panel_title(ax, cell_label(label), f'M = {c["M"]}, b$_t$ = {c["b_t"]}; {note}')
    for ax in axes[-1, :]:
        ax.set_xlabel('n')
    for ax in axes[:, 0]:
        ax.set_ylabel('absolute error')
    figure_title(fig, 'Each source of error against n: measured (points) and calculated (lines)',
                 'Rotation synthesis and readout are allowances (a compiler and amplitude '
                 'estimation spend them), so they have no measured points.', y=0.99)
    fig.legend(handles=handles([('truncation', PORT, 'o', True), ('selector', REF, 'D', True),
                                ('time grid', ORDER04, 's', True),
                                ('allowances: rotations, readout', MUTED, None, True)]),
               loc='upper left', bbox_to_anchor=(0.055, 0.935), ncol=4)
    fig.subplots_adjust(left=0.08, right=0.89, top=0.83, bottom=0.07, hspace=0.38, wspace=0.30)
    fig.savefig(os.path.join(OUT, 'fig7_error_sources_vs_n.png'))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------

def write_tables(d):
    with open(os.path.join(OUT, 'errors_gate_level.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['family', 'n', 'M', 'b_t', 'selector precision', 'Clifford+T', 'gates',
                    '|amp - reference|', 'measured error', 'calculated error',
                    'truncation', 'time grid', 'selector'])
        for r in d['gate']:
            fam = FAMILY_NAME[(r['args'][0], r['args'][1])]
            w.writerow([fam, r['n'], r['M'], r['b_t'], r['precision'], r['clifford_t'], r['ops'],
                        f"{r['amp_difference']:.3e}", f"{r['error']:.4e}",
                        f"{r['calculated_total']:.4e}", f"{r['truncation']:.4e}",
                        f"{r['time']:.4e}", f"{r['selector']:.4e}"])
    with open(os.path.join(OUT, 'errors_vs_n.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell', 'n', 'measured error', 'measured truncation', 'measured selector',
                    'measured time grid', 'calculated truncation', 'calculated time grid',
                    'calculated selector', 'calculated rotations', 'readout allowance', 'eps'])
        for label in CELL_ORDER:
            c = d['cells'][label]
            for r in c['rows']:
                k = r['calculated']
                w.writerow([label, r['n'], f"{r['error']:.4e}", f"{r['truncation']:.4e}",
                            f"{r['selector']:.4e}",
                            '' if r['time'] is None else f"{r['time']:.4e}",
                            f"{k['truncation']:.4e}", f"{k['time']:.4e}", f"{k['selector']:.4e}",
                            f"{k['rotations']:.4e}", f"{c['budget']['readout']:.4e}", c['eps']])


def main():
    os.makedirs(OUT, exist_ok=True)
    setup()
    d = json.load(open(DATA))
    if all(k in d for k in ('gate', 'scatter', 'time', 'truncation', 'selector')):
        fig_small(d)
    if 'cells' in d and len(d['cells']) == len(CELL_ORDER):
        fig_cells(d)
        fig_sources(d)
    if 'gate' in d and 'cells' in d:
        write_tables(d)
    print('figures in', os.path.abspath(OUT))


if __name__ == '__main__':
    main()
