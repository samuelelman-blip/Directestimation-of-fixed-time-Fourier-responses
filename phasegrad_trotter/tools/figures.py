"""The figures of the paper, from data/layer.json and data/trotter.json, as PDF (for the paper)
and PNG, in figures/."""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, '..', 'data')
FIG = os.path.join(HERE, '..', 'figures')

# one colour per method, in a fixed order, with a marker and a line style as well
STYLE = {
    'gridsynth': dict(color='#2a78d6', ls='--', marker=None, label='gridsynth'),
    'bothe': dict(color='#eb6834', ls='--', marker=None, label='Bothe et al., fallback'),
    'bothe_af': dict(color='#008300', ls='-.', marker=None, label='Bothe et al., ancilla-free'),
    'det': dict(color='#1baf7a', ls='-', marker='o', label='gradient, nearest'),
    'rand': dict(color='#eda100', ls='-', marker='s', label='gradient, randomised'),
    'sparse': dict(color='#e87ba4', ls='-', marker='^', label='gradient, sparse pair'),
}
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'lines.linewidth': 1.4,
    'lines.markersize': 3.5, 'figure.dpi': 150, 'savefig.bbox': 'tight', 'font.family': 'serif',
    'mathtext.fontset': 'cm'})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=200)
    plt.close(fig)


def fig_layer():
    d = json.load(open(os.path.join(DATA, 'layer.json')))
    epss = (1e-6, 1e-10, 1e-14)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=False)
    for ax, eps in zip(axs, epss):
        rs = [r for r in d['rows'] if r['eps'] == eps and r['mode'] == 'rand']
        x = np.array([r['theta'] for r in rs])
        st = STYLE['gridsynth']
        ax.axhline(d['gridsynth'][str(eps)], color=st['color'], ls=st['ls'], label=st['label'])
        for m, key in (('bothe', 'bothe_fallback'), ('bothe_af', 'bothe_ancilla_free')):
            st = STYLE[m]
            ax.plot(x, [r[key] for r in rs], color=st['color'], ls=st['ls'], label=st['label'])
        for mode in ('det', 'rand', 'sparse'):
            rs = [r for r in d['rows'] if r['eps'] == eps and r['mode'] == mode]
            st = STYLE[mode]
            ax.plot([r['theta'] for r in rs], [r['t_per_rot'] for r in rs], color=st['color'],
                    ls=st['ls'], marker=st['marker'], label=st['label'])
        ax.axvline(np.sqrt(eps), color=INK2, lw=0.6, ls=':')
        ax.text(np.sqrt(eps) * 1.4, 0.97 * d['gridsynth'][str(eps)], r'$\theta=\sqrt{\epsilon}$',
                color=INK2, fontsize=6.5, va='top')
        ax.set_xscale('log')
        ax.set_xlim(2e-9, 1.3)
        ax.set_ylim(0, 1.08 * d['gridsynth'][str(eps)])
        ax.set_title(rf'$\epsilon=10^{{{int(np.log10(eps))}}}$', fontsize=8, color=INK)
        ax.set_xlabel(r'angle scale $\theta$')
    axs[0].set_ylabel('T gates per rotation')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.08),
               fontsize=6.3)
    save(fig, 'fig_layer')


def fig_batch():
    d = json.load(open(os.path.join(DATA, 'layer.json')))
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.1), gridspec_kw={'wspace': 0.45})
    for mode in ('det', 'rand'):
        rs = [r for r in d['batches'] if r['mode'] == mode]
        B = [r['B'] for r in rs]
        st = STYLE[mode]
        axs[0].plot(B, [r['t_per_rot'] for r in rs], color=st['color'], marker=st['marker'],
                    label=st['label'])
        axs[1].plot(B, [r['peak_ancilla'] for r in rs], color=st['color'], marker=st['marker'],
                    label=st['label'])
    st = STYLE['gridsynth']
    axs[0].axhline(d['gridsynth']['1e-10'], color=st['color'], ls=st['ls'], label=st['label'])
    st = STYLE['bothe']
    bt = d['batches'][0]['bothe_fallback']
    axs[0].axhline(bt, color=st['color'], ls=st['ls'], label=st['label'])
    for ax in axs:
        ax.set_xscale('log', base=2)
        ax.set_xlabel('batch size $B$')
    axs[1].set_yscale('log')
    axs[0].set_ylabel('T gates per rotation')
    axs[1].set_ylabel('peak ancillas')
    axs[0].set_ylim(0, 135)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.16))
    save(fig, 'fig_batch')


def fig_trotter():
    d = json.load(open(os.path.join(DATA, 'trotter.json')))
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for i, eps in enumerate(d['eps_totals']):
        for j, chain in enumerate(('random bonds', 'uniform bonds')):
            ax = axs[i, j]
            for m in ('gridsynth', 'bothe', 'bothe_af', 'det', 'rand', 'sparse'):
                rs = [r for r in d['rows'] if r['chain'] == chain and r['method'] == m
                      and r['eps_total'] == eps]
                st = STYLE[m]
                ax.plot([r['dt'] for r in rs], [r['t_step'] for r in rs], color=st['color'],
                        ls=st['ls'], marker=st['marker'], label=st['label'])
            ax.set_xscale('log')
            ax.set_yscale('log')
            if i == 1:
                ax.set_xlabel(r'Trotter step $\delta t$')
            ax.set_title(rf'{chain}, $\epsilon_{{\rm tot}}=10^{{{int(np.log10(eps))}}}$',
                         fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel('T gates per step')
    axs[0, 0].invert_xaxis()
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03),
               fontsize=6.3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, 'fig_trotter')


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    fig_layer()
    fig_batch()
    fig_trotter()
