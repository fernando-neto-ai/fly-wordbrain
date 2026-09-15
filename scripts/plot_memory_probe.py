"""Plot primary readouts alongside the explicitly exploratory scaling check."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(output):
    output = Path(output)
    primary = json.loads((output/'readouts.json').read_text())['results']
    sensitivity = json.loads((output/'scale-sensitivity.json').read_text())['results']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    features = ['input_history', 'on_pooled_early', 'on_pooled_final', 'on_cells_final', 'on_cells_temporal']
    labels = ['Input codes', 'Early pooled\nresponse', 'Final pooled\nresponse',
              'Final selected\nneurons', 'Four selected\nneuron snapshots']
    for ax, task, title, chance in zip(axes, ['identity', 'order'],
            ['Earlier word identity · 10 classes', 'Earlier word order · 2 classes'], [10, 50]):
        x = np.arange(len(features))
        a = [100*primary[task+'/'+f+'/linear']['metrics']['test']['accuracy'] for f in features]
        b = [100*sensitivity[task+'/'+f+'/linear']['metrics']['test']['accuracy'] for f in features]
        ax.bar(x-.18, a, .36, color='#526f98', label='Original probe scaling')
        ax.bar(x+.18, b, .36, color='#e58b40', label='Lower threshold (exploratory)')
        for at, value in zip(x+.18, b):
            ax.text(at, value+2, f'{value:g}%', ha='center', va='bottom', fontsize=9)
        ax.axhline(chance, color='#777', linestyle='--', linewidth=1)
        ax.text(-.4, chance+2, 'chance', ha='left', color='#666', fontsize=9)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 113), title=title,
               ylabel='Held-out probe accuracy (%)')
        ax.tick_params(axis='x', length=0, labelsize=9)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=.15)
    step = json.loads((output/'protocol.json').read_text())['snapshot']['checkpoint_step']
    fig.suptitle('The early pooled response contains signal; final readouts remain near chance', fontsize=13, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .072), ncol=2, frameon=False)
    fig.text(.5, .035, f'Frozen step {step:,} · fast weights ON · linear heads · 8 held-out context groups per task',
             ha='center', fontsize=9, color='#555')
    fig.text(.5, .003, 'Normalization check reuses the test set. These are synthetic memory probes, not next-word accuracy.',
             ha='center', fontsize=9, color='#555')
    fig.tight_layout(rect=(0, .16, 1, .92))
    fig.savefig(output/'probe-results.png', dpi=180, bbox_inches='tight')
    fig.savefig(output/'probe-results.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    plot(parser.parse_args().output)
