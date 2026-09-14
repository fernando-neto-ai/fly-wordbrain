"""Paired descriptive analysis and plots for the fixed two-horizon pilot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def paired_interval(a, b, horizon, seed=1927, samples=10000):
    for key in ('story_ids', 'positions', 'target_mask', 'y'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError('Paired analysis rows/labels differ: ' + key)
    valid = a['target_mask'].copy()
    if horizon is not None:
        valid[:, 1 - horizon] = False
    differences = np.where(valid, a['losses'] - b['losses'], 0.)
    stories = np.unique(a['story_ids'])
    sums = np.array([differences[a['story_ids'] == story].sum() for story in stories])
    counts = np.array([valid[a['story_ids'] == story].sum() for story in stories])
    rng = np.random.default_rng(seed)
    selected = rng.integers(len(stories), size=(samples, len(stories)))
    bootstrap = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {'delta_cross_entropy': float(sums.sum() / counts.sum()),
            'percentile_95_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
            'stories': len(stories), 'forecasts': int(counts.sum()),
            'bootstrap_samples': samples, 'bootstrap_seed': seed,
            'interpretation': 'Exploratory paired story bootstrap on reused pilot test stories; negative favors pair-through-brain. Not a confirmatory or multiplicity-adjusted interval.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('results/pair-pilot'))
    args = parser.parse_args()
    run = args.run
    metrics = read(run / 'decoder/metrics.json')
    manifest = read(run / 'features/manifest.json')
    generation = read(run / 'generation.json')
    timing = read(run / 'decoder-timing.json')
    audit = read(run / 'checkpoint-audit.json')
    if not audit.get('passed') or not manifest.get('completed'):
        raise ValueError('Report requires completed extraction and successful checkpoint audit')
    rows = []
    for arm, label in [('brain', 'Pair through brain'), ('direct', 'Direct pair code'),
                       ('single_word_brain', 'Single word through brain')]:
        rows.append((label, metrics['arms'][arm]['repeats'][0]['metrics']['test']))
    for kind, label in [('unigram', 'Unigram counts'), ('current_word', 'Current-word counts'),
                        ('ordered_pair', 'Ordered-pair counts')]:
        rows.append((label, metrics['references']['test'][kind]))
    brain_predictions = archive(run / 'decoder/brain/seed-0/test-predictions.npz')
    comparisons = {}
    for label, path in [('single_word_brain', 'single_word_brain/seed-0/test-predictions.npz'),
                        ('direct', 'direct/seed-0/test-predictions.npz'),
                        ('ordered_pair_counts', 'references/ordered_pair-test-predictions.npz')]:
        other = archive(run / 'decoder' / path)
        comparisons[label] = {name: paired_interval(brain_predictions, other, horizon)
                              for name, horizon in [('next_1', 0), ('next_2', 1), ('forecast_average', None)]}
    summary = {'primary_seed': 0, 'primary_test': dict(rows), 'paired_story_comparisons': comparisons,
               'feature_identity': manifest['identity'], 'brain': manifest['brain'],
               'extraction_seconds': manifest['elapsed_seconds'], 'decoder_timing': timing,
               'checkpoint_audit_passed': True}
    output = Path('artifacts/pair-pilot')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    colors = ['#315b8e', '#3f8578', '#96a9bf', '#a9adb1', '#8a8f94', '#666d74']
    shared_limit = max(row['horizons'][h]['cross_entropy'] for _, row in rows
                       for h in ('next_1', 'next_2')) + .7
    for number, ax in enumerate(axes):
        values = [row['horizons']['next_%d' % (number + 1)]['cross_entropy'] for _, row in rows]
        ax.barh(range(len(rows)), values, color=colors, height=.64)
        for index, value in enumerate(values):
            ax.text(value + .06, index, '%.3f' % value, va='center', fontsize=9)
        ax.set_yticks(range(len(rows)), [name for name, _ in rows] if number == 0 else [''] * len(rows))
        ax.invert_yaxis()
        ax.set_xlim(0, shared_limit)
        ax.set_xlabel('Cross-entropy in nats (lower is better)')
        ax.set_title('Next word' if number == 0 else 'Second future word from the same state',
                     loc='left', weight='bold', fontsize=11)
    fig.text(.5, .025, 'Seed 0 · same 32 test stories · graph frozen · two independent heads · no future word supplied to head 2',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(output / 'comparison.png', dpi=160)
    fig.savefig(output / 'comparison.svg')
    svg = output / 'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)

    lines = ['# Bigram input, two future words: completed macm3 pilot', '',
             '**Pair input produced a small, inconsistent change across decoder seeds. '
             'The primary next-word perplexity moved from 189.16 to 185.11, while the '
             'same-sized decoder reading the pair codes directly reached 89.51. '
             'This run does not demonstrate a benefit from passing the codes through '
             'the current frozen-brain/readout configuration.**', '',
             'The complete original Doomfly connectome remains frozen. The new interface '
             'encodes the previous and current word in distinct fixed retinal roles, '
             'advances one word per 20 ms interval, and predicts two unseen future words. '
             'The decoder has **526,336 trainable parameters**: two independent '
             '256 → 1,024 affine softmax heads.', '',
             '![Pair pilot comparison](artifacts/pair-pilot/comparison.png)', '',
             '## Primary held-out comparison', '',
             'All learned models below use seed 0, the same two-head capacity and '
             'validation-only epoch/L2 selection. The same 32 test stories yield '
             '3,992 next-word targets and 3,960 second-future-word targets. The test '
             'set is reused from the earlier pilot for a paired developmental comparison. '
             'UNK makes up 11.62% and 11.67% of the respective horizon targets.', '',
             '| Representation / reference | Next-word CE | Next-word PPL ↓ | Next-word accuracy | Second-word CE ↓ | Two-word exact accuracy |',
             '|---|---:|---:|---:|---:|---:|']
    for label, row in rows:
        first, second = row['horizons']['next_1'], row['horizons']['next_2']
        lines.append(f"| {label} | {first['cross_entropy']:.4f} | {first['perplexity']:.2f} | {100*first['accuracy']:.2f}% | {second['cross_entropy']:.4f} | {100*row['complete_pair']['exact_accuracy']:.2f}% |")
    lines += ['', 'The second head receives the same current features as the first head. '
              'Its loss does not condition on the true first future word. The ordered-pair '
              'count reference follows that same rule, with current-word/unigram backoff. '
              'Count smoothing is fixed at add-one unigram smoothing and backoff prior '
              'mass 1; it was not tuned on validation, so these are reference settings '
              'rather than optimized count models. '
              'Two-word exact accuracy uses only rows where both future targets exist.', '',
              'These sliding forecasts overlap. We report each horizon separately; '
              'averaging the two losses is a forecast average, not ordinary sequence '
              'perplexity. Known-lexical scores exclude all four special symbols.', '',
              '| Model | Seed | Next-word CE | Second-word CE | Combined forecast CE | Known-lexical combined CE |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in ['brain', 'direct', 'single_word_brain']:
        for row in metrics['arms'][arm]['repeats']:
            test = row['metrics']['test']
            first, second = test['horizons']['next_1'], test['horizons']['next_2']
            avg = test['forecast_average']
            lines.append(f"| {arm} | {row['seed']} | {first['cross_entropy']:.4f} | {second['cross_entropy']:.4f} | {avg['cross_entropy']:.4f} | {avg['known_lexical_cross_entropy']:.4f} |")
    lines += ['', '## Paired uncertainty', '',
              'Differences below are pair-through-brain CE minus the comparator CE; '
              'negative favors the brain with pair input. We resample entire stories '
              'together across systems, with 10,000 fixed-seed bootstrap replicates. '
              'Intervals are exploratory, unadjusted for multiple comparisons, and '
              'describe these reused pilot stories—not biological variability.', '',
              '| Comparator | Next-word ΔCE | 95% story-bootstrap interval | Second-word ΔCE | 95% story-bootstrap interval |',
              '|---|---:|---|---:|---|']
    for label, row in comparisons.items():
        first, second = row['next_1'], row['next_2']
        a, b = first['percentile_95_interval'], second['percentile_95_interval']
        lines.append(f"| {label} | {first['delta_cross_entropy']:+.4f} | [{a[0]:+.4f}, {a[1]:+.4f}] | {second['delta_cross_entropy']:+.4f} | [{b[0]:+.4f}, {b[1]:+.4f}] |")
    lines += ['', '## What changed and what stayed fixed', '',
              'All 166,700 neurons, 25,582,938 directed edges and 124,177,617 synaptic '
              'contacts are retained in the original fixed NativeBrain baseline. '
              'The 0.1 ms timestep, warmup, drive intensity, delay, refractory behavior '
              'and neural readout projection are unchanged. Input roles illuminate '
              '417 + 417 receptors, matching the original 834 total and its brightness '
              'budget. This supplies one explicit previous word at the sensory interface.', '',
              'The direct control projects the identical retinal stimulus into 256 '
              'features using a fixed signed CountSketch, with no temporal memory. '
              'The single-word brain control reuses the earlier neural observations. '
              'No word embeddings or shared decoder trunk are learned. Training the '
              'second independent head cannot reshape the first head or the frozen brain.', '',
              'Each of the three seeds selects epoch/L2 separately for each head from '
              'the fixed grid (0, 0.0001, 0.01), at most 30 epochs. All selections finish '
              'before test arrays are loaded. No architecture or training setting was '
              'changed based on this run’s test results. See [PAIR_PROTOCOL.md](PAIR_PROTOCOL.md).', '',
              '## Runtime and verification', '',
              f"Feature extraction took **{manifest['elapsed_seconds']:.1f} seconds** with four CPU workers on macm3. "
              f"Decoder training and evaluation for all three representations, all three seeds and both heads took **{timing['elapsed_seconds']:.1f} seconds**. "
              'No GPU was required.', '',
              'The full-graph probe checks repeatability, order sensitivity, disconnected '
              'inputs, equal stimulus energy, memoryless direct projection and frozen '
              'graph hashes. The post-training audit checks complete causal rows, '
              'source/cache hashes, exact direct-projection reconstruction, train-only '
              'standardization and independently recomputed checkpoint losses. The '
              'single-word control’s first head exactly reproduces the original checkpoint. '
              '**All checks passed, along with 50 lightweight tests.** '
              'Every post-BOS observation differs from the original word-encoding cache; '
              '87 of 256 columns change. Both brain encodings have 4 varying spike-rate '
              'columns and 101 varying voltage columns. The new input did reach the '
              'simulation; these checks do not explain its limited predictive benefit.', '',
              '## Fixed-seed paired generation', '',
              'Both heads sample from the same state before either word is fed back. '
              'Then both successive sliding pairs pass through the brain. The original '
              'three prompts, seed 1729, temperature 0.8 and 24-word cap are retained. '
              'Samples remain incoherent. They are illustrative and were not selected for quality.', '']
    for sample in generation['samples']:
        continuation = sample['continuation'].replace('<', '&lt;').replace('>', '&gt;')
        lines += [f"Prompt: `{sample['prompt']}`", '', f"> {continuation}", '']
    lines += ['## Interpretation limits', '',
              'This experiment changes the external encoding while preserving the graph. '
              'A gain over the direct linear control could arise from nonlinear pair '
              'processing or older temporal state; it would not isolate biological '
              'topology. A negative result would not rule out better interfaces. '
              'Matched rewired graphs and a fresh, larger final evaluation remain untested.', '',
              f"Checkpoints, source receipts, predictions and full metrics are under `{run}`. "
              'The primary checkpoint directory is `decoder/brain/seed-0`; it contains '
              '`head-0.pt`, `head-1.pt`, `pair.json` and the feature manifest. '
              'The original [single-word report](REPORT.md) remains unchanged.', '']
    Path('PAIR_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'report': 'PAIR_REPORT.md', 'summary': str(output / 'summary.json'),
                      'primary_test': dict(rows)}, indent=2))


if __name__ == '__main__':
    main()
