"""Render a concise report and scientific plot from completed pilot artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('results/pilot-verified'))
    args = parser.parse_args()
    run = args.run
    metric = read(run / 'decoder/metrics.json')
    recall = read(run / 'recall/report.json')
    brain = read(run / 'features/manifest.json')
    audit = read(run / 'checkpoint-audit.json')
    replay = read(run / 'replay-check.json')
    generation = read(run / 'generation.json')
    timing = read(run / 'decoder-timing.json')
    assert brain['completed'] and audit['passed'] and replay['all_arrays_bit_exact']
    primary = metric['primary']['metrics']['test']
    refs = metric['references']['test']
    contexts = [8, 16, 32, 64, 128]
    models = [('Frozen brain + linear (seed 0)', primary),
              ('Unigram', refs['unigram']), ('Smoothed bigram', refs['smoothed_bigram'])]
    out = Path('artifacts')
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), gridspec_kw={'width_ratios': [1.1, 1]})
    ax = axes[0]
    ax.barh([0, 1, 2], [row['perplexity'] for _, row in models],
            color=['#315b8e', '#a4adb6', '#707d88'], height=.6)
    for i, (_, row) in enumerate(models):
        ax.text(row['perplexity'] + 3, i, f"{row['perplexity']:.1f}", va='center')
    ax.set_yticks([0, 1, 2], [name for name, _ in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 225)
    ax.set_xlabel('Test perplexity (lower is better)')
    ax.set_title('Next-word prediction · 3,992 targets', loc='left', weight='bold')
    ax = axes[1]
    values = np.array([[r['test']['by_context_words'][str(c)]['accuracy'] * 100
                        for c in contexts] for r in recall['runs']])
    ax.plot(range(5), values.mean(axis=0), color='#315b8e', marker='o', label='Mean of 3 decoder seeds')
    for i, y in enumerate(values):
        ax.scatter(np.arange(5) + (i - 1) * .07, y, color='#687b90', s=17, alpha=.7,
                   label='Individual seeds' if i == 0 else None)
    ax.axhline(25, color='#a0633e', linestyle='--', label='Chance (25%)')
    ax.set_xticks(range(5), contexts)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Lexical context length (words)')
    ax.set_ylabel('Name classification accuracy (%)')
    ax.set_title('Separate delayed-name diagnostic', loc='left', weight='bold')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.text(.5, .015, 'Recall uses the same 8 test prompts per context for all seeds. Seed spread is not a confidence interval.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(out / 'pilot-results.png', dpi=160)
    fig.savefig(out / 'pilot-results.svg')
    svg = out / 'pilot-results.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)

    lines = [
        '# First frozen-brain pilot on macm3', '',
        '**The complete frozen graph runs and a small decoder trains successfully. '
        'This configuration did not beat unigram or bigram next-word perplexity, '
        'and did not establish reliable delayed-name recall.**', '',
        '![Pilot results](artifacts/pilot-results.png)', '',
        '## Model and data', '',
        'The unchanged original Doomfly `NativeBrain` baseline contains 166,700 neurons, '
        '25,582,938 directed connections and 124,177,617 synaptic contacts. '
        'It is not the later v6 physiological model. All original graph edges and weights '
        'remain frozen; the original 0.1 ms timestep is retained.', '',
        'Fixed equal-energy word codes stimulate 3,335 retinal inputs for 20 ms per word. '
        'A fixed projection reads spike rates and voltages from all 1,314 annotated '
        'descending neurons into 256 features. The sole language trainable component is '
        '**256 → 1,024 affine softmax: 263,168 parameters, about 1 MiB in float32**. '
        'There are no learned embeddings, hidden layers or decoder memory.', '',
        'The pilot uses 128/32/32 disjoint TinyStories train/validation/test stories, '
        'with at most 128 lexical words per story. There are 16,017/3,991/3,992 scored '
        'next-word targets. Vocabulary is train-only. The natural-text test targets '
        'include 11.62% UNK; known lexical words cover 88.34% of retained test lexical words. '
        'The brain resets between stories.', '',
        '## Held-out next-word results', '',
        '| Model | Cross-entropy (nats) | Perplexity ↓ | Top-1 accuracy | Known lexical perplexity ↓ |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, row in models:
        lines.append(f"| {name} | {row['cross_entropy']:.4f} | {row['perplexity']:.2f} | {100*row['accuracy']:.2f}% | {row['known_lexical_perplexity']:.2f} |")
    lines += ['', 'All three decoder seeds are reported; seed 0 was designated primary before '
              'evaluation. Epoch/L2 selection used validation only. The references are fitted '
              'on the same training targets and vocabulary.', '',
              '| Decoder seed | Selected L2 | Selected epoch | Test perplexity | Test accuracy |',
              '|---|---:|---:|---:|---:|']
    for row in metric['repeats']:
        selected, test = row['selected'], row['metrics']['test']
        lines.append(f"| {selected['seed']} | {selected['l2']} | {selected['selected_epoch']} | {test['perplexity']:.2f} | {100*test['accuracy']:.2f}% |")
    lines += ['', 'The neural decoder has higher top-1 accuracy than the unigram reference, '
              'but worse cross-entropy/perplexity; the bigram reference wins on both. '
              'There is no overall predictive advantage in this pilot. Seed differences '
              'do not estimate uncertainty over new stories or biological samples.', '',
              '## Separate name-memory diagnostic', '',
              'A separate 1,028-parameter four-way linear classifier is trained on 80 synthetic '
              'prompts, selected on 40 validation prompts and tested on 40 held-out prompts. '
              'The four names are balanced; paired prompts differ only in the early name. '
              'This is a test of final-state decodability, not the natural-text decoder or '
              'natural-language query understanding.', '',
              '| Context words | Seed 0 correct / 8 | Mean accuracy, 3 seeds | Mean CE (nats) |',
              '|---|---:|---:|---:|']
    for c in contexts:
        row = recall['test_by_context_words'][str(c)]
        first = recall['runs'][0]['test']['by_context_words'][str(c)]
        lines.append(f"| {c} | {first['correct']}/8 | {100*row['accuracy_mean']:.2f}% | {row['cross_entropy_mean']:.4f} |")
    lines += ['', 'Chance is 25% accuracy and a uniform classifier has CE 1.3863. '
              'Overall test accuracies for seeds 0/1/2 are 25%, 32.5%, and 25%; '
              'all three select epoch 1. At 128 words, mean accuracy is 20.83%. '
              'Only two paired test groups exist at each length. These observations '
              'do not establish reliable recall or a memory-capacity limit.', '',
              'The query/filler templates are held out. Some carrier words map to UNK '
              '(including test `person`, `tell`, `moves`), equally across labels. '
              'A poor result cannot isolate forgetting from changes in input distribution '
              'or limitations of the fixed input/readout interface.', '',
              '## Verification and runtime', '',
              '- All three raw graph sources and the vendored Doomfly files match pinned upstream hashes.',
              '- Every neuron/edge remains; per-story graph hashes stayed unchanged.',
              '- Reset replay is exact; disconnected inputs ignore word identity; connected histories affect the readout.',
              '- Two complete extractions produce bit-identical features and targets across all three splits.',
              '- The deployed source matches the local source lock. Direct NumPy checkpoint evaluation matches logged CE within 1e-6.',
              '- Every expected causal input/target pair is present exactly once. Only training statistics standardize features.',
              '- 22 lightweight tests passed; the native full-graph checks and generation provenance checks also passed.', '',
              f"The verified language extraction took **{brain['elapsed_seconds']:.1f} seconds** with four CPU workers; "
              f"all three decoder seeds and their L2 candidates took **{timing['elapsed_seconds']:.1f} seconds** in total. "
              'Hardware: M3 Max, 16 CPU cores, 128 GB unified memory. No GPU was required. '
              'These are small-pilot timings, not extrapolated large-corpus benchmarks.', '',
              'An audit of the lower validation loss found 4.00% UNK training targets versus '
              '12.10% validation targets. Reweighting validation losses to training UNK '
              'frequency changes CE from 5.1621 to 5.3398, versus training CE 5.4403. '
              'Thus coverage explains much of the inversion, with a remaining difference '
              'between the story samples. In the cached training features, 105 of 256 '
              'columns vary; the checkpoint logits respond to those features.', '',
              '## Fixed-seed samples', '',
              'Temperature 0.8; up to 24 new words; the same primary checkpoint and '
              'brain configuration; no sample selection. These outputs are not coherent language.', '']
    for sample in generation['samples']:
        continuation = sample['continuation'].replace('<', '&lt;').replace('>', '&gt;')
        lines += [f"Prompt: `{sample['prompt']}`", '', f"> {continuation}", '']
    lines += ['## Scope and artifacts', '',
              'This is a completed first interface experiment. It neither establishes nor '
              'rules out a useful inductive bias in the biological topology. That claim '
              'requires matched rewired-graph controls, more data and interface comparisons. '
              'The preserved object is Doomfly’s directed graph and point-neuron dynamics, '
              'not a spatial cable model of biological morphology. Causal history sensitivity '
              'also includes the inherited retinal filter and is not equivalent to memory.', '',
              f'Artifacts are under `{run}`: `decoder/checkpoint.pt`, `decoder/metrics.json`, '
              '`features/manifest.json`, `recall/report.json`, `generation.json`, '
              '`checkpoint-audit.json`, and `replay-check.json`. Full per-story caches and '
              'the original graph data remain on macm3 at `/Users/fernando/fly_wordbrain`.', '',
              'See [README.md](README.md) for reproduction and [DATA_PROVENANCE.md](DATA_PROVENANCE.md) '
              'for source and tokenization details.', '']
    Path('REPORT.md').write_text('\n'.join(lines))
    print('REPORT.md and artifacts/pilot-results.{png,svg}')


if __name__ == '__main__':
    main()
