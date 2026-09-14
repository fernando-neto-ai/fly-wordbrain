"""Verify deployed source, complete causal rows, and checkpoint loss directly."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-lock', type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.source_lock.read_text())
    actual = {name: digest(ROOT / name) for name in expected}
    assert actual == expected, 'Deployed source differs from source lock'
    for cached in (ROOT / 'fly_wordbrain/__pycache__').glob('decoder.*.pyc'):
        cached.unlink()
    import numpy as np
    import torch
    import fly_wordbrain.decoder as module
    torch.set_num_threads(1)
    assert Path(module.__file__).resolve() == ROOT / 'fly_wordbrain/decoder.py'
    assert 'loss.backward()' in inspect.getsource(module.fit_decoder)
    dataset_path = ROOT / 'data/pilot/dataset.json'
    dataset = json.loads(dataset_path.read_text())
    manifest = json.loads((args.run / 'features/manifest.json').read_text())
    metrics = json.loads((args.run / 'decoder/metrics.json').read_text())
    assert manifest['completed'] and digest(dataset_path) == manifest['dataset_sha256']
    model = module.Decoder.load(args.run / 'decoder')
    assert model.config['feature_sha256'] == manifest['split_hashes']
    assert sum(p.numel() for p in model.model.parameters()) == 263168
    assert np.array_equal(model.scaler.mean, module.Standardizer.fit(
        module.load_split(args.run / 'features/train.npz', 1024)['X']).mean)
    report = {'source_sha256': actual, 'executed_module': module.__file__,
              'parameters': 263168, 'splits': {}}
    train_unknown_fraction = None
    for split in ('train', 'val', 'test'):
        path = args.run / 'features' / (split + '.npz')
        assert digest(path) == manifest['split_hashes'][split]
        data = module.load_split(path, 1024)
        module.validate_split_membership(data, dataset, split)
        expected_pairs = {(story['id'], p) for story in dataset['splits'][split]
                          for p in range(len(story['word_ids']) - 1)
                          if story['target_mask'][p + 1]}
        pairs = list(zip(data['story_ids'].tolist(), data['positions'].tolist()))
        assert len(pairs) == len(set(pairs)) and set(pairs) == expected_pairs
        assert max(data['positions']) <= 128
        logits = model.logits(data['X']).astype(np.float64)
        weight = model.model.weight.detach().numpy().astype(np.float64)
        bias = model.model.bias.detach().numpy().astype(np.float64)
        direct = model.scaler.transform(data['X']).astype(np.float64) @ weight.T + bias
        assert np.allclose(logits, direct, atol=2e-5, rtol=2e-5)
        shifted = direct - direct.max(axis=1, keepdims=True)
        loss = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(direct)), data['y']]
        reported = metrics['primary']['metrics'][split]['cross_entropy']
        assert abs(float(loss.mean()) - reported) < 1e-6
        unknown = data['y'] == 1
        if train_unknown_fraction is None:
            train_unknown_fraction = float(unknown.mean())
        entry = {'examples': len(loss), 'all_causal_rows_present_once': True,
                 'manual_cross_entropy': float(loss.mean()), 'reported_cross_entropy': reported,
                 'unknown_fraction': float(unknown.mean()),
                 'unknown_cross_entropy': float(loss[unknown].mean()),
                 'known_target_cross_entropy': float(loss[~unknown].mean()),
                 'cross_entropy_at_train_unknown_fraction': float(
                     train_unknown_fraction * loss[unknown].mean() +
                     (1 - train_unknown_fraction) * loss[~unknown].mean()),
                 'nonconstant_feature_columns': int((data['X'].std(axis=0) > 1e-8).sum()),
                 'logits_change_between_examples': bool(np.any(logits[0] != logits[-1]))}
        assert entry['nonconstant_feature_columns'] and entry['logits_change_between_examples']
        report['splits'][split] = entry
    report['passed'] = True
    output = args.run / 'checkpoint-audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
