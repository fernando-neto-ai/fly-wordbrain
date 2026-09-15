#!/usr/bin/env python3
"""Losslessly compact verified replay JSON for the static demo; no inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_verified(path, expected):
    if digest(path) != expected:
        raise ValueError('Replay source hash mismatch: ' + str(path))
    return json.loads(path.read_text())


def compact(path, value):
    body = json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n'
    if json.loads(body) != value:
        raise ValueError('Packaging changed decoded data')
    path.write_text(body)
    return digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--story-id', action='append', default=[])
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or (args.output / 'manifest.json').exists():
        raise ValueError('Choose a fresh output, separate from the original export')
    manifest_path = args.source / 'manifest.json'
    source_digest = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    stories = {story['id']: story for story in manifest['stories']}
    chosen = args.story_id or list(stories)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen).issubset(stories):
        raise ValueError('Choose a nonempty set of distinct recorded stories')
    names = [manifest['neuron_layout_url'], *(stories[key]['trace_url'] for key in chosen)]
    if any(Path(name).name != name for name in names):
        raise ValueError('Replay assets must have simple filenames')
    args.output.mkdir(parents=True, exist_ok=True)
    layout = read_verified(args.source / names[0], manifest['neuron_layout_sha256'])
    manifest['neuron_layout_sha256'] = compact(args.output / names[0], layout)
    retained, original_hashes = [], {names[0]: digest(args.source / names[0])}
    for key in chosen:
        entry = dict(stories[key])
        trace = read_verified(args.source / entry['trace_url'], entry['sha256'])
        if not trace['frames'] or not trace['generated_ids']:
            raise ValueError('Empty recorded story')
        original_hashes[entry['trace_url']] = entry['sha256']
        entry['sha256'] = compact(args.output / entry['trace_url'], trace)
        retained.append(entry)
    manifest['stories'] = retained
    manifest['packaging'] = {
        'method': 'Lossless JSON minification; decoded texts and neuron values are unchanged.',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_manifest_sha256': source_digest,
        'source_file_sha256': original_hashes,
        'display_order': chosen,
        'curation': 'Selected recorded demonstrations, not a quality benchmark.',
    }
    compact(args.output / 'manifest.json', manifest)
    print(json.dumps({'output': str(args.output), 'stories': chosen,
                      'bytes': sum(path.stat().st_size for path in args.output.glob('*.json'))}))


if __name__ == '__main__':
    main()
