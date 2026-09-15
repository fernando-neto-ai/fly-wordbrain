#!/usr/bin/env python3
"""Independently audit the anchored 8192/1024/1024 word corpus from raw HF bytes.

No graph is loaded, no model is instantiated, and no network/GPU work occurs.
Tokenization, story identities, vocabulary and causal targets are reconstructed
here; the pair/trainer helpers are checked against those independent results.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPLITS = ('train', 'val', 'test')
COUNTS = {'train': 8192, 'val': 1024, 'test': 1024}
PILOT_COUNTS = {'train': 128, 'val': 32, 'test': 32}
PILOT_SHA256 = '740fac37a10dc6df82a74a7613b2888e10dc282e2de82c2281905ebebd94c43f'
SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>']
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)*")
ROW_FILENAME = re.compile(r'^train_rows_(\d{7})_(\d+)\.json$')
INFO_URL = 'https://huggingface.co/api/datasets/roneneldan/TinyStories'


class AuditFailure(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tokenize(text):
    # Independently implements the declared word contract; do not call builder.
    return TOKEN_PATTERN.findall(text.replace('\u2018', "'").replace('\u2019', "'").lower())


def verify_cache_file(path):
    body = path.read_bytes()
    receipt_path = path.with_suffix('.receipt.json')
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get('bytes') == len(body), 'Source byte-count mismatch: ' + str(path))
    require(receipt.get('sha256') == sha256(body), 'Source SHA256 mismatch: ' + str(path))
    return json.loads(body), receipt, {'file': path.name, 'sha256': sha256(body),
                                     'bytes': len(body), 'receipt_sha256': file_hash(receipt_path)}


def read_raw_source(directory):
    info, info_receipt, info_hash = verify_cache_file(directory / 'dataset_info.json')
    require(info_receipt['url'] == INFO_URL, 'Unexpected source repository URL')
    require(info.get('id') == 'roneneldan/TinyStories', 'Unexpected source repository ID')
    revision = info.get('sha', '')
    require(re.fullmatch(r'[0-9a-f]{40}', revision) is not None, 'Missing pinned source revision')
    rows, receipts, files, page_sizes = {}, {info_receipt['url']: info_receipt}, [info_hash], {}
    for path in sorted(directory.iterdir()):
        match = ROW_FILENAME.fullmatch(path.name)
        if match is None:
            continue
        offset, length = map(int, match.groups())
        page, receipt, source_hash = verify_cache_file(path)
        parsed = urlparse(receipt['url'])
        require(parsed.scheme == 'https' and parsed.netloc == 'datasets-server.huggingface.co'
                and parsed.path == '/rows', 'Unexpected HF rows endpoint: ' + str(path))
        require(parse_qs(parsed.query) == {'dataset': ['roneneldan/TinyStories'], 'config': ['default'],
                                         'split': ['train'], 'offset': [str(offset)], 'length': [str(length)]},
                'Source page query does not match its filename: ' + str(path))
        require(receipt.get('x_revision') == revision, 'Source page/repository revision mismatch: ' + str(path))
        require(receipt['url'] not in receipts, 'Duplicate source receipt URL')
        page_rows = page.get('rows')
        require(isinstance(page_rows, list) and 0 < len(page_rows) <= length, 'Empty/malformed source rows page')
        for index, row in enumerate(page_rows):
            row_id = row.get('row_idx')
            require(type(row_id) is int and row_id == offset + index, 'Nonsequential or invalid HF source row index')
            identity = 'tinystories/train/' + str(row_id)
            require(identity not in rows, 'Raw source row ID duplicated across pages: ' + identity)
            text = row['row']['text']
            require(isinstance(text, str), 'Nontext HF story: ' + identity)
            words = tokenize(text)
            rows[identity] = {'id': identity, 'text': text, 'full_words': words,
                'words': words[:128], 'source_text_sha256': sha256(text.encode('utf-8')),
                'normalized_story_sha256': sha256(' '.join(words).encode('utf-8')),
                'retained_words_sha256': sha256(' '.join(words[:128]).encode('utf-8')),
                'original_word_count': len(words), 'ended_naturally': len(words) <= 128,
                'page_url': receipt['url']}
        receipts[receipt['url']] = receipt
        page_sizes[receipt['url']] = len(page_rows)
        files.append(source_hash)
    require(bool(rows), 'No raw HF rows found in source cache')
    nonempty = [row for row in rows.values() if row['full_words']]
    summary = {'directory': str(directory.resolve()), 'revision': revision,
               'rows': len(rows), 'pages': len(page_sizes), 'nonempty_rows': len(nonempty),
               'unique_normalized_stories': len({r['normalized_story_sha256'] for r in nonempty}),
               'unique_retained_prefixes': len({r['retained_words_sha256'] for r in nonempty}),
               'verified_bytes': sum(f['bytes'] for f in files),
               'source_receipts_digest': sha256(json.dumps(files, sort_keys=True).encode()),
               'files': files}
    return rows, receipts, page_sizes, summary


def verify_source_metadata(dataset, cache, label):
    rows, receipts, page_sizes, summary = cache
    source = dataset['metadata']['source']
    require(source.get('dataset_id') == 'roneneldan/TinyStories', label + ': wrong source dataset ID')
    require(source.get('revision') == summary['revision'], label + ': declared revision differs from raw cache')
    require(source.get('source_split') == 'train', label + ': wrong official source split')
    declared = source.get('receipts', [])
    require(declared and len({r['url'] for r in declared}) == len(declared), label + ': missing/duplicate declared receipts')
    for receipt in declared:
        require(receipt['url'] in receipts and receipts[receipt['url']] == receipt,
                label + ': dataset receipt differs from verified raw-source receipt')
    declared_urls = {r['url'] for r in declared}
    require(INFO_URL in declared_urls, label + ': source repository receipt not declared')
    declared_rows = sum(count for url, count in page_sizes.items() if url in declared_urls)
    require(source.get('source_rows_read') == declared_rows, label + ': declared raw-row count mismatch')
    selected_urls = {rows[record['id']]['page_url'] for split in SPLITS for record in dataset['splits'][split]}
    require(selected_urls <= declared_urls, label + ': selected story comes from an undeclared source page')
    return {'declared_rows': declared_rows, 'declared_pages': len(declared_urls) - 1,
            'selected_source_pages': len(selected_urls), 'extra_verified_cache_pages': len(page_sizes) - len(declared_urls) + 1}


def verify_record(record, raw, vocabulary, label):
    identity = record['id']
    require(identity in raw, label + ': selected ID absent from raw source: ' + identity)
    expected = raw[identity]
    require(bool(expected['words']), label + ': empty selected story: ' + identity)
    for key in ('words', 'source_text_sha256', 'normalized_story_sha256', 'retained_words_sha256',
                'original_word_count', 'ended_naturally'):
        require(record.get(key) == expected[key], label + ': raw-source mismatch for ' + identity + '/' + key)
    require(type(record['ended_naturally']) is bool and type(record['original_word_count']) is int,
            label + ': invalid source-length/EOS metadata types')
    word_to_id = vocabulary
    lexical = [word_to_id.get(word, 1) for word in expected['words']]
    ids = [2] + lexical + ([3] if expected['ended_naturally'] else [])
    mask = [False] + [True] * (len(ids) - 1)
    require(record.get('word_ids') == ids and all(type(i) is int for i in record['word_ids']),
            label + ': word IDs/BOS/EOS differ from raw text and vocabulary: ' + identity)
    require(record.get('target_mask') == mask and all(type(m) is bool for m in record['target_mask']),
            label + ': target masks differ from independent token contract: ' + identity)
    return ids


def expected_causal_rows(ids):
    result = []
    for p in range(len(ids) - 1):
        second_valid = p + 2 < len(ids) and ids[p + 1] != 3
        result.append((p, ids[p - 1] if p > 0 else 2, ids[p], ids[p + 1],
                       ids[p + 2] if second_valid else 0, second_valid))
    return result


def verify_dataset_records(dataset, raw, expected_counts, label, crosscheck_helpers):
    require(set(dataset['splits']) == set(SPLITS), label + ': unexpected split names')
    vocabulary = dataset['vocabulary']
    require(len(vocabulary) == 1024 and vocabulary[:4] == SPECIAL and len(set(vocabulary)) == 1024,
            label + ': vocabulary size/specials/uniqueness mismatch')
    word_to_id = {word: index for index, word in enumerate(vocabulary)}
    require(dataset['metadata']['max_words'] == 128 and dataset['metadata']['vocabulary_fit_split'] == 'train',
            label + ': context or vocabulary-fit contract mismatch')
    frequency = Counter(word for record in dataset['splits']['train'] for word in record['words'])
    independent_vocab = SPECIAL + sorted(frequency, key=lambda w: (-frequency[w], w))[:1020]
    require(vocabulary == independent_vocab, label + ': vocabulary is not exact train-only frequency ranking')
    if crosscheck_helpers:
        from fly_wordbrain.pair_extract import story_rows
        from fly_wordbrain.plastic_train import causal_rows
    all_ids, all_full, all_prefix = set(), set(), set()
    sets, stats, story_lookup = {}, {}, {}
    for split in SPLITS:
        records = dataset['splits'][split]
        require(len(records) == expected_counts[split], label + ': wrong story count in ' + split)
        split_ids, split_full, split_prefix = set(), set(), set()
        totals = Counter()
        unknown = Counter()
        for record in records:
            identity = record['id']
            ids = verify_record(record, raw, word_to_id, label)
            require(identity not in all_ids, label + ': repeated selected ID: ' + identity)
            require(record['normalized_story_sha256'] not in all_full, label + ': normalized story duplicate')
            require(record['retained_words_sha256'] not in all_prefix, label + ': retained prefix duplicate')
            all_ids.add(identity); all_full.add(record['normalized_story_sha256']); all_prefix.add(record['retained_words_sha256'])
            split_ids.add(identity); split_full.add(record['normalized_story_sha256']); split_prefix.add(record['retained_words_sha256'])
            story_lookup[identity] = (split, record)
            expected = expected_causal_rows(ids)
            if crosscheck_helpers:
                pairs = [(r['position'], r['previous_id'], r['current_id'], *r['targets'], r['target_mask'][1])
                         for r in story_rows(record)]
                require(pairs == expected, 'Pair extraction causality differs at ' + identity)
                require(all(r['target_mask'][0] is True for r in story_rows(record)), 'Pair head1 mask mismatch')
                trainer = list(causal_rows(record))
                require(trainer == [row[1:] for row in expected], 'Trainer causal targets differ at ' + identity)
            totals.update(stories=1, lexical_words=len(record['words']),
                          original_lexical_words=record['original_word_count'],
                          naturally_ended_stories=int(record['ended_naturally']),
                          truncated_stories=int(not record['ended_naturally']),
                          next1_targets=len(expected), next2_targets=sum(r[-1] for r in expected),
                          observed_positions=len(expected), eos_targets=int(record['ended_naturally']))
            unknown.update(word for word in record['words'] if word not in word_to_id)
        totals['unknown_words'] = sum(unknown.values())
        totals['unknown_word_types'] = len(unknown)
        totals['missing_second_horizon_rows'] = totals['observed_positions'] - totals['next2_targets']
        coverage = dataset['metadata']['coverage'][split]
        for key, expected in (('stories', totals['stories']), ('lexical_words', totals['lexical_words']),
                              ('unknown_words', totals['unknown_words']), ('unknown_word_types', len(unknown)),
                              ('truncated_stories', totals['truncated_stories']), ('supervised_targets', totals['next1_targets'])):
            require(coverage.get(key) == expected, label + ': coverage metadata mismatch ' + split + '/' + key)
        known_fraction = 1 - totals['unknown_words'] / totals['lexical_words']
        require(abs(coverage['known_word_fraction'] - known_fraction) < 1e-12, label + ': known-word fraction mismatch')
        stats[split] = {**dict(totals), 'known_word_fraction': known_fraction}
        sets[split] = {'ids': split_ids, 'normalized': split_full, 'prefixes': split_prefix}
    ranking_receipt = sha256(json.dumps(sorted(frequency.items()), ensure_ascii=False, separators=(',', ':')).encode())
    return {'splits': stats, 'total_stories': len(all_ids), 'unique_ids': len(all_ids),
            'unique_normalized_stories': len(all_full), 'unique_retained_prefixes': len(all_prefix),
            'vocabulary': {'size': len(vocabulary), 'fit_split': 'train', 'unique_train_lexical_types': len(frequency),
                           'ranking': 'descending selected-training-prefix count, alphabetical tie break',
                           'train_frequency_sha256': ranking_receipt,
                           'vocabulary_sha256': sha256(json.dumps(vocabulary, ensure_ascii=False).encode())}}, sets, story_lookup


def audit_padding(dataset, batch_size=8):
    from fly_wordbrain.plastic_train import collate_stories
    import torch
    torch.set_num_threads(1)
    totals = {}
    for split in SPLITS:
        counts = Counter()
        records = dataset['splits'][split]
        for start in range(0, len(records), batch_size):
            stories = records[start:start + batch_size]
            expected = [expected_causal_rows(story['word_ids']) for story in stories]
            lengths = [len(rows) for rows in expected]
            shape = (len(stories), max(lengths))
            previous = np.full(shape, 2, np.int64); current = np.full(shape, 2, np.int64)
            targets = np.zeros(shape + (2,), np.int64); masks = np.zeros(shape + (2,), bool)
            active = np.zeros(shape, bool)
            for i, rows in enumerate(expected):
                for p, prev, curr, first, second, second_valid in rows:
                    previous[i, p] = prev; current[i, p] = curr
                    targets[i, p] = first, second
                    masks[i, p] = True, second_valid
                    active[i, p] = True
            actual = collate_stories(stories, device='cpu')
            for key, value in (('previous', previous), ('current', current), ('targets', targets),
                               ('target_mask', masks), ('active', active), ('lengths', np.asarray(lengths))):
                require(np.array_equal(getattr(actual, key).numpy(), value), 'Trainer padding/target mismatch: ' + split + '/' + key)
            require(actual.story_ids == [str(s['id']) for s in stories], 'Trainer padding reordered story identities')
            counts.update(batches=1, active_positions=int(active.sum()), padding_positions=int((~active).sum()),
                          scored_next1=int(masks[..., 0].sum()), scored_next2=int(masks[..., 1].sum()),
                          unscored_second_tail_rows=int((active & ~masks[..., 1]).sum()))
        totals[split] = dict(counts)
    return {'batch_size': batch_size, 'ordering': 'Stored split order, consecutive batches; training reshuffling may change the amount of padding',
            'checked': 'BOS/BOS padded inputs, zero padded targets, false masks and activity, exact lengths/IDs; no recurrent simulation',
            'splits': totals}


def overlap_matrix(left, right):
    return {kind: {a: {b: len(left[a][kind] & right[b][kind]) for b in SPLITS} for a in SPLITS}
            for kind in ('ids', 'normalized', 'prefixes')}


def audit(dataset_path, pilot_path):
    dataset = json.loads(dataset_path.read_text()); pilot = json.loads(pilot_path.read_text())
    require(file_hash(pilot_path) == PILOT_SHA256, 'Pilot differs from the original verified192-story dataset')
    expanded_cache = read_raw_source(dataset_path.parent / 'source')
    pilot_cache = read_raw_source(pilot_path.parent / 'source')
    require(expanded_cache[3]['revision'] == pilot_cache[3]['revision'], 'Expanded source revision differs from pilot source')
    expanded, sets, selected = verify_dataset_records(dataset, expanded_cache[0], COUNTS, 'expanded', True)
    original, pilot_sets, pilot_selected = verify_dataset_records(pilot, pilot_cache[0], PILOT_COUNTS, 'pilot', False)
    expanded_source = verify_source_metadata(dataset, expanded_cache, 'expanded')
    pilot_source = verify_source_metadata(pilot, pilot_cache, 'pilot')
    anchors = {}
    for split in SPLITS:
        expected_ids = pilot_sets[split]['ids']
        retained = expected_ids & sets[split]['ids']
        require(retained == expected_ids, 'A pilot anchor disappeared or changed split: ' + split)
        for identity in retained:
            require(pilot_cache[0][identity]['source_text_sha256'] == expanded_cache[0][identity]['source_text_sha256'],
                    'Pilot/expanded raw source differs at shared selected ID: ' + identity)
            for key in ('words', 'source_text_sha256', 'normalized_story_sha256', 'retained_words_sha256',
                        'original_word_count', 'ended_naturally'):
                require(pilot_selected[identity][1][key] == selected[identity][1][key], 'Anchored content changed: ' + identity + '/' + key)
        anchors[split] = {'pilot_stories': len(expected_ids), 'positively_matched_same_split': len(retained),
                          'new_stories': len(sets[split]['ids'] - expected_ids),
                          'anchor_ids_sha256': sha256(json.dumps(sorted(retained)).encode())}
    overlaps = overlap_matrix(sets, pilot_sets)
    for kind, matrix in overlaps.items():
        for expanded_split in SPLITS:
            for pilot_split in SPLITS:
                require(matrix[expanded_split][pilot_split] == (PILOT_COUNTS[pilot_split] if expanded_split == pilot_split else 0),
                        'Unexpected expanded/pilot overlap matrix: ' + kind)
    raw_shared_ids = set(expanded_cache[0]) & set(pilot_cache[0])
    for identity in raw_shared_ids:
        require(expanded_cache[0][identity]['source_text_sha256'] == pilot_cache[0][identity]['source_text_sha256'],
                'Shared raw-source ID changed contents: ' + identity)
    raw_overlap = {'shared_row_ids': len(raw_shared_ids), 'all_shared_raw_text_hashes_match': True}
    for key in ('normalized_story_sha256', 'retained_words_sha256'):
        raw_overlap[key] = len({r[key] for r in expanded_cache[0].values()} & {r[key] for r in pilot_cache[0].values()})
    padding = audit_padding(dataset)
    for split in SPLITS:
        require(padding['splits'][split]['scored_next1'] == expanded['splits'][split]['next1_targets']
                and padding['splits'][split]['scored_next2'] == expanded['splits'][split]['next2_targets'],
                'Padding aggregate forecast count mismatch')
    manifest_path = dataset_path.with_name('manifest.json')
    manifest = json.loads(manifest_path.read_text())
    declared_anchor_ids = dataset['metadata'].get('pilot_anchor_ids', {})
    require(dataset['metadata'].get('pilot_anchor_counts') == PILOT_COUNTS, 'Declared pilot anchor counts mismatch')
    require(dataset['metadata'].get('pilot_dataset_sha256') == PILOT_SHA256, 'Declared pilot dataset SHA256 mismatch')
    for split in SPLITS:
        declared = declared_anchor_ids.get(split, [])
        require(len(declared) == PILOT_COUNTS[split] and set(declared) == pilot_sets[split]['ids'],
                'Declared anchor IDs differ from positively verified pilot anchors: ' + split)
    receipts_bytes = (json.dumps(dataset['metadata']['source']['receipts'], sort_keys=True,
                                 ensure_ascii=False, indent=2) + '\n').encode()
    expected_manifest = {
        'dataset_sha256': file_hash(dataset_path), 'pilot_dataset_sha256': PILOT_SHA256,
        'source_dataset': 'roneneldan/TinyStories', 'source_revision': expanded_cache[3]['revision'],
        'split_counts': COUNTS, 'source_receipts_sha256': sha256(receipts_bytes),
        'builder_source_sha256': file_hash(ROOT / 'fly_wordbrain/expanded_data.py'),
        'base_data_source_sha256': file_hash(ROOT / 'fly_wordbrain/data.py')}
    for key, expected in expected_manifest.items():
        require(manifest.get(key) == expected, 'Expanded manifest mismatch: ' + key)
    return {'passed': True, 'dataset_sha256': file_hash(dataset_path), 'pilot_dataset_sha256': file_hash(pilot_path),
            'manifest_sha256': file_hash(manifest_path), 'manifest_fields_verified': list(expected_manifest),
            'expanded': expanded, 'pilot': original,
            'anchors': anchors, 'expanded_vs_pilot_overlap': overlaps,
            'expanded_internal_overlap': overlap_matrix(sets, sets), 'raw_cache_overlap': raw_overlap,
            'source_caches': {'expanded': {**expanded_cache[3], **expanded_source}, 'pilot': {**pilot_cache[3], **pilot_source}},
            'trainer_padding': padding,
            'interpretation': 'Raw text, complete/prefix hashes, anchored split membership, train-only vocabulary and all causal forecast rows verified. Old evaluation anchors are reused explicitly; this is not a fresh independent test set.',
            'source_sha256': {name: file_hash(ROOT / 'fly_wordbrain' / name) for name in
                               ('expanded_data.py', 'data.py', 'pair_extract.py', 'plastic_train.py')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        report = audit(args.dataset, args.pilot)
    except Exception as error:
        report = {'passed': False, 'error': str(error), 'exception_type': type(error).__name__}
    report.update(auditor_sha256=file_hash(__file__), elapsed_seconds=time.perf_counter() - started,
                  dataset_path=str(args.dataset.resolve()), pilot_path=str(args.pilot.resolve()),
                  graph_loaded=False, neural_workload=False, network_requests=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.partial')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in ('passed', 'elapsed_seconds')}, allow_nan=False), flush=True)
    if not report['passed']:
        raise AuditFailure(report['error'])
    print(json.dumps({'audit': str(args.output), 'splits': report['expanded']['splits'],
                      'anchors': report['anchors'], 'raw_cache_overlap': report['raw_cache_overlap']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
