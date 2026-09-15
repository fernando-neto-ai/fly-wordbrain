#!/usr/bin/env python3
"""Build an auditable TinyStories split for the reconstructed ngxson recipe."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import sys
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly_wordbrain.data import DATASET_ID, _fetch_json
from prepare_ngxson import DEFAULT_MODEL, FILES, REVISION, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "ngxson-tinystories-v1")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    destination = args.output / "dataset.json"
    if destination.exists():
        parser.error("Dataset already exists; use a fresh output directory")
    tokenizer_receipt = verify(args.model / "tokenizer.json", FILES["tokenizer.json"])
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    assert tokenizer.get_vocab_size() == 1024
    assert [tokenizer.token_to_id(x) for x in ["<pad>", "<s>", "</s>"]] == [0, 1, 2]
    cache = args.output / "source"
    info, info_receipt = _fetch_json(f"https://huggingface.co/api/datasets/{DATASET_ID}", cache / "dataset_info.json")
    revision = info["sha"]
    receipts = [info_receipt]
    seen = set()
    stats = {}

    def fetch_page(split, offset):
        url = "https://datasets-server.huggingface.co/rows?" + urlencode({
            "dataset": DATASET_ID, "config": "default", "split": split,
            "offset": offset, "length": 100,
        })
        page, receipt = _fetch_json(url, cache / f"{split}_{offset:07d}.json")
        if receipt.get("x_revision") != revision:
            raise ValueError("Dataset revision differs from repository receipt")
        return page, receipt

    def select(split, needed):
        selected = []
        discarded_long = discarded_duplicate = examined = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0, 20000, 400):
                pages = list(pool.map(lambda offset: fetch_page(split, offset), range(start, start + 400, 100)))
                for page, receipt in pages:
                    receipts.append(receipt)
                    for row in page["rows"]:
                        if len(selected) == needed:
                            break
                        examined += 1
                        text = row["row"]["text"]
                        normal = " ".join(text.casefold().split())
                        digest = hashlib.sha256(normal.encode()).hexdigest()
                        if not normal or digest in seen:
                            discarded_duplicate += 1
                            continue
                        ids = [1, *tokenizer.encode(text, add_special_tokens=False).ids, 2]
                        if len(ids) > 320:
                            discarded_long += 1
                            continue
                        seen.add(digest)
                        selected.append({
                            "id": f"tinystories/{split}/{row['row_idx']}", "text": text, "ids": ids,
                            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "normalized_sha256": digest,
                        })
                if len(selected) == needed:
                    break
        if len(selected) != needed:
            raise ValueError(f"Only {len(selected)} qualifying {split} stories within bound")
        stats[split] = {"examined": examined, "selected": len(selected),
                        "discarded_long": discarded_long, "discarded_duplicate_or_empty": discarded_duplicate}
        print(split, stats[split], flush=True)
        return selected

    train = select("train", 1000)
    heldout = select("validation", 200)
    random.Random(args.seed).shuffle(heldout)
    data = {"train": train, "validation": heldout[:100], "test": heldout[100:]}
    data["provenance"] = {
        "dataset": DATASET_ID, "revision": revision, "receipts": receipts,
        "model_revision": REVISION, "tokenizer": tokenizer_receipt,
        "seed": args.seed, "source_stats": stats, "max_tokens_including_bos_eos": 320,
        "token_counts": {key: sum(len(r["ids"]) - 1 for r in rows) for key, rows in data.items()},
        "selection": f"First 1000 qualifying official training stories; first 200 qualifying official validation stories split 100/100 by seed{args.seed} shuffle",
        "deduplication": "Whitespace-normalized casefolded full-text SHA256 across all three splits; not semantic deduplication",
        "tokenizer_policy": "Reuse the exact released 1024-token BPE; do not retrain it on the new split",
        "replication_limit": "Author did not release sample IDs, seeds, or training script. This is a documented recipe reconstruction, not the author's exact corpus. Released tokenizer fitting-set overlap with these held-out texts is unknown.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()
    destination.write_bytes(body)
    (args.output / "manifest.json").write_text(json.dumps({
        "dataset_sha256": hashlib.sha256(body).hexdigest(),
        "splits": {key: len(data[key]) for key in ["train", "validation", "test"]},
        "predicted_tokens": data["provenance"]["token_counts"],
    }, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
