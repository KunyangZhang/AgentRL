"""Deterministic, database-disjoint protocol; manifest frozen before model evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from .environment import execute, schema


def prepare(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("refusing to replace a frozen split")
    train = json.loads((source / "train_spider.json").read_text())
    validation = json.loads((source / "dev.json").read_text())
    db_root = source / "database"
    def eligible(rows, name):
        selected, excluded = [], {}
        for i, r in enumerate(rows):
            db = db_root / r["db_id"] / (r["db_id"] + ".sqlite")
            reason = None
            if not db.exists():
                reason = "missing_database"
            elif len("\n".join(schema(db).values())) > 4800:
                reason = "schema_over_4800_chars"
            elif len(r["query"]) > 500:
                reason = "gold_over_500_chars"
            else:
                result = execute(db, r["query"])
                if not result["ok"]:
                    reason = "gold_execution_failed"
                elif not result["rows"]:
                    reason = "empty_gold_result"
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            selected.append({"uid": f"{name}-{i}", "db_id": r["db_id"],
                             "question": r["question"], "query": r["query"],
                             "ordered": bool(r["sql"].get("orderBy"))})
        return selected, excluded
    train, train_excluded = eligible(train, "spider-train")
    validation, val_excluded = eligible(validation, "spider-dev")
    rng = random.Random(20260926)
    dbs = sorted({x["db_id"] for x in validation})
    rng.shuffle(dbs)
    dev_dbs = set(dbs[:len(dbs)//2])
    dev = [x for x in validation if x["db_id"] in dev_dbs]
    test = [x for x in validation if x["db_id"] not in dev_dbs]
    splits = {}
    for name, rows, n in [("train", train, 256), ("dev", dev, 24), ("test", test, 64)]:
        rng.shuffle(rows)
        # Deduplicate exact question/database pairs before sampling.
        seen, kept = set(), []
        for row in rows:
            key = (row["db_id"], row["question"].strip().lower())
            if key not in seen:
                seen.add(key)
                kept.append(row)
        splits[name] = kept[:n]
        assert len(splits[name]) == n
        (output / f"{name}.json").write_text(json.dumps(splits[name], ensure_ascii=False, indent=2))
    sets = {s: {x["db_id"] for x in rows} for s, rows in splits.items()}
    assert not sets["train"] & (sets["dev"] | sets["test"])
    assert not sets["dev"] & sets["test"]
    manifest = {"sampling_seed": 20260926, "dataset": "Spider 1.0, xlangai/spider",
                "metric": "single-database execution result match; not official test-suite accuracy",
                "filters": {"train_excluded": train_excluded, "official_dev_excluded": val_excluded},
                "splits": {s: {"count": len(rows), "databases": sorted(sets[s]),
                           "sha256": hashlib.sha256((output/f"{s}.json").read_bytes()).hexdigest()} for s, rows in splits.items()},
                "database_sha256": {db: hashlib.sha256((db_root/db/(db+'.sqlite')).read_bytes()).hexdigest()
                                    for db in sorted(set.union(*sets.values()))}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "database_sha256"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("output")
    args = p.parse_args()
    prepare(args.source, args.output)
