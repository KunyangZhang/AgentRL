import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import torch
from .environment import dataset_manifest
from .policy import SmallPolicy
from .training import Config, evaluate, load_policy, train


def main():
    parser = argparse.ArgumentParser(description="AgentRL Tool Policy Lab")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--config", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--resume")
    evaluate_cmd = sub.add_parser("evaluate")
    evaluate_cmd.add_argument("--checkpoint", required=True)
    evaluate_cmd.add_argument("--output", required=True)
    evaluate_cmd.add_argument("--count", type=int, default=256)
    suite = sub.add_parser("suite")
    suite.add_argument("--output", required=True)
    suite.add_argument("--updates", type=int, default=100)
    suite.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        cfg = Config(**json.loads(Path(args.config).read_text()))
        manifest = dataset_manifest(max(cfg.train_count, cfg.eval_count))
        (output / "dataset.json").write_text(json.dumps(manifest, indent=2))
        train(cfg, output, args.resume)
    elif args.command == "evaluate":
        policy, cfg = load_policy(args.checkpoint)
        report = {s: evaluate(policy, s, args.count, cfg, trace_path=output/f"{s}-traces.jsonl")
                  for s in ("test", "stress")}
        (output/"evaluation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    else:
        (output / "dataset.json").write_text(json.dumps(dataset_manifest(), indent=2))
        report = {"scope": "synthetic tool dispatch; small neural policy; not LLM fine-tuning",
                  "protocol": "fixed final checkpoint; greedy held-out evaluation; no test-based model selection",
                  "runs": []}
        for seed in args.seeds:
            for shaped in (True, False):
                cfg = Config(seed=seed, updates=args.updates, shaped=shaped)
                name = f"seed-{seed}-{'shaped' if shaped else 'terminal'}"
                torch.manual_seed(seed)
                initial = SmallPolicy()
                before = {s: evaluate(initial, s, 256, cfg) for s in ("test", "stress")}
                policy = train(cfg, output/name)
                after = {s: evaluate(policy, s, 256, cfg, trace_path=output/name/f"{s}-traces.jsonl")
                         for s in ("test", "stress")}
                reloaded, _ = load_policy(output/name/"last.pt")
                assert evaluate(reloaded, "test", 256, cfg) == after["test"]
                report["runs"].append({"seed": seed, "shaped": shaped, "before": before, "after": after,
                                       "checkpoint_sha256": hashlib.sha256((output/name/"last.pt").read_bytes()).hexdigest()})
                (output/"report.json").write_text(json.dumps(report, indent=2))
        report["baselines"] = {b: {s: evaluate(None, s, 256, Config(), baseline=b) for s in ("test", "stress")}
                               for b in ("random", "oracle")}
        report["runtime"] = {"python": platform.python_version(), "torch": str(torch.__version__),
                             "device": "cpu", "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                             "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))}
        (output/"report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
