"""One-shot prompt-only control: prevents attributing format adaptation to reasoning."""
import argparse
import json
from pathlib import Path
from .train import Config, Policy, evaluate

# Fixed synthetic demonstration; no Spider train/dev/test answers used.
DEMONSTRATION = [
    {"role":"user", "content":"Database schema:\nCREATE TABLE products (id INTEGER, price REAL);\nQuestion: How many products are there?"},
    {"role":"assistant", "content":'{"tool":"query","sql":"SELECT COUNT(*) FROM products"}'},
    {"role":"user", "content":'Tool result: {"ok":true,"columns":["COUNT(*)"],"rows":[[7]],"row_count":1}'},
    {"role":"assistant", "content":'{"tool":"submit","sql":"SELECT COUNT(*) FROM products"}'},
]


class PromptPolicy(Policy):
    def prompt_ids(self, messages):
        return super().prompt_ids(messages[:1]+DEMONSTRATION+messages[1:])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text()))
    policy = PromptPolicy(cfg)
    cases = json.loads((Path(cfg.data)/'test.json').read_text())
    result = evaluate(policy, cases, cfg, Path(args.output)/'prompt-only-test.jsonl')
    (Path(args.output)/'prompt-only-report.json').write_text(json.dumps(result, indent=2))
