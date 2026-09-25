"""On-policy grouped rollouts using the upstream AgentRL GRPO and PPO kernels."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import torch
from agentrl.trainer.algorithms.core_algos import compute_grpo_outcome_advantage, compute_policy_loss
from .environment import make_cases
from .policy import LanguagePolicy, SmallPolicy, trainable_state
from .rollout import collect, metrics


@dataclass
class Config:
    seed: int = 17
    backend: str = "small"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    updates: int = 100
    batch_size: int = 16
    group_size: int = 8
    ppo_epochs: int = 2
    micro_batch: int = 256
    learning_rate: float = 0.003
    clip: float = 0.2
    kl_coef: float = 0.01
    entropy_coef: float = 0.01
    shaped: bool = True
    train_count: int = 256
    eval_count: int = 256
    eval_every: int = 20

    def validate(self):
        if self.backend not in ("small", "lora") or self.group_size < 2:
            raise ValueError("backend must be small/lora and group_size >= 2")
        for key in ("updates", "batch_size", "ppo_epochs", "micro_batch", "train_count", "eval_count", "eval_every"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")


def signature(config):
    values = asdict(config)
    values.pop("updates")  # increasing total updates is the only permitted resume change
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def prepare_batch(episodes):
    """Flatten ONLY policy actions. Tool observations never become training targets."""
    groups = defaultdict(list)
    for i, ep in enumerate(episodes):
        groups[ep.uid].append(i)
    advantages = [None] * len(episodes)
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError("GRPO requires >=2 trajectories for each prompt")
        rewards = [torch.tensor(episodes[i].rewards) for i in indices]
        masks = [torch.ones_like(r) for r in rewards]
        adv, _ = compute_grpo_outcome_advantage(rewards, masks)
        for i, a in zip(indices, adv):
            advantages[i] = a
    return {"observations": [x for e in episodes for x in e.observations],
            "actions": torch.tensor([x for e in episodes for x in e.actions]),
            "old": torch.tensor([x for e in episodes for x in e.old_log_probs]),
            "advantages": torch.cat(advantages)}


def optimize(policy, reference, optimizer, batch, cfg):
    device = next(policy.parameters()).device
    total = len(batch["actions"])
    stats = defaultdict(float)
    for _ in range(cfg.ppo_epochs):
        optimizer.zero_grad()
        for start in range(0, total, cfg.micro_batch):
            end = min(start + cfg.micro_batch, total)
            observations = batch["observations"][start:end]
            logits = policy(observations)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(batch["actions"][start:end].to(device))
            with torch.no_grad():
                ref_logits = reference(observations) if reference is not None else policy(observations, reference=True)
            ref_logp = ref_logits.log_softmax(-1)
            # Exact categorical KL over the tool menu, not a sampled token estimator.
            kl = (dist.probs * (logits.log_softmax(-1) - ref_logp)).sum(-1).mean()
            pg, clipped, approx_kl, _ = compute_policy_loss(
                batch["old"][start:end].to(device), log_prob,
                batch["advantages"][start:end].to(device), torch.ones_like(log_prob),
                cliprange=cfg.clip)
            loss = pg + cfg.kl_coef * kl - cfg.entropy_coef * dist.entropy().mean()
            # Weight by ACTION count; prevents short last microbatch being overweighted.
            weight = (end-start)/total
            (weight * loss).backward()
            for name, value in {"loss": loss, "reference_kl": kl, "clip_fraction": clipped}.items():
                stats[name] += float(value.detach()) * weight / cfg.ppo_epochs
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient")
        optimizer.step()
    return dict(stats)


def save_checkpoint(path, policy, reference, optimizer, cfg, step, rng):
    path = Path(path)
    state = {"config": asdict(cfg), "signature": signature(cfg), "step": step,
             "policy": trainable_state(policy),
             "reference": reference.state_dict() if reference is not None else None,
             "optimizer": optimizer.state_dict(), "python_rng": rng.getstate(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def train(cfg, output, resume=None):
    cfg.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError("output contains a run; use --resume or a fresh directory")
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    policy = SmallPolicy() if cfg.backend == "small" else LanguagePolicy(cfg.model, cfg.revision)
    reference = deepcopy(policy).requires_grad_(False) if cfg.backend == "small" else None
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=cfg.learning_rate, weight_decay=0)
    start = 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["signature"] != signature(cfg):
            raise ValueError("resume configuration mismatch")
        policy.load_state_dict(state["policy"], strict=cfg.backend == "small")
        if reference is not None:
            reference.load_state_dict(state["reference"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["step"]
        log = output / "train.jsonl"
        if log.exists():
            # A crash may leave metrics newer than the durable checkpoint.
            retained = [line for line in log.read_text().splitlines()
                        if json.loads(line)["step"] <= start]
            log.write_text("\n".join(retained) + ("\n" if retained else ""))
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    cases = make_cases("train", cfg.train_count)
    if cfg.batch_size > len(cases):
        raise ValueError("batch_size exceeds unique training cases")
    for step in range(start + 1, cfg.updates + 1):
        selected = rng.sample(cases, cfg.batch_size)
        episodes = collect(policy, [c for c in selected for _ in range(cfg.group_size)], shaped=cfg.shaped)
        record = {"step": step, **metrics(episodes),
                  **optimize(policy, reference, optimizer, prepare_batch(episodes), cfg)}
        if step % cfg.eval_every == 0 or step == cfg.updates:
            record["dev"] = evaluate(policy, "dev", cfg.eval_count, cfg)
            save_checkpoint(output / "last.pt", policy, reference, optimizer, cfg, step, rng)
            print(json.dumps(record), flush=True)
        with (output / "train.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
    return policy


def evaluate(policy, split, count, cfg, baseline=None, trace_path=None):
    episodes = []
    cases = make_cases(split, count)
    # Bound LM inference memory independently from dataset size.
    chunk = 8 if cfg.backend == "lora" else 256
    for start in range(0, count, chunk):
        episodes.extend(collect(policy, cases[start:start+chunk], shaped=cfg.shaped,
                                greedy=True, baseline=baseline, seed=cfg.seed+start))
    if trace_path:
        with Path(trace_path).open("w") as f:
            for ep in episodes:
                f.write(json.dumps(asdict(ep)) + "\n")
    return metrics(episodes)


def load_policy(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = Config(**state["config"])
    policy = SmallPolicy() if cfg.backend == "small" else LanguagePolicy(cfg.model, cfg.revision)
    policy.load_state_dict(state["policy"], strict=cfg.backend == "small")
    return policy, cfg
