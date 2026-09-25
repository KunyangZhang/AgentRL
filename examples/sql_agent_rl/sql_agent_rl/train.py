"""Single-GPU SFT -> on-policy multi-turn GRPO with auditable action tokens."""
import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from agentrl.trainer.algorithms.core_algos import compute_grpo_outcome_advantage, compute_policy_loss
from .environment import SQLEnv


@dataclass
class Config:
    seed: int = 17
    model: str = "/root/autodl-tmp/agent-lab/model"
    data: str = "/root/autodl-tmp/agent-lab/splits"
    databases: str = "/root/autodl-tmp/agent-lab/spider/database"
    max_context: int = 3072
    max_new_tokens: int = 160
    max_turns: int = 3
    temperature: float = 0.8
    lora_rank: int = 16
    sft_epochs: int = 1
    sft_accumulate: int = 4
    sft_lr: float = 0.0001
    rl_lr: float = 0.00001
    rl_updates: int = 40
    group_size: int = 4
    clip: float = 0.2
    kl_coef: float = 0.01
    eval_batch: int = 4

    def __post_init__(self):
        if self.group_size < 2 or self.max_turns != 3 or self.temperature <= 0:
            raise ValueError("group_size >= 2, max_turns = 3 and temperature > 0 required")
        if min(self.rl_updates, self.sft_epochs, self.sft_accumulate, self.eval_batch) < 1:
            raise ValueError("positive training/evaluation budgets required")
        if not 0 < self.max_new_tokens < self.max_context or self.lora_rank < 1:
            raise ValueError("invalid token budget or LoRA rank")


class Policy:
    def __init__(self, cfg, adapter=None):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(cfg.model, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        if adapter:
            self.model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        else:
            self.model = get_peft_model(model, LoraConfig(r=cfg.lora_rank, lora_alpha=2*cfg.lora_rank,
                lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False

    def prompt_ids(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, prompts, greedy=False):
        self.model.eval()
        pad = self.tokenizer.pad_token_id
        width = max(map(len, prompts))
        inputs = torch.tensor([[pad]*(width-len(p))+p for p in prompts], device="cuda")
        mask = torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts], device="cuda")
        kwargs = {} if greedy else {"temperature": self.cfg.temperature, "top_p": 1.0, "top_k": 0}
        outputs = self.model.generate(input_ids=inputs, attention_mask=mask, do_sample=not greedy,
                    max_new_tokens=self.cfg.max_new_tokens, repetition_penalty=1.0,
                    pad_token_id=pad, eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True, **kwargs)[:, width:].tolist()
        replies = []
        for row in outputs:
            if self.tokenizer.eos_token_id in row:
                row = row[:row.index(self.tokenizer.eos_token_id)+1]
            replies.append(row)
        return replies

    def log_probs(self, prompt, response, model=None):
        model = model if model is not None else self.model
        if not prompt or not response:
            raise ValueError("nonempty prompt and action required")
        ids = torch.tensor([prompt+response], device=next(model.parameters()).device)
        # Compute vocabulary logits only for generated action positions, saving memory.
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(response)+1).logits[0, :-1].float()
        targets = ids[0, -len(response):]
        return -torch.nn.functional.cross_entropy(logits/self.cfg.temperature, targets, reduction="none")

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


@torch.no_grad()
def rollout(policy, cases, cfg, greedy=False, old_probs=False):
    envs = [SQLEnv(c, cfg.databases, cfg.max_turns) for c in cases]
    trajectories = [{"uid": c["uid"], "db_id": c["db_id"], "turns": [], "truncated": False} for c in cases]
    for _ in range(cfg.max_turns):
        active, prompts = [], []
        for i, env in enumerate(envs):
            if env.done:
                continue
            ids = policy.prompt_ids(env.messages)
            if len(ids) + cfg.max_new_tokens > cfg.max_context:
                env.done = True
                trajectories[i]["truncated"] = True
                continue
            active.append(i)
            prompts.append(ids)
        if not active:
            break
        replies = policy.generate(prompts, greedy)
        for i, prompt, response in zip(active, prompts, replies):
            text = policy.tokenizer.decode(response, skip_special_tokens=True)
            turn = {"prompt_ids": prompt, "action_ids": response, "text": text}
            if old_probs:
                turn["old_log_probs"] = policy.log_probs(prompt, response).cpu().tolist()
            trajectories[i]["turns"].append(turn)
            envs[i].step(text)
    for record, env in zip(trajectories, envs):
        record.update(env.score())
        record["trace"] = env.trace
    return trajectories


def summarize(rows):
    n = len(rows)
    success = sum(x["success"] for x in rows)
    errors = [x for x in rows if x["tool_errors"] > 0]
    return {"n": n, "correct": success, "execution_match": success/n,
            "executable_rate": sum(x["executable"] for x in rows)/n,
            "json_valid_rate": sum(x["valid_json"] for x in rows)/n,
            "mean_turns": sum(len(x["turns"]) for x in rows)/n,
            "mean_action_tokens": sum(sum(len(t["action_ids"]) for t in x["turns"]) for x in rows)/n,
            "truncated": sum(x["truncated"] for x in rows),
            "tool_error_episodes": len(errors),
            "recovered_episodes": sum(x["error_recovered"] for x in rows)}


def evaluate(policy, cases, cfg, destination):
    started = time.monotonic()
    records = []
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as f:
        for i in range(0, len(cases), cfg.eval_batch):
            rows = rollout(policy, cases[i:i+cfg.eval_batch], cfg, greedy=True)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+"\n")
                f.flush()
            records.extend(rows)
    result = {**summarize(records), "seconds": time.monotonic()-started}
    print(json.dumps({"evaluation": str(destination), **result}), flush=True)
    return result


def sft_examples(cases, cfg, tokenizer):
    examples = []
    for case in cases:
        env = SQLEnv(case, cfg.databases)
        for tool in ("query", "submit"):
            prompt = tokenizer.apply_chat_template(env.messages, tokenize=True, add_generation_prompt=True)
            text = json.dumps({"tool": tool, "sql": case["query"]})
            response = tokenizer.encode(text, add_special_tokens=False)+[tokenizer.eos_token_id]
            if len(prompt)+len(response) <= cfg.max_context:
                examples.append((prompt, response))
            env.step(text)
    return examples


def log(path, obj):
    with Path(path).open("a") as f:
        f.write(json.dumps(obj)+"\n")
    print(json.dumps(obj), flush=True)


def sft(policy, cases, cfg, output):
    examples = sft_examples(cases, cfg, policy.tokenizer)
    rng = random.Random(cfg.seed)
    optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=cfg.sft_lr)
    started = time.monotonic()
    step = 0
    for epoch in range(cfg.sft_epochs):
        rng.shuffle(examples)
        for start in range(0, len(examples), cfg.sft_accumulate):
            batch = examples[start:start+cfg.sft_accumulate]
            optimizer.zero_grad()
            policy.model.train()
            total_loss = 0
            for prompt, response in batch:
                # Undo rollout temperature for standard SFT cross entropy.
                original = cfg.temperature
                cfg.temperature = 1.0
                loss = -policy.log_probs(prompt, response).mean()
                cfg.temperature = original
                (loss/len(batch)).backward()
                total_loss += loss.detach().item()/len(batch)
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 8 == 0 or start+len(batch) == len(examples):
                log(output/"sft.jsonl", {"step": step, "loss": total_loss, "seconds": time.monotonic()-started})
    policy.save(output/"sft")
    return {"examples": len(examples), "updates": step, "seconds": time.monotonic()-started}


def grpo(policy, cases, dev, cfg, output):
    # Freeze the SFT policy, not the pre-SFT base, as the GRPO reference.
    reference = deepcopy(policy.model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=cfg.rl_lr)
    rng = random.Random(cfg.seed)
    started = time.monotonic()
    for step in range(1, cfg.rl_updates+1):
        case = rng.choice(cases)
        rows = rollout(policy, [case]*cfg.group_size, cfg, old_probs=True)
        lengths = [sum(len(t["action_ids"]) for t in row["turns"]) for row in rows]
        if not all(lengths):
            log(output/"grpo.jsonl", {"step": step, "skipped": "empty_action", "uid": case["uid"]})
            continue
        rewards = [torch.tensor([r["reward"]]) for r in rows]
        adv, _ = compute_grpo_outcome_advantage(rewards, [torch.ones(1) for _ in rewards])
        optimizer.zero_grad()
        policy.model.train()
        total_tokens = sum(lengths)
        loss_total, kl_total = 0.0, 0.0
        for row, advantage in zip(rows, adv):
            for turn in row["turns"]:
                prompt, response = turn["prompt_ids"], turn["action_ids"]
                old = torch.tensor(turn["old_log_probs"], device="cuda")
                current = policy.log_probs(prompt, response)
                with torch.no_grad():
                    ref = policy.log_probs(prompt, response, reference)
                pg, _, _, _ = compute_policy_loss(old, current, advantage.to("cuda").expand_as(current),
                                                 torch.ones_like(current), cliprange=cfg.clip)
                delta = (ref-current).clamp(-10, 10)
                kl = (delta.exp()-delta-1).mean()
                weight = len(response)/total_tokens
                objective = (pg + cfg.kl_coef*kl)*weight
                objective.backward()
                loss_total += objective.detach().item()
                kl_total += weight*kl.detach().item()
        norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite GRPO gradient")
        optimizer.step()
        record = {"step": step, "uid": case["uid"], "successes": sum(r["success"] for r in rows),
                  "reward_std": torch.tensor([r["reward"] for r in rows]).std().item(),
                  "loss": loss_total, "reference_kl": kl_total, "action_tokens": total_tokens,
                  "seconds": time.monotonic()-started}
        log(output/"grpo.jsonl", record)
        if step % 10 == 0 or step == cfg.rl_updates:
            policy.save(output/f"grpo-{step}")
            checkpoint = {"step": step, "optimizer": optimizer.state_dict(), "python_rng": rng.getstate(),
                          "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                          "config": asdict(cfg)}
            tmp = output/"trainer.tmp"
            torch.save(checkpoint, tmp)
            os.replace(tmp, output/"trainer.pt")
    policy.save(output/"grpo")
    del reference, optimizer
    torch.cuda.empty_cache()
    return {"updates": cfg.rl_updates, "seconds": time.monotonic()-started}


def run(cfg, output, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"config": asdict(cfg),
       "selection": "fixed final checkpoints; held-out test evaluated once per stage after all training",
       "split_manifest_sha256": hashlib.sha256((Path(cfg.data)/"manifest.json").read_bytes()).hexdigest()}
    if (output/"protocol.json").exists():
        if not resume or json.loads((output/"protocol.json").read_text()) != protocol:
            raise ValueError("existing run requires --resume and identical frozen protocol")
    else:
        (output/"protocol.json").write_text(json.dumps(protocol, indent=2))
    def seed():
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    seed()
    torch.set_num_threads(8)
    cases = {s: json.loads((Path(cfg.data)/f"{s}.json").read_text()) for s in ("train", "dev", "test")}
    torch.cuda.reset_peak_memory_stats()
    report = json.loads((output/"report.json").read_text()) if (output/"report.json").exists() else {
        "scope": "Spider filtered subset, single-database execution match, one training seed",
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "cuda": torch.version.cuda},
        "config": asdict(cfg), "results": {}}
    def persist():
        report["peak_allocated_gib"] = max(report.get("peak_allocated_gib", 0), torch.cuda.max_memory_allocated()/2**30)
        (output/"report.tmp").write_text(json.dumps(report, indent=2))
        os.replace(output/"report.tmp", output/"report.json")
    if "base_dev" not in report:
        policy = Policy(cfg)
        report["parameters"] = {"total": sum(p.numel() for p in policy.model.parameters()),
            "trainable": sum(p.numel() for p in policy.model.parameters() if p.requires_grad)}
        policy.save(output/"base")
        report["base_dev"] = evaluate(policy, cases["dev"], cfg, output/"base-dev.jsonl")
        persist()
        del policy
        torch.cuda.empty_cache()
    if "sft_training" not in report:
        seed()
        policy = Policy(cfg, output/"base")
        report["sft_training"] = sft(policy, cases["train"], cfg, output)
        persist()
        del policy
        torch.cuda.empty_cache()
    if "sft_dev" not in report:
        policy = Policy(cfg, output/"sft")
        report["sft_dev"] = evaluate(policy, cases["dev"], cfg, output/"sft-dev.jsonl")
        persist()
        del policy
        torch.cuda.empty_cache()
    if "grpo_training" not in report:
        seed()
        policy = Policy(cfg, output/"sft")
        report["grpo_training"] = grpo(policy, cases["train"], cases["dev"], cfg, output)
        persist()
        del policy
        torch.cuda.empty_cache()
    if "grpo_dev" not in report:
        policy = Policy(cfg, output/"grpo")
        report["grpo_dev"] = evaluate(policy, cases["dev"], cfg, output/"grpo-dev.jsonl")
        persist()
        del policy
        torch.cuda.empty_cache()
    # Stage-boundary resume skips completed evaluations; no checkpoint selection on test.
    for stage in ("base", "sft", "grpo"):
        if stage in report["results"]:
            continue
        policy = Policy(cfg, output/stage)
        report["results"][stage] = evaluate(policy, cases["test"], cfg, output/f"{stage}-test.jsonl")
        persist()
        del policy
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(Config(**json.loads(Path(args.config).read_text())), args.output, args.resume)
