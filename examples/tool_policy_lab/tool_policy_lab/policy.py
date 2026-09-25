"""Interchangeable small-network and constrained language-model policies."""
from contextlib import nullcontext
import json
import torch
from torch import nn
from .environment import ACTIONS, OPS


def features(obs):
    op = obs["instruction"]["operation"] if obs["instruction"] else None
    return [float(obs["loaded"]), float(op is None), *[float(op == x) for x in OPS],
            float(obs["last_error"] == "temporary_unavailable"),
            float(obs["last_error"] not in (None, "temporary_unavailable"))]


class SmallPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.Tanh(), nn.Linear(32, len(ACTIONS)))

    def forward(self, observations, reference=False):
        return self.net(torch.tensor([features(x) for x in observations], dtype=torch.float32))


class LanguagePolicy(nn.Module):
    """LoRA over next-token tool IDs. No unrestricted JSON/text generation.

    Both rollout and update normalize over the SAME five tool tokens. Observed
    environment text is context only; the objective is on selected actions.
    """
    def __init__(self, model_name, revision=None, device="cuda", model=None, tokenizer=None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa")
        self.model = get_peft_model(self.model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"))
        self.model.to(device)
        self.model.config.use_cache = False
        if device.startswith("cuda"):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        encoded = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in range(len(ACTIONS))]
        if any(len(x) != 1 for x in encoded) or len({x[0] for x in encoded}) != len(ACTIONS):
            raise ValueError("tool IDs 0..4 must each encode to one distinct token")
        self.register_buffer("action_tokens", torch.tensor([x[0] for x in encoded], device=device))

    def forward(self, observations, reference=False):
        prompts = []
        for obs in observations:
            messages = [{"role": "system", "content":
                         "Execute the current arithmetic instruction using tools. First fetch, "
                         "then follow each operation in order, then submit. Retry transient failures. "
                         "Reply with one digit only: " + ", ".join(f"{i}={a}" for i, a in enumerate(ACTIONS))},
                        {"role": "user", "content": json.dumps(obs, sort_keys=True)}]
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False,
                           add_generation_prompt=True))
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        if inputs.input_ids.shape[1] > 1024:
            raise ValueError("observation exceeds 1024-token context budget")
        inputs = inputs.to(self.action_tokens.device)
        # eval mode disables dropout but still allows gradients in policy updates.
        self.model.eval()
        with self.model.disable_adapter() if reference else nullcontext():
            logits = self.model(**inputs).logits[:, -1, self.action_tokens].float()
        return logits


def trainable_state(policy):
    names = {n for n, p in policy.named_parameters() if p.requires_grad}
    return {n: t.detach().cpu().clone() for n, t in policy.state_dict().items() if n in names}
