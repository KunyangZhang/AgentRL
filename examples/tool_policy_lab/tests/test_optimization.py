from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from tool_policy_lab.environment import make_cases
from tool_policy_lab.policy import LanguagePolicy, SmallPolicy, trainable_state
from tool_policy_lab.rollout import collect
from tool_policy_lab.training import Config, optimize, prepare_batch


def test_microbatch_accumulation_preserves_full_action_mean():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    a = SmallPolicy()
    b, reference = deepcopy(a), deepcopy(a).requires_grad_(False)
    episodes = collect(a, make_cases("train", 2) * 4)
    batch = prepare_batch(episodes)
    ref_before = deepcopy(reference.state_dict())
    cfg = Config(ppo_epochs=1)
    optimize(a, reference, torch.optim.SGD(a.parameters(), lr=0.01), batch, replace(cfg, micro_batch=10000))
    optimize(b, reference, torch.optim.SGD(b.parameters(), lr=0.01), batch, replace(cfg, micro_batch=3))
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)
    assert all(torch.equal(reference.state_dict()[k], v) for k, v in ref_before.items())
    assert any(not torch.equal(a.state_dict()[k], v) for k, v in ref_before.items())


def test_real_transformer_lora_forward_backward_and_reference():
    # Offline random tiny Qwen model: validates code path, NOT pretrained quality.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    tokenizer = Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "<eos>": 2,
                                    **{str(i): 3+i for i in range(5)}}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
                                       unk_token="<unk>", eos_token="<eos>")
    tokenizer.chat_template = "{% for message in messages %}{{message['content']}} {% endfor %}"
    config = Qwen2Config(vocab_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, attention_dropout=0)
    model = Qwen2ForCausalLM(config)
    policy = LanguagePolicy("offline-tiny", model=model, tokenizer=tokenizer, device="cpu")
    obs = [{"loaded": False, "instruction": {"operation": "add", "operand": 2}, "last_error": None}]
    ref = policy(obs, reference=True).detach().clone()
    original = {n: p.detach().clone() for n, p in policy.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.01)
    before = trainable_state(policy)
    loss = -policy(obs).log_softmax(-1)[0, 0]
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(v, trainable_state(policy)[k]) for k, v in before.items())
    torch.testing.assert_close(policy(obs, reference=True), ref)
    assert all(torch.equal(p, original[n]) for n, p in policy.named_parameters() if n in original)
    assert all("lora_" in name for name in trainable_state(policy))
