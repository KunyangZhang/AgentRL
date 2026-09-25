import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from sql_agent_rl.train import Config, Policy


def test_action_only_likelihood_and_frozen_backbone():
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    policy = Policy.__new__(Policy)
    policy.cfg = Config(temperature=0.8)
    policy.model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
        target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    policy.model.eval()
    prompt, response = [1, 2, 3, 4], [5, 6]
    actual = policy.log_probs(prompt, response)
    logits = policy.model(input_ids=torch.tensor([prompt+response])).logits[0, len(prompt)-1:-1]
    expected = logits.div(0.8).log_softmax(-1).gather(1, torch.tensor(response)[:, None]).squeeze(1)
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected)
    frozen = {n:p.detach().clone() for n,p in policy.model.named_parameters() if not p.requires_grad}
    trainable = {n:p.detach().clone() for n,p in policy.model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=0.01)
    (-actual.mean()).backward()
    optimizer.step()
    assert all(torch.equal(frozen[n], p) for n,p in policy.model.named_parameters() if n in frozen)
    assert any(not torch.equal(trainable[n], p) for n,p in policy.model.named_parameters() if n in trainable)
