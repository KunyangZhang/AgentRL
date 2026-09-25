# SQL Agent: real-model SFT, multi-turn GRPO and held-out evaluation

A database analysis agent built as an independent extension of [THUDM/AgentRL](https://github.com/THUDM/AgentRL), using its **actual GRPO advantage and clipped policy-loss kernels** and its native `Task` / `Session` interface. The single-GPU trainer is new code; this experiment does **not** exercise AgentRL's distributed trainer, vLLM rollout workers or Kubernetes deployment.

Unlike the earlier `tool_policy_lab` arithmetic microbenchmark, this experiment trains **Qwen2.5-Coder-1.5B-Instruct LoRA adapters on a real GPU**. The model sees natural-language questions and public SQLite schemas, generates unrestricted SQL inside a JSON tool action, receives database results/errors, and can repair a query before final submission.

## Measured results

See [the complete experiment report](evidence/EXPERIMENT.md). On the 64-question filtered held-out set: prompt-only **38/64**, SFT **37/64**, GRPO **37/64**. The project delivers the training/evaluation pipeline; this run does not demonstrate a correctness gain from training. The report includes failure buckets, paired comparisons and the zero-variance rollout diagnostic.

## Design

```text
question + schema -> model query/schema action -> bounded read-only SQLite
                           ^                             |
                           +--------- feedback ----------+
                                      |
                                 final submit
                                      |
              offline gold-result evaluator -> terminal reward
                                      |
              grouped on-policy rollouts -> action-token GRPO update
```

- Three tools (`schema`, `query`, `submit`), maximum three turns. Gold SQL and gold results stay outside model-visible observations.
- SQLite read-only URI and an authorizer reject writes, attachment, pragmas and extension loading. Single-statement execution, 1,000-row result limit, VM-step/time checks and bounded feedback prevent common runaway queries. This is a local read-only tool, **not an OS sandbox or a guarantee against arbitrary memory exhaustion**.
- SFT uses query → execution feedback → submit demonstrations derived only from training examples. Prompt and tool-observation tokens are excluded from the supervised loss.
- GRPO samples four trajectories for the same question, computes group-relative terminal advantages with the upstream kernel, and applies clipped policy loss only to generated action tokens. A frozen SFT reference constrains updates using sampled KL. Sampling temperature is also applied to old/new/reference log probabilities.
- Reward: exact execution-result match `1.0`, executable submitted SQL `0.1`, every action respecting the JSON contract `0.02`. Correctness dominates the shaping terms; executable-but-wrong queries do not pass evaluation.
- Per-trajectory logs preserve prompts, action token IDs, SQL, tool feedback and outcomes. Training records reward variance, KL, action-token counts and timing. Stage completion reports are atomic. `--resume` skips completed stages; an interrupted GRPO stage restarts from SFT, **not** an exact mid-step continuation.
- `SQLAgentTask` exposes the same environment to native AgentRL sessions, including cancellation, invalid response, turn budget and terminal reward. The single-GPU experiment invokes the environment directly; it is not a distributed-worker benchmark.

## Frozen evaluation protocol

Public [Spider 1.0](https://github.com/taoyds/spider), distributed by [xlangai/spider](https://huggingface.co/datasets/xlangai/spider), archive revision `6232cc3fad6d54c62b3ba23a364083a98ff36a17`. The dataset retains its **CC-BY-SA-4.0** license; code in this fork retains upstream MIT. Cite Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*, EMNLP 2018. Downloaded data/model files are not committed.

A deterministic filter removes missing databases, failing/empty gold executions, schemas over 4,800 characters and gold SQL over 500 characters. The manifest records exclusions. The fixed sample comprises **256 training questions across 101 databases, 24 development questions across 9 databases, and 64 test questions across 10 databases**. Training uses official Spider training data; development/test divide official Spider development databases into disjoint groups. All three database sets are disjoint. The manifest includes split and database SHA-256 hashes.

The metric is **single-database execution-result match**, preserving duplicate rows and respecting ordered gold queries. It is conservative, type-sensitive and column-order-sensitive. It is **not official Spider test-suite accuracy**, which evaluates across additional database instances. This is a filtered 64-question test subset and a single training seed; do not describe it as the full benchmark or a robust multi-seed gain. Public data may have appeared in model pretraining.

Base / final SFT / final GRPO checkpoints are evaluated after training, with identical greedy decoding and budgets. No test-based checkpoint selection is used. A separately declared fixed synthetic one-shot prompt-only control checks whether gains merely reflect learning the JSON interface. Its extra prompt tokens can affect truncation; results include that count.

## Reproduce on one GPU

Tested: RTX 4090 D 24 GB, Ubuntu 22.04, Python 3.12, PyTorch 2.8.0+cu128. A single card is sufficient; no distributed infrastructure is required.

```bash
git clone --branch feat/tool-policy-lab https://github.com/KunyangZhang/AgentRL.git
cd AgentRL
# Create a venv with the host's CUDA-enabled PyTorch, or install the matching torch build.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r examples/sql_agent_rl/requirements.txt
export PYTHONPATH="$PWD/trainer/src:$PWD/worker/src:$PWD/examples/sql_agent_rl"
python examples/sql_agent_rl/download.py /root/autodl-tmp/agent-lab
python -m sql_agent_rl.data /root/autodl-tmp/agent-lab/spider /root/autodl-tmp/agent-lab/splits
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-1.5B-Instruct', revision='2e1fd397ee46e1388853d2af2c993145b0f1098a', local_dir='/root/autodl-tmp/agent-lab/model')"
python -m pytest examples/sql_agent_rl/tests -q
python -m sql_agent_rl.train --config examples/sql_agent_rl/configs/single_4090.json --output /root/autodl-tmp/agent-lab/run
python -m sql_agent_rl.prompt_baseline --config examples/sql_agent_rl/configs/single_4090.json --output /root/autodl-tmp/agent-lab/run
```

Change model/data/database paths in the JSON config for another machine. Keep a fresh run directory for a new experiment; use `--resume` only for identical frozen configuration and split-manifest hash. Adapters and logs remain local. Native worker config is in `configs/tasks.yaml`; add the three directories above to its Python path.

## Source map

| Module | Responsibility |
| --- | --- |
| `environment.py` | Action validation, read-only execution, feedback and label-isolated scoring |
| `data.py` | Deterministic database-disjoint sampling and provenance |
| `train.py` | LoRA SFT, batched multi-turn sampling, upstream GRPO/PPO kernels, evaluation |
| `prompt_baseline.py` | Fixed synthetic one-shot control |
| `task.py` | Native AgentRL Task/Session integration |
| `tests/` | Read-only boundaries, error recovery, session lifecycle, action loss and frozen backbone |

Developed with Codex assistance. This is a reproducible research/engineering project, not evidence of production traffic, enterprise adoption or distributed training experience.
