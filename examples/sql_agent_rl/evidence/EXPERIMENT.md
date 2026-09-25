# Experiment: 2026-09-26

Actual single-GPU run, seed 17, Qwen2.5-Coder-1.5B-Instruct revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`, on RTX 4090 D 24 GB. The model weights were retrieved from a mirror and checked against SHA-256 `c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8` from the pinned Hugging Face artifact. Dataset provenance and immutable split manifest are included.

## Results

**64 filtered Spider questions across 10 databases unseen in this training split.** Single-database execution-result match, not official Spider test-suite accuracy. Greedy decoding; three-turn limit; 160 new tokens per turn; 3,072 context tokens. No trajectories hit the context limit.

| Policy | Correct / 64 | Execution match | Executable final SQL | All actions valid JSON | Mean output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base, zero-shot tool protocol | 1 | 1.56% | 4.69% | 6.25% | 81.72 |
| Base, fixed synthetic one-shot | 38 | **59.38%** | 79.69% | 100% | 75.09 |
| SFT | 37 | 57.81% | **84.38%** | 100% | 71.75 |
| SFT + GRPO | 37 | 57.81% | **84.38%** | 100% | 70.94 |

The single prompt example fixes most base-model protocol failures. **Neither trained policy beats the prompt-only baseline on correctness. GRPO has no test correctness gain over SFT.** Do not present the zero-shot-to-trained difference as a GRPO reasoning improvement. SFT versus one-shot changes 7 questions to correct and 8 to incorrect: paired bootstrap difference −1.56 percentage points, 95% interval [−12.5, +10.94]. This is inconclusive at this sample size. Identical SFT/GRPO per-question success indicators give a zero paired interval for this dataset only; that does not imply zero uncertainty on other tasks or seeds.

SFT's 57.81% execution match has an approximate Wilson 95% interval [45.61%, 69.13%]. The one-shot baseline interval is [47.14%, 70.54%]. Public benchmark contamination from pretraining has not been excluded.

## What the traces reveal

- Base failures: 60 invalid protocol, 2 executable-but-wrong results, 1 SQL execution error.
- Prompt-only failures: 13 wrong results, 9 SQL execution errors, 4 episodes without final submission.
- SFT/GRPO failures: 17 wrong results, 10 SQL execution errors; every episode follows the query → submit structure.
- No evaluated policy successfully recovered from an encountered tool error. The environment's repair path is verified by tests, but this run does **not** establish learned repair capability.
- **21 of 40 GRPO groups have zero reward variance**, hence zero group-relative policy advantages. Nineteen groups supply nonzero policy-gradient signals. This is a concrete sampling diagnostic, not proof of the sole cause of the flat metric.
- Development correctness changes from 18/24 after SFT to 17/24 after GRPO. Test was evaluated only after training, with fixed final checkpoints.

A useful next experiment would add controlled error → feedback → repair demonstrations drawn exclusively from training databases, compare reward/rollout sampling variants, increase held-out coverage, and run at least three seeds. Register a new evaluation protocol before tuning; do not repeatedly optimize on this test set.

## Training cost and scope

- Total parameters including adapters: 1,545,893,376; **trainable LoRA parameters: 2,179,072**.
- SFT: 256 questions → 512 action examples; 128 optimizer steps, 114.10 seconds.
- GRPO: 40 updates × 4 sampled trajectories = 160 trajectories, 239.55 seconds.
- Combined training-loop time: **353.65 seconds (~5.9 minutes)**, excluding model loads, downloads and evaluation.
- Peak PyTorch **allocated** GPU memory: **6.365 GiB** during the formal pipeline. This excludes allocator reservation and driver memory, so it is not a minimum card specification.
- The 24 GB card used is sufficient. No FSDP, multi-GPU, vLLM or production service throughput was evaluated.
- **23 CPU contract tests passed**, including a tiny actual Qwen model test that checks action-token likelihood and frozen backbone parameters.

## Evidence and reruns

`report.json`, `prompt-only-report.json`, `analysis.json` and `*-outcomes.json` contain complete stage metrics and all paired question outcomes. `sft.jsonl` / `grpo.jsonl` contain every logged training update. `executed-code-sha256.txt` identifies the training/environment/upstream kernels actually executed. `adapter-manifest.json` records final adapter checksums. Full token-level trajectories and adapter weights were backed up locally; they are not committed to Git.

Reproduction commands are in the parent README. After a run, execute:

```bash
python -m sql_agent_rl.analyze /path/to/run
```

The earlier arithmetic policy experiments remain in `examples/tool_policy_lab` as separate history. Their multi-seed results and exact-resume test do not apply to this SQL model experiment.
