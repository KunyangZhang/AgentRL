# AgentRL portfolio extensions

This fork preserves [THUDM/AgentRL](https://github.com/THUDM/AgentRL) and its MIT license. Extensions are on `feat/tool-policy-lab`; they are independent work, not accepted upstream contributions.

## Main project: SQL Agentic RL

[Implementation and reproduction](examples/sql_agent_rl/README.md) · [measured experiment](examples/sql_agent_rl/evidence/EXPERIMENT.md)

Real Qwen2.5-Coder-1.5B LoRA SFT and multi-turn GRPO on RTX 4090 D, using public Spider SQLite databases, native AgentRL task sessions and upstream GRPO/PPO kernels. Read-only tool execution, database-disjoint splits, action-token training, prompt-only control, paired evaluation and complete failure analysis.

Observed correctness on the filtered 64-question holdout: prompt-only 38/64, SFT 37/64, GRPO 37/64. This experiment does not show a training gain over the prompt baseline; the report separates protocol adherence from SQL correctness and records 21/40 zero-variance GRPO groups.

## Earlier microbenchmark

[Tool policy lab](examples/tool_policy_lab/README.md) is a separate arithmetic-environment experiment with a small neural policy, reward ablations and exact CPU checkpoint resume. Its metrics must not be attributed to the SQL LLM project.
