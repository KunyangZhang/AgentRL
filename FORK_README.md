# Tool Policy Lab — Agentic RL 工程实验

本分支在 [THUDM/AgentRL](https://github.com/THUDM/AgentRL) 上新增可复现的多轮工具策略训练实验。
原项目的训练引擎、论文成果与基础算法归上游作者；本分支的增量位于
[`examples/tool_policy_lab`](examples/tool_policy_lab)。保留上游 MIT 许可证和 README。

**本地已完成：**真实小型神经网络 GRPO 训练、3 个种子 × 2 种奖励的消融、留出集评测、
故障重试、参数级断点续训一致性检查、原生 Task/Session 集成、离线微型 Transformer 的 LoRA 前后向验证。

**等待 GPU：**Qwen2.5-0.5B-Instruct 预训练权重上的单卡 LoRA 实验。入口已经实现，
尚无该模型的 GPU 训练结果。没有宣称复现 AgentRL 论文、训练通用工具智能体或多卡吞吐提升。

查看 [复现说明与实验边界](examples/tool_policy_lab/README.md)、
[实验报告](examples/tool_policy_lab/evidence/EXPERIMENT.md) 和
[AutoDL 交接](examples/tool_policy_lab/AUTODL.md)。
