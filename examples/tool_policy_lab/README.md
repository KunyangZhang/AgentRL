# Agentic RL Tool Policy Lab

这是一个基于 AgentRL 的工具策略训练与验证项目。研究对象是**多轮工具路由、顺序执行和失败重试**。
环境给出结构化的当前操作指令，工具实际改变累加器状态；策略需要先读取、逐步执行、处理临时不可用并最终提交。
任务是可控的合成微基准，不代表自然语言理解、数学推理或通用 Agent 能力。

## 本分支实现

| 组件 | 实现 | 上游复用 |
|---|---|---|
| 环境 | 5 个工具、前置条件、调用预算、故障注入、不可伪造的最终校验 | Task/Session 协议 |
| Rollout | 批量运行活动环境，保存 observation/action/reward/done/old log-prob | 无 |
| 优化 | 按任务 ID 分组；仅动作进入损失；冻结参考策略、精确离散 KL、熵、梯度裁剪 | GRPO outcome advantage、PPO clipped loss |
| 模型 | 421 参数 MLP；Qwen LoRA 约束工具解码接口 | PyTorch / Transformers / PEFT |
| 可恢复性 | 原子 checkpoint，优化器与 Python/Torch/CUDA RNG，配置指纹 | 无 |
| 验证 | 数据内容哈希隔离、随机/初始/规则基线、奖励消融、压力集、JSONL 轨迹 | 原生 Session 测试 |

`environment.py → rollout.py → training.prepare_batch → AgentRL GRPO → training.optimize → checkpoint → evaluate`。
`task.py` 把同一个环境接入原生 AgentRL；没有复制另一套奖励逻辑。

## CPU 复现

从仓库根目录运行，建议 Python 3.12：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.6' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r examples/tool_policy_lab/requirements-cpu.txt
export PYTHONPATH="$PWD/trainer/src:$PWD/worker/src:$PWD/examples/tool_policy_lab"
python -m pytest examples/tool_policy_lab/tests -q
bash examples/tool_policy_lab/run.sh train \
  --config examples/tool_policy_lab/configs/cpu.json --output runs/my-cpu-run
bash examples/tool_policy_lab/run.sh evaluate \
  --checkpoint runs/my-cpu-run/last.pt --output runs/my-cpu-eval
bash examples/tool_policy_lab/run.sh suite --output runs/my-six-run-suite
```

CPU 依赖包含 Ray，是因为上游算法工具模块导入了 Ray；这里不会启动 Ray 集群，也不安装 FlashAttention。
完整的已测版本见 `evidence/runtime.json`。CI 在干净环境执行契约测试。

运行目录包括配置、数据清单、训练日志和 `last.pt`；suite 另含 6 组完整评测轨迹、对照与汇总。
`runs/` 和模型权重不提交到 Git。精简证据（指标、日志、代表轨迹、版本与 SHA256）在 `evidence/`。

## 实验协议

- 固定 256 个 train、256 个 dev、256 个 test；另有 256 个 stress 任务。
  普通任务 1–3 次操作、单阶段至多 1 次暂时故障；压力任务 4–6 次操作、至多 2 次故障。
- 独立生成种子；内容哈希排除重叠。**任务模板相同**，这是数值与长度泛化，不能声称跨领域泛化。
- 每个训练步从 train 采样 16 个不同任务，每任务 8 条轨迹；100 步，每步 2 次 PPO 更新。
  不做 SFT；终局奖励和过程奖励各运行 17/29/43 三个种子。
- 使用固定末步 checkpoint；不根据 test 选模型。dev 每 20 步记录，test/stress 训练后评测。
- 初始模型、均匀随机工具策略与规则执行器作基线。规则执行器理解全部任务协议，是环境校验上界。
- 成功、调用数、非法动作比例、故障任务成功率和遇到故障后的成功率分别报告。
  后者有暴露偏差，提前退出会降低遇故障数量，因此同时记录**预先分配故障的任务**作为分母。
- Wilson 区间仅反映单个固定策略在有限任务集的二项结果，不是跨种子置信区间。

## 奖励与负面结果

终局成功 +1，失败提交/超预算 -0.3。过程奖励版本另有每次调用 -0.01、
正确阶段推进总计 +0.2、非暂时性错误 -0.1。
本次过程惩罚版本在 3 个种子上均学会提前提交；终局版本完成所有留出任务，
但部分种子仍有多余调用。不能据此声称过程奖励普遍无效。
默认 CPU/LoRA **配置文件**选择终局奖励；suite 始终保留两组实验，失败结果不删除。

## 语言模型路径

`LanguagePolicy` 将观测序列化到聊天模板，模型在 0–4 五个单 token 工具 ID 上采样。
Rollout 与优化使用同一归一化分布；它是**受约束的工具选择策略**，不训练自由文本推理或 JSON 参数生成。
LoRA 只更新 q/v projection 的低秩参数；关闭 adapter 的基础模型充当冻结参考策略。
工具反馈仅作为 prompt，不参与动作损失。每个决策重建当前状态上下文，不使用完整历史长上下文。

离线测试使用随机初始化的微型 Qwen Transformer，实际验证 adapter 梯度和冻结参考输出；
这不等于预训练 Qwen 效果验证。真实 GPU 启动详见 [AUTODL.md](AUTODL.md)。

## 原生 AgentRL 接入

安装上游 worker 后，可用 `configs/tasks.yaml` 注册 `tool-policy-train` / `tool-policy-dev`。
遵循上游 [部署文档](https://github.com/THUDM/AgentRL/blob/main/docs/deployment.md) 启动控制器，
并设置本目录的 PYTHONPATH 后运行 worker。
原生适配器通过实际 SessionController 信号量往返测试：正常工具执行、取消、非法参数、并行调用拒绝。
原生分布式 trainer、controller 服务与 SGLang 多卡集群**尚未端到端运行**；不把它们列为本项目实验成果。

## 已知限制与下一步

这个微基准过于结构化，规则策略已能满分，100% 不能证明比规则系统更有价值。
终局奖励容易容忍浪费调用；奖励设计还需引入独立 held-out 开发协议，避免重新调参污染现有 test。
下一阶段用 AutoDL 验证预训练模型 LoRA、显存/耗时、相同 token 预算下的调用效率，再扩展到含参数的真实任务。
本仓库交付的是可运行训练闭环和验证框架，不是已经验证商业价值的 Agent 产品。
