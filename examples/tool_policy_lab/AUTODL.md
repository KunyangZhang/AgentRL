# AutoDL 单卡训练交接

当前状态：CPU 真实训练已完成；离线随机微型 Qwen 的 LoRA 梯度测试通过。
**尚未执行预训练 Qwen 的 CUDA 训练，不提供推测的效果、耗时或显存数据。**

建议先用已有的单卡 24GB CUDA PyTorch 实例，实际卡型和软件版本拿到机器后核验。
脚本预检要求至少 12GiB，只是首版配置的保守门槛，不是实测最小显存。
原生 AgentRL 分布式引擎需要独立 actor/rollout GPU 池；本项目提供的 PEFT 入口是单卡路径。

```bash
git clone -b feat/tool-policy-lab https://github.com/KunyangZhang/AgentRL.git
cd AgentRL
# 在 AutoDL 已有 CUDA PyTorch 环境中，先确认版本；不要安装 CPU torch wheel。
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m pip install -r examples/tool_policy_lab/requirements-cpu.txt
export PYTHONPATH="$PWD/trainer/src:$PWD/worker/src:$PWD/examples/tool_policy_lab"
python -m pytest examples/tool_policy_lab/tests -q
bash examples/tool_policy_lab/autodl.sh --output runs/qwen-lora-seed17
```

首次加载需下载公开模型 `Qwen/Qwen2.5-0.5B-Instruct`，固定 revision 见 `configs/lora.json`。
只训练工具 token 条件分布，不生成思维链。第一次先把复制后的配置 updates 改为 2、eval_count 改为 4 做冒烟，
正式训练使用全新目录和正式配置。避免用 smoke checkpoint 续接不同配置。

正式阶段：

1. 保存 `nvidia-smi`、pip 版本清单、Git SHA、配置、模型 revision、峰值显存和墙钟耗时。
2. 在初始模型评测 test 前先固定全部训练配置；开发期只查看 dev。
3. 固定种子 17/29/43，分别用相同训练预算比较训练前后、奖励消融和工具调用成本。
4. `last.pt` 包含 LoRA 参数、优化器、step 和 RNG。仅提高 `updates` 后可按下方续训；其他配置变化会拒绝。
5. 评测产物写入新目录；把结果核对后再升级简历里的实验结论。

```bash
bash examples/tool_policy_lab/autodl.sh --output runs/qwen-lora-seed17 \
  --resume runs/qwen-lora-seed17/last.pt
bash examples/tool_policy_lab/run.sh evaluate \
  --checkpoint runs/qwen-lora-seed17/last.pt --output runs/qwen-lora-eval --count 256
```

完整日志本地持久化；每 20 步及末步 checkpoint。中断会丢失最近 checkpoint 后的更新，
恢复时回退日志到 checkpoint 的 step。CPU 已验证续训参数逐元素一致；CUDA 是否逐位一致需在目标硬件验证。
仓库无需存 SSH 凭据、API key 或访问 Token。
