# 第十八单元：真实模型计时对照

入口 experiments.gpu_timing 已实现，用户已在 2026-11-07 完成实际 GPU 运行。
CPU 负载控制测试不等于 GPU 性能验证。

## 实测结果与边界

证据：results/unit18/gpu_timing/20261107T051248_426826Z_033389e4/report.json。
十个测量样本均为 36 轮、532 个实际输入 token、A 输出16个及 B1..4 各输出1个。
A 首次 Decode 轮次在每个原版样本均为21，每个 bounded 样本均为3。

| 指标（五次运行中位数，ms） | prefill_first | bounded_prefill |
|---|---|---|
| A 首次 Decode | 590.73 | 101.73 |
| A 完成 | 878.64 | 868.09 |
| B 最大首 token 等待 | 466.66 | 639.61 |
| 全部完成 | 878.64 | 868.10 |

此固定轮次到达负载中，A 首次 Decode 中位数减少约82.78%，
B 最大首 token 等待增加约37.06%。这是取舍，不是所有请求都更快。
全部完成时间中位数仅降低约1.20%；原版范围819.34～972.05ms，
bounded 范围774.83～1001.27ms，明显重叠，不能宣称稳定吞吐提升或统计显著。
每个策略仅五个测量样本，未做显著性推断；不是固定墙钟到达率或真实业务负载。

模型加载一次，单卡 eager，预算 32，max_num_seqs=4，max_model_len=256，
Decode 和缓存固定 original；只比较 prefill_first / bounded_prefill（上限2）。
A 在第0轮到达，prompt=1、生成16个；B1..4 在第1..4轮到达，各 prompt=129、生成1个。
输入由相同 tokenizer token 重复构造；ignore_eos=True 固定生成长度。
这是按调度轮次到达的受控负载，实际到达墙钟时刻会随策略变化，
不能说成相同外部请求速率下的服务性能测试，也不是自然语言质量测试。

每次运行复制配置、重建 Scheduler，令缓存元数据为空；GPU 物理存储不清零，
但新输入必须先写再读。两种策略共用同一个已加载模型。
每种策略预热两轮，随后各测五轮，重复之间交替策略执行顺序。
模型加载、初始化、随机种子设置、Scheduler 重建和预热不计入样本。
每步完成后 CUDA synchronize，用 perf_counter 测同步墙钟耗时；
计入请求入队、CPU 调度、模型步骤、同步和记录开销，不是纯 kernel 事件时间。

指标包含 A 首次 Decode、A 完成、B 最大首 token 等待和全部完成的毫秒数，
保存每次原始值及 min/median/max。输出 token 可因调度次序影响采样随机数而不同，
不能把跨策略输出不一致直接当作错误。
报告保留来源指纹、环境、模型配置哈希和逐轮轨迹；完整数值对照另行验证。
小负载、重复次数有限、笔记本功耗与热状态都可能影响数值，不强制断言某策略更快。

运行（PowerShell）：

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
wsl -d Ubuntu-24.04 --cd "/mnt/c/Users/tong/Documents/Codex/2026-11-01/referenced-chatgpt-conversation-this-is-an/outputs/nano-vllm" -- env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/tong/.venvs/nano-vllm-gpu/bin/python -m experiments.gpu_timing --model "/mnt/c/Users/tong/Documents/Codex/2026-11-01/referenced-chatgpt-conversation-this-is-an/outputs/models/Qwen3-0.6B"
```

成功状态 gpu_timing_completed 仅表示采样完成，不代表有性能收益。
结果位于 results/unit18/gpu_timing 下，每次独立目录；失败也保留错误及已有样本。
