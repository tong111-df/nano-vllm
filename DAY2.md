# Day 2：多请求调度基线

2026-11-02。上游基线：bb823b3e06983d71485a8e1f23715ebd87d98ef8。
沿用 Day 1 的导入隔离；今天未修改调度算法。

## 复现

PowerShell 完整命令：

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.day2_baseline
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```

实测 Python 3.12.13；全量 14 passed。
实验脚本写入 results/day2_baseline.json，重复运行会覆盖该结果文件。
报告包含源码哈希、环境和逐轮记录；未来修改策略前应保留本次结果。

## 场景与定义

块大小 256，块池 4 块，max_num_seqs=4。无 EOS、无前缀命中、无空间压力。
A 的 prompt 为 [10]，max_tokens=3。第 0 轮先让 A 完成 Prefill 并生成一个 token。
长 Prefill 场景：第 1 轮前加入 B，prompt 长度 5，max_tokens=1。
连续到达场景：第 1～3 轮调度前各加入一个 prompt 长度 1、max_tokens=1 的请求。

A_blocked_rounds：第 0 轮之后、A 首次 Decode 之前，A 已生成数量保持为 1 的轮数。
A_first_decode_round：A 第一次执行 Decode 的轮号，第 0 轮是准备阶段。
所有模型输出均为合成 token ID，实际调用真实 Scheduler 和 BlockManager。

## 实测结果

| 场景 | Prefill 预算 | A 被推迟轮数 | A 首次 Decode 轮号 |
|---|---:|---:|---:|
| B 长度 5 | 1 | 5 | 6 |
| B 长度 5 | 2 | 3 | 4 |
| B 长度 5 | 5 | 1 | 2 |
| 连续三个短请求 | 2 | 3 | 4 |

预算为 2 的长请求轨迹：第 1 轮 B 处理 2 个，第 2 轮 B 处理 2 个，第 3 轮 B 处理 1 个并结束，第 4 轮 A Decode，第 5 轮 A Decode 后结束。

## 如何读记录

- waiting_before / running_before：schedule 调用前的队列。
- phase：本轮返回的执行阶段。
- scheduled_tokens：schedule 后、postprocess 前抓取的每个请求计算量；postprocess 会把该字段清零。
- states_after / completion_counts_after：postprocess 后的状态与生成数量。

先阅读 experiments/day2_baseline.py 的 advance()，对应 schedule → 抓取本轮计算量 → postprocess → 保存结果。
再在 JSON 中找 long_prefill、budget=2 的 trace，解释第 3 轮为何只处理一个 token，A 仍未执行。

## 结论与限制

原版一轮只执行 Prefill 或 Decode，可以执行多个请求的同一阶段。只要选中了 Prefill，就提前返回，不混合 Decode；RUNNING 请求不保证每轮有进展。
连续可调度的新 Prefill 可能持续推迟 Decode。增加预算可以降低这个固定长 prompt 场景的等待轮数，但不能保证真实延迟降低；每轮耗时可能变化。
此实验不验证真实 KV tensor、模型结果、GPU 兼容性、TTFT 或吞吐量，不把 CPU 脚本运行时间作为推理性能。

## 后续设计假设（尚未实现）

有 Decode 等待时，最多连续 2 轮 Prefill，再让出一轮 Decode。
预算为 2 的上述场景预计 A 首次 Decode 提前到第 3 轮，代价是 B 首 token 推迟到第 4 轮。这是推演，不是新策略实测结果。
后续需实现策略开关，保留原版基线，并覆盖多请求、空队列、空间不足、抢占和计数重置边界。
