# 单元 5：Decode 请求间公平性（进行中）

2026-11-04，按学习单元推进，不限制每天只完成一个单元。

## 已复现的偏置

三请求均在第 0 轮前到达，prompt 长度均为 1；A、B 输出长度为 L，C 输出长度为 2。
单批 max_num_seqs=2、Prefill 预算 8、块池 16 块、每块 256；没有空间压力、EOS 或前缀命中。
第 0 轮 A/B Prefill，第 1 轮 C Prefill，此后无 waiting 请求。
注意原版 max_num_seqs 约束本轮调度数量，并没有在 Prefill 准入时扣除已在 running 中的请求，因此 running 可以有三个请求。

| A/B 输出长度 | 原版 C 首次 Decode | bounded_prefill C 首次 Decode | 两者 C 最长生成间隔 |
|---|---:|---:|---:|
| 4 | 5 | 5 | 4 |
| 8 | 9 | 9 | 8 |
| 16 | 17 | 17 | 16 |

C 第 1 轮产生首 token，随后一直等待 A/B 完成。所有请求均有限结束，不称为已证明永久饥饿。
源码原因：Decode 从 running 队首取最多两个，然后 extendleft(reversed(scheduled_seqs)) 将选中者放回队首，恢复原先优先顺序。
bounded_prefill 只处理 Prefill/Decode 两阶段之间的机会分配，无法改变 Decode 请求间的队列优先顺序。

## 复现与记录

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.unit5_fairness
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```

结果：results/unit5_fairness.json，含逐轮队列、被选请求、生成轮次及源码哈希。
全量 37 passed，新增六组测试确认当前偏置与最终资源释放，不代表已修复公平性。
所有数据单位为调度轮次，无 GPU 计时。

## 已实现：可切换轮转

Config 新增 decode_scheduling_policy，默认 original 保留原版顺序，round_robin 将已调度的请求追加到 running 队尾。
这是与 scheduling_policy 独立的配置：前者决定 Decode 阶段内部顺序，后者决定 Prefill/Decode 阶段选择。配置经真实 Config 暴露，但完整 GPU 调用尚未验证。
请求先在 schedule 中重新入队，postprocess 再移除已完成请求；本次未改 token 处理、释放块或抢占算法。

| A/B 输出长度 | 原版 C 首次 Decode | 轮转 C 首次 Decode | 原版 B 完成 | 轮转 B 完成 |
|---|---:|---:|---:|---:|
| 4 | 5 | 3 | 4 | 5 |
| 8 | 9 | 3 | 8 | 9 |
| 16 | 17 | 3 | 16 | 17 |

各组 A 完成轮次不变。轮转让 C 更早执行，代价是 B 完成延后一轮；两策略组合在该负载下结果相同，因为后续没有 Prefill 等待。
轮转时第 2 轮选 A/B 后队列变为 C/A/B，第 3 轮选 C/A，C 生成第二个 token 完成。
新增六组轮转结果与完成回收验证；空间不足时抢占其他 running 请求的测试扩展为两种 Decode 顺序。全量 44 passed。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.unit5_fairness --compare
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```

对照记录写入 results/unit5_round_robin.json，保留之前的偏置基线文件。
空间不足、持续 Prefill 与请求数量动态变化下不预先承诺一般性公平上界；当前只验证列明的场景。后续已完成组合实验与聊天口述验收，单元 5 学习验收通过。

## 组合实验：持续到达 + 多个 Decode 请求

A/B 输出长度固定 8，C 输出长度 2，均在第 0 轮前到达。第 0 轮 A/B Prefill，第 1 轮 C Prefill。
新增 D1～D6 在第 2～7 轮调度前到达，各 prompt 长度 1、输出长度 1。其他设置与前述实验相同。
四组使用相同的到达计划、输出长度和按生成位置确定的合成 token；无真实 GPU 计时。

| 阶段策略 | Decode 顺序 | A 首次 Decode | C 首次 Decode | D 最大首 token 等待 |
|---|---|---:|---:|---:|
| prefill_first | original | 8 | 15 | 0 |
| bounded_prefill | original | 3 | 13 | 1 |
| prefill_first | round_robin | 8 | 9 | 0 |
| bounded_prefill | round_robin | 3 | 6 | 1 |

只改阶段选择，C 仍受队首偏置影响；只改轮转，Decode 阶段仍被持续 Prefill 推迟。
组合策略第 3 轮执行 A/B Decode，队列轮转为 C/A/B；第 4 轮 D2/D3 Prefill，第 5 轮 D4 Prefill，第 6 轮执行 C/A Decode，C 完成。
C 从第 1 轮首 token 到第 6 轮第二个 token 的间隔为 5，而不是之前单个 Decode 等待者实验的最多 3 轮。再次说明阶段让行不等于每请求每三轮执行一次。
D 请求在 Prefill 后立即完成，无 Decode 轮次，first_decode_round 为 null；单输出无相邻生成间隔，脚本约定 max_generation_gap=0，不表示零耗时。
新加四组组合测试，验证请求未遗漏、每批无重复选中、批大小限制及结果正确；全量 48 passed。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.unit5_fairness --mixed
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```

组合结果保存到 results/unit5_mixed.json，其他基线文件保留。
