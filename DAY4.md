# Day 4：连续到达负载（进行中）

2026-11-04。今天未改调度算法，新增 experiments/day4_arrivals.py 与 tests/test_day4_arrivals.py。
首次实验与全量测试通过：25 passed，Python 3.12.13。

## 固定负载

A 第 0 轮到达，prompt 长度 1，最多生成 4 个 token。
B1～B6 分别在第 1～6 轮调度前到达，各有 1 个 prompt token、最多生成 1 个 token。
两策略使用完全相同的到达时刻、输入与按输出位置确定的合成 token ID。
Prefill 预算 2，max_num_seqs=4，16 块，每块 256，无空间压力、EOS 或前缀命中。
有限流结束后继续运行直至所有请求完成，以 arrival_count + a_max_tokens + 10 为轮数上限防止无限循环。

## 初次实测

| 指标 | prefill_first | bounded_prefill |
|---|---|---|
| A 生成 token 的轮号 | 0,7,8,9 | 0,3,6,8 |
| A 首次 Decode | 7 | 3 |
| A 相邻生成 token 的轮号差 | 7,1,1 | 3,3,2 |
| A 相邻 Decode 的轮号差 | 1,1 | 3,2 |
| A 完成轮号 | 9 | 8 |
| B3 首 token 等待轮数 | 0 | 1 |
| B6 首 token 等待轮数 | 0 | 1 |
| 其他 B 首 token 等待轮数 | 0 | 0 |

第 0 轮的输出来自 A 的 Prefill，其余来自 Decode。生成间隔包括首个 Prefill 输出到首次 Decode 输出的间隔；Decode 间隔只比较相邻 Decode，不能混淆。
first_token_wait_rounds=first_token_round-arrival_round；同轮执行记为 0，不代表实际耗时为零。

新策略第 3 轮让行 Decode，B3 留在 waiting；第 4 轮 B4 到达，B3、B4 可以在同一轮 Prefill 完成，因为总 token 数为 2，满足预算。
第 6 轮再让行，B6 到第 7 轮完成。第 8 轮已没有 Prefill 等待，A 可直接 Decode，无需刻意凑满两轮。

这说明该有限负载下 A 恢复机会与完成轮次改善，但部分新请求首 token 被推迟。没有 GPU 计时，不声称 TTFT、吞吐量收益或所有负载下的公平性保证。

## 复现

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.day4_arrivals
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```

结果保存为 results/day4_arrivals.json，包含每请求指标、逐轮记录与源码哈希。重复运行覆盖该文件。
## 扩展实验：6、12、24 个连续到达请求

A 输出长度固定为 32，以使它在整个到达期间持续等待 Decode。不要与初次实验 A 输出长度 4 的完成轮次直接比较。
各轮仍只到达一个 B，每个 B prompt 长度 1、输出长度 1。预算仍为 2，其他配置不变。

| B 数量 | 原版 A 首次 Decode | 新策略 A 首次 Decode | 原版/新策略 A 最长生成间隔 | 原版/新策略 A 完成轮号 |
|---|---:|---:|---|---|
| 6 | 7 | 3 | 7 / 3 | 37 / 36 |
| 12 | 13 | 3 | 13 / 3 | 43 / 40 |
| 24 | 25 | 3 | 25 / 3 | 55 / 48 |

三组新策略的 B 最大首 token 等待均为 1 轮，均值为 1/3 轮；原版均为 0。
waiting 在每轮调度前最多 2 个（包括新到达请求），执行后最多残留 1 个；原版对应为 1 和 0。所有有限流最终清空，未在这三组负载观察到等待队列随到达数量持续增长。
每三轮两个 Prefill 批次最多处理四个此类短请求，而这里只到达三个；这解释了积压能被追上。到达率加倍、prompt 变长、块空间不足或多个 Decode 请求时，不保证同样结果。
最长生成间隔 3 轮是这组条件下的观测值，不是一般性的公平或时间上界。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.day4_arrivals --sweep
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```

扩展结果保存为 results/day4_sweep.json，初次实验结果仍保留。新增三组回归测试后全量 28 passed。
## 高到达率：每轮两个短请求

增加 --overload 模式，输出 results/day4_overload.json，同时包含每轮 1 个与 2 个到达的对照。
arrival_count 参数沿用旧名称，但含义为到达轮数；total_arrivals 为实际短请求数。A 输出长度保持 32。

| 到达轮数 | 短请求总数 | 新策略在最后到达轮执行后残留 waiting | 新策略 B 最大首 token 等待 | 新策略 waiting 清空轮号 |
|---|---:|---:|---:|---:|
| 6 | 12 | 4 | 2 | 8 |
| 12 | 24 | 8 | 5 | 17 |
| 24 | 48 | 16 | 11 | 35 |

原版在以上负载中每轮 Prefill 完成两个短请求，到达停止时无 waiting 残留，B 首 token 等待为 0；代价是 A 首次 Decode 分别为第 7、13、25 轮，新策略始终为第 3 轮。
高到达率下两策略 A 完成轮号相同，分别为 37、43、55，不能声称整体完成提速。
新策略每三轮到达六个 B、最多服务四个 B，在 A 活跃且到达持续期间净积压两个。停止到达后最终清空，不代表无限流也稳定。
waiting 中未分配的请求不占 KV 块；本次积压不来自 KV 内存不足。
waiting_clear_round_after_arrivals 表示最后到达轮及之后，首次执行后 waiting 为空的轮号，不表示所有 running 请求完成；all_finished_round 单独记录整体完成。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.day4_arrivals --overload
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```

新增三组积压增长与最终清空回归测试，全量 31 passed。
学习验收尚待继续，不以测试通过视为整天已完成。
