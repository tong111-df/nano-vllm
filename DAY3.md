# Day 3：可切换的 Prefill 让行策略

日期：2026-11-03。固定上游基线：bb823b3e06983d71485a8e1f23715ebd87d98ef8。

## 改动与规则

Config 新增 scheduling_policy（默认 prefill_first）和 max_consecutive_prefills（默认 2）。
Scheduler 默认保留原版阶段选择；选择 bounded_prefill 时启用让行。

1. 调度开始时保存 had_running。没有 running 请求则计数归零。
2. 本轮成功安排 Prefill，且调度开始时已有 running 请求，计数加一。
3. 达到上限，且至少一个 running 请求满足 can_append 时，跳过 Prefill，进入原有 Decode 调度。
4. 成功安排一轮 Decode 后计数归零。

计数按轮次而非请求数或 token 数累计；不支持 Prefill/Decode 混合批次。
新策略并不保证每个 running 请求都在上限之后执行，仍受请求数量限制、队列顺序和空间影响。

## 空间不足时的修正

测试复现：池中仅两块，A 占一块且 Decode 需新块，B 在 waiting 中做分块 Prefill并占另一块。
无条件强制 Decode 会抢占 A 自己，然后因 scheduled_seqs 为空触发断言。
新增 can_append 条件，允许 B 先完成剩余 Prefill、结束释放块，随后 A 执行。
因此连续两轮并非无条件的等待上界。若所有 Decode 都缺空间，可能执行更多 Prefill。
原版的不可调度输入、所有请求均无法推进等情况尚未全面处理；不能声称完全解决空间不足。

另一个三块场景验证：A Decode 需扩块、C 在 waiting 做分块 Prefill时，仍可抢占另一个 running 请求 B，为 A 腾出空间。B 已生成 token 保留，之后恢复并结束。
空间测试在准备阶段使用预算 512 建立跨块请求，再切换到预算 2 制造分块竞争；这是测试设置，不是新策略自动调预算。

## 对照结果

同一工作负载：预算 2，块大小 256，4 块，最多 4 个请求，限额 2，无 EOS、缓存命中或空间压力。
A prompt=[10]、max_tokens=3，第 0 轮完成初始 Prefill；B prompt 长度 5、max_tokens=1，第 1 轮前到达。

| 策略 | A 首次 Decode | B 首 token | A 完成 |
|---|---:|---:|---:|
| prefill_first | 4 | 3 | 5 |
| bounded_prefill | 3 | 4 | 5 |

轮号从准备阶段的第 0 轮开始。新策略让 A 首次 Decode 提前一轮，B 首 token 推迟一轮，A 完成轮次不变。
这些是合成输出 token 下的真实控制面调度记录，不是 GPU 时间、TTFT 或吞吐量收益。
新旧策略使用相同输入 token 和输出长度；输出 token ID 由轮号合成，不是实际模型生成。

## 验证与复现

实测 Python 3.12.13，23 passed。覆盖默认基线、新策略主路径、无 Decode 等待、重复计数周期、空间受限回退、抢占恢复、上限 1/2/3、多请求 Prefill 每轮只计一次。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
.\.venv\Scripts\python.exe -m experiments.day3_compare
```

新记录：results/day3_compare.json，包含配置、源码哈希、逐轮轨迹；重复运行覆盖该文件。
Day 2 的结果保留在 results/day2_baseline.json。

## 学习验收

已在聊天中完成口述验收。had_running 的计数含义与“B 首 token 推迟”的代价经过提示后理解，后续需独立复述巩固。
运行与口述验收均已完成，不代表 GPU 推理路径已验证。
