# 第八单元：让行上限与 Prefill 预算对照

实现、亲手复跑和口述验收已完成；本单元完成时全量 87 passed。
用户已解释 A/B 取舍、首次 Decode 与完成的区别、预算可能未用满、批次计数单位，
以及调度轮次不能直接代表 GPU 耗时。
沿用第七单元的统一入口；调度算法本身没有在本单元改变。

## 固定条件与两个自变量

第 0 轮 A 到达，prompt=1，max_tokens=8。
第 1～6 轮，每轮一个 B 请求到达，各 prompt=5，max_tokens=1。
EOS=-1，使用固定合成输出，不提前结束；块池 16、块大小 256，max_num_seqs=4。
这组最多七个请求，每个只需一块，不以缺块或抢占制造调度差异。

预算取 1/2/5，每种预算均比较 prefill_first 与 bounded_prefill 上限 1/2/3，
共 12 组。Decode 顺序固定 original，不同时改变队列公平性策略。
每组具有相同的请求到达、prompt 和输出长度。
比较上限时先固定预算；比较预算时先固定阶段策略及上限。

`budget` 是一轮 Prefill 最多计算的 token 数；
`max_consecutive_prefills` 是有 Decode 等待时允许连续执行的 Prefill 批次数。
它不是请求数，也不是毫秒上限；仍受 Decode 是否可执行的条件约束。
本实验只有 A 需要 Decode，不能据此证明多 running 请求的普遍公平性。

## 运行

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.unit8_sweep
```

入口读取 `experiments/configs/unit8_arrivals.json`，依次覆盖 budget 为 1/2/5，
各自调用统一入口 run_experiment。该配置可单独用于 experiments.run，默认预算为 2。
未对真实 Scheduler 另写模拟算法。

输出存于 `results/unit8/<本次UTC时间与后缀>/`：

- `summary.csv`、`summary.json`：12 组指标；每行记录原始 report 相对路径与 case_index。
- 三个子目录：分别保存各预算的完整 config.json、report.json，含源码指纹、环境和轨迹。

汇总文件在全部成功后写出；中途失败可能留下原始报告，但不会生成完整汇总。
每次运行使用新目录，不覆盖此前记录。

## 已验证结果

表中全是从第 0 轮开始编号的调度轮次；B_max_wait 是各 B 的首 token 轮次减到达轮次。
`-` 表示原策略不使用让行上限。

| 预算 | 策略 | 上限 | A 首次 Decode | A 完成 | B 最大首 token 等待 | 全部完成 |
|---|---|---|---|---|---|---|
| 1 | prefill_first | - | 31 | 37 | 24 | 37 |
| 1 | bounded_prefill | 1 | 2 | 14 | 31 | 37 |
| 1 | bounded_prefill | 2 | 3 | 21 | 31 | 37 |
| 1 | bounded_prefill | 3 | 4 | 28 | 31 | 37 |
| 2 | prefill_first | - | 19 | 25 | 12 | 25 |
| 2 | bounded_prefill | 1 | 2 | 14 | 19 | 25 |
| 2 | bounded_prefill | 2 | 3 | 21 | 19 | 25 |
| 2 | bounded_prefill | 3 | 4 | 25 | 17 | 25 |
| 5 | prefill_first | - | 7 | 13 | 0 | 13 |
| 5 | bounded_prefill | 1 | 2 | 13 | 5 | 13 |
| 5 | bounded_prefill | 2 | 3 | 13 | 2 | 13 |
| 5 | bounded_prefill | 3 | 4 | 13 | 1 | 13 |

先只读预算 2 的四行，区分收益（A 更早 Decode）与代价（B 等待增加）。
再看预算 5：首次 Decode 提前并没有让 A 更早完成；
最后看预算 1：上限更严格也未让 B 最大等待继续严格增加，不能声称所有指标随上限单调变化。
这些负载下，同一预算各策略全部完成轮次相同，不代表一般负载必然如此。

## 用轨迹解释预算

预算 2 时每个 B 的 Prefill 分三轮，计算量为 2、2、1，最后一轮才生成输出。
当前 Scheduler 只允许批次中的第一个请求执行部分 Prefill；
计算最后 1 个 token 后剩下的 1 个预算，无法容纳下一个完整的 5-token prompt，
因此本负载每个 B 都需要三轮，而不是把 30 个 prompt token 理想地打包成 15 轮。
原策略六个 B 共占 18 轮 Prefill，A 第一次 Decode 在第 19 轮。

## 代码与验证

到达实验新增 short_prompt_tokens，默认仍为 1，因此旧命令行为保持。
轮数保护随总 prompt 计算量扩大，防止 budget=1 时被旧的过短保护上限截断。
测试检查三块 Prefill 完成前不采样、12 组保持相同负载、逐轮预算限制、
各 B 恰好完成 5 个 prompt token、输出数量和资源释放，以及汇总与原始报告一致。
原实验脚本内仍验证所有请求完成后块引用归零。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```

## 验收目标

能说明两个参数的单位；能固定一个参数比较另一个；
能用预算 2 的结果解释 A 的收益和 B 的代价；
能用预算 5 说明首次 Decode 提前不保证更早完成；
能解释轮次减少不等于真实 GPU 时间减少，不依据这一负载挑选普遍最优参数。
