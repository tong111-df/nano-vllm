# 第十五单元：热点、扫描和混合负载对照

实现、实验与助手 CPU 回归完成，全量 141 passed；新增七项验证。
用户已亲手运行负载实验、复跑七项测试并全部通过，学习与口述验收已完成。
已讲解冷热与容量影响、257-token 请求需要两块、候选检查数 13/40 的计算，
并澄清策略 CPU 方法耗时不能换算成整个 GPU 推理的速度倍数。
未修改运行时缓存策略。

## 固定负载

experiments/configs/unit15_workloads.json 保存前缀 ID 序列、块池容量和策略名称。
每个请求顺序到达，前一个完成后才处理下一个。prompt=257、输出长度=1。
同一前缀 ID 表示同一组 256 个前缀 token，尾 token 随请求位置变化。
前缀 k 的 token 为 range(k*1000,k*1000+256)，尾 token 为 1000000+请求序号。

| 名称 | 前缀 ID 序列 | 想观察什么 |
|---|---|---|
| hot | 连续 12 次 0 | 冷启动后的重复热点是否保留 |
| scan | 1 到 12，各一次 | 没有重用机会时的表现与管理成本 |
| mixed | 0,1,2,0,3,4,0,5,6,0,7,8 | 热点被扫描请求穿插，容量能否保留它 |

每个负载在 4/6 块池和 original/lru 下对照，共 12 组。
块大小 256，预算 1024，max_num_seqs=1，Prefill 优先，Decode 顺序为 original。
每个请求只生成一个 token，不会进入 Decode。并发交互不由本单元负载覆盖，
此前第十三/十四单元的退化与交互检查继续保留。
输出由合成 token 代替，真实 Scheduler/BlockManager 运行；没有计算 GPU KV 张量。

## 已验证的 token 账目

每组共 12×257=3084 个 prompt token。

| 池容量 | 负载 | original 命中/需计算 | lru 命中/需计算 |
|---|---|---|---|
| 4 | hot | 2816 / 268 | 2816 / 268 |
| 4 | scan | 0 / 3084 | 0 / 3084 |
| 4 | mixed | 0 / 3084 | 0 / 3084 |
| 6 | hot | 2816 / 268 | 2816 / 268 |
| 6 | scan | 0 / 3084 | 0 / 3084 |
| 6 | mixed | 768 / 2316 | 768 / 2316 |

每行满足 cached_tokens + computed_prompt_tokens = total_prompt_tokens。
computed_prompt_tokens 表示 Scheduler 安排的实际计算输入数量，不是 GPU 已执行的测量。
其中包含首次冷计算和未命中的尾 token，不应把所有 computed 都称为重复计算。
hot：首个请求计算 257；其余 11 个每次命中 256、只算尾 token 一个，
所以命中 11×256=2816，需计算 257+11=268。
mixed：增大容量后，后续三个热点重访各命中 256，共 768。

报告的 prompt_token_hit_rate 以所有 prompt token 为分母；
requests_with_prefix_hit 是发生过前缀命中的请求数，不能将两种比例混淆。
本组两种策略命中量相同；它不推翻第十三单元的退化案例，
也不能证明两种策略对所有负载等价或 LRU 平均更好。

## 管理开销的两种证据

第一种是稳定的算法工作量 selection_candidate_entries：
当前原版每次选块读队首一个候选，LRU 的 min 会检查全部空闲候选。
此计数由两种已知实现的候选规模计算，不是读取硬件性能计数器。

| 容量 | 负载 | original 候选检查数 | lru 候选检查数 |
|---|---|---|---|
| 4 | hot | 13 | 40 |
| 4 | scan / mixed | 24 | 84 |
| 6 | hot | 13 | 66 |
| 6 | scan | 24 | 132 |
| 6 | mixed | 21 | 114 |

同时记录 select/acquire/release 调用次数和 recency_metadata_entries。
本负载每个请求取得并释放两个引用，所以每组 acquire/release 各 24 次。

第二种是 perf_counter_ns 的探索性 CPU 计时：

- policy_delegate：选择候选与 acquire/release 的被包装策略方法调用，包含计时开销，
  不包含 BlockManager 的队列移除、哈希计算等操作。
- instrumented_control_plane：add/schedule/postprocess 累计时间，包含策略测量包装开销，
  不包含模型计算、输入构造、轨迹构造与 Scheduler 初始化。

默认每组先预热一轮再测五轮，每轮都重新创建 Scheduler，缓存池从空开始。
预热用于解释器/执行路径，不把前一次的缓存带进测量造成假命中。
每轮的指标和轨迹必须一致；耗时保留原始样本及 min/median/max。
策略调用开销很小，计时器、操作系统调度、固定测试顺序等均可能影响数值，
不根据几个微秒宣称稳定速度倍数或统计显著性，更不能当作 GPU 推理延迟。
相同命中下仍可发现候选检查成本差异，但不能据此精确预测整个推理耗时。

## 运行与结果

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.unit15_cache_workloads
```

输出 cached/computed 为 token 数，candidate_entries 为候选检查数，
policy_cpu_us_med 为一整个 12 请求负载的策略方法累计耗时中位数，单位微秒。
不是每个 token 的时间，也不是整个调度流程总耗时；重新运行该列可以变化。

结果保存到 results/unit15/<本次UTC时间与唯一后缀>/：
summary.json 是各组汇总；report.json 包含完整负载配置、输入生成规则、
计时范围、各次样本、逐请求轨迹、环境依赖版本、源码指纹和 Git 状态。
不覆盖以前运行，源文件在运行期间改变会拒绝归档。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
$workloadTemp = Join-Path $env:TEMP ('nano-vllm-workloads-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider --basetemp "$workloadTemp" tests/test_unit15_cache_workloads.py
```

预期 7 passed。检查完整矩阵与 token 守恒、测量包装不改变缓存行为、
同一负载重复运行得到相同计数/轨迹、报告落盘、非法配置在实验前拒绝。
不写“LRU 耗时必须大于原版”这种依赖运行噪声的断言。

## 验收目标

能解释热点与扫描、冷启动与稳态重用；能计算命中/需计算 token 总量；
能区分 token 命中比例和请求命中数量；能解释容量影响和相同策略结果；
能区分候选检查数、探索性 CPU 管理计时与真实 GPU 推理时间。
