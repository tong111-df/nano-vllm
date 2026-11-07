# 第十一单元：共享缓存块的生命周期

新增生命周期实验与四项测试；保留原版 BlockManager 实现，尚未引入 LRU。
代码、生命周期实验、用户亲手复跑与口述验收已完成。
用户复跑四项测试全部通过；本单元实现后助手全量回归 101 passed。
用户已解释引用计数 1→2→1→0、空闲不等于缓存失效、覆盖后不能凭物理编号复用，
以及完整块计算完成后即可共享，不必等待整个 prompt 完成。GPU KV 张量仍未验证。

## 本单元的新内容

从两个请求共享同一个完整前缀块开始，追踪引用计数、物理块编号、
空闲/占用集合和前缀索引；再观察全部释放、空闲命中、覆盖后失效与重新计算。
此前测试主要验证一个请求释放后另一个请求命中，本次加入同时引用的情况。

用真实 Scheduler/Sequence/BlockManager，人工 token 代替模型输出。
这里验证控制面元数据，不创建或检查 GPU 上的 KV 张量。

## 固定实验

块大小 256、池共三块、max_num_seqs=1、Prefill 预算 1024，默认阶段与队列策略。
A/B 的 prompt 各 257 个 token：前 256 个相同，最后一个不同；各生成两个 token。
A 先执行完整 Prefill，B 再到达。因为共享块已算满，B 可命中一个完整前缀块。

| 事件 | 被跟踪物理块的引用数 | 原前缀索引 | 含义 |
|---|---|---|---|
| A Prefill 完成 | 1 | 有 | A 独自引用完整前缀 |
| B 分配并命中 | 2 | 有 | A/B 的 block_table 指向同一前缀块，尾块各自独立 |
| A 完成 | 1 | 有 | 仅释放 A 的引用，共享块仍被 B 使用 |
| B 完成 | 0 | 有 | 最后引用释放，块进入空闲队列，缓存元数据保留 |
| C 命中空闲前缀 | 1 | 有 | 块重新变为占用，避免重复计算 256 个 token |
| C 完成 | 0 | 有 | 再次变为空闲缓存 |
| D 分配全部三块 | 1 | 无 | 旧物理块被分配作其他用途，旧索引被移除，元数据重置 |
| E 请求原前缀 | 1 | 无 | 不能再命中旧前缀，需要重新计算 |
| E Prefill 完成 | 0 | 重新有 | E 算完建立了新的有效缓存，并随请求完成释放 |

注意 D 分配之后，跟踪的仍是同一个物理块编号，但它已不再代表原前缀。
不能把物理编号相同理解成内容相同。分配时 reset 重置的是管理元数据；
原 GPU 存储不需要在此被清零，新计算会写入对应槽位，本 CPU 实验没有验证那些写入。

## 关键调用链

```text
Scheduler.schedule
  → can_allocate：沿哈希链查询可命中的完整前缀，检查所需空闲空间
  → allocate：共享已占用块时引用 +1；重新引用空闲缓存时移出空闲队列
  → 模型计算（本实验用合成输出替代）
Scheduler.postprocess
  → hash_blocks：为本轮算满的块建立内容与前缀哈希索引
  → 请求完成时 deallocate：引用 -1，只有减至 0 才进入空闲队列
```

原版 _allocate_block 从 free_block_ids 队首取块；若该块仍拥有旧哈希索引，
则移除相应索引并 reset，再用作新块。当前未新增显式 LRU 策略。
can_allocate 是查询，不分配块，不修改探测请求的 cached_tokens。

## 亲手运行与阅读

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.unit11_cache_lifecycle
```

先只看 A_prefill_done、B_prefix_hit、A_finished、B_finished 四行。
block_ref 是固定物理块的引用数，free_blocks 是空闲物理块编号列表，
prefix_index 是原前缀当前索引的物理块编号；None 表示未找到此索引，0 是有效编号。
结果保存到本次 results/unit11/<UTC时间与唯一后缀>/report.json。
报告含逐事件状态、固定配置、Python/系统信息与源码指纹。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
$cacheTemp = Join-Path $env:TEMP ('nano-vllm-cache-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider --basetemp "$cacheTemp" tests/test_unit11_cache_lifecycle.py
```

四项检查：

- 同时共享与最后引用释放；A 完成后共享块仍被 B 引用。
- 空闲命中与覆盖失效；旧索引移除后原前缀重算。
- 每个快照的块不变量：引用数等于各 block_table 的引用总数，free/used 互斥且覆盖整个池，
  空闲队列无重复，FINISHED 请求不再持块，所有请求最终完成。
- 分块 Prefill：第一轮算 128 个 token 不能共享；第二轮累计算满 256 后，
  即使整个 257-token prompt 尚未完成，其完整前缀已可被查询命中。

## 验收目标

能推演引用数 1→2→1→0；能解释最后一个引用释放与缓存失效的差别；
能指出物理编号、内容和前缀索引不是同一概念；
能解释为什么分配空间不代表该块内容已经计算完成，完整计算后才能建立可复用前缀。
