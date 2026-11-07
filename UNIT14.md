# 第十四单元：缓存共享、哈希链与抢占的交互

新增实验和 16 项交互检查，均通过；全量 CPU 回归 134 passed。
用户已亲手运行实验、复跑 16 项测试并全部通过，口述验收已完成。
已讲解恢复时重算原 prompt 尾 token 和已有生成 token 共两个位置，
并确认哈希链依赖前置上下文、删除索引须检查其是否仍指向被覆盖的物理块。
本单元未修改运行时调度或缓存算法。
检查覆盖真实 Scheduler/BlockManager 元数据，不是 GPU KV 数值正确性的验证。

## 新知识一：抢占后不一定全部重算

固定池三块、块大小 256，max_num_seqs=1、预算 1024。
A/B prompt 各 257 个 token，共用前 256 个，尾 token 不同；
A 最多生成三个 token，B 最多生成两个，均已 Prefill 并生成一个 token。

```text
A.block_table = [共享前缀, A尾块]
B.block_table = [共享前缀, B尾块]
共享前缀 ref_count = 2
```

实验手动触发抢占 A：先从 running 移除 A，再调用 preempt，
符合 Scheduler 内部先取出请求再抢占的调用约定。
这是控制变量场景，不将其说成资源不足自动触发；自动抢占另有测试。

A 的块表清空、num_cached_tokens 归零，已生成 token IDs 保留；
共享块引用数降到 1，由 B 继续持有，不能被回收。
A 恢复时 can_allocate 再次查到共享前缀，命中 256 个 token，
只重算原 prompt 尾部一个 token 和之前生成的一个 token，共两个。
恢复 Prefill 完成后追加第二个生成 token，最后一次 Decode 生成第三个并结束。

```text
事件                         A_cached shared_ref A_generated
A_prefill_done                   257          1           1
B_prefill_done                   257          2           1
A_preempted_B_holds_prefix         0          1           1
A_resume_scheduled               256          2           1
A_resumed_prefill_done           258          2           2
最终全部结束                       0          0           3
```

A_cached 是请求当前计算进度计数，不是全局缓存中保存了多少 token。
清零不等于清空整个池。恢复是否命中取决于有效前缀是否还存在。
本次 original/lru 结果一致，八种缓存/阶段/Decode 组合均验证共享保护与恢复。

## 新知识二：后块相同，不等于 KV 上下文相同

对于有两个完整块和一个尾 token 的 513-token prompt，索引按前缀链计算：

```text
h0 = compute_hash(第一块token)
h1 = compute_hash(第二块token, h0)
```

另一个请求即使第二块的 256 个 token 完全相同，只要第一块不同，
第二块所处的上下文就不同，不能拿原第二块当作相同前缀直接复用。
本 CPU 测试验证索引与调度行为，不执行 attention 数值对照。

在两个缓存策略下分别检查：原 prompt 命中两块；改变第一块命中零块；
只改变第二块命中一块。随后实际调度改变第一块的请求，验证全部 513 token 重算。

## 新知识三：重复副本与索引归属

同一批次接纳 A/B 时，两者相同前缀都可能尚未计算，因而各分配一份物理块。
postprocess 顺序发布相同前缀后，hash_to_block_id 指向后发布的副本，
但前一份仍可能有效且被引用；该字典不是“一个哈希到所有副本”的列表。

测试让旧副本所在请求先完成，再回收其物理块，另一副本仍被 B 持有。
旧副本被覆盖时只能删除仍指向自己的索引，不能误删指向新副本的映射：

```python
if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
    del self.hash_to_block_id[block.hash]
```

随后 D 实际复用 B 持有的新副本，验证 cached_tokens=256、共享引用数增加。
测试先用批次上限 2 建立双副本，再将上限调整为 1 以控制后续完成次序。
此用例 original/lru 均通过。没有新增多副本索引或搜索旧副本的算法。

## 自动抢占与覆盖

另外四组检查覆盖 original/lru 和两种 Decode 顺序，阶段固定 bounded_prefill。
两块池，A 的 256-token prompt 生成一个 token 后需要新块；B 也占一块。
schedule 自动抢占 B，将它的块分配给 A；A 完成后，B 重新 Prefill。
B 原 prompt 一个 token 加已生成一个 token，共两个需要重算，
原输出 ID 保留，恢复后累计输出恰好两个。
这个场景没有可命中的完整前缀，不与上述共享保留的场景混为一谈。

检查还包括引用数等于块表引用总数、空闲与占用集合互斥且覆盖整个池、
空闲队列无重复、哈希索引指向对应完整块以及最终资源释放。
不能据此声称已穷尽任意异常恢复、并发资源死锁或 GPU 错误。

## 运行

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.unit14_cache_interactions
```

先只看 original 的 A_preempted_B_holds_prefix 与 A_resume_scheduled 两行。
报告保存到 results/unit14/<本次UTC时间与唯一后缀>/report.json，
包含触发方式、固定参数、源码指纹、系统/Python 信息以及各策略逐事件状态。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
$interactionTemp = Join-Path $env:TEMP ('nano-vllm-interactions-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider --basetemp "$interactionTemp" tests/test_unit14_cache_interactions.py
```

预期 16 passed。八组共享恢复、两组哈希链、两组重复副本索引、四组自动抢占覆盖。

## 验收目标

能区分请求缓存进度清零与全局缓存失效；能算出恢复时共享前缀和重算 token 数；
能解释相同块内容仍需要相同前置上下文；能解释删除旧索引前检查当前指向的必要性；
能区分手动控制抢占场景、自动资源不足抢占以及 CPU 元数据验证的范围。
