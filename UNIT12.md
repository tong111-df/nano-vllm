# 第十二单元：原版回收基线与策略接口

代码、基线实验与 CPU 回归已完成，全量 107 passed。
新增六项接口与顺序测试；用户已亲手运行基线实验并通过口述验收，随后要求进入第十三单元。
用户未提供六项测试单独复跑输出；助手已验证其与原有测试一起通过（107 passed）。
本单元只提供原版空闲队列策略，不宣称已实现 LRU 或获得缓存性能收益。

## 原版到底选谁

_allocate_block 原本从 free_block_ids 队首 popleft；
_deallocate_block 在最后引用归零时，将块编号 append 到队尾。
deallocate(seq) 按 reversed(seq.block_table) 的顺序减少引用，
因此同一请求的尾块通常先于前缀块进入空闲队列。
若某块仍被其他请求共享，不会在这次释放时入队。

这描述的是原版队列机制，不能简单说成“最早分配的块先回收”。
命中空闲缓存时，该块从空闲队列移除；再次释放又排到队尾。
所以一些工作负载下，它也会体现类似最近使用的效果；后续与显式 LRU
结果相同也应如实记录，不能预先假设新策略一定更好。

## 新增接口与职责

新增 nanovllm/engine/cache_policy.py：

| 方法 | 触发时机 | 含义 |
|---|---|---|
| select_victim(free_block_ids) | 需要新块时 | 从当前空闲候选中选择一个物理块编号，不修改队列 |
| on_acquire(block_id) | 一个请求成功取得块引用 | 包括新分配、命中占用缓存、重新使用空闲缓存 |
| on_release(block_id) | 最后一个引用释放，块进入空闲队列 | 不是每次引用数减少都会触发 |

这里“访问事件”明确指取得块引用，不把 can_allocate 的查询当作成功访问，
也没有把每轮 Decode 的 attention 读取或 GPU 内存访问记作 on_acquire。
后续 LRU 必须先明确采用的访问定义，不能把不同定义的策略混成同一结论。

BlockManager 构造函数可通过 eviction_policy 注入满足接口的策略；省略时使用
OriginalFreeQueuePolicy，选择队首且不维护显式访问时间。
当前 Scheduler/Config 尚未新增策略名称开关；可选 LRU 与用户配置接入留在下一单元。
实验的 RecordingOriginalPolicy 只记录这些事件，选择行为与默认策略相同。

BlockManager 继续负责候选合法性检查、空闲/占用集合、ref_count、
旧哈希索引删除、block.reset 和 block_table，不把这些安全职责交给回收策略。
默认策略保留队首 O(1) 弹出；非队首候选通过 deque.remove 移出。
接口实现应遵守不修改候选队列的约定，hook 也不是事务回滚接口。

## 固定基线实验

块池四块，大小 256。默认阶段和 Decode 策略，max_num_seqs=1，预算 1024。
A 先到达：prompt=257、max_tokens=2，分配 [0,1] 并完成 Prefill，仍待 Decode。
B 再到达：不同前缀的 prompt=257、max_tokens=1，分配 [2,3]，Prefill 后立即完成。
A 然后 Decode 完成。

```text
事件                  空闲队列          原因
A_prefill_done        [2,3]             A 占用 0、1
B_finished            [3,2]             B 按尾块到前缀块释放
A_finished            [3,2,1,0]         A 的释放追加到队尾
C_allocated           [2,1,0]           C 取队首块 3
C_finished            [2,1,0,3]         C 完成后追加块 3
D_probe_only          [2,1,0,3]         只查询 A 的前缀，不修改顺序或访问事件
D_prefix_acquired     [1,3]             D 命中块 0，再分配队首块 2 作尾块
D_finished            [1,3,2,0]         D 释放尾块 2、前缀块 0
```

D 命中的 256-token 前缀来自 A；C 使用 B 的旧尾块，不破坏 A 的前缀。
D 分配自己的尾块时则回收了 B 的旧前缀块，其旧哈希索引由 BlockManager 删除。
本实验观察的是 CPU 元数据与队列顺序，没有 GPU KV 张量或推理计时。

## 运行

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.unit12_reclaim_order
```

报告位于 results/unit12/<本次UTC时间与唯一后缀>/report.json，包含固定参数、
系统/Python 信息、源码指纹、队列快照和成功取得引用/归零释放/选择候选的事件。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
$cacheTemp = Join-Path $env:TEMP ('nano-vllm-cache-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider --basetemp "$cacheTemp" tests/test_unit12_reclaim_order.py
```

六项新增检查覆盖原版顺序、查询与实际取得引用的区别、最后引用才触发 release、
非队首选择仍清理旧哈希、拒绝回收仍有引用的块，以及没有空闲块时不调用候选选择。
非队首选择用的是测试专用 TailPolicy，不能将它当成 LRU 实现。
此前 101 项测试也保持通过，默认策略的缓存和调度基线未回归。

## 验收目标

能根据释放顺序推演空闲队列，解释为何不是最先分配的块必然先回收；
能区分查询命中与成功取得引用；能说明策略只能选空闲块、管理器负责索引和引用安全；
知道下一单元的 LRU 需要明确访问定义并实测，不预设新策略一定改善命中率。
