# 第十三单元：显式 LRU 与原版回收对照

实现、CPU 实验、用户亲手复跑与口述验收均已完成。用户复跑 11 项测试全部通过。
用户已解释空闲候选限制、查询与取得引用的区别、释放不刷新访问时间，
以及规则正确仍可能退化、管理开销可能抵消缓存命中收益。
新增 11 项测试，全量 118 passed。真实 GPU 验证仍未进行。

## 访问定义先于实现

本项目的 lru 明确定义为“最近成功取得块引用”的 LRU：

- 新分配、共享已占用前缀、重新取得空闲缓存块，都调用 on_acquire。
- 每次 on_acquire 增加逻辑计数器，并记录该物理块的 last_access。
- can_allocate 只查询，不增加计数、不改变 last_access。
- on_release 表示块变为空闲，不刷新访问时间。
- 每轮 Decode、Prefill 的 attention 读取没有接入此访问事件，不能把它叫作所有实际 KV 读取的 LRU。

策略从 free_block_ids 中选 last_access 最小者；从未取得过引用的块按时间 0 处理，
时间相同时保持原空闲队列顺序。只在物理块池编号上维护时间，不随请求数无限增加条目。
不从 used_block_ids 回收块；引用、旧索引删除和内容元数据重置仍由 BlockManager 管理。
已有候选检查也继续保护仍被引用的块。

## 实现与选择入口

cache_policy.py 中 LRUCachePolicy 实现第十二单元的接口；
create_cache_policy 支持 original/lru，对错误名称明确报错。
Config 新增 cache_eviction_policy，默认 original；Scheduler 缺省该字段也保持 original。
可直接通过 LLM(..., cache_eviction_policy="lru") 传入真实 Config，
但该 GPU 路径尚未实测。GPU smoke 对应新增 --cache-policy lru。
第七单元的混合调度 JSON 格式未扩展为任意缓存实验配置，本单元使用单独的固定对照入口。

当前 LRU 选择候选使用 min，时间复杂度 O(F)，F 为当前空闲块数；
若选中非队首块，deque.remove 也是 O(F)。原策略保持队首弹出 O(1)。
多了逻辑计数和访问时间维护，不把新策略当作零开销优化。

## 同一负载下的退化案例

四块池，A/B 的 prompt 各 257 个 token、前缀不同。
A 先取得 [0,1]，B 再取得 [2,3]；B 先完成，A 随后 Decode 完成。
两种策略此时空闲队列都是 [3,2,1,0]。

LRU 的访问逻辑时间为：

```text
块 0：1
块 1：2
块 2：3
块 3：4
```

A 最后执行 Decode 不会重新取得引用，因此不会刷新这些时间。
C 随后需要一个新块：原版按队首取 3，LRU 按最旧取得引用时间取 0。
D 然后到达，prompt 的前 256 个 token 与 A 相同，最后一个不同。

| 策略 | C 回收块 | D 命中前缀 token | D 实算 prompt token |
|---|---|---|---|
| original | 3 | 256 | 1 |
| lru | 0 | 0 | 257 |

LRU 此时已经移除块 0 的旧前缀索引，D 必须重算。
这组负载中原版更好，体现本访问定义的取舍；不能为了“新策略必须更好”而隐藏结果。
只比较当前例子的命中与重算量，不把它推广成所有 LRU 实现或负载的结论。
相反负载和更完整热点/扫描对照在后续单元进行，当前没有平均收益或 GPU 性能结论。

## 亲手运行

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.unit13_lru_compare
```

报告保存到 results/unit13/<本次UTC时间与唯一后缀>/report.json，
记录固定参数、访问定义、源码指纹、Python/系统及各策略的逐事件队列、块表、
引用数、前缀索引与访问时间。原策略不维护 last_access，报告对应字段为空字典。
输出 token 仍是合成值，只验证真实管理代码的 CPU 元数据。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
$lruTemp = Join-Path $env:TEMP ('nano-vllm-lru-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider --basetemp "$lruTemp" tests/test_unit13_lru.py
```

11 项新增检查覆盖固定对照退化、未访问块与平局处理、查询不刷新/命中刷新、
释放不刷新、仍有引用的旧块不被回收、四种阶段/Decode 组合下共享和空闲复用、
非法策略配置。已有 107 项测试保持通过。
缓存哈希链、抢占等更深入的 LRU 交互验证在第十四单元展开，不声称本单元穷尽所有边界。

## 验收目标

能根据访问时间在空闲候选中选择 LRU，能解释为什么刚释放不一定是最近取得引用；
能区分查询/成功取得引用/实际 attention 读取，说明本 LRU 的访问定义；
能解释原版与 LRU 不同的选块结果以及本例的退化，保留仍被引用块和哈希索引的安全边界。
