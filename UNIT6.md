# 单元 6：可靠性（进行中）

## 第一步：拒绝根本装不下的 prompt

当前每块 256 token。池只有 1 块而 prompt 长度 257 时，等待也无法获得所需的 2 块。
旧 add() 无条件入队，后续 can_allocate 返回 -1；若没有可执行 running 请求，schedule 最后触发空批次断言，报错没有说明请求容量问题。
现在 add() 在入队和分配前检查 seq.num_blocks 与整个池的块数，不足时抛 ValueError，信息明确显示所需与可用总块数。
检查使用整个池容量，不使用当前 free 数量；暂时被其他请求占用不等于请求永远无法执行。

测试先复现未拒绝问题（2 failed, 1 passed），修改后全量 51 passed。
新增覆盖 1 块/257-token prompt、2 块/513-token prompt，拒绝不改变队列和块分配，既有有效请求仍可完成。
边界：1 块、256-token prompt、max_tokens=1 可以执行。最终长度 257，但最后生成 token 无需再作为输入计算 KV，不需要第二块。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider tests/test_unit6_reliability.py
```

## 第二步：按最大生成长度做保守准入

仅检查 prompt 不足：1 块、256-token prompt、max_tokens=2，初始可以 Prefill，但随后 Decode 需要不存在的第二块。
add() 现在进一步校验最大 KV token 数：prompt_length + max_tokens - 1，对块大小向上取整后不能超过整个池容量。
减一的原因：最后一个输出 token 不再输入模型，无须为其计算 KV。
这是明确的保守契约，不依赖提前 EOS；某些实际上会提前 EOS、能够完成的请求也会被拒绝。
只检查单请求理论上能否容纳，不提前分配或预留整个输出空间，不保证多请求并发永不缺块。
当前策略没有 offloading 或滑动窗口，无法靠丢弃旧 KV 来支持超过池总容量的长序列。

新用例：256+2、255+3、1+257 在 1 块池中均入队前拒绝；255-token prompt、生成 2 个仍允许，最终序列长 257，最后输出无需第二块。
测试验证拒绝时未产生输出或分配块；边界请求实际 Prefill/Decode 完成后释放资源。全量 55 passed。

## 第三步：执行前拒绝无法推进的参数

SamplingParams 要求 max_tokens 是正整数；当前不支持 max_tokens=0 表示零输出请求。
原结束检查发生在 append_token 之后，0 或负数不会因长度相等而结束；没有 EOS 时可能继续执行直至其他错误。
Sequence 在读取 token_ids[-1] 前明确拒绝空列表，报 ValueError 而非 IndexError；不自动添加 BOS。
Scheduler 要求 max_num_seqs、max_num_batched_tokens、num_kvcache_blocks、kvcache_block_size 为正整数，避免零预算、零批次上限或零容量运行。
这些检查发生在对应对象开始执行或创建 BlockManager 之前；GPU 完整初始化流程仍未经验证。

六个新增用例先复现失败，再修复；全量 61 passed。当前测试覆盖 0/-1 输出上限、空 prompt、三项零配置。
暂时资源竞争、无法推进的调度与错误后的恢复仍待处理，不能宣称可靠性已完成。完整 GPU 路径未验证。

## 第四步：输出数量错误不能导致部分更新

原 postprocess 用 zip(seqs, token_ids)，输出过少会漏掉请求，输出过多会被丢弃，均不报错。
现在进入循环前检查长度一致，抛 ValueError 显示预期和实际数量。
两个新用例验证：调度两个请求却返回一个或三个 token 时，生成数量、计算进度、队列和块分配保持不变；随后给同一批次正确数量的结果仍可完成并释放块。
先复现两组失败，修复后全量 63 passed。
这只是数量不匹配的更新前检查，不是任意异常下的事务回滚，也不代表可重新执行整个模型步骤；测试重用的是原批次的结果处理。

## 第五步：无法调度的明确诊断与真实暂时缺块恢复

引入 SchedulerNoProgressError(RuntimeError)。空队列上调用 schedule 明确报 no pending requests。
调度尝试后仍无可执行批次时，用包含 waiting/running 数、free/used/total block 数的异常替代 assert，不返回空批次，也不自动无限重试。
测试通过故障注入让 can_allocate 暂时拒绝有效请求，验证诊断字段、未丢请求；移除注入原因后原请求可执行。这验证错误处理，不把人为故障称为真实负载中的死锁。
若调度尝试中已经发生抢占，异常并不回滚此前抢占；调用者需检查当前状态，不能声称所有错误均无副作用。
LLMEngine.step 未吞掉该异常，因此不会在正常 generate 循环中默默重复空批次；完整 GPU 路径仍未实测。

另有真实暂时缺块用例：池共两块，A/B 各需两块且 max_tokens=1。两者均允许入队，A 分配两块后 B 暂时无法分配；A 完成释放后 B 正常执行。
四种阶段/Decode 顺序组合均通过，证明总容量检查没有错误地要求所有并发请求同时装下。
本单元完成时全量 69 passed。口述验收已完成，包括保守准入可能拒绝本可提前 EOS 的请求这一代价；不声称已自动解决所有资源死锁、OOM 或 GPU 故障。
