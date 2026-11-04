# Day 1：CPU 调度实验

日期：2026-11-01。目标：12 月 1 日开始投递；今天只建立正确性基线。

## 固定版本与实测

- 上游：https://github.com/GeeeekExplorer/nano-vllm
- 基线 commit：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`
- 本地分支：`feat/day1-cpu-tests`（新增修改尚未提交）
- 实测：Windows，Python 3.12.13，`python -m pytest -q`：7 passed。
- 当前 .venv 来自 `D:\anaconda\envs\pytorch\python.exe`，隔离安装 CPU 依赖。
- 用户提供的 `C:\Users\tong\AppData\Local\Programs\Python\Python312\python.exe` 已确认版本为 3.12.9，也满足上游 >=3.10,<3.13 要求。

## 自己运行（PowerShell）

先进入本文件所在仓库目录，已有 .venv 可直接运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -v tests/test_cpu_scheduler.py::test_request_lifecycle
```

在新目录复现时，先 clone 并 checkout 上述 commit，再应用本仓库的修改与测试文件；只 checkout 上游 commit 不包含这些文件。然后创建环境：

```powershell
& 'C:\Users\tong\AppData\Local\Programs\Python\Python312\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pytest -q
```

不需要激活环境，不需要下载模型。不要执行 `pip install -e .`：那会安装上游声明的整套 GPU 依赖。前三个直接测试依赖已固定版本，pytest 的间接依赖未做完整锁定。

## 先理解实际导入链（20 分钟）

1. 原包入口 `nanovllm/__init__.py` 会立即导入 LLM，继而加载 LLMEngine、ModelRunner 和推理依赖。现在仅在请求 `LLM` 时延迟导入，`SamplingParams` 仍可直接导入。
2. Scheduler 原本为类型注解导入 Config，而 Config 顶层导入 transformers。现在使用延迟注解和 TYPE_CHECKING，使 Config 只参与静态类型检查。
3. Sequence 仅依赖标准库和 SamplingParams。BlockManager 需要真实的 numpy 和 xxhash；测试不替换其哈希或分配逻辑。
4. 测试用 SimpleNamespace 提供 Scheduler 实际读取的六个配置字段，绕过 Config 的模型路径、模型配置读取和 GPU 配置校验。它不代表生产配置验证。

没有改 schedule/postprocess/BlockManager 算法。`test_import_does_not_load_inference_dependencies` 在新 Python 进程中验证不加载 torch、triton、transformers、flash_attn 和 ModelRunner。
延迟导入保留了 `from nanovllm import LLM` 入口，但完整 GPU 推理未验证；运行时反射解析 Scheduler 的 Config 注解也不在本次验证范围。

## 按调用顺序阅读（40 分钟）

`LLMEngine.add_request` → `Sequence` → `Scheduler.add` → waiting

`LLMEngine.step` → `Scheduler.schedule` → `ModelRunner.call('run', ...)` → `Scheduler.postprocess`

测试直接调用真实 Scheduler，用固定 token ID 列表替代模型输出，不实例化 LLMEngine 或 ModelRunner。

- schedule 优先处理 waiting：检查 token 预算和 block 容量，调用 can_allocate/allocate。每轮设置 num_scheduled_tokens；最后一个 Prefill 分块被调度时进入 running。
- 本轮有 Prefill 就返回，不与 Decode 混合。没有 Prefill 时从 running 调度 Decode，调用 can_append/may_append，必要时抢占尾部请求。
- postprocess 先 hash_blocks，再累加已计算 token 数。分块 Prefill 未结束时跳过采样结果；完成 Prefill 或 Decode 后追加一个生成 token。
- EOS（未 ignore）或达到 max_tokens 时设 FINISHED，deallocate 释放引用，从 running 移除。
- block 引用归零后回到 free 队列，但完整块的 hash/token 元数据可能保留，用于后续前缀复用。此实验没有实际 KV tensor，只验证管理元数据。

## 第一个测试：先预测，再运行（30 分钟）

打开 `tests/test_cpu_scheduler.py` 的 test_request_lifecycle。prompt=[10,11,12]，max_tokens=2，块大小保持真实默认值 256。

| 时刻 | 状态 | num_tokens | num_cached_tokens | 输出 | 占用块 |
|---|---|---:|---:|---|---:|
| add 后 | WAITING | 3 | 0 | [] | 0 |
| 第一次 postprocess 后 | RUNNING | 4 | 3 | [20] | 1 |
| 第二次 postprocess 后 | FINISHED | 5 | 0 | [20,21] | 0 |

完成后 cached_tokens 被 deallocate 清零，所以最后不是 4。

自己做三次小修改，每次先写预测，再运行对应测试，最后恢复：

1. 将生命周期测试 max_tokens 改为 3：原来的“第二步已结束”断言应失败。补上第三次 Decode 和相应断言，使测试再次通过。
2. 分块测试 budget=2、prompt 长度 3：解释为什么第一次固定输出 20 被丢弃，最终只有 [21]。
3. 跨块测试 prompt 长度 256：解释 Prefill 后长度是 257，但为什么第二块直到下次 schedule 才分配。

不要独立把 Sequence.block_size 改成 4；它必须与 BlockManager 的块大小一致。今天保持 256。

## 验收（20 分钟）

- 全量测试得到 7 passed；导入测试、生命周期、分块、两种 EOS、跨块、前缀复用均通过。
- 能脱离代码讲出 waiting → running → finished，说明分块 Prefill 为什么仍可能在 waiting。
- 能解释 schedule 决定执行哪些 token，postprocess 消费执行结果并回收资源。
- 能解释 block_table、free_block_ids、used_block_ids、ref_count 的关系。
- 能完成上面的 max_tokens=3 练习并说清每个状态变化。

已验证的是控制面正确性样例，不是吞吐量、TTFT 或真实 GPU KV 内容；不要把 CPU 测试时间作为推理性能结果。抢占、内存耗尽及多请求公平性留作后续用例。今天技术环境就绪，学习验收由你复跑与口述完成。
