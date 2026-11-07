# 第九单元：调度代码审查与 GPU 验证准备

代码审查、边界修复、CPU 回归已完成，全量 97 passed。
用户已完成阅读、亲手复跑、环境记录和口述验收；最终全量回归 97 passed。
真实 GPU 推理仍未验证。用户已区分模型上下文与块池容量、阶段让行与每请求公平性，
并理解环境记录/GPU smoke/性能验证的不同证据范围及错误处理不等于事务回滚。
Windows 复跑时原 pytest 共享临时目录出现权限错误；通过 --basetemp 指向
TEMP 下每次新建的唯一目录后，单元测试 10 passed、全量 97 passed。

## 本次审查的实际发现

原入队检查仅保证最大 KV 需求不超过整个块池，没有检查模型上下文长度。
ModelRunner 使用 max_model_len 准备 CUDA graph 的 block_tables 等数据，
并按序列长度构造 positions；空间充足并不能允许无限扩大输入位置。

Scheduler 现在保存可选 max_model_len，在 add() 入队前检查：

```python
required_tokens = seq.num_prompt_tokens + seq.max_tokens - 1
if self.max_model_len is not None and required_tokens > self.max_model_len:
    raise ValueError(...)
```

这里上限限制实际送进模型计算 KV 的位置数，最终输出 token 不再送回模型。
不将它解释为强制限制输出列表的总长度，也不依赖提前 EOS。
GPU 的真实 Config 总会传入该字段；旧 CPU fixture 可省略，省略时只保留块池检查。
非正整数模型上限在构造 Scheduler 时明确报错。

例子：块池 4 × 256 = 1024 个位置，但 max_model_len=256。

- prompt=256，max_tokens=2：需要 257 个 KV 位置，入队前拒绝。
- prompt=255，max_tokens=2：需要 256 个 KV 位置，允许；最终 token_ids 可长达 257。
- prompt=257，max_tokens=1：仅 prompt 就超长，拒绝。

四种阶段/Decode 顺序组合均检查拒绝不修改队列和块分配，原有请求仍可继续执行。
真实模型路径的兼容性仍需 GPU 验证，CPU 测试没有执行位置编码或 attention kernel。

## 审查结论与保留的限制

| 范围 | 当前证据 | 保留边界 |
|---|---|---|
| 默认策略 | 原 CPU 基线结果仍通过；新增字段使用兼容默认值 | 非法/超长请求现在主动拒绝，错误行为并非完全保持原版 |
| bounded_prefill | 有 running 时按批次计数，Decode 后清零，无法追加时允许 Prefill 继续 | 是阶段让行，不是每请求的时间或公平性保证 |
| round_robin | Decode 已选请求移到队尾，多请求与抢占回归仍通过 | 不消除 Prefill 占用、缺块、队首等待等所有问题 |
| 入队 | 总池容量、模型位置上限、保守输出需求检查 | 不预留全部输出空间，不能保证所有并发负载免于资源停滞 |
| postprocess | 输出数量不匹配在更新前拒绝 | 不是任意错误的事务回滚，不能盲目重跑整个模型步骤 |
| 无法推进 | 明确异常，包含资源状态 | 异常前可能已经抢占；不承诺所有错误无副作用 |
| 实验记录 | 固定负载、配置、指纹、版本、逐轮轨迹 | 指纹不能还原未提交代码；CPU 轮次不能替代 GPU 性能 |
| Engine 接口 | kwargs 中合法 Config 字段可传入新策略 | 原有未知 kwargs 静默过滤行为尚未修改，调用时须用明确参数名 |

没有重构已有调度算法，也未修改缓存回收策略；缓存生命周期留到后续缓存单元。

## 当前电脑可运行

只记录环境，不导入 torch/triton/transformers/flash_attn，不加载模型：

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.gpu_smoke
```

应看到 `Status: environment_recorded_gpu_not_run`。
report 中包含 Python/系统、依赖发行版版本（未装则 null）、nvidia-smi 查询结果和源码指纹。
nvidia-smi 查询失败不影响环境记录，也不会被当成 GPU 推理通过。
结果每次独立保存到 results/gpu_smoke 下，终端 Saved 显示具体位置。

回归：

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```

GPU 运行步骤见 GPU_VALIDATION.md。此时不要求在 Windows CPU 环境安装 GPU 依赖。

## 验收目标

能区分模型上下文限制与 KV 块池容量，能解释最大 KV 需求公式；
能说明环境记录成功、GPU smoke 通过、性能/数值正确性验证之间的区别；
能指出阶段公平性、并发容量和错误恢复的保留限制。
实际 GPU 推理保持待验证，不以本单元 CPU 验收替代。
