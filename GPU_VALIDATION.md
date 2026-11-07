# GPU 验证入口与未完成项

2026-11-06 实际检测到 RTX 5060 Laptop GPU（8151 MiB），机器具备 NVIDIA 硬件。
Windows CPU 虚拟环境缺少 GPU 依赖，Linux/WSL 环境仍待核对；
不是“已确定没有 GPU”。详细证据见 VALIDATION_STATUS.md。

当前状态（2026-11-06）：用户已在 WSL2 / RTX 5060 上通过八种策略组合的真实模型 smoke，
使用 torch 2.8.0+cu128 / triton 3.4.0 / transformers 4.57.1 / flash-attn 2.8.3。
Attention 六项小规模参考对照也通过。报告汇总和边界见 VALIDATION_STATUS.md。
2026-11-07 已完成固定小负载 GPU 计时对照，结果与限制见 GPU_TIMING.md。
完整模型数值正确性与更广泛性能仍未验证；下文保留复现步骤及后续检查项目。

## 准备目标环境

本仓库 ModelRunner 硬编码 NCCL，使用 CUDA、Triton、FlashAttention，并实现 Qwen3 模型。
首次验证目标为 Linux、单张 NVIDIA CUDA GPU、已在本地保存的兼容 Qwen3 权重和 tokenizer。
模型可从原仓库示例使用的 Qwen3-0.6B 开始；显存是否足够依实际环境检查。
不承诺其他模型架构或原生 Windows GPU 支持。

保留具体 Python、torch/CUDA、驱动、triton、transformers、flash-attn、
numpy、xxhash、safetensors、tqdm 版本。安装需求参考当前 pyproject.toml 及实际源码导入；
requirements-cpu.txt 仅用于 CPU 控制面，不是完整 GPU 环境清单。
在目标环境中先使依赖导入与模型文件可用，再执行下面步骤。
不直接把未知机器上的未验证版本声明为锁定环境。

## 第一步：环境记录

以下 Bash 命令中的仓库和模型目录须改成目标 Linux 机器上的实际路径。

```bash
cd /path/to/nano-vllm
python -m experiments.gpu_smoke
```

默认入口只查版本、源码指纹和 nvidia-smi，不进行 torch 导入或推理。
状态应为 environment_recorded_gpu_not_run；这只证明报告生成成功。

## 第二步：显式运行兼容性 smoke

```bash
cd /path/to/nano-vllm
python -m experiments.gpu_smoke --run --model /path/to/Qwen3-0.6B --policy bounded_prefill --decode-policy round_robin
```

入口要求 Linux、CUDA 和 NCCL、存在 config.json 的本地模型目录。
不自动下载模型，不使用 CPU 合成 token 冒充推理。
固定 tensor_parallel_size=1、enforce_eager=True、max_model_len=256、
max_num_batched_tokens=16、max_num_seqs=4、让行上限 2、gpu_memory_utilization=0.5。
分配出的块数和生效参数写入报告。

通过真实 LLMEngine.add_request/step 调用 Scheduler → ModelRunner → postprocess：
第 0 轮 A/C（各一个 prompt token、生成四个 token），第 1 轮 B 到达
（33 个 prompt token、生成一个 token），确保发生分块 Prefill、Decode、全部完成和块释放。
prompt token 从真实 tokenizer 得到，输出来自模型；ignore_eos=True 仅为固定 smoke 长度。
成功后显式清理 runner，避免重复执行已注册的退出函数。

成功状态为 gpu_smoke_passed；可捕获的异常记为 gpu_smoke_failed，并返回非零退出码。
强制杀进程、崩溃或初始化未返回等情况仍可能没有完整报告，需保留终端日志。
初始化中途失败不保证所有资源已经被清理。

## 第三步：四种组合分别运行

```bash
cd /path/to/nano-vllm
for policy in prefill_first bounded_prefill; do
  for decode in original round_robin; do
    python -m experiments.gpu_smoke --run --model /path/to/Qwen3-0.6B --policy "$policy" --decode-policy "$decode" || exit 1
  done
done
```

分别检查报告 status、effective_config、trace、outputs，确认四种组合均完成。
随机种子固定为 0 便于排查，但不同调度顺序会改变采样随机数的分配，
不能要求四组输出 token 完全相同，也不能据此判断数值正确性。
报告记录模型 config.json 指纹，不包含完整权重哈希；保留模型来源和权重版本另行记录。

## 仍需另行验证

- 数值正确性：与可信模型实现对照 logits 或设计合适的确定性参考。
- GPU 性能：相同模型与负载、预热、同步计时、重复运行、记录显存和波动。
- CUDA graph、Tensor Parallel、EOS、跨块、共享前缀及 GPU 资源紧张/抢占。
- 若冒烟测试失败，先保留报告和日志定位原因，再扩大测试范围。

此入口是小规模兼容性检查，不能代表性能 benchmark 或完整 GPU 正确性验收。

第十三单元增加 `--cache-policy original|lru`，默认 original。
将其设为 lru 可以接入取得块引用事件定义下的 LRU；requested 与 effective_config
会记录该选项。目前两种缓存策略的真实 GPU 路径仍未验证，原有四组合命令默认仅覆盖 original 缓存策略。
