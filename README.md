# NanoSched / CacheLab：nano-vLLM 控制面实验

基于 GeeeekExplorer/nano-vllm 固定提交 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，
研究调度机会、公平性与 KV 缓存回收。一个仓库、两个模块，使用真实 Scheduler/BlockManager，
在 CPU 上用合成 token 替代模型输出。当前全量 CPU 测试 146 passed；
WSL2 / RTX 5060 上八种组合的单卡 eager 模型 smoke 已通过，
并完成固定小负载 GPU 计时对照；完整模型数值对照仍待验证。

| 模块 | 新增修改 | 已观察到的结果 |
|---|---|---|
| NanoSched | bounded_prefill 阶段让行、Decode round_robin | 部分请求更早 Decode，可能增加首 token 等待；更早 Decode 不保证更早完成 |
| CacheLab | 回收策略接口、按成功取得引用事件定义的 LRU | 固定热点/扫描组与原版命中相同；另有 LRU 多重算 256 token 的退化案例 |

原版已具备 KV 分块、共享引用、前缀缓存与抢占；本项目没有将这些机制当作原创实现。
LRU 的查询和释放不刷新访问时间，尚未接入逐轮 attention 读取事件。

项目概览与调用图见 [项目讲解](PROJECT_BRIEF.md)，代表性数据及范围见 [实验结果](EXPERIMENT_RESULTS.md)。
GPU固定负载中A首次Decode中位数590.73→101.73ms，但B最大首token等待增加；
这不是首token延迟TTFT改善，也不证明稳定整体吞吐提升。

先阅读 [CPU 复现与实验索引](CPU_REPRO.md)。当前 Windows 工作副本快速运行：

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/integrated.json --output-root results/unit17
```

八组配置覆盖阶段规则、Decode 顺序、缓存回收的 2×2×2 组合。
该 mixed 短 prompt 负载用于接入检查，不证明 LRU 的缓存命中收益。
共享、覆盖、哈希链和抢占见 [交互验证](UNIT14.md)，缓存负载见 [对照实验](UNIT15.md)。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
$regressionTemp = Join-Path $env:TEMP ('nano-vllm-regression-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider --basetemp "$regressionTemp"
```

报告保存实际配置、源码指纹、环境与轨迹；源码指纹不能还原未提交代码。
CPU 轮次不是毫秒，CPU 管理计时不是 GPU 推理耗时。
GPU 已有小规模兼容性证据，详见 [验证状态](VALIDATION_STATUS.md)；
小负载计时结果与代价见 [GPU 计时](GPU_TIMING.md)；
完整模型数值正确性和更广场景仍按 [GPU 验证](GPU_VALIDATION.md) 待验证。

---

下方保留上游项目说明和上游性能表，仅作来源说明；它们不是本分支修改后的实测结果。

<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
