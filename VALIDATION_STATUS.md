# 验证状态与证据范围

更新于 2026-11-07。环境报告是快照，不能据此宣布后续推理已通过。

| 项目 | 当前状态 | 证据或缺口 |
|---|---|---|
| 真实 Scheduler/BlockManager CPU 回归 | 通过 | 最新助手回归146 passed，含新增的两项计时负载控制测试 |
| 同机独立 Python 环境 | 通过 | 用户 .venv-clean，Python 3.12.9，144 passed，八组实验结果一致 |
| 跨机器复现 | 未验证 | 目前仍是同一台 Windows 机器 |
| NVIDIA 硬件与驱动可查询 | 已确认 | RTX 5060 Laptop GPU，8151 MiB，驱动 592.19 |
| 原 .venv GPU 依赖 | 不齐全 | 报告中 torch/triton/transformers/flash-attn 等为 null |
| Linux/WSL 运行环境 | 用户已确认 | Ubuntu-24.04 / WSL2，nvidia-smi 能查询显卡 |
| Linux PyTorch 安装 | 已完成 | 用户日志：torch 2.8.0+cu128、triton 3.4.0，位于 ~/.venvs/nano-vllm-gpu |
| CUDA 张量运算 | 用户运行通过 | torch 2.8.0+cu128，sm_120，GPU 64×64 矩阵乘法结果和为 262144 |
| Attention 小规模检查 | 用户运行通过 | FP16/BF16 各含普通 Prefill、前缀 Prefill、Decode，六项通过，KV 写入精确匹配 |
| 真实 GPU smoke 基线 | 用户运行通过 | prefill_first / original Decode / original 缓存；单卡 eager 小负载 |
| 全部八种 GPU 策略组合 | 用户运行通过 | 八份报告配置、相同源码指纹和完成长度已核对；单卡 eager 小负载 |
| 完整模型 GPU 数值正确性 | 未验证 | Attention 的小规模 FP32 数学参考对照已通过，但未与可信完整模型实现对照 |
| 固定小负载 GPU 计时对照 | 用户已完成 | 两种阶段策略各五次；结果与限制见 GPU_TIMING.md，不代表通用性能提升 |

最新助手环境快照：results/unit18/20261106T003208_587958Z_801af495/report.json。
该报告来自原 .venv（Python 3.12.13），不是 .venv-clean；
Codex 进程无法启动后者所依赖的用户 Python，不推翻用户终端中的复现证据。
硬件查询成功不代表模型可加载、显存足够或当前依赖兼容。

## 已有实验可以陈述的结果

- bounded_prefill 和 round_robin 在固定负载下改善部分请求的 Decode 机会，
  但可能增加首 token 等待，不保证每请求时间上界或更早完成。
- 第十三单元 LRU 退化案例：后续请求需计算 prompt token 从 1 增至 257。
- 第十五单元固定热点/扫描/混合矩阵中，两种策略命中量相同；
  当前 LRU 候选检查更多，策略 CPU 方法计时只是探索性管理开销。
- 共享、覆盖、前缀哈希链、重复副本和抢占的已覆盖 CPU 元数据场景通过测试。

## GPU 证据升级顺序

八组汇总：results/unit18/gpu_smoke_matrix_20261106.json。
各报告均覆盖部分 Prefill、Decode、A/C 输出各四个 token、B 输出一个 token，
并由 smoke 入口检查块释放与 runner 清理。短 prompt 场景不证明完整前缀命中或回收收益。
2026-11-07 已完成预热/同步/重复计时的固定小负载实验，原始报告为
results/unit18/gpu_timing/20261107T051248_426826Z_033389e4/report.json。
A 首次 Decode 中位数590.73→101.73ms，B 最大首 token 等待466.66→639.61ms。
总时间中位数878.64→868.10ms，波动范围重叠，不能宣称稳定总吞吐改善。
独立完整模型数值参考对照及更广泛性能负载仍待验证。

真实模型基线证据：results/gpu_smoke/20261106T022901_866357Z_9fa0b458/report.json，
status=gpu_smoke_passed。模型 Qwen/Qwen3-0.6B，固定 revision
c1899de289a04d12100db370d81485cdf75e47ca，model.safetensors SHA256
f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b（用户下载后校验一致）。
模型从本地离线加载；此证据是小规模流程检查，不是完整数值对照或性能基准。

Attention 证据：results/unit18/attention_probe/20261106T013634_505359Z_21ef685a/report.json。
环境使用 flash-attn 2.8.3 官方 torch2.8/cp312/cxx11abiTRUE wheel；
SHA256 为 f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb。
首次运行因缺少 Python.h 失败，安装 python3.12-dev 后六项通过。
这些检查不等于完整模型参考对照，也未测性能。

张量检查证据：results/unit18/cuda_probe/20261106T005201_906463Z_4f2de07a/report.json。
nccl_backend_built=true 仅说明后端已构建，尚未实际初始化通信组；
基础张量检查也没有执行 nano-vLLM attention kernel 或模型加载。

先确认 Linux、CUDA/NCCL 和依赖、模型路径可用，再运行 GPU_VALIDATION.md 的 smoke。
每种组合成功后仍需独立数值对照和性能实验。
首次 smoke 使用单卡、eager；不能据此声称 CUDA graph、多卡或全部缓存压力场景已验证。
不在 Windows CPU 虚拟环境中直接安装一整套 GPU 依赖以替代目标运行环境准备。
