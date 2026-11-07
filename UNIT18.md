# 第十八单元：GPU 可行性与验证状态

本单元已完成：环境准备、GPU smoke、固定负载计时和用户结果分析验收均完成。
用户已确认 Ubuntu-24.04 / WSL2 能查询 RTX 5060，
创建 /home/tong/.venvs/nano-vllm-gpu，安装 torch 2.8.0+cu128、triton 3.4.0 成功。
用户已运行 experiments.cuda_probe 并通过，sm_120、cuda:0，矩阵结果和 262144。
NumPy 等基础依赖已安装，transformers 4.57.1；flash-attn 2.8.3 与 einops 0.8.2 安装成功。
WSL 下载 GitHub wheel 返回 403，用户从 Windows 下载、核对官方 SHA256 后本地安装。
缺 Python.h 的 Triton 编译错误在安装 python3.12-dev 后解除。
用户六项 Attention 检查通过：FP16/BF16 的 Prefill、前缀 Prefill、Decode，
小规模 FP32 数学对照和 KV 写入通过。
用户完成 Qwen3-0.6B 固定版本权重校验，离线运行原版三策略基线 GPU smoke 通过，
证据为 results/gpu_smoke/20261106T022901_866357Z_9fa0b458/report.json。
用户其余七种组合也均通过；八份报告的生效策略、相同源码指纹、输出长度与阶段覆盖已核对，
汇总为 results/unit18/gpu_smoke_matrix_20261106.json。
小负载兼容性检查已完成。2026-11-07 用户完成两种阶段策略各五次 GPU 计时对照，
已核对所有样本的输出长度、36轮计算与532个输入token一致。
结果与边界见 GPU_TIMING.md；用户已解释 A 首次 Decode 与 B 等待的取舍，
并确认单负载、每策略五次测量的82.8%改善不具有通用性。
完整模型数值对照、更广泛负载和压力场景仍未完成，不以本单元验收替代。
此单元没有新增运行时算法，也不因硬件存在就宣布 GPU 通过。

2026-11-06 读取实际环境：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB，驱动 592.19。
已有 .venv 为 Windows/Python 3.12.13，缺少 torch/triton/transformers/flash-attn 等 GPU 路径依赖。
ModelRunner 直接初始化 nccl，gpu_smoke 的实际执行入口要求 Linux。
因此区分“机器有显卡”和“具备已验证的推理运行环境”。

Codex 无法启动用户 .venv-clean 的基础 Python，本次环境记录改用原 .venv；
用户新环境测试 144 passed 的证据保持有效。
Codex 的 wsl --list --quiet 返回 E_ACCESSDENIED，不能判断用户是否已安装发行版。
需要用户终端核对，不能据此宣称本机没有 WSL。

新增 VALIDATION_STATUS.md 汇总 CPU 已有证据与 GPU 待验证项。
GPU_VALIDATION.md 保留真实运行步骤，包括每种缓存/阶段/Decode 组合的后续检查。
若暂无目标环境，CPU 结果与限制说明可按计划交付，GPU 项继续明确标为未完成。
如用户已有可用环境，则继续准备和实际执行，不能用文档代替运行证据。

验收目标：能区分 GPU 硬件、框架后端、模型兼容性、数值与性能这几层证据；
明确接下来可执行的验证路径及当前缺口。
