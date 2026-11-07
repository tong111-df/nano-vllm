# 最终交付清单

2026-11-07，核心交付与最终代码推送已完成，补记本交付状态。

| 项目 | 状态 |
|---|---|
| 调度/缓存核心实现及CPU交互回归 | 已完成，最新146项 |
| 同机独立环境 | 用户最终复跑146 passed，2026-11-07 |
| GPU基础与Attention六项 | 用户运行通过 |
| GPU八组合单卡eager smoke | 用户运行通过，原始报告已保留 |
| GPU固定负载重复计时 | 用户运行完成，收益与代价已记录 |
| README、结果表、架构图、讲解初稿 | 已整理，用户讲解验收通过 |
| 本地源文件/报告快照 | 第二十单元生成，清单及哈希见results/final |
| Linux GPU实际安装清单 | 用户已导出results/final/gpu-requirements-observed.txt；是环境记录，不是通用安装锁文件 |
| Git核心提交及GitHub同步 | 用户已提交并推送1f0b549到origin/feat/day1-cpu-tests；本状态更新单独收尾提交 |
| 完整模型数值参考、CUDA graph、多卡、广泛压力 | 未完成，保留为限制 |

当前本地分支feat/day1-cpu-tests，origin为tong111-df/nano-vllm。
GPU依赖记录中的FlashAttention为本机wheel路径及哈希；迁移机器需按GPU_VALIDATION.md
使用对应官方wheel，不能直接把该本机路径当成可移植下载地址。
归档前HEAD为8c7c62b；核心交付提交为1f0b549，用户推送输出确认8c7c62b..1f0b549成功。
分页器误保存的差异统计文件已移至忽略的results/final/console-stat-*.log，不作为代码提交。
本地跟踪引用不是远端最新状态的证明；未进行强制推送或重写历史。

快照用于保存当前可审查文件，不能替代Git提交、用户的新环境复跑或最终推送。
后续源码/报告变化后应重新生成快照，不能把旧快照哈希套到新文件。

运行入口见CPU_REPRO.md、GPU_TIMING.md，实验与局限见EXPERIMENT_RESULTS.md、VALIDATION_STATUS.md。
