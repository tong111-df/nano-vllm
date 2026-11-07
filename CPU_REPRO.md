# CPU 复现与实验索引

当前环境为 Windows / Python 3.12；CPU 依赖固定在 requirements-cpu.txt。
从仓库根目录运行，无需安装 torch、triton、transformers 或 flash-attn。
本路径不会加载模型，也不验证 GPU attention 数值。

## 当前工作副本

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/integrated.json --output-root results/unit17
```

预期八行：前四组 cache=original，后四组 cache=lru，
C 的首次 Decode 轮次均依次为 15、13、9、6。
每次产生新的报告目录，不覆盖以前结果。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
$regressionTemp = Join-Path $env:TEMP ('nano-vllm-regression-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider --basetemp "$regressionTemp"
```

第十七单元代码全量 144 passed。独立临时路径用于避开 Windows 共享临时目录所有权冲突；
不要反复使用固定 --basetemp 指向包含个人文件的目录，pytest 会清理其指定的测试目录。

## 独立环境步骤

以下步骤使用新 .venv-clean，不影响已有 .venv。安装需要可用的软件包索引或本地 wheel。
2026-11-06 用户已在同一 Windows 机器新建 .venv-clean、按依赖清单安装，
全量 144 passed，并复现八种组合的 15/13/9/6 轮次结果。
这是同机独立 Python 环境验证，不代表另一台机器或 GPU 环境已经验证。
在其他机器上，将 cd 和 Python 路径改为该机实际位置。
若目录已存在，先使用另一个新环境名称，避免覆盖已有环境。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
& "C:\Users\tong\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv-clean
.\.venv-clean\Scripts\python.exe -m pip install -r requirements-cpu.txt
$cleanTemp = Join-Path $env:TEMP ('nano-vllm-clean-' + [guid]::NewGuid().ToString('N'))
.\.venv-clean\Scripts\python.exe -m pytest -v -p no:cacheprovider --basetemp "$cleanTemp"
.\.venv-clean\Scripts\python.exe -m experiments.run --config experiments/configs/integrated.json --output-root results/clean_repro
```

CPU 路径不需要运行 pyproject.toml 中安装 GPU 依赖的完整项目安装命令。
要让另一个人复现，还必须取得包含本分支修改的源码；上游基线 SHA 不包含这些修改。
提交并保存最终分支及配置是后续归档任务，不能仅提供上游 SHA 或未提交文件的指纹。

## 三种策略分别控制什么

| 配置字段 | 值 | 控制范围 |
|---|---|---|
| policy | prefill_first / bounded_prefill | 本轮选择 Prefill 还是 Decode |
| decode_policy | original / round_robin | Decode 批次的请求选择顺序 |
| cache_eviction_policy | original / lru | 新分配时选择哪个空闲块 |

策略对象由真实 Scheduler 创建；统一 JSON 入口会拒绝拼错字段和非法值。
旧 JSON 省略 cache_eviction_policy 时自动补 original。
直接使用模型引擎时字段名是 scheduling_policy、decode_scheduling_policy、cache_eviction_policy；
前两个 JSON 字段是实验入口的参数名，不要直接当作 LLM 的参数名。
真实模型引擎路径尚未在 GPU 上验证。

## 实验索引

以下模块均从仓库根目录，用 .venv/Scripts/python.exe -m 加模块名执行。

| 目的 | 模块或配置 | 证据范围 |
|---|---|---|
| 八种组合接入 | experiments.run，integrated.json | 参数实际传入及完整请求流程；不证明缓存收益 |
| 预算/让行上限 | experiments.unit8_sweep | 固定负载下请求等待与完成轮次 |
| 缓存生命周期 | experiments.unit11_cache_lifecycle | 共享引用、空闲复用、覆盖失效 |
| 原版回收顺序 | experiments.unit12_reclaim_order | 空闲队列与取得/释放引用事件 |
| LRU 退化案例 | experiments.unit13_lru_compare | 同负载的回收差异、命中和需计算量 |
| 缓存交互 | experiments.unit14_cache_interactions | 共享抢占恢复；其他交互由同名测试覆盖 |
| 热点/扫描 | experiments.unit15_cache_workloads | 命中、需计算量、候选检查数、探索性 CPU 管理计时 |
| 环境记录 | experiments.gpu_smoke，默认不加 --run | 仅记录环境，非 GPU 推理成功 |

实验必须保持输入与到达条件一致再比较策略。时间戳不同是正常的；
核对代码、环境和配置后比较确定性 cases/轨迹；管理计时样本允许波动。
