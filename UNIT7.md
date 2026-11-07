# 第七单元：统一实验入口与配置

代码、亲手复跑与口述验收均已完成。能够区分配置、源码指纹、请求汇总和逐轮轨迹；
知道在确认配置、代码与环境一致后比较 cases，而非要求含时间戳的整个报告逐字相同。
本单元后全量 82 passed。此前脚本的命令继续可用。

## 先看什么

先看 `experiments/configs/mixed.json`，再看 `experiments/run.py` 中的
`resolve_config → run_experiment → run_mixed/run_arrivals`。
负载函数继续调用真实的 `Scheduler.schedule → Scheduler.postprocess`；
统一入口没有另写一套调度算法。

配置分成三部分：

- `schema_version`：配置格式版本，目前必须为 1，不是调度算法版本。
- `workload` 和 `parameters`：固定请求输入与到达安排，以及 token 预算。
- `strategies`：对相同负载依次比较哪些策略。

`mixed`：A/B/C 在第 0 轮前到达，prompt 都为 1 个 token。
A/B 各生成 `leading_tokens` 个 token，C 生成 2 个。
随后在第 2～N+1 轮，每轮到达一个只生成 1 个 token 的短请求，N 为
`short_arrival_rounds`。`budget` 是每轮 Prefill 计算 token 的预算。

`arrivals`：A 在第 0 轮到达、生成 `a_max_tokens` 个 token。
第 1～`arrival_count` 轮，每轮到达 `arrivals_per_round` 个短请求，均生成 1 个 token。
注意 `arrival_count` 是到达轮数，总短请求数是两者乘积。

策略字段：`policy` 选择阶段优先规则，`decode_policy` 选择 Decode 队列顺序，
`max_consecutive_prefills` 是 bounded_prefill 的让行上限。
prefill_first 不使用该上限，但配置仍明确记录这个值。
预算是 token 数，上限是批次轮数，它们的单位不同。

两个负载固定块大小 256、块数 16、EOS=-1；mixed 的 max_num_seqs=2，
arrivals 为 4。它们也记录在报告的 fixed_scheduler 中。
合成输出固定为 `1000 + 当前已生成数量`，不假设提前 EOS。

## 亲手运行

在 PowerShell 复制完整命令：

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/mixed.json
```

预期四行 C 的 decode_rounds 依次为 `[15]`、`[13]`、`[9]`、`[6]`。
终端的 Saved 会给出本次结果目录。

另一负载：

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m experiments.run --config experiments/configs/arrivals.json
```

预期 A 的 decode_rounds 为原策略 `[7, 8, 9]`，bounded_prefill `[3, 6, 8]`。

## 每次保存什么

每次成功比较生成 `results/unit7/<UTC时间和唯一后缀>/`，不覆盖上次运行：

- `config.json`：实际生效配置，包含补齐的默认值。
- `report.json`：同一配置、固定调度参数、环境和依赖版本、Git HEAD/dirty、
  Python 源文件及 CPU 依赖清单的 SHA256、每种策略的请求指标和逐轮轨迹。

同配置、同代码、同环境再次运行，应得到相同 cases；时间戳和目录名会不同。
源码指纹用于识别运行代码是否相同，并不保存或恢复源码；未提交代码仍需提交或备份。
HEAD 相同也不代表本地修改相同。Git 不可用时相关字段为 null，不伪造版本。
运行期间发现源码改变，会拒绝归档，要求重新运行。

报告中的轮次不是毫秒，不把 pytest 耗时或首 Decode 轮次作为 GPU 性能结果。

## 验证与边界

新增检查覆盖：配置驱动 CLI 的 CPU-only 导入、相同负载四策略复现、
重复运行不覆盖、预算和上限确实改变调度轨迹、错误配置在运行前被拒绝、
短 A/B 完成后仍等待后续请求到达。

```powershell
cd "C:\Users\tong\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\nano-vllm"
.\.venv\Scripts\python.exe -m pytest -v -p no:cacheprovider
```

配置字段或策略值拼错、预算为 0、把 true 当整数等均明确报错。
较大请求仍需通过第六单元的容量准入；未承诺任意配置均能完成。
本入口未新增 GPU 推理、任意异常事务回滚或完整断点恢复。

## 本单元验收目标

能够运行一份配置并找到本次报告；能够解释负载、策略与固定参数；
能够说明为什么比较策略时必须保持请求到达与输出长度相同，
以及为什么相同配置仍需记录源码和环境。
让行上限 1/2/3 和预算的系统对照放到下一单元。
