# 海光九齿 Attention 独立验证包 v1.2

v1.2 根据取回的全零失败张量和实际生成内核，修正九齿DSL中的动态K/V分块向上取整：将 `-(-actual_kv_len // block)` 改成 `(actual_kv_len + block - 1) // block`。Triton张量整数除法向零截断，旧式在单key/32块时得到0，跳过计算。原始 `upstream_attention.py` 保留，新增 `candidate_attention.py` 仅改这一处表达式，按精确差异核验并继续通过九齿生成编译。参考、权重、输入用例和容差不变。尚未取得v1.2设备结果。

v1.1 针对回传日志中的 LLVM `getelementptr inbounds nuw` 解析失败，在当前作业私有 Triton 后端的 IR→目标汇编边界加入方言兼容处理：只移除 GEP 的 `nuw/nusw` 标记，保留 `inbounds`、类型、索引与数值指令，并在翻译前保存原始/处理后 IR。固定原始九齿 Attention 表达、模型和容差均不变。此候选尚未取得目标设备修复结果。

本包执行真实九齿 JIT dot、softmax 归约和 Attention 数值验证。它不安装覆盖正式库、不加载 MiniMax-H3 权重、不运行完整视频生成，也不代表比赛验收或性能通过。九齿 Attention 来自固定 InfiniCore commit；原始文件及MIT许可随包提供，候选修复在独立副本中明确记录。

## 在网页中运行

1. 在本机解压 `hygon-ninetoothed-attention-probe-v1.2-upload.zip`，将解压得到的 **全部13个文件** 上传到网站项目的同一目录，同名文件保存为新版本。包括新增 `candidate_attention.py` 及更新后的脚本、说明、清单，不要只上传 Bash 文件。源文件总量很小，不需上传模型、旧源码包或 wheel。
2. 入口选择 `run_hygon_ninetoothed_attention.sh`，程序参数留空，使用之前成功的 `scnet-dcu`/gfx936环境，时长选 **28分钟**。本验证仅用可见设备0，不应借此修改正式模型评测的卡数；若平台只提供整节点，可沿用整节点资源。
3. 启动一次即可。日志每20秒输出一次 worker 心跳；首次编译可以需要数分钟。外层最多25分钟，内部 worker 最多20分钟，超时会失败并保留已完成记录，绝不将部分通过写成全部成功。

项目目录可与已提交项目分开。若平台入口只能放项目根，请把所有文件放根目录。无需使用不稳定的交互 Bash 终端，不要运行旧的完整安装脚本。

## 服务器必须已有

- `IFAB_AFS_HOME/minimax-h3/setup_infinifabric_env.sh`。
- `state/ninetoothed-gfx936.json` 和其中指向的持久私有九齿环境。这是之前九齿加法 JIT 成功时保存的状态，不是新配置文件。
- 之前已上传的 `packages/minimax-h3-hygon-runtime-inputs/triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl`，SHA256 固定为 `e397b16365da22508353ce44a45f5be82f8ee58e49853a3df06aa8f6f83d5581`。

控制器从固定 wheel 在当前可写作业目录恢复短生命周期 Triton，复用已有九齿/NumPy 私有环境。不会尝试使用之前日志里已经过期的 `/dev/shm/.../vendor-site-packages`。依赖缺失会明确失败，无网络安装，也不更换共享 Torch/NumPy。

入口会把这13个小文件保存到 `$WORK_HOME/scripts/nine-attention-probe-<本次编号>/` 再执行并核验包哈希；日志仍统一写 AFS 的 `logs`。复用环境前按固定 wheel 的 RECORD 校验九齿源文件，仅允许已归档的作业缓存位置补丁；源码有其他改动时会停止，保留错误而不静默覆盖你的环境。LLVM兼容处理只加在本次作业刚解包的临时Triton中，不修改持久九齿安装、系统LLVM或正式库。新日志开头应为 `version=1.2-nine-attention-ceil-div`。

## 需要取回的文件

结束后在网页“个人存储”打开：

`minimax-h3/logs/infinifabric-nine-attention-probe-<本次时间和进程号>/`

请保存以下结果文件，保留本次目录名，不覆盖旧实验：

| 文件 | 用途 |
| --- | --- |
| `summary.txt` | 批作业退出码及验证范围。 |
| `run.log` | 全部阶段输出、每轮结果及错误堆栈。 |
| `attention-controller.json` | 离线环境恢复、真实来源与最终控制器结果。 |
| `attention-probe.json` | GPU目标、原语及12个用例、两轮误差和边界检查。 |
| `llvm-compat-last.json`（存在时） | 最近一次兼容处理的IR哈希及移除标记数量；编译崩溃时也先保存此证据。 |

如果预检阶段失败，第四、第五个文件可能尚未创建，取回已有文件即可。`llvm-compat/` 保存后端精确差异和每次原始/处理后IR；`generated/` 保存九齿生成源码，`compiled/` 保存至多64MiB的编译证据，先保留在 AFS，本轮不用逐个下载。出现某用例失败还可能生成 `*-failed.pt`，用于诊断，不是候选输出或正式模型缓存。

本地建议保存到项目的 `artifacts/hygon-ninetoothed-attention-probe-v1/results/<本次目录名>/`。本地 Windows 没有 DCU，因此交付前验证只能覆盖脚本行为、来源/哈希及打包；是否能在 gfx936 上编译和通过必须以本次服务器结果为准。

## 结果含义

只有四类条件同时满足才记录 `JIT_PROBE_PASSED`：实际 HIP/gfx936 目标和固定包来源正确；dot/归约通过；12用例各两组新输入满足开发容差、输入保持及哨兵检查；已保存 hsaco 目标编译证据。FP16容差 `atol=rtol=0.01`，BF16 `0.025`，只用于开发探针，不能当官方阈值；失败不自动扩大容差。

包内原 DSL 的 online softmax/Q预缩放与现有组合 MHA 的中间舍入不同，结果需要对照固定参考分析。即使本轮通过，也只能说明这些开发用例的 JIT 可行性；AOT 宿主启动、InfiniOp/Core 正式登记、真实模型形状、八卡实际调用和收益均为 `NOT_TESTED`。
