# gfx936 九齿 Attention AOT／宿主启动验证 v1.1

本包尚无设备验证结果。它沿用已经通过 gfx936 JIT 的九齿 DSL 和 ceil-div 修复，验证独立编译、文件导出、另一个进程的 HIP Module API 启动。不替换正式库、不修改 state、权重或模型。无性能收益声明。

## 上传与运行

1. 解压 `hygon-ninetoothed-attention-aot-v1.1-upload.zip`。在平台新建或使用仅放本轮脚本的项目，把解压后的 **19 个文件全部放在同一目录**；包含许可证、UPLOAD.md、package-manifest.json，不只上传入口。
2. 使用以前通过的 gfx936 / Torch2.9.0 / Python3.11 运行环境，入口选 `run_hygon_attention_aot.sh`，参数留空，批处理作业，时长 **28 分钟**。卡数沿用平台整节点配置，不需要修改。
3. 脚本使用现有 `$IFAB_AFS_HOME/minimax-h3/state/ninetoothed-gfx936.json` 的私有九齿安装，以及 `packages/minimax-h3-hygon-runtime-inputs` 内固定的海光 Triton wheel。不要删除这些现有依赖；无需再上传 wheel 或模型。本轮需要环境已有的 `g++`，只编译宿主启动器。
4. 必须提交一个**新的作业**，不要查看旧实例的日志。开头应显示 `version=1.1-nine-attention-aot-dynamic-lds`；日志目录前缀为 `infinifabric-nine-attention-aot-`。

## 阶段与输出

`aot_export`：生成6种专化，动态尺寸/步长显式 i64，scale fp64，无自动输入值/对齐特化；导出源码、HSACO、IR、程序集、参数表和哈希。未执行 Attention。

`aot_worker`：新进程只加载导出 HSACO，通过宿主 C++ HIP API 启动；禁止九齿 make、Triton编译/JIT启动；用原验证器的12个开发形状、每形状2组随机输入比较 CPU double 参考，并检查哨兵、输入、流和实际24次宿主启动。沿用的验证器仅数值比较/数据准备使用 Torch；GPU候选计算在九齿生成文件内。

HIP启动/临时编译均在当前作业工作区；脚本归档到 AFS scripts，报告归档到 AFS logs。不使用 /dev/fd，不对共享解释器安装包。两个进程共享20分钟预算、外层25分钟超时，每20秒心跳。

## 下载回来

在个人存储打开 `minimax-h3/logs/infinifabric-nine-attention-aot-本轮时间-编号/`，下载：

- `summary.txt`
- `run.log`
- `attention-aot-controller.json`
- `attention-export.json`（导出失败也应保留）
- `attention-aot.json`（进入启动阶段后才产生）
- `attention-aot-evidence.zip`（内含本轮生成的源码、IR、HSACO、ABI及局部失败张量；总原始材料超过120MiB会报告未打包，原文件仍保留）

本地建议保存到本候选目录的 `results/本轮时间-编号/`。若失败，先提供现有报告和 evidence ZIP，不重复安装或重建正式库。

成功要求：summary退出0，controller和attention-aot都为 `AOT_PROBE_PASSED`，6种导出、12形状/24比较、24次真实宿主启动、全部输入/哨兵保持。只证明小形状独立AOT链路，InfiniOp/Core接入、真实长序列、多卡模型和性能仍未验证。

## 来源和边界

`upstream_attention.py` 是 InfiniCore f32e2fbb2a9b2ca912b5711f4f3836c403196a74 原文件，MIT许可证随包保留。`candidate_attention.py` 只有已验证的正整数 K/V 分块 ceil-div 修复。`attention_worker.py` 等旧验证/引导模块复用 JIT v1.2，主入口不会调用其旧 main。LLVM桥只在临时私有vendor内移除不支持的GEP标志，不更改操作数或计算。Vendor3.5源码固定将global scratch传零且无相应metadata字段；profile scratch需显式为0，程序集必须匹配两个尾部指针，否则停止。

## v1.1 修复（2026-11-11）

真实作业20261111T004840Z-336595中，静态LDS为0、metadata.shared为4096，LLVM IR声明external global_smem；v1误把静态/动态大小要求相等。v1.1仅修正宿主ABI契约：严格要求静态0，按metadata传动态大小，并在导出和加载两侧检查LLVM外部LDS声明。其他ABI检查、九齿计算和C++启动器不变。已返回的真实产物可通过本地回归，修复包设备数值结果仍待运行。旧v1压缩包保留；请上传新版全部19文件并重新提交作业。
