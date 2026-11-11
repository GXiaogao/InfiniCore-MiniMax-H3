# gfx936 九齿 Attention 独立实验

此目录公开独立JIT、AOT宿主启动和BF16诊断的实际源码。未覆盖src下正式模型源文件或启用正式九齿分发。
计算来自固定上游InfiniCore f32e2fbb2a9b2ca912b5711f4f3836c403196a74的InfiniOp九齿Attention DSL，保留原文件与MIT许可。
candidate_attention.py仅将动态K/V块数表达式改为正数加法上取整，源摘要5378d0f8dfa5c34aac83413e34094863864b273cbfb2bad4e43729c5748738c4。
Nine生成/编译链保留，参考结果不代替计算，不放宽开发容差。AOT C++仅负责HIP装载、参数绑定、stream启动与资源管理，不含手写设备计算。

## 实测边界

- JIT作业20261110T063310Z-2643063：12个小形状用例各两组输入，24次比较通过；FP16最大绝对误差0.0007062888，BF16为0.0048552823。
- AOT作业20261111T015938Z-364509：六种内核导出完成，14次比较通过，第15次BF16-tail失败；不是AOT整体通过。
- BF16诊断作业20261111T060308Z-612075：创建首个转换JIT句柄时报launch_nine_cast_fp64_direct缺失，未执行转换和Attention成对测试；发布源码保留该失败状态。
- 真实模型形状、正式InfiniOp/Core接入、八卡模型调用、质量收益和性能均未由这些实验验证。

原包UPLOAD.md是提交作业时的说明，保留原字节以供清单核验；以本页及总提交仓库results/ninetoothed-attention-20261111的状态和去隐私证据为当前结论。包清单中的source-upload-not-device-verified描述打包时状态，不覆盖后续实际测试结论。

## 从Git源码取得网页上传文件

在本仓库根执行`python competition/attention-experiments/export_upload.py jit /可写目录/jit-upload`，将输出目录的全部文件上传到网站项目同一目录，入口run_hygon_ninetoothed_attention.sh。
模式aot对应run_hygon_attention_aot.sh；模式bf16对应run_hygon_bf16_diagnostic.sh。每项独立申请28分钟，参数留空。
bf16模式是重现当前诊断失败的材料，不是修复包。AOT BF16未通过时不得将该候选启用为正式模型实现。
服务器须具备既有私有Nine0.26.0环境、state/ninetoothed-gfx936.json、固定海光Triton wheel和setup_infinifabric_env.sh；不安装到共享Python。
BF16诊断另外只读复用作业20261111T015938Z-364509的AFS导出目录，下载到本地的日志摘要不能替代该运行前提。
此目录不含wheel、HSACO、共享库或模型权重，源码提交以公开Git为准。

## 宿主契约测试

分别运行`python -m unittest discover -s competition/attention-experiments/<包目录>/tests -v`，每个包使用单独进程。
设置TEST_BASH为Bash可执行文件路径以启用shell测试；H3_TEST_TRITON_WHEEL和H3_TEST_NINETOOOTHED_WHEEL分别指向固定厂商Triton与Nine0.26.0 wheel，以启用可选源身份测试。未提供依赖的项目明确跳过。
宿主测试不编译或启动GPU，不验证JIT模块实际导入；启动函数绑定缺陷见总提交仓库实验报告。源清单校验使用export_upload.py，各运行文件与实际作业包保持逐字节一致。
