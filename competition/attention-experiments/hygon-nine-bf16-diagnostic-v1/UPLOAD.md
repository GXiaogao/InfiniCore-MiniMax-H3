# gfx936 BF16转换／JIT-AOT同输入诊断 v1.1

此包只定位故障，不是BF16修复或正式推理实现。旧v1.1中14次比较通过，第15次BF16-tail失败；不调整原容差，不修改正式库、权重、Nine Attention计算或C++启动器。

## 运行

解压 `hygon-nine-bf16-diagnostic-v1.1-upload.zip`，将**全部20文件上传到同一目录**，选择入口 **`run_hygon_bf16_diagnostic.sh`**，程序参数空，沿用gfx936环境、28分钟，新建一次批处理作业。

开头版本：`1.1-bf16-cast-loaded-value-diagnostic`。

必须保留个人存储中的原成功导出目录：

`minimax-h3/logs/infinifabric-nine-attention-aot-20261111T015938Z-364509/`

诊断将只读复用其中6种内核导出及宿主so，固定原export报告SHA并校验文件，使用BF16/D64/noncausal一份内核。不要删除、移动、编辑该目录。无需重新上传wheel、模型或本地下载的证据ZIP。私有Nine环境和海光Triton wheel位置与之前一致。

## 测什么

1. 三个小型Nine DSL转换：FP64直接转BF16、FP64经FP32转BF16、FP32直接转BF16。第二条仅是诊断控制组，不接入模型；其参考也显式经FP32，避免把二次舍入误判为缺陷。
2. 两轮相同新输入，分别运行已有AOT HSACO、原Nine DSL JIT及CPU double参考；记录误差、输入／哨兵、实际宿主启动。保存q/k/v和输出NPZ，采集JIT实际IR和HSACO，核对动态类型和编译路径。

疑似FP64→BF16转换问题尚未确证，存在其他泛化编译/装载因素的可能。此包结果用于区分，不默认认定根因。测试对照中的Torch不替代正式计算。

## 下载结果

个人存储打开本轮：

`minimax-h3/logs/infinifabric-nine-bf16-diagnostic-时间-编号/`

下载 `summary.txt`、`run.log`、`bf16-diagnostic-controller.json`、`bf16-diagnostic.json`、`attention-aot-evidence.zip`。ZIP包含3个转换NPZ、2轮输入/输出NPZ、生成源码和编译证据。

`DIAGNOSTIC_COLLECTED`／退出0仅表示观察材料收集完整，**不表示数值通过**；`correctness_passed`固定false。若发生异常、内存哨兵失败或材料缺失则非零退出；非有限转换结果会以字符串写入JSON并完整保留NPZ。设备结果尚未运行，本地仅验证契约与打包。

## v1.1 源码生成修复

作业20261111T025748Z-462049在第一个转换内核生成时停止，尚无转换或成对测试结果。Nine0.26不能在应用参数描述符上直接value.to，现改为loaded=value先生成load，随后loaded.to；没有增加乘加或修改转换链。已用真实九齿Python生成器复现旧错误并验证三种新源码能生成（设备发现使用stub，未运行GPU编译）。入口、20文件、28分钟及旧HSACO复用目录均不变；请上传新版全部文件并提交新作业。
