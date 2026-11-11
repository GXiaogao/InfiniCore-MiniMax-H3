# 九齿 FP32 门控残差舍入与负零修复验证包 v3

v2 在 2026-10-27 15:18 的设备探针中停止：原生 mul→add 已确认采用向下舍入；新内核最大绝对差为0，但逐位结果不同。已从固定 Triton 3.8.0 wheel 的 `triton/language/semantic.py::scalar_constant` 复现字面量 `-0.0` 被 `value == 0` 分支转成正零。v3 仅将负零输出改成 int32 位模式 `0x80000000` 的选择，再 bitcast 为 FP32；不放宽逐位检查。

2026-10-27 14:16 回传表明：新增公共接口36项数值/8项拒绝、H3开关各18次、Encoder桥接开关各12次、85项MHA全部通过；新内核计数也正确。但最终开关全等失败，差异仅在FP32（最大绝对差2.384185791015625e-7），整轮仍为失败。

固定基线的 `src/infiniop/ops/add/cuda/kernel.cuh` 用 `__fadd_rd` 做FP32加法，乘法用 `__fmul_rn`；原九齿表达是默认就近加法。本包保留原基线，用九齿算术、where、bitcast表达向下舍入，保持FP16/BF16不变，不手写GPU内核、不扩大模型容差。宿主启动器和C++公共接口不变。

## 执行

上传 `h3-gated-rounding-v3.tar.gz` 到 `$WORK_HOME/scripts/`，在初始化且已有v1接入的Pod运行：

```bash
tar -xzf "$WORK_HOME/scripts/h3-gated-rounding-v3.tar.gz" -C "$WORK_HOME/scripts" &&
run_logged h3_gated_rounding_v3.log bash "$WORK_HOME/scripts/h3-gated-rounding-v3/run.sh"
```

流程：源码只读校验 → 同输入原生mul+add/旧九齿舍入探针 → 隔离候选环境导出修正内核 → 新探针 → 完整六阶段回归及开关输出全等 → 最后才更新 `logs/h3_gated_active_bundle.txt`。

无wheel安装，不修改setup_env.sh、仓库源码或已安装框架库；不重编译InfiniCore/InfiniLM。仅编译新HSACO和同一宿主启动器。原资产保留，原选中路径备份在本轮setup日志下。任一步失败不切换活动目录、不把失败改记为通过，保留日志。

成功标志：`ROUNDING REPAIR AND INTEGRATION PASSED`、exit code 0。回传外层log和末尾`EVIDENCE ZIP:`压缩包，其中自动包含两次探针的JSON/NPZ。若探针或编译阶段失败、尚未生成ZIP，回传本轮`h3_gated_rounding_v3_setup_*`目录即可定位。新探针会打印位差数量、正负零分类及前8个不同元素的十六进制位模式。

## 检查与范围

- 修复前后探针各3种布局，1024组数对，包含半ULP、符号零、极大指数差；逐位比对独立CPU向下舍入参考。
- 原36个公共算子用例新增同输入的真实InfiniCore `mul`→`add` 对照，仍有8项拒绝检查。FP32 CPU单算子参考明确向下舍入，其余dtype和原用例数据保持。
- H3/Encoder模型参考脚本、容差、模型输入/权重、MHA脚本全部原样复用；开关对照仍要求逐元素全等。
- 本地以精确有理数oracle验证FP32 DSL数学表达和独立float64补偿CPU参考；本地没有HIP/DCU，真实九齿编译与设备修复结论仍待服务器执行。
- 本地另从实际 Triton 3.8.0 wheel 提取并执行 `scalar_constant` 方法，用有类型的NumPy存储替代IR builder验证常量转换边界；旧表达出现5个负零位模式错误，修正后6个抵消/正负零用例通过。这项宿主测试不等同于整套GPU编译器或DCU验证。
- TwoSum使用严格FP32中间计算，禁止FMA收缩；正有限溢出、负溢出、精确抵消和带符号零单独处理。模型数据有限；不扩展NaN payload契约。
- 不代表完整官方模型、视频、多卡或性能验证，未宣称提速。新增FP32补偿运算的成本需后续实测。
