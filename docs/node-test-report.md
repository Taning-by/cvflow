# 节点功能黑盒测试报告

生成时间：2026-10-04 05:21　环境：Python 3.10.12, OpenCV 5.0.0, numpy 2.2.6, onnxruntime 1.23.2, Linux 5.15.0-117-generic

测试方式：只通过节点的公开接口（输入端口、参数 → 输出端口）验证，期望值来自可解析计算的合成图像。契约测试对注册表中全部 42 个节点自动执行（元数据、输出端口与声明一致、错误输入被隔离）；功能测试按节点覆盖各模式与参数。

**合计 175 个用例：通过 165，失败 0，跳过 10（跳过的均为“只有 ANY 类型输入、无类型约束可验证”的契约用例）。**

| 分类 | 节点 | type_id | 契约用例 | 功能用例 | 结果 | 说明 |
|---|---|---|---|---|---|---|
| 分析 | Blob 分析 | `analysis.blob` | 3 | 1 | ✅ 通过 |  |
| 分析 | 二维码 | `analysis.qrcode` | 3 | 1 | ✅ 通过 |  |
| 分析 | 圆查找 | `analysis.circle` | 3 | 1 | ✅ 通过 |  |
| 分析 | 模板匹配 | `analysis.template_match` | 3 | 1 | ✅ 通过 |  |
| 分析 | 灰度统计 | `analysis.intensity` | 3 | 1 | ✅ 通过 |  |
| 分析 | 轮廓查找 | `analysis.contours` | 3 | 1 | ✅ 通过 |  |
| 分析 | 边缘卡尺 | `analysis.caliper` | 3 | 1 | ✅ 通过 |  |
| 学习 | Bandit 阈值调优 | `learning.bandit_threshold` | 3 | 1 | ✅ 通过 |  |
| 学习 | PyTorch 模型（模板） | `learning.torch_template` | 3 | 1 | ✅ 通过 |  |
| 深度学习 | ONNX 分类 | `dl.onnx_classifier` | 3 | 1 | ✅ 通过 |  |
| 深度学习 | ONNX 推理 | `dl.onnx` | 3 | 1 | ✅ 通过 |  |
| 深度学习 | ONNX 检测 (YOLO) | `dl.onnx_detector` | 3 | 1 | ✅ 通过 |  |
| 脚本 | Python 脚本 | `script.python` | 3 | 1 | ✅ 通过 |  |
| 输出 | Modbus 写入 | `output.modbus_write` | 3 | 1 | ✅ 通过 |  |
| 输出 | 保存图像 | `output.save_image` | 3 | 1 | ✅ 通过 |  |
| 输出 | 发布结果 | `output.publish` | 3 | 1 | ✅ 通过 |  |
| 输出 | 发送报文 | `output.comm_send` | 3 | 1 | ✅ 通过 |  |
| 输出 | 日志 | `output.log` | 3 | 1 | ✅ 通过 |  |
| 输出 | 渲染叠加层 | `output.render` | 3 | 1 | ✅ 通过 |  |
| 逻辑 | 写变量 | `logic.set_variable` | 3 | 1 | ✅ 通过 |  |
| 逻辑 | 判定 | `logic.judge` | 3 | 13 | ✅ 通过 |  |
| 逻辑 | 延时 | `logic.delay` | 3 | 1 | ✅ 通过 |  |
| 逻辑 | 文本格式化 | `logic.format` | 3 | 1 | ✅ 通过 |  |
| 逻辑 | 表达式 | `logic.expression` | 3 | 1 | ✅ 通过 |  |
| 逻辑 | 计数器 | `logic.counter` | 3 | 1 | ✅ 通过 |  |
| 逻辑 | 读变量 | `logic.get_variable` | 2 | 1 | ✅ 通过 |  |
| 逻辑 | 门控 | `logic.gate` | 3 | 1 | ✅ 通过 |  |
| 采集 | 图像文件 | `source.image_file` | 2 | 1 | ✅ 通过 |  |
| 采集 | 图像文件夹 | `source.image_folder` | 2 | 1 | ✅ 通过 |  |
| 采集 | 常量 | `source.constant` | 2 | 1 | ✅ 通过 |  |
| 采集 | 相机 | `source.camera` | 2 | 1 | ✅ 通过 |  |
| 采集 | 触发数据 | `source.trigger` | 2 | 1 | ✅ 通过 |  |
| 预处理 | Canny 边缘 | `preprocess.canny` | 3 | 1 | ✅ 通过 |  |
| 预处理 | 位运算 | `preprocess.bitwise` | 3 | 1 | ✅ 通过 |  |
| 预处理 | 图像增强 | `preprocess.enhance` | 3 | 1 | ✅ 通过 |  |
| 预处理 | 形态学 | `preprocess.morphology` | 3 | 1 | ✅ 通过 |  |
| 预处理 | 旋转 / 翻转 | `preprocess.transform` | 3 | 1 | ✅ 通过 |  |
| 预处理 | 滤波 | `preprocess.blur` | 3 | 4 | ✅ 通过 |  |
| 预处理 | 缩放 | `preprocess.resize` | 3 | 1 | ✅ 通过 |  |
| 预处理 | 裁剪 ROI | `preprocess.crop` | 3 | 1 | ✅ 通过 |  |
| 预处理 | 阈值分割 | `preprocess.threshold` | 3 | 1 | ✅ 通过 |  |
| 预处理 | 颜色转换 | `preprocess.color` | 3 | 1 | ✅ 通过 |  |

## 覆盖缺口

- 所有节点都有功能用例。

## 说明

- 契约用例对每个节点各 2～3 条：元数据/序列化往返、输出端口齐全且类型符合声明、错误输入以 ERROR 结束且不崩溃。
- 功能用例的真值来自合成图像：已知面积与质心的方块、已知边缘位置的阶跃、已知圆心半径的圆、用 OpenCV 生成的二维码、用 onnx 构造的微型模型（恒等、按通道均值分类、固定输出的 YOLO 布局）。
- 未在真实硬件上验证的部分（海康 MVS 取图、GenICam 取图、PyTorch 插件）以“明确报错”作为通过标准。

重新生成：`python tools/node_test_report.py`
