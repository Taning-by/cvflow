# 界面功能黑盒测试报告

生成时间：2026-10-05 01:54　环境：Python 3.10.12, PySide6 6.11.2, Linux 5.15.0-117-generic, QT_QPA_PLATFORM=offscreen

测试方式：只通过用户可见的操作驱动界面（菜单、工具栏、鼠标按下/拖动/释放、键盘、对话框、表格编辑），检查用户可见的结果（画布上的节点与连线、面板内容、标题栏、写出的文件、通信设备状态）。模态对话框在离屏环境里会阻塞，用桩替换并记录调用；设置存储隔离到临时目录。

**合计 42 个用例：通过 42，失败 0，跳过 0。**

| 功能区 | 场景 | 结果 | 耗时(s) | 说明 |
|---|---|---|---|---|
| 启动与布局 | layout and title | ✅ 通过 | 0.2 |  |
| 启动与布局 | without file reopens last | ✅ 通过 | 0.2 |  |
| 文件菜单（新建/打开/保存/另存/最近/快捷方式/关闭） | new open save saveas recent | ✅ 通过 | 0.3 |  |
| 文件菜单（新建/打开/保存/另存/最近/快捷方式/关闭） | close with unsaved changes asks | ✅ 通过 | 0.1 |  |
| 文件菜单（新建/打开/保存/另存/最近/快捷方式/关闭） | create shortcut menu | ✅ 通过 | 0.1 |  |
| 流程管理（新建/重命名/删除/切换/检查） | add rename remove switch validate | ✅ 通过 | 0.2 |  |
| 节点库（筛选/双击/拖放） | filter and double click adds node | ✅ 通过 | 0.1 |  |
| 节点库（筛选/双击/拖放） | drag drop into editor | ✅ 通过 | 0.1 |  |
| 节点编辑器（选择/移动/删除/连线/右键菜单/缩放） | select move delete node | ✅ 通过 | 0.2 |  |
| 节点编辑器（选择/移动/删除/连线/右键菜单/缩放） | connect ports by drag and reject type mismatch | ✅ 通过 | 0.2 |  |
| 节点编辑器（选择/移动/删除/连线/右键菜单/缩放） | context menu add duplicate disable | ✅ 通过 | 0.2 |  |
| 节点编辑器（选择/移动/删除/连线/右键菜单/缩放） | zoom and fit | ✅ 通过 | 0.1 |  |
| 参数面板（各类控件/高级参数/代码/ROI） | edit spinbox combo checkbox rename | ✅ 通过 | 0.2 |  |
| 参数面板（各类控件/高级参数/代码/ROI） | advanced toggle and json code apply | ✅ 通过 | 0.2 |  |
| 参数面板（各类控件/高级参数/代码/ROI） | roi draw show clear | ✅ 通过 | 0.2 |  |
| 参数面板（各类控件/高级参数/代码/ROI） | typing into port count does not crash | ✅ 通过 | 0.3 |  |
| 参数面板（各类控件/高级参数/代码/ROI） | typing then focus out applies and survives | ✅ 通过 | 0.3 |  |
| 参数面板（各类控件/高级参数/代码/ROI） | repeated port edits keep panel usable | ✅ 通过 | 0.8 |  |
| 参数面板（各类控件/高级参数/代码/ROI） | reducing ports drops links | ✅ 通过 | 0.4 |  |
| 参数面板（各类控件/高级参数/代码/ROI） | rename field does not crash | ✅ 通过 | 0.3 |  |
| 参数面板（各类控件/高级参数/代码/ROI） | roi clear button does not crash | ✅ 通过 | 0.3 |  |
| 参数面板（各类控件/高级参数/代码/ROI） | advanced toggle does not crash | ✅ 通过 | 0.4 |  |
| 运行控制（运行一次/自动运行/连续/运行模式） | once autorun continuous | ✅ 通过 | 1.1 |  |
| 运行控制（运行一次/自动运行/连续/运行模式） | mode locks editing and triggers | ✅ 通过 | 1.1 |  |
| 图像窗口（显示/叠加层/缩放/像素信息） | shows selected node overlays and pixel info | ✅ 通过 | 0.2 |  |
| 图像窗口（显示/叠加层/缩放/像素信息） | input picker hidden for single input node | ✅ 通过 | 0.2 |  |
| 图像窗口（显示/叠加层/缩放/像素信息） | input picker switches between two inputs | ✅ 通过 | 0.2 |  |
| 图像窗口（显示/叠加层/缩放/像素信息） | multi input dl node shows each input and its overlays | ✅ 通过 | 0.5 |  |
| 图像窗口（显示/叠加层/缩放/像素信息） | all overlays ignores input filter | ✅ 通过 | 0.2 |  |
| 结果面板 | banner tree ok and ng | ✅ 通过 | 0.2 |  |
| 结果面板 | error text is visible in value column | ✅ 通过 | 0.2 |  |
| 结果面板 | ctrl c copies error row | ✅ 通过 | 0.2 |  |
| 结果面板 | context menu copies cell row and all | ✅ 通过 | 0.2 |  |
| 结果面板 | long value is truncated but copied in full | ✅ 通过 | 0.2 |  |
| 结果面板 | banner text is selectable | ✅ 通过 | 0.1 |  |
| 变量面板 | add edit remove | ✅ 通过 | 0.1 |  |
| 日志面板 | shows messages level filter clear | ✅ 通过 | 0.1 |  |
| 通信面板（设备/规则/监视/测试连接） | device add edit test connect remove | ✅ 通过 | 0.6 |  |
| 通信面板（设备/规则/监视/测试连接） | rules add edit remove and monitor | ✅ 通过 | 1.2 |  |
| 相机管理对话框 | dialog search add by ip test force add | ✅ 通过 | 0.2 |  |
| 插件菜单 | load folder shows in palette | ✅ 通过 | 0.1 |  |
| 帮助 | about | ✅ 通过 | 0.1 |  |

## 说明

- 每个用例都会新建主窗口并打开示例方案，用例之间互不影响。
- 节点编辑器的连线、移动、右键菜单通过向视图发送真实的鼠标事件完成；拖放从节点库到画布在离屏平台没有拖放会话，改为把放下事件直接交给视图。
- 运行模式用例会真正启动流程线程并连接示例方案的 TCP 服务端（6000 端口）。

重新生成：`python tools/ui_test_report.py`
