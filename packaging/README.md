# 打包成 Windows 安装程序

产物是**一个** `CVFlow-Setup-<版本>-x64.exe`，GPU 推理支持做成安装时可勾选的组件。

| 组件 | 体积（解压后） | 内容 |
|---|---|---|
| 主程序 | 约 **530 MB** | Python 运行时、PySide6 界面、OpenCV、onnxruntime 核心 |
| GPU 推理支持 | 约 **1.9 GB** | `onnxruntime_providers_cuda.dll`（325 MB）+ CUDA 12 / cuDNN 9 运行库（约 1.6 GB） |
| 示例方案 | 几 MB | 示例流程与图像，装到 `%ProgramData%\CVFlow\examples` |

安装程序本身经 LZMA2 压缩，完整版约 1 GB 上下，仅 CPU 约 200 MB。

## 为什么能这么切

onnxruntime 的 CUDA 后端是**单独一个 DLL**，运行时才去加载。不装它（以及 CUDA/cuDNN 运行库）时
软件照常启动，只是推理跑在 CPU 上，`cvflow gpu` 会说明"安装时没勾 GPU 组件"。
所以一套程序 + 一个可选组件就够了，不必出两个安装包。

## 构建

前置条件：

* 64 位 Python 3.10–3.13，并且**已经按 GPU 方式装好环境**（`.\install.ps1`）——
  安装包里的 CUDA/cuDNN 就是从这个环境里收走的
* [Inno Setup 6](https://jrsoftware.org/isdl.php)：`winget install JRSoftware.InnoSetup`
  （没装也能先跑，冻结那一步照常完成，到封装时才会提示；装完用 `-InstallerOnly`
  接着封装即可，**不必重新冻结**——那一步要十几分钟）

```powershell
.\packaging\build.ps1                    # 完整构建，产出 dist\CVFlow-Setup-0.1.0-x64.exe
.\packaging\build.ps1 -SkipGpu           # 只打基础部分（约 530 MB），CI 用这个验证链路
.\packaging\build.ps1 -TrimCudnn         # GPU 组件去掉 cudnn_adv64_9.dll（258 MB，RNN/注意力才用）
.\packaging\build.ps1 -FreezeOnly        # 只做 PyInstaller，不封安装包（调试打包问题）
.\packaging\build.ps1 -InstallerOnly     # 反过来：跳过冻结，把已有的 dist\CVFlow 直接封成安装包
.\packaging\build.ps1 -Clean             # 先清 build\ 和 dist\
```

构建完会打印两个组件各占多少，当场就能核对切分对不对。

## 安装后的样子

```
C:\Program Files\CVFlow\          程序（只读）
  CVFlow.exe                      界面，双击打开
  cvflow.exe                      命令行：cvflow gpu / serve / run
  _internal\                       运行时、依赖、（勾了的话）CUDA/cuDNN
%ProgramData%\CVFlow\             数据（可写——Program Files 下用户改不了）
  solutions\  logs\  captures\  examples\
```

安装向导做的事：检查 64 位、Windows 10 以上；没检测到 NVIDIA 显卡时提醒别白勾 GPU 组件；
可选创建桌面快捷方式、关联 `.cvflow` 文件；装完可选立即跑一次 `cvflow gpu` 自检——
**现场当场就知道 CUDA 行不行**，不用等跑流程才发现。

静默安装（产线批量部署）：

```powershell
CVFlow-Setup-0.1.0-x64.exe /SILENT /DIR="C:\CVFlow"
CVFlow-Setup-0.1.0-x64.exe /SILENT /COMPONENTS="main,examples"    # 不要 GPU 组件
```

## 已知限制

* **没有代码签名**：双击会看到 SmartScreen 警告（"Windows 已保护你的电脑"→ 更多信息 → 仍要运行）。
  要去掉只能买代码签名证书，然后在 build.ps1 里加一步 signtool。
* 卸载会保留 `%ProgramData%\CVFlow`（方案和存图在里面），需要的话手工删。
* GPU 组件装上之后，换一台没有 NVIDIA 显卡的机器仍然能启动，只是推理回到 CPU。
