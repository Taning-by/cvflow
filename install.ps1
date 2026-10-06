<#
.SYNOPSIS
    一键装好 CVFlow 的运行环境（Windows）。有 NVIDIA 显卡时默认装 GPU 版，装完自动验证 CUDA。

.DESCRIPTION
    做的事情和 docs\install.md 里的手工步骤完全一样，只是不会漏步、不会装错顺序：
      1) 检查 Python 版本和位数
      2) 检查显卡驱动够不够新（CUDA 12 需要 >= 527.41）
      3) 建虚拟环境、升级 pip
      4) 把 onnxruntime / onnxruntime-gpu 卸干净（两个包共用一个模块，混装必然出问题）
      5) 只装需要的那一个：.[dev,gpu] 或 .[dev,cpu]
      6) 跑 cvflow gpu 自检；要求 GPU 时 CUDA 没跑起来就报错退出

.EXAMPLE
    .\install.ps1
    有 NVIDIA 显卡：装 GPU 版（onnxruntime-gpu + CUDA 12 + cuDNN 9，约 2 GB）

.EXAMPLE
    .\install.ps1 -Cpu
    没有 NVIDIA 显卡：装 CPU 版

.EXAMPLE
    .\install.ps1 -Model D:\models\best.onnx
    装完再拿这个模型实测一次：实际用哪个后端、占多少显存

.EXAMPLE
    .\install.ps1 -Recreate
    之前装乱了：删掉旧虚拟环境从头装一遍（代码不受影响）
#>
[CmdletBinding()]
param(
    [switch] $Cpu,                  # 强制装 CPU 版（没有 NVIDIA 显卡时用）
    [switch] $Recreate,             # 删掉旧的虚拟环境从头装（装乱了用这个，代码不受影响）
    [string] $Python = "python",    # 用哪个 Python 建虚拟环境，可写 py -3.12 或绝对路径
    [string] $Venv   = ".venv",     # 虚拟环境目录
    [string] $Model  = ""           # 可选：装完用这个 ONNX 模型实测一次
)

$ErrorActionPreference = "Stop"
$MinDriver = [version] "527.41"     # CUDA 12 在 Windows 上要求的最低驱动版本

function Say      { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok       { param($m) Write-Host "    $m"   -ForegroundColor Green }
function Note     { param($m) Write-Host "    $m"   -ForegroundColor DarkGray }
function Die      { param($m) Write-Host "`n✗ $m`n" -ForegroundColor Red; exit 1 }

# 以脚本所在目录为准，双击或从别的盘执行都不会跑错地方
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path "pyproject.toml")) { Die "这个脚本要放在 CVFlow 仓库根目录里执行" }

# --------------------------------------------------------------------- 1. Python
Say "检查 Python"
$pyCmd = $Python.Split(" ")[0]
$pyArgs = @($Python.Split(" ") | Select-Object -Skip 1)
try   { $probe = & $pyCmd @pyArgs "-c" "import sys,struct;print('%d.%d %d' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P')*8))" }
catch { Die "找不到 Python（试的是 '$Python'）。装 64 位 Python 3.10-3.14，或用 -Python 指定，例如 .\install.ps1 -Python 'py -3.12'" }
# 有些 Python 启动时会多打几行（比如虚拟环境提示），所以合成一串再按空白切
$parts = (($probe | Out-String).Trim() -split '\s+')
if ($parts.Count -lt 2) { Die "探测 Python 版本失败，输出是：$probe" }
$ver, $bits = $parts[-2], $parts[-1]
$v = [version] $ver
if ($bits -ne "64")                                { Die "Python 是 $bits 位的，onnxruntime 只有 64 位轮子。请装 64 位 Python" }
if ($v -lt [version]"3.10" -or $v -ge [version]"3.15") { Die "Python $ver 不行：onnxruntime-gpu 只有 3.10-3.14 的轮子。用 -Python 指定一个合适的版本" }
Ok "Python $ver（$bits 位）"

# --------------------------------------------------------------------- 2. 显卡驱动
$useGpu = -not $Cpu
if ($useGpu) {
    Say "检查 NVIDIA 显卡和驱动"
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        Die @"
找不到 nvidia-smi，说明这台机器没有 NVIDIA 显卡、或者驱动没装好。
  有独显的话：去 nvidia.com 装/更新显卡驱动后重跑本脚本
  确实没有独显：改成 .\install.ps1 -Cpu
"@
    }
    $rows = & nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
    if ($LASTEXITCODE -ne 0 -or -not $rows) { Die "nvidia-smi 执行失败，先确认显卡驱动是否正常" }
    $driver = [version] (($rows | Select-Object -First 1).Split(",")[2].Trim())
    foreach ($r in $rows) { Note ($r -replace "\s*,\s*", "  ") }
    if ($driver -lt $MinDriver) {
        Die @"
驱动版本 $driver 太老，CUDA 12 至少要 $MinDriver。
先去 nvidia.com 更新显卡驱动（不用装 CUDA Toolkit），再重跑本脚本。
只想先把软件跑起来、推理用 CPU：.\install.ps1 -Cpu
"@
    }
    Ok "驱动 $driver，满足 CUDA 12 的要求（>= $MinDriver）"
} else {
    Say "按 -Cpu 指定装 CPU 版，跳过显卡检查"
}

# --------------------------------------------------------------------- 3. 虚拟环境
$venvPy = Join-Path $Venv "Scripts\python.exe"
if ($Recreate -and (Test-Path $Venv)) {
    # 环境正被当前会话激活时 python.exe 被占用，删不掉，先让用户 deactivate
    if ($env:VIRTUAL_ENV -and (Resolve-Path -LiteralPath $env:VIRTUAL_ENV).Path -eq (Resolve-Path -LiteralPath $Venv).Path) {
        Die "虚拟环境 $Venv 正处于激活状态，删不掉。先执行 deactivate，再重跑 .\install.ps1 -Recreate"
    }
    Say "按 -Recreate 删掉旧的虚拟环境 $Venv"
    Remove-Item -LiteralPath $Venv -Recurse -Force
}
if (Test-Path $venvPy) {
    Say "复用已有虚拟环境 $Venv"
} else {
    Say "创建虚拟环境 $Venv"
    & $pyCmd @pyArgs "-m" "venv" $Venv
    if (-not (Test-Path $venvPy)) { Die "虚拟环境没建起来，检查一下 $Venv 目录的写权限" }
}
Ok $venvPy

Say "升级 pip（旧 pip 会悄悄忽略 [cuda,cudnn] 这类 extra，导致 CUDA 运行库一个都不装）"
& $venvPy -m pip install -q -U pip
if ($LASTEXITCODE -ne 0) { Die "pip 升级失败，检查网络或公司内网镜像设置" }
Ok (& $venvPy -m pip --version)

# --------------------------------------------------------------------- 4. 卸干净推理运行时
Say "隔离虚拟环境"
Note "系统 site-packages 里的 CPU 版 onnxruntime 会盖掉环境里的 GPU 版，而且在环境内 pip 卸不掉它"
& $venvPy "tools\venv_clean.py" isolate | ForEach-Object { Note $_ }

Say "把 onnxruntime / onnxruntime-gpu 卸干净"
Note "两个包装的是同一个 Python 模块，谁后装谁覆盖，混装的症状就是「装了 GPU 版却只有 CPU 后端」"
for ($i = 0; $i -lt 5; $i++) {
    $listed = & $venvPy -m pip list --local --format=freeze 2>$null |
              Where-Object { $_ -match "^onnxruntime(-gpu)?==" }
    if (-not $listed) { break }
    & $venvPy -m pip uninstall -y -q onnxruntime onnxruntime-gpu | Out-Null
}
# 混装过的环境常是"dist-info 还在、模块目录已经没了"，这时 pip 以为装过，install 会直接跳过
& $venvPy "tools\venv_clean.py" purge | ForEach-Object { Note $_ }
Ok "已清空"

# --------------------------------------------------------------------- 5. 安装
$extra = if ($useGpu) { "dev,gpu" } else { "dev,cpu" }
Say "安装 cvflow[$extra]"
if ($useGpu) { Note "要下 onnxruntime-gpu 和 CUDA 12 / cuDNN 9 的 pip 包，约 2 GB，第一次会比较久" }
& $venvPy -m pip install -e ".[$extra]"
if ($LASTEXITCODE -ne 0) { Die "安装失败。上面的 pip 报错是第一手线索；常见原因是网络中断或镜像里没有 onnxruntime-gpu 1.21+" }
Ok "装好了"

# --------------------------------------------------------------------- 6. 自检
Say "自检推理环境"
$selfcheck = @("-m", "cvflow", "gpu")
if ($Model) { $selfcheck += $Model }
if ($useGpu) { $selfcheck += "--require-gpu" }
& $venvPy @selfcheck
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Die @"
环境装上了，但 CUDA 没跑起来（退出码 $rc）。按上面自检输出里的提示处理，详见 docs\install.md。
最常见的三种情况：驱动太老、公司镜像里只有旧版 onnxruntime-gpu、虚拟环境里还有残留。
"@
}

Write-Host ""
Ok "环境就绪。接下来："
Write-Host @"
    $Venv\Scripts\activate                 激活环境（之后直接敲 cvflow）
    cvflow gui                              打开桌面程序
    cvflow gpu 你的模型.onnx                 实测模型跑在哪、占多少显存
    pytest -q                               跑一遍测试
"@
