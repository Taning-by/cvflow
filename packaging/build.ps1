<#
.SYNOPSIS
    打出 CVFlow 的 Windows 安装包：PyInstaller 冻结程序 → Inno Setup 封成 setup.exe。

.DESCRIPTION
    产物是**一个**安装程序，GPU 推理支持做成可勾选的组件：

      主程序        约 500 MB   装完就能用，推理跑在 CPU 上
      GPU 推理支持  约 1.9 GB   onnxruntime 的 CUDA provider（325 MB）+ CUDA 12 / cuDNN 9（约 1.6 GB）

    前置条件：
      · 64 位 Python 3.10-3.13（可以用项目自带的：install.ps1 -BootstrapPython）
      · Inno Setup 6（winget install JRSoftware.InnoSetup，或 choco install innosetup）

.EXAMPLE
    .\packaging\build.ps1
    完整构建：装构建依赖 → PyInstaller → Inno Setup，产出 dist\CVFlow-Setup-0.1.0-x64.exe

.EXAMPLE
    .\packaging\build.ps1 -SkipGpu
    不收 CUDA/cuDNN 的运行库，只打基础部分（约 500 MB）。CI 上用这个验证打包链路，快得多

.EXAMPLE
    .\packaging\build.ps1 -TrimCudnn
    GPU 组件里去掉 cudnn_adv64_9.dll（258 MB，循环网络/注意力才用得到，
    纯卷积的检测分割模型用不上）。省体积，但模型用到那些算子时会退回 CPU

.EXAMPLE
    .\packaging\build.ps1 -FreezeOnly
    只做 PyInstaller 那一步，产出 dist\CVFlow\，不封安装包（调试打包问题时用）
#>
[CmdletBinding()]
param(
    [switch] $SkipGpu,                      # 不收 CUDA/cuDNN 运行库
    [switch] $TrimCudnn,                    # GPU 组件里去掉 cudnn_adv（省 258 MB）
    [switch] $FreezeOnly,                   # 只冻结，不封安装包
    [switch] $Clean,                        # 先清掉 build/ 和 dist/
    [string] $Python = "",                  # 用哪个 Python，默认优先用项目里的 .venv
    [string] $Version = "0.1.0"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

# 构建过程里的输出会被重定向（CI 抓日志、管道），而本地编码在英文 Windows 上是 cp1252，
# 打中文会抛 UnicodeEncodeError 把构建弄挂。把子进程的 I/O 编码定死成 UTF-8。
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Say  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok   { param($m) Write-Host "    $m"   -ForegroundColor Green }
function Note { param($m) Write-Host "    $m"   -ForegroundColor DarkGray }
function Die  { param($m) Write-Host "`n✗ $m`n" -ForegroundColor Red; exit 1 }

Set-Location -LiteralPath $Root

# --------------------------------------------------------------------- 1. Python
if (-not $Python) {
    foreach ($cand in @("$Root\.venv\Scripts\python.exe", "$Root\.python\python.exe")) {
        if (Test-Path $cand) { $Python = $cand; break }
    }
}
if (-not $Python) { $Python = "python" }
try { $ver = & $Python -c "import sys;print('%d.%d %d' % (sys.version_info[0], sys.version_info[1], __import__('struct').calcsize('P')*8))" }
catch { Die "找不到 Python（试的是 '$Python'）。先跑 .\install.ps1 建好环境，或用 -Python 指定" }
Say "构建用的 Python"
Ok "$Python  →  $ver"
if (-not ($ver -match " 64$")) { Die "必须是 64 位 Python" }

# --------------------------------------------------------------------- 2. 构建依赖
Say "检查构建依赖"
& $Python -c "import cvflow" 2>$null
if ($LASTEXITCODE -ne 0) { Die "这个 Python 环境里没装 cvflow。先在仓库根目录跑 .\install.ps1" }
& $Python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Note "装 PyInstaller…"
    & $Python -m pip install -q "pyinstaller>=6.6"
    if ($LASTEXITCODE -ne 0) { Die "PyInstaller 装不上，检查网络或内网镜像" }
}
$ortPkg = & $Python -c "from importlib.metadata import distributions; n={d.metadata['Name'].lower() for d in distributions() if d.metadata['Name']}; print('gpu' if 'onnxruntime-gpu' in n else ('cpu' if 'onnxruntime' in n else 'none'))"
Ok "PyInstaller 就绪；推理运行时：onnxruntime-$ortPkg"
if ($ortPkg -eq "none") { Die "环境里没有 onnxruntime。跑 .\install.ps1 装上（有显卡装 GPU 版）" }
if (($ortPkg -eq "cpu") -and (-not $SkipGpu)) {
    Die @"
环境里装的是 CPU 版 onnxruntime，打不出带 GPU 组件的安装包。
  要带 GPU 组件：先 .\install.ps1（会装 onnxruntime-gpu 和 CUDA/cuDNN），再重跑本脚本
  只打基础部分：.\packaging\build.ps1 -SkipGpu
"@
}

# --------------------------------------------------------------------- 3. 冻结
if ($Clean) {
    Say "清理 build\ 和 dist\"
    foreach ($d in @("$Root\build", "$Root\dist")) { if (Test-Path $d) { Remove-Item -LiteralPath $d -Recurse -Force } }
}
Say "PyInstaller 冻结程序（第一次比较久）"
$env:CVFLOW_SKIP_NVIDIA = if ($SkipGpu) { "1" } else { "" }
& $Python -m PyInstaller --noconfirm --clean --distpath "$Root\dist" --workpath "$Root\build" "$Root\packaging\cvflow.spec"
if ($LASTEXITCODE -ne 0) { Die "PyInstaller 失败，上面的输出是第一手线索" }
$app = "$Root\dist\CVFlow"
foreach ($exe in @("$app\CVFlow.exe", "$app\cvflow.exe")) {
    if (-not (Test-Path $exe)) { Die "没产出 $exe，打包配置可能有问题" }
}
$size = [math]::Round(((Get-ChildItem $app -Recurse -File | Measure-Object Length -Sum).Sum / 1GB), 2)
Ok "冻结完成：$app（$size GB）"

if ($TrimCudnn) {
    $adv = Get-ChildItem "$app\_internal\nvidia\cudnn\bin\cudnn_adv64_*.dll" -ErrorAction SilentlyContinue
    foreach ($f in $adv) {
        Note "按 -TrimCudnn 去掉 $($f.Name)（$([math]::Round($f.Length/1MB)) MB）"
        Remove-Item -LiteralPath $f.FullName -Force
    }
}

# 报一下两个组件各占多少：和 installer.iss 里的模式保持一致，构建完当场能核对
Say "组件切分"
$gpuPatterns = @("_internal\nvidia\*", "*onnxruntime_providers_cuda.dll", "*onnxruntime_providers_tensorrt.dll",
                 "_internal\cudnn*.dll", "_internal\cublas*.dll", "_internal\cudart64*.dll",
                 "_internal\cufft*.dll", "_internal\curand*.dll", "_internal\nvrtc*.dll", "_internal\nvJitLink*.dll")
$all = Get-ChildItem $app -Recurse -File
$gpuFiles = $all | Where-Object { $rel = $_.FullName.Substring($app.Length + 1); $p = $rel; ($gpuPatterns | Where-Object { $p -like $_ }).Count -gt 0 }
$gpuSize = [math]::Round((($gpuFiles | Measure-Object Length -Sum).Sum / 1MB), 0)
$baseSize = [math]::Round((($all | Measure-Object Length -Sum).Sum / 1MB), 0) - $gpuSize
Note "主程序       $baseSize MB（$($all.Count - $gpuFiles.Count) 个文件）"
Note "GPU 推理支持 $gpuSize MB（$($gpuFiles.Count) 个文件）"
if ($gpuSize -eq 0 -and -not $SkipGpu) {
    Die "GPU 组件一个文件都没收到。环境里多半是 CPU 版 onnxruntime，或者 CUDA/cuDNN 的包没装上"
}

# 节点数量要和源码环境一致：少了说明 spec 里的 hidden imports 漏了某个模块，
# 这种错 PyInstaller 只打一行 ERROR 就继续，不盯着的话会打出一个悄悄少节点的程序
Say "核对节点数量"
$expect = (& $Python -m cvflow nodes --json | ConvertFrom-Json).Count
$got = (& "$app\cvflow.exe" nodes --json | ConvertFrom-Json).Count
Note "源码环境 $expect 个，打包后 $got 个"
if ($got -lt $expect) { Die "打包后少了 $($expect - $got) 个节点，检查 packaging\cvflow.spec 里的 hidden imports" }
Ok "一致"

# 冻结出来的程序先自检一次：连 onnxruntime 都加载不了的话，封成安装包也没意义
Say "自检冻结后的程序"
& "$app\cvflow.exe" gpu
if ($LASTEXITCODE -ne 0) { Note "自检退出码 $LASTEXITCODE（没装 GPU 组件或这台机器没显卡时属正常）" }

if ($FreezeOnly) { Ok "按 -FreezeOnly 到此为止：$app"; exit 0 }

# --------------------------------------------------------------------- 4. 封安装包
Say "Inno Setup 封装"
$iscc = $null
foreach ($p in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe")) {
    if (Test-Path $p) { $iscc = $p; break }
}
if (-not $iscc) { $iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue)?.Source }
if (-not $iscc) {
    Die @"
找不到 Inno Setup 6（ISCC.exe）。装一个：
    winget install JRSoftware.InnoSetup
    或 choco install innosetup
只想要冻结好的程序目录（不封安装包）：.\packaging\build.ps1 -FreezeOnly
"@
}
Note $iscc
& $iscc "/DAppVersion=$Version" "$Root\packaging\installer.iss"
if ($LASTEXITCODE -ne 0) { Die "Inno Setup 失败" }

$setup = Get-ChildItem "$Root\dist\CVFlow-Setup-*.exe" | Sort-Object LastWriteTime | Select-Object -Last 1
Ok "安装包：$($setup.FullName)（$([math]::Round($setup.Length/1MB)) MB）"
Write-Host @"

    双击安装，或静默安装（产线批量部署）：
      $($setup.Name) /SILENT /DIR="C:\CVFlow"
      $($setup.Name) /SILENT /COMPONENTS="main,examples"      只装基础部分，不要 GPU
"@
