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

.EXAMPLE
    .\packaging\build.ps1 -InstallerOnly
    反过来：跳过冻结，直接把已有的 dist\CVFlow 封成安装包。
    冻结要十几分钟（光 CUDA/cuDNN 就 1.9 GB），补装完 Inno Setup 之后用这个，不必重来
#>
[CmdletBinding()]
param(
    [switch] $SkipGpu,                      # 不收 CUDA/cuDNN 运行库
    [switch] $TrimCudnn,                    # GPU 组件里去掉 cudnn_adv（省 258 MB）
    [switch] $FreezeOnly,                   # 只冻结，不封安装包
    [switch] $InstallerOnly,                # 跳过冻结，直接把已有的 dist\CVFlow 封成安装包
    [string] $Iscc = "",                    # Inno Setup 的 ISCC.exe 路径（自动找不到时手工指定）
    [switch] $Clean,                        # 先清掉 build/ 和 dist/
    [string] $Python = "",                  # 用哪个 Python，默认优先用项目里的 .venv
    [string] $Version = "0.1.0"
)

# Windows PowerShell 5.1 的坑：$ErrorActionPreference = "Stop" 时，**原生命令**（python、pip、
# PyInstaller、ISCC…）只要往 stderr 写一个字，就会被包成 NativeCommandError 当成终止错误，
# 连 2>$null 都拦不住——PyInstaller 的日志恰恰全写 stderr。所以这里用 Continue，
# 原生命令一律显式查 $LASTEXITCODE（下面每处都查了）；会动文件的 cmdlet 单独加 -ErrorAction Stop。
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot

# 构建过程里的输出会被重定向（CI 抓日志、管道），而本地编码在英文 Windows 上是 cp1252，
# 打中文会抛 UnicodeEncodeError 把构建弄挂。把子进程的 I/O 编码定死成 UTF-8。
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# 没有真实控制台时（CI 里输出被重定向）这一句可能失败，失败了也不影响构建
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

function Say  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok   { param($m) Write-Host "    $m"   -ForegroundColor Green }
function Note { param($m) Write-Host "    $m"   -ForegroundColor DarkGray }
function Die  { param($m) Write-Host "`n✗ $m`n" -ForegroundColor Red; exit 1 }

# 找 Inno Setup 的命令行编译器 ISCC.exe。
# 装的位置比想象中多：winget 可能按用户装（%LOCALAPPDATA%\Programs），也可能按机器装
# （Program Files / Program Files (x86)）；版本目录名还会随大版本变（Inno Setup 6 / 7…）。
# 所以除了常见路径，还要查注册表的卸载项（按用户装的在 HKCU）和 PATH。
function Find-Iscc {
    param([ref] $Searched)
    $tried = @()

    if ($Iscc) {                                   # 手工指定优先
        $tried += $Iscc
        if (Test-Path $Iscc) { $Searched.Value = $tried; return $Iscc }
    }

    $roots = @("${env:ProgramFiles(x86)}", "$env:ProgramFiles",
               "$env:LOCALAPPDATA\Programs", "$env:ProgramW6432") | Where-Object { $_ }
    foreach ($r in $roots) {
        foreach ($d in @("Inno Setup 6", "Inno Setup 5")) {      # 固定名字先试，最快
            $p = Join-Path (Join-Path $r $d) "ISCC.exe"
            $tried += $p
            if (Test-Path $p) { $Searched.Value = $tried; return $p }
        }
        # 大版本号变了也能找到（Inno Setup 7…）
        $dirs = Get-ChildItem -LiteralPath $r -Directory -Filter "Inno Setup*" -ErrorAction SilentlyContinue
        foreach ($d in $dirs) {
            $p = Join-Path $d.FullName "ISCC.exe"
            $tried += $p
            if (Test-Path $p) { $Searched.Value = $tried; return $p }
        }
    }

    # 注册表的卸载项里有 InstallLocation，按用户装的在 HKCU
    $keys = @("HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
              "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
              "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
    $tried += "注册表卸载项（HKLM / HKLM WOW6432Node / HKCU）"
    foreach ($k in $keys) {
        $subs = Get-ChildItem -LiteralPath $k -ErrorAction SilentlyContinue
        foreach ($sub in $subs) {
            $prop = Get-ItemProperty -LiteralPath $sub.PSPath -ErrorAction SilentlyContinue
            if ($prop -and $prop.DisplayName -like "Inno Setup*" -and $prop.InstallLocation) {
                $p = Join-Path $prop.InstallLocation "ISCC.exe"
                if (Test-Path $p) { $Searched.Value = $tried; return $p }
            }
        }
    }

    $tried += "PATH 上的 ISCC.exe"
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { $Searched.Value = $tried; return $cmd.Source }

    $Searched.Value = $tried
    return $null
}

Set-Location -LiteralPath $Root -ErrorAction Stop
# .NET 的"当前目录"和 PowerShell 的"当前位置"是两回事：Set-Location 只改后者。
# 不同步的话，任何 [System.IO.*] 调用拿到相对路径都会跑去进程启动时的目录找
# （典型现象：明明在仓库里建的 .python.tmp，却报 C:\Users\xxx\.python.tmp 找不到）。
[Environment]::CurrentDirectory = (Get-Location).Path

$app = "$Root\dist\CVFlow"

if ($InstallerOnly) {
    # 冻结那一步很贵（要搬 1.9 GB 的 CUDA/cuDNN），补装完 Inno Setup 之后没必要重做一遍
    Say "按 -InstallerOnly 跳过冻结，直接用已有的 $app"
    if (-not (Test-Path "$app\CVFlow.exe")) {
        Die "没有 $app\CVFlow.exe。先跑一次完整构建（或 -FreezeOnly）把程序冻结出来"
    }
    $built = (Get-Item "$app\CVFlow.exe").LastWriteTime
    Ok "沿用 $built 冻结的那一份"
}

# --------------------------------------------------------------------- 1. Python
if (-not $InstallerOnly) {
if (-not $Python) {
    foreach ($cand in @("$Root\.venv\Scripts\python.exe", "$Root\.python\python.exe")) {
        if (Test-Path $cand) { $Python = $cand; break }
    }
}
if (-not $Python) { $Python = "python" }
# 用 Get-Command 显式判断"在不在"：ErrorActionPreference 是 Continue，命令不存在不会抛异常，
# try/catch 捕不到，报错会变成后面莫名其妙的"版本探测失败"
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Die "找不到 Python（试的是 '$Python'）。先跑 .\install.ps1 建好环境，或用 -Python 指定"
}
$ver = & $Python -c "import sys;print('%d.%d %d' % (sys.version_info[0], sys.version_info[1], __import__('struct').calcsize('P')*8))"
if ($LASTEXITCODE -ne 0 -or -not $ver) { Die "'$Python' 跑不起来，探测版本失败" }
Say "构建用的 Python"
Ok "$Python  →  $ver"
if (-not ($ver -match " 64$")) { Die "必须是 64 位 Python" }

# --------------------------------------------------------------------- 2. 构建依赖
Say "检查构建依赖"
& $Python -c "import cvflow" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Die "这个 Python 环境里没装 cvflow。先在仓库根目录跑 .\install.ps1" }
& $Python -c "import PyInstaller" 2>&1 | Out-Null
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
    foreach ($d in @("$Root\build", "$Root\dist")) { if (Test-Path $d) { Remove-Item -LiteralPath $d -Recurse -Force -ErrorAction Stop } }
}
Say "PyInstaller 冻结程序（第一次比较久）"
$env:CVFLOW_SKIP_NVIDIA = if ($SkipGpu) { "1" } else { "" }
& $Python -m PyInstaller --noconfirm --clean --distpath "$Root\dist" --workpath "$Root\build" "$Root\packaging\cvflow.spec"
if ($LASTEXITCODE -ne 0) { Die "PyInstaller 失败，上面的输出是第一手线索" }
foreach ($exe in @("$app\CVFlow.exe", "$app\cvflow.exe")) {
    if (-not (Test-Path $exe)) { Die "没产出 $exe，打包配置可能有问题" }
}
$size = [math]::Round(((Get-ChildItem $app -Recurse -File | Measure-Object Length -Sum).Sum / 1GB), 2)
Ok "冻结完成：$app（$size GB）"
}   # -InstallerOnly 到此为止，下面的组件切分和封装两种模式都要做

if ($TrimCudnn) {
    $adv = Get-ChildItem "$app\_internal\nvidia\cudnn\bin\cudnn_adv64_*.dll" -ErrorAction SilentlyContinue
    foreach ($f in $adv) {
        Note "按 -TrimCudnn 去掉 $($f.Name)（$([math]::Round($f.Length/1MB)) MB）"
        Remove-Item -LiteralPath $f.FullName -Force -ErrorAction Stop
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
if ($InstallerOnly) { Note "跳过（-InstallerOnly 没有参照的源码环境）" } else {
$expect = (& $Python -m cvflow nodes --json | ConvertFrom-Json).Count
$got = (& "$app\cvflow.exe" nodes --json | ConvertFrom-Json).Count
Note "源码环境 $expect 个，打包后 $got 个"
if ($got -lt $expect) { Die "打包后少了 $($expect - $got) 个节点，检查 packaging\cvflow.spec 里的 hidden imports" }
Ok "一致"
}

# 冻结出来的程序先自检一次：连 onnxruntime 都加载不了的话，封成安装包也没意义
Say "自检冻结后的程序"
& "$app\cvflow.exe" gpu
if ($LASTEXITCODE -ne 0) { Note "自检退出码 $LASTEXITCODE（没装 GPU 组件或这台机器没显卡时属正常）" }

if ($FreezeOnly) { Ok "按 -FreezeOnly 到此为止：$app"; exit 0 }

# --------------------------------------------------------------------- 4. 封安装包
Say "Inno Setup 封装"
$searched = @()
$iscc = Find-Iscc ([ref]$searched)
if (-not $iscc) {
    Die @"
找不到 ISCC.exe（Inno Setup 的命令行编译器）。找过这些地方：
$($searched -join "`n")

还没装的话装一个：
    winget install JRSoftware.InnoSetup
    或 choco install innosetup
    或从 https://jrsoftware.org/isdl.php 下载安装

**已经装了**却没找到（装到别的盘、或者是按用户装的）：先找出来再指过去
    Get-ChildItem $env:LOCALAPPDATA, $env:ProgramFiles, "C:\" -Filter ISCC.exe -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 3 -ExpandProperty FullName
    .\packaging\build.ps1 -InstallerOnly -Iscc "找到的那个路径"

**装完不用重新冻结**（那一步要十几分钟），直接接着封装即可：
    .\packaging\build.ps1 -InstallerOnly

只想要冻结好的程序目录、不封安装包：.\packaging\build.ps1 -FreezeOnly
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
