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

.EXAMPLE
    .\install.ps1 -Python "C:\Python312\python.exe"
    机器上装的都是 conda 时，明确用一个普通 CPython 建 venv（推荐这么做）

.EXAMPLE
    .\install.ps1 -BootstrapPython
    机器上没有 Python、或者只有 conda：把便携版 CPython 下到项目里的 .python\，
    再用它建 .venv。装完整个项目自带 Python，和系统里的 Python / conda 完全无关
#>
[CmdletBinding()]
param(
    [switch] $Cpu,                  # 强制装 CPU 版（没有 NVIDIA 显卡时用）
    [switch] $Recreate,             # 删掉旧的虚拟环境从头装（装乱了用这个，代码不受影响）
    [switch] $AllowConda,           # 允许用 conda 环境里的解释器建 venv（默认拒绝，见下）
    [switch] $BootstrapPython,      # 把一份便携版 CPython 下到 .python\，整个项目自带 Python
    [string] $PythonArchive = "",   # 已经手工下好的便携版压缩包（网络慢时用，照样校验 SHA256）
    [string] $Python = "python",    # 用哪个 Python 建虚拟环境，可写 py -3.12 或绝对路径
    [string] $Venv   = ".venv",     # 虚拟环境目录
    [string] $Model  = ""           # 可选：装完用这个 ONNX 模型实测一次
)

# Windows PowerShell 5.1 的坑：$ErrorActionPreference = "Stop" 时，**原生命令**（python、pip、
# PyInstaller、ISCC…）只要往 stderr 写一个字，就会被包成 NativeCommandError 当成终止错误，
# 连 2>$null 都拦不住——PyInstaller 的日志恰恰全写 stderr。所以这里用 Continue，
# 原生命令一律显式查 $LASTEXITCODE（下面每处都查了）；会动文件的 cmdlet 单独加 -ErrorAction Stop。
$ErrorActionPreference = "Continue"
$MinDriver = [version] "527.41"     # CUDA 12 在 Windows 上要求的最低驱动版本

# -BootstrapPython 用的便携版 CPython（python-build-standalone，可重定位，自带 pip 和 venv）。
# 版本和哈希都写死：装环境这一步必须可复现，也必须能校验下载内容。
# 换版本时去 https://github.com/astral-sh/python-build-standalone/releases 取对应的 SHA256SUMS。
$PyRelease = "20261003"
$PyAsset   = "cpython-3.12.15+20261003-x86_64-pc-windows-msvc-install_only.tar.gz"
$PySha256  = "4b6f0beebbb695a0f3ea237b8c3eaa5bd424f47a7bc25b2fbe3a43390c770f08"
$PyDir     = ".python"

# 流式下载并显示进度。
# 不用 Invoke-WebRequest：PS 5.1 下它画进度条会把大文件拖慢十倍以上，而关掉进度条
# （$ProgressPreference = "SilentlyContinue"）又变成几分钟一点反馈都没有，看着像卡死。
function Save-File {
    param([string] $Url, [string] $Dest)
    # 相对路径先转成绝对：[System.IO.File]::Create 按 .NET 的当前目录解析，不是 $PWD
    if (-not [System.IO.Path]::IsPathRooted($Dest)) { $Dest = Join-Path (Get-Location).Path $Dest }
    $Dest = [System.IO.Path]::GetFullPath($Dest)
    # PS 5.1 默认可能还在用 TLS 1.0/1.1，而 GitHub 只收 1.2 以上
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }
    $req = [System.Net.HttpWebRequest]::Create($Url)
    $req.UserAgent = "cvflow-installer"
    $req.Timeout = 30000              # 连不上就别干等
    $req.ReadWriteTimeout = 120000
    try { $req.Proxy = [System.Net.WebRequest]::GetSystemWebProxy()
          $req.Proxy.Credentials = [System.Net.CredentialCache]::DefaultCredentials } catch { }
    $resp = $req.GetResponse()
    $total = $resp.ContentLength
    $in = $resp.GetResponseStream()
    $out = [System.IO.File]::Create($Dest)
    $buf = New-Object byte[] 262144
    $done = [long]0
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $lastMs = [long]0
    try {
        while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
            $out.Write($buf, 0, $n)
            $done += $n
            if (($sw.ElapsedMilliseconds - $lastMs) -ge 400) {
                $lastMs = $sw.ElapsedMilliseconds
                $secs = [math]::Max($sw.Elapsed.TotalSeconds, 0.001)
                $speed = ($done / 1MB) / $secs
                if ($total -gt 0) {
                    $pct = [int]($done * 100 / $total)
                    $left = if ($speed -gt 0.01) { [TimeSpan]::FromSeconds((($total - $done) / 1MB) / $speed).ToString("mm\:ss") } else { "--:--" }
                    $msg = "`r    {0,6:N1} / {1:N1} MB   {2,3}%   {3,5:N1} MB/s   剩余 {4}   " -f ($done / 1MB), ($total / 1MB), $pct, $speed, $left
                    Write-Host $msg -NoNewline
                } else {
                    Write-Host ("`r    已下载 {0,6:N1} MB   {1,5:N1} MB/s   " -f ($done / 1MB), $speed) -NoNewline
                }
            }
        }
    } finally {
        $out.Close(); $in.Close(); $resp.Close()
        Write-Host ("`r    {0:N1} MB 下载完成，用时 {1:N0} 秒{2}" -f ($done / 1MB), $sw.Elapsed.TotalSeconds, (" " * 20))
    }
}

function Say      { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok       { param($m) Write-Host "    $m"   -ForegroundColor Green }
function Note     { param($m) Write-Host "    $m"   -ForegroundColor DarkGray }
function Die      { param($m) Write-Host "`n✗ $m`n" -ForegroundColor Red; exit 1 }

# 以脚本所在目录为准，双击或从别的盘执行都不会跑错地方
Set-Location -LiteralPath $PSScriptRoot -ErrorAction Stop
# .NET 的"当前目录"和 PowerShell 的"当前位置"是两回事：Set-Location 只改后者。
# 不同步的话，任何 [System.IO.*] 调用拿到相对路径都会跑去进程启动时的目录找
# （典型现象：明明在仓库里建的 .python.tmp，却报 C:\Users\xxx\.python.tmp 找不到）。
[Environment]::CurrentDirectory = (Get-Location).Path
if (-not (Test-Path "pyproject.toml")) { Die "这个脚本要放在 CVFlow 仓库根目录里执行" }

# --------------------------------------------------------------- 0. 项目自带的 Python
$bootstrapPy = Join-Path $PyDir "python.exe"
if ($BootstrapPython) {
    if (Test-Path $bootstrapPy) {
        Say "复用项目里已有的便携版 Python"
        Ok (Resolve-Path -LiteralPath $bootstrapPy).Path
    } else {
        Say "把便携版 CPython 下到 $PyDir\（约 44 MB，机器上不需要预装 Python）"
        $url = "https://github.com/astral-sh/python-build-standalone/releases/download/$PyRelease/$PyAsset"
        $tmp = "$PyDir.tmp"
        if (Test-Path $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction Stop }
        New-Item -ItemType Directory -Path $tmp -ErrorAction Stop | Out-Null
        # 压缩包直接下到解压目录里：待会儿用相对文件名调 tar，命令行里就不会出现带盘符的路径
        $tgz = Join-Path $tmp $PyAsset
        if ($PythonArchive) {
            # 网络慢或者下不动时：自己用浏览器/下载工具下好，再指过来（照样校验 SHA256）
            if (-not (Test-Path $PythonArchive)) { Die "找不到 -PythonArchive 指定的文件：$PythonArchive" }
            Note "用本地压缩包：$PythonArchive"
            Copy-Item -LiteralPath $PythonArchive -Destination $tgz -ErrorAction Stop
        } else {
            Note $url
            try { Save-File -Url $url -Dest $tgz }
            catch {
                Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
                Die @"
下载失败：$($_.Exception.Message)

这个包在 GitHub 上，国内直连有时很慢甚至连不上。两条退路：
  · 自己用浏览器或下载工具把它下下来，再指过去（会照样校验 SHA256）：
      $url
      .\install.ps1 -BootstrapPython -PythonArchive "D:\下载\$PyAsset"
  · 机器上装一个普通的 64 位 Python 3.10-3.14，然后不用 -BootstrapPython：
      .\install.ps1 -Python "C:\Python312\python.exe"
"@
            }
        }
        $got = (Get-FileHash -Algorithm SHA256 -LiteralPath $tgz).Hash.ToLower()
        if ($got -ne $PySha256) {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
            Die "下载内容校验不过：期望 $PySha256，实到 $got。网络被劫持或文件损坏，别用它"
        }
        Note "SHA256 校验通过"
        # Windows 自带的 tar 是 bsdtar（System32\tar.exe），处理 C:\ 这种路径没问题；而 PATH 上
        # 常常先命中 Git for Windows 的 GNU tar，它把 "C:\..." 当成远程磁带机的 host:path，
        # 报 "Cannot connect to C: resolve failed"。所以指名用 System32 的那个，
        # 并且切进解压目录用相对文件名调用——两种 tar 都能过。
        $tarExe = Join-Path $env:SystemRoot "System32\tar.exe"
        if (-not (Test-Path $tarExe)) { $tarExe = "tar" }
        Push-Location $tmp
        try { & $tarExe -xf $PyAsset } finally { Pop-Location }
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $tmp "python\python.exe"))) {
            Die @"
解压失败（用的是 $tarExe）。
Windows 10 1803 以后自带 System32\tar.exe；如果 PATH 上是 Git for Windows 的 GNU tar，
它不认带盘符的路径。手工解开也行：把 $tgz 解压后，里面的 python 目录改名成 $PyDir
"@
        }
        Move-Item -LiteralPath (Join-Path $tmp "python") -Destination $PyDir -ErrorAction Stop
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        Ok "装好了：$((Resolve-Path -LiteralPath $bootstrapPy).Path)"
    }
    $Python = (Resolve-Path -LiteralPath $bootstrapPy).Path
} elseif ((Test-Path $bootstrapPy) -and $Python -eq "python") {
    # 项目里已经有便携版了，默认就用它，不去碰系统的 Python / conda
    Say "项目里有便携版 Python，直接用它（不想用就加 -Python 指定别的）"
    $Python = (Resolve-Path -LiteralPath $bootstrapPy).Path
    Ok $Python
}

# --------------------------------------------------------------------- 1. Python
Say "检查 Python"
$pyCmd = $Python.Split(" ")[0]
$pyArgs = @($Python.Split(" ") | Select-Object -Skip 1)
# 显式判断命令在不在：ErrorActionPreference 是 Continue（原因见文件开头），
# 命令不存在不会抛异常，try/catch 捕不到
if (-not (Get-Command $pyCmd -ErrorAction SilentlyContinue)) { Die @"
找不到 Python（试的是 '$Python'）。三条路：
  · 让项目自带一份（机器上什么都不用装）： .\install.ps1 -BootstrapPython
  · 用已有的某个解释器：                   .\install.ps1 -Python "C:\Python312\python.exe"
  · 自己装 64 位 Python 3.10-3.14：        winget install Python.Python.3.12
"@ }
# conda-meta 目录是 conda 安装/环境的标志，conda 环境本身不是 venv（sys.prefix == sys.base_prefix），认不出来
$PROBE = "import os,sys,struct;print('%d.%d %d %s' % (sys.version_info[0], sys.version_info[1], " +
         "struct.calcsize('P')*8, 'conda' if os.path.isdir(os.path.join(sys.base_prefix, 'conda-meta')) else 'plain'))"
$probe = & $pyCmd @pyArgs "-c" $PROBE
# 有些 Python 启动时会多打几行（比如虚拟环境提示），所以合成一串再按空白切
$parts = (($probe | Out-String).Trim() -split '\s+')
if ($parts.Count -lt 3) { Die "探测 Python 版本失败，输出是：$probe" }
$ver, $bits, $kind = $parts[-3], $parts[-2], $parts[-1]
$v = [version] $ver
if ($bits -ne "64")                                { Die "Python 是 $bits 位的，onnxruntime 只有 64 位轮子。请装 64 位 Python" }
if ($v -lt [version]"3.10" -or $v -ge [version]"3.15") { Die "Python $ver 不行：onnxruntime-gpu 只有 3.10-3.14 的轮子。用 -Python 指定一个合适的版本" }
if ($kind -eq "conda" -and -not $AllowConda) {
    Die @"
'$Python' 是 conda 里的解释器（Python $ver）。默认拒绝用它建 venv，原因有两个：
  1) conda 的 python.exe 要靠 <env>\Library\bin 里的 DLL，那个目录只在 conda 环境激活时才在 PATH 上；
     基于它建的 venv 继承这个依赖，以后不激活 conda 就可能报 DLL load failed / No module named '_ssl'
  2) conda 环境里若装过 cudnn / cudatoolkit，它们的 DLL 也在 PATH 上，会和 pip 装进 venv 的
     CUDA 12 / cuDNN 9 撞版本，排查起来非常费劲
怎么办（任选其一）：
  · 装一个普通 CPython（python.org 或 winget install Python.Python.3.12），然后指定它：
      .\install.ps1 -Python "C:\Python312\python.exe"        （或 -Python "py -3.12"）
  · 先 conda deactivate 退出 conda，再确认 where python 指向的不是 conda
  · 确实要用 conda 的解释器：加 -AllowConda
"@
}
Ok "Python $ver（$bits 位，$(if ($kind -eq 'conda') { 'conda 解释器' } else { '普通 CPython' })）"
if ($env:CONDA_PREFIX) {
    Note "注意：当前有 conda 环境处于激活状态（$env:CONDA_PREFIX）。装之前先 conda deactivate 更干净，"
    Note "      免得 conda 的 Library\bin 往 PATH 里塞 CUDA / cuDNN 的 DLL，和环境里 pip 装的那套撞版本"
}

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
    Remove-Item -LiteralPath $Venv -Recurse -Force -ErrorAction Stop
}
if (Test-Path $venvPy) {
    Say "复用已有虚拟环境 $Venv"
    $venvKind = & $venvPy -c "import os,sys;print('conda' if os.path.isdir(os.path.join(sys.base_prefix, 'conda-meta')) else 'plain')"
    if ($venvKind -eq "conda" -and -not $AllowConda) {
        Die @"
已有的 $Venv 是用 conda 的解释器建的（它的 sys.base_prefix 指向一个 conda 环境）。
要换成普通 CPython 建的干净环境：
    .\install.ps1 -Recreate -Python "C:\Python312\python.exe"
就想接着用它：加 -AllowConda
"@
    }
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
