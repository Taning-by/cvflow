#!/usr/bin/env bash
# 一键装好 CVFlow 的运行环境（Linux / macOS）。有 NVIDIA 显卡时默认装 GPU 版，装完自动验证 CUDA。
#
#   ./install.sh                        有 NVIDIA 显卡：装 .[dev,gpu]（CUDA 12 + cuDNN 9 一起装，约 2 GB）
#   ./install.sh --cpu                  没有 NVIDIA 显卡：装 .[dev,cpu]
#   ./install.sh --model best.onnx      装完再拿这个模型实测一次
#   ./install.sh --recreate             删掉旧的虚拟环境从头来（装乱了用这个，代码不受影响）
#   ./install.sh --python /usr/bin/python3.12   指定解释器（机器上只有 conda 时要这样）
#   ./install.sh --allow-conda          允许用 conda 的解释器建 venv（默认拒绝）
#   ./install.sh --bootstrap-python     把便携版 CPython 下到项目里的 .python/，整个项目自带 Python
#   ./install.sh --python python3.12 --venv .venv
#
# 做的事和 docs/install.md 里的手工步骤一样，只是不会漏步、不会装错顺序。
set -euo pipefail

PYTHON=python3
VENV=.venv
MODEL=""
USE_GPU=1
RECREATE=0
ALLOW_CONDA=0
BOOTSTRAP=0
PY_DIR=.python
# 便携版 CPython（python-build-standalone，可重定位，自带 pip 和 venv）。版本和哈希写死：
# 装环境必须可复现、下载必须能校验。换版本时去
# https://github.com/astral-sh/python-build-standalone/releases 取对应的 SHA256SUMS。
PY_RELEASE=20261003
PY_VER=3.12.15
MIN_DRIVER=525.60          # CUDA 12 在 Linux 上要求的最低驱动版本

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu)    USE_GPU=0; shift ;;
    --allow-conda) ALLOW_CONDA=1; shift ;;
    --bootstrap-python) BOOTSTRAP=1; shift ;;
    --recreate) RECREATE=1; shift ;;
    --python) PYTHON="$2"; shift 2 ;;
    --venv)   VENV="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数：$1（看 ./install.sh --help）" >&2; exit 1 ;;
  esac
done

say()  { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
ok()   { printf '\033[32m    %s\033[0m\n' "$1"; }
note() { printf '\033[90m    %s\033[0m\n' "$1"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n\n' "$1" >&2; exit 1; }

cd "$(dirname "$(readlink -f "$0")")"
[ -f pyproject.toml ] || die "这个脚本要放在 CVFlow 仓库根目录里执行"

# ------------------------------------------------------ 0. 项目自带的 Python（--bootstrap-python）
portable_asset() {
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)         echo "x86_64-unknown-linux-gnu  f937814031eab4698ca6d07ec606ede1825768f3f3e99af76d9db3900bee03c5" ;;
    Linux-aarch64)        echo "aarch64-unknown-linux-gnu 95c01982c9fcb9d95b0acfdb5eb8a6e0099dd11edf062314a474228d93f2b765" ;;
    Darwin-arm64)         echo "aarch64-apple-darwin      316a463172740e71d8dca1f2730784e325f3f720941137b5d674d5801a632213" ;;
    Darwin-x86_64)        echo "x86_64-apple-darwin       a8fd7a91852f19b6d959793ef41fad048631ccb2a334a9ecdf573255298f7978" ;;
    *) return 1 ;;
  esac
}

bootstrap_python() {
  if [ -x "$PY_DIR/bin/python3" ]; then
    say "复用项目里已有的便携版 Python"; ok "$(cd "$PY_DIR" && pwd)/bin/python3"; return 0
  fi
  read -r triple sha <<<"$(portable_asset)" || die "没有 $(uname -s)-$(uname -m) 的便携版 Python，请自己装一个再用 --python 指定"
  asset="cpython-$PY_VER+$PY_RELEASE-$triple-install_only.tar.gz"
  url="https://github.com/astral-sh/python-build-standalone/releases/download/$PY_RELEASE/$asset"
  say "把便携版 CPython 下到 $PY_DIR/（机器上不需要预装 Python）"
  rm -rf "$PY_DIR.tmp"; mkdir -p "$PY_DIR.tmp"
  # 压缩包下到解压目录里，解压时用相对文件名：路径里含冒号时 GNU tar 会当成远程磁带机的 host:path
  curl -fL --progress-bar "$url" -o "$PY_DIR.tmp/$asset" || die "下载失败：$url"
  got=$( { sha256sum "$PY_DIR.tmp/$asset" 2>/dev/null || shasum -a 256 "$PY_DIR.tmp/$asset"; } | cut -d' ' -f1)
  [ "$got" = "$sha" ] || { rm -rf "$PY_DIR.tmp"; die "下载内容校验不过：期望 $sha，实到 $got。网络被劫持或文件损坏，别用它"; }
  note "SHA256 校验通过"
  ( cd "$PY_DIR.tmp" && tar -xf "$asset" ) || die "解压失败：$PY_DIR.tmp/$asset"
  [ -x "$PY_DIR.tmp/python/bin/python3" ] || die "解压出来的内容不对，没找到 python/bin/python3"
  mv "$PY_DIR.tmp/python" "$PY_DIR"; rm -rf "$PY_DIR.tmp"
  ok "装好了：$(cd "$PY_DIR" && pwd)/bin/python3"
}

if [ "$BOOTSTRAP" = 1 ]; then
  bootstrap_python
  PYTHON="$(cd "$PY_DIR" && pwd)/bin/python3"
elif [ -x "$PY_DIR/bin/python3" ] && [ "$PYTHON" = python3 ]; then
  # 项目里已经有便携版了，默认就用它，不去碰系统的 Python / conda
  say "项目里有便携版 Python，直接用它（不想用就加 --python 指定别的）"
  PYTHON="$(cd "$PY_DIR" && pwd)/bin/python3"
  ok "$PYTHON"
fi

# ------------------------------------------------------------------ 1. Python
say "检查 Python"
command -v "$PYTHON" >/dev/null 2>&1 || die "找不到 $PYTHON。三条路：
  · 让项目自带一份（机器上什么都不用装）：./install.sh --bootstrap-python
  · 用已有的解释器：                      ./install.sh --python /usr/bin/python3.12
  · 自己装 Python 3.10-3.14"
# conda-meta 目录是 conda 安装/环境的标志；conda 环境本身不是 venv，靠 sys.prefix 认不出来
read -r PYVER PYBITS PYKIND <<<"$("$PYTHON" -c 'import os,sys,struct;print("%d.%d %d %s" % (sys.version_info[0], sys.version_info[1], struct.calcsize("P")*8, "conda" if os.path.isdir(os.path.join(sys.base_prefix, "conda-meta")) else "plain"))')"
[ "$PYBITS" = "64" ] || die "Python 是 $PYBITS 位的，onnxruntime 只有 64 位轮子"
"$PYTHON" -c 'import sys;sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,15) else 1)' \
  || die "Python $PYVER 不行：onnxruntime-gpu 只有 3.10-3.14 的轮子。用 --python 指定一个合适的版本"
if [ "$PYKIND" = conda ] && [ "$ALLOW_CONDA" = 0 ]; then
  die "'$PYTHON' 是 conda 里的解释器（Python $PYVER）。默认拒绝用它建 venv：conda 环境里的
CUDA / cuDNN 会跟 pip 装进环境的那套撞版本，而且基于 conda 解释器的 venv 以后不激活 conda 可能起不来。
  · 用普通 CPython：./install.sh --python /usr/bin/python3.12
  · 或先 conda deactivate，确认 which python 不再指向 conda
  · 确实要用：加 --allow-conda"
fi
ok "Python $PYVER（$PYBITS 位，$([ "$PYKIND" = conda ] && echo 'conda 解释器' || echo '普通 CPython')）"
if [ -n "${CONDA_PREFIX:-}" ]; then
  note "注意：当前有 conda 环境处于激活状态（$CONDA_PREFIX），装之前先 conda deactivate 更干净"
fi

# ------------------------------------------------------------------ 2. 显卡驱动
if [ "$USE_GPU" = 1 ]; then
  say "检查 NVIDIA 显卡和驱动"
  command -v nvidia-smi >/dev/null 2>&1 || die "找不到 nvidia-smi，说明没有 NVIDIA 显卡或驱动没装好。
  有独显：装/更新显卡驱动后重跑（不用装 CUDA Toolkit）
  没独显：改成 ./install.sh --cpu"
  rows=$(nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader) \
    || die "nvidia-smi 执行失败，先确认显卡驱动是否正常"
  printf '%s\n' "$rows" | while IFS= read -r r; do note "$r"; done
  driver=$(printf '%s\n' "$rows" | head -1 | awk -F', *' '{print $3}')
  # 只比前两段（主.次），610.43.02 这种三段版本号直接用字符串比会出错
  if [ "$(printf '%s\n%s\n' "$MIN_DRIVER" "$(echo "$driver" | cut -d. -f1,2)" | sort -V | head -1)" != "$MIN_DRIVER" ]; then
    die "驱动版本 $driver 太老，CUDA 12 至少要 $MIN_DRIVER。更新显卡驱动后重跑；
只想先用 CPU 跑起来：./install.sh --cpu"
  fi
  ok "驱动 $driver，满足 CUDA 12 的要求（>= $MIN_DRIVER）"
else
  say "按 --cpu 指定装 CPU 版，跳过显卡检查"
fi

# ------------------------------------------------------------------ 3. 虚拟环境
VENV_PY="$VENV/bin/python"
if [ "$RECREATE" = 1 ] && [ -d "$VENV" ]; then
  # 环境正被当前 shell 激活时，删掉会留下一堆失效的 PATH 指向，先让用户 deactivate
  if [ -n "${VIRTUAL_ENV:-}" ] && [ "$(cd "$VIRTUAL_ENV" 2>/dev/null && pwd)" = "$(cd "$VENV" && pwd)" ]; then
    die "虚拟环境 $VENV 正处于激活状态。先执行 deactivate，再重跑 ./install.sh --recreate"
  fi
  say "按 --recreate 删掉旧的虚拟环境 $VENV"
  rm -rf "$VENV"
fi
if [ -x "$VENV_PY" ]; then
  say "复用已有虚拟环境 $VENV"
  if [ "$ALLOW_CONDA" = 0 ] && [ "$("$VENV_PY" -c 'import os,sys;print("conda" if os.path.isdir(os.path.join(sys.base_prefix, "conda-meta")) else "plain")')" = conda ]; then
    die "已有的 $VENV 是用 conda 的解释器建的。换成普通 CPython 建的干净环境：
    ./install.sh --recreate --python /usr/bin/python3.12
就想接着用它：加 --allow-conda"
  fi
else
  say "创建虚拟环境 $VENV"
  "$PYTHON" -m venv "$VENV"
  [ -x "$VENV_PY" ] || die "虚拟环境没建起来，检查 $VENV 目录的写权限"
fi
ok "$VENV_PY"

say "升级 pip（旧 pip 会悄悄忽略 [cuda,cudnn] 这类 extra，导致 CUDA 运行库一个都不装）"
"$VENV_PY" -m pip install -q -U pip || die "pip 升级失败，检查网络或内网镜像设置"
ok "$("$VENV_PY" -m pip --version)"

# ------------------------------------------------------------------ 4. 卸干净推理运行时
say "隔离虚拟环境"
note "系统 site-packages 里的 CPU 版 onnxruntime 会盖掉环境里的 GPU 版，而且在环境内 pip 卸不掉它"
"$VENV_PY" tools/venv_clean.py isolate | while IFS= read -r l; do note "$l"; done

say "把 onnxruntime / onnxruntime-gpu 卸干净"
note "两个包装的是同一个 Python 模块，谁后装谁覆盖，混装的症状就是「装了 GPU 版却只有 CPU 后端」"
for _ in 1 2 3 4 5; do
  "$VENV_PY" -m pip list --local --format=freeze 2>/dev/null | grep -qE '^onnxruntime(-gpu)?==' || break
  "$VENV_PY" -m pip uninstall -y -q onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
done
# 混装过的环境常是"dist-info 还在、模块目录已经没了"，这时 pip 以为装过，install 会直接跳过
"$VENV_PY" tools/venv_clean.py purge | while IFS= read -r l; do note "$l"; done
ok "已清空"

# ------------------------------------------------------------------ 5. 安装
EXTRA=$([ "$USE_GPU" = 1 ] && echo "dev,gpu" || echo "dev,cpu")
say "安装 cvflow[$EXTRA]"
[ "$USE_GPU" = 1 ] && note "要下 onnxruntime-gpu 和 CUDA 12 / cuDNN 9 的 pip 包，约 2 GB，第一次会比较久"
"$VENV_PY" -m pip install -e ".[$EXTRA]" \
  || die "安装失败。上面的 pip 报错是第一手线索；常见原因是网络中断或镜像里没有 onnxruntime-gpu 1.21+"
ok "装好了"

# ------------------------------------------------------------------ 6. 自检
say "自检推理环境"
set +e
check=("-m" "cvflow" "gpu")
[ -n "$MODEL" ] && check+=("$MODEL")
[ "$USE_GPU" = 1 ] && check+=("--require-gpu")
"$VENV_PY" "${check[@]}"
rc=$?
set -e
[ "$rc" = 0 ] || die "环境装上了，但 CUDA 没跑起来（退出码 $rc）。按上面自检输出里的提示处理，详见 docs/install.md。
最常见的三种情况：驱动太老、内网镜像里只有旧版 onnxruntime-gpu、虚拟环境里还有残留。"

echo
ok "环境就绪。接下来："
cat <<TXT
    source $VENV/bin/activate          激活环境（之后直接敲 cvflow）
    cvflow gui                          打开桌面程序
    cvflow gpu 你的模型.onnx             实测模型跑在哪、占多少显存
    pytest -q                           跑一遍测试
TXT
