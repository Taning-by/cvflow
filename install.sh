#!/usr/bin/env bash
# 一键装好 CVFlow 的运行环境（Linux / macOS）。有 NVIDIA 显卡时默认装 GPU 版，装完自动验证 CUDA。
#
#   ./install.sh                        有 NVIDIA 显卡：装 .[dev,gpu]（CUDA 12 + cuDNN 9 一起装，约 2 GB）
#   ./install.sh --cpu                  没有 NVIDIA 显卡：装 .[dev,cpu]
#   ./install.sh --model best.onnx      装完再拿这个模型实测一次
#   ./install.sh --recreate             删掉旧的虚拟环境从头来（装乱了用这个，代码不受影响）
#   ./install.sh --python /usr/bin/python3.12   指定解释器（机器上只有 conda 时要这样）
#   ./install.sh --allow-conda          允许用 conda 的解释器建 venv（默认拒绝）
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
MIN_DRIVER=525.60          # CUDA 12 在 Linux 上要求的最低驱动版本

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu)    USE_GPU=0; shift ;;
    --allow-conda) ALLOW_CONDA=1; shift ;;
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

# ------------------------------------------------------------------ 1. Python
say "检查 Python"
command -v "$PYTHON" >/dev/null 2>&1 || die "找不到 $PYTHON。装 Python 3.10-3.13，或用 --python 指定"
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
