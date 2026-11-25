# 验证集 JSONDecodeError：读取器修复

训练 600 题生成及 598 条激活提取已完成。本地同步的 validation.jsonl 是有效的 1120 条 JSONL，其中有两个 U+2028（Unicode 行分隔符）。原读取器先调用 str.splitlines()，会把 JSON 字符串内的 U+2028 当成记录边界，从而把一个合法 JSON 对象拆坏。错误发生在读取整个验证集，故即使 LIMIT=200 也会触发；不是模型、CUDA 或数据下载错误。

新增 scripts/generate_safe.py：按文件物理行读取 JSONL，保留字符串中的 Unicode 分隔符。它只在进程内替换原 generate 的记录读取语句，且检查原语句必须唯一匹配；不改变生成算法、prompt、停止规则、答案评分或 pipeline.py 文件。这样 generation.json 中原有 adapter 哈希保持一致，已有训练生成无需重跑或篡改配置。input-reader.json 单独记录读取实现及数据哈希。真实损坏的 JSON 仍会报告文件与物理行号，不静默修补数据。

本地同步：

```powershell
cd E:\Research\LLMCoT\methods\StepCluster
git add scripts/generate_safe.py scripts/check_jsonl.py scripts/run.sh RECOVER_JSONL_READER.md
git commit -m "Fix JSONL reading for embedded Unicode line separators"
git push
```

服务器：

```bash
cd ~/projects/StepCluster
conda activate stepcluster
git pull --ff-only
python -B scripts/check_jsonl.py
```

保留之前的所有路径与 RUN_NAME 设置，用新日志从 validation/test 继续；无需重跑 train 生成或提取。

```bash
export DATA_ROOT=/mnt/data/wxh/wdl
export STEP_ROOT="$DATA_ROOT/results/StepCluster"
export HF_HOME="$DATA_ROOT/cache/huggingface"
export MODEL_PATH="$DATA_ROOT/models/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28"
export RUN_NAME=qwen25-pilot-chat-v2
export PROMPT=chat
export STOP_MODE=final-line
export MAX_TOKENS=2048
export LAYER=27
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6
mkdir -p "$DATA_ROOT/logs/StepCluster"
LOG="$DATA_ROOT/logs/StepCluster/$RUN_NAME-continue-$(date +%Y%m%d-%H%M%S).log"
nohup setsid bash -c '
  set -euo pipefail
  for split in validation test; do
    LIMIT=200 bash scripts/run.sh generate gsm8k "$split"
    bash scripts/run.sh extract gsm8k "$split"
  done
  for feature in early final late; do
    FEATURE="$feature" bash scripts/run.sh heldout gsm8k test
  done
  bash scripts/run.sh tsne gsm8k test
  bash scripts/run.sh steps gsm8k train
  bash scripts/run.sh distances gsm8k test
  echo "PILOT COMPLETE"
' > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG.pid"
echo "PID/PGID: $PID"
echo "LOG: $LOG"
tail -n 30 -f "$LOG"
```

GPU 6 替换为空闲 GPU。Ctrl+C 只退出日志；断开 SSH 后继续。不要删除或重写数据文件，也无需重新 prepare。保留同名实验与现有 598 条 train 激活。若某些分析目录此前已存在，从未完成阶段恢复，避免重跑拒绝覆盖的 heldout。
