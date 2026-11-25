# StepCluster 服务器操作指引

## 1. 本地同步

本地完整项目位于 `E:\Research\LLMCoT\methods\StepCluster`。作者源码已随工程保存，无需服务器再克隆作者仓库。

如果尚未创建 GitHub 仓库，最直接的是在本地 PowerShell 同步目录。

```powershell
scp -r 'E:\Research\LLMCoT\methods\StepCluster' wxh@rtx6000.yangtan.top:~/projects/
```

如果使用 GitHub，在本地项目创建自己的仓库并 push，再在服务器克隆自己的实际地址。

```powershell
cd E:\Research\LLMCoT\methods\StepCluster
git init -b codex/stepcluster-reproduction
git add .
git commit -m "Add official trajectory reproduction and server adapters"
# 将下面地址换为你已创建的仓库
git remote add origin git@github.com:ZERMORE/StepCluster.git
git push -u origin codex/stepcluster-reproduction
```

```bash
mkdir -p ~/projects
cd ~/projects
git clone --branch codex/stepcluster-reproduction git@github.com:ZERMORE/StepCluster.git StepCluster
# 后续更新
cd ~/projects/StepCluster
git pull --ff-only origin codex/stepcluster-reproduction
```

上述仓库地址是操作示例。本地没有替你创建远程仓库或 push。已有 `~/wdl/data -> /mnt/data/wxh/wdl` 链接可以继续用，所有命令直接指向真实数据盘路径。

## 2. 环境

推荐复制现有 ood-repro 环境，保留原实验依赖。

```bash
conda create -n stepcluster --clone ood-repro
conda activate stepcluster
cd ~/projects/StepCluster
python -m pip install -r requirements.txt
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__); print(torch.cuda.is_available())'
```

依赖沿用 torch 2.5.1 / Transformers 4.46.3 / datasets 3.6.0，requirements 不重装 CUDA torch。服务器已有这些包可直接使用。临时复用原环境也可 `conda activate ood-repro`，但先确认安装不会改变其他实验。

## 3. 通用路径，每次重新 SSH 后设置

```bash
cd ~/projects/StepCluster
conda activate stepcluster
export DATA_ROOT=/mnt/data/wxh/wdl
export HF_HOME="$DATA_ROOT/cache/huggingface"
export STEP_ROOT="$DATA_ROOT/results/StepCluster"
export MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
export MODEL_PATH="$DATA_ROOT/models/Qwen2.5-7B-Instruct/$MODEL_REVISION"
export RUN_NAME=qwen25-pilot-v1
export CUDA_VISIBLE_DEVICES=6
export LAYER=27
export MPLBACKEND=Agg
mkdir -p "$DATA_ROOT/logs/StepCluster"
nvidia-smi
test -f "$MODEL_PATH/config.json"
python -B scripts/check.py
```

GPU 6 仅为示例，按 nvidia-smi 的空闲情况选择。指定一个物理 GPU 后，进程内 cuda:0 对应这张卡。Qwen2.5-7B 通常有 28 个 block，0-based 最后层为 27；服务器可 `python -c 'from transformers import AutoConfig; import os; print(AutoConfig.from_pretrained(os.environ["MODEL_PATH"], local_files_only=True).num_hidden_layers)'` 核实。

## 4. 准备 GSM8K，复用下载缓存

```bash
bash scripts/run.sh prepare gsm8k
```

默认离线读取之前下载的 GSM8K revision `740312add88f781978c0658806c59bc2815b9866`，不重新下载模型。训练题固定种子划出 15% validation，官方 test 保留。数据盘保存准备好的 JSONL 与哈希。

若缺少该 revision 的 HF 缓存，先检查自己之前使用的 HF_HOME。网络可用时只对这一步允许联网。

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 bash scripts/run.sh prepare gsm8k
```

也可以指定现有原始 JSONL 路径模式 `DATA_JSONL='/实际路径/{split}.jsonl' bash scripts/run.sh prepare gsm8k`。支持原始 question/answer，也支持已有 OOD 预测记录的 question/gold_answer，但后者仅含之前跑过的子集，准备后检查 manifest 的题数。改变数据来源请另设 STEP_ROOT。旧 OOD 回答不带本实验 Step 协议，因此只复用题目与真值，重新生成回答。

## 5. 先 smoke

```bash
bash scripts/background.sh smoke gsm8k
```

8 题生成、边界激活提取、作者 t-SNE，默认独立 `qwen25-smoke` 目录。日志文件和 PID/PGID 会打印。8 题只检查路径、token 边界、GPU、输出格式，不能评价正确性检测。

```bash
tail -n 30 -f /mnt/data/wxh/wdl/logs/StepCluster/实际日志文件.log
```

按 Ctrl+C 退出日志查看。nohup + 独立进程组保证断开 SSH 后继续运行。

## 6. Pilot，再扩大规模

先用训练 600、验证 200、测试 200 题检查步骤分布与错误数量。以下整条任务是一个后台进程组，生成完才提取。默认 max_new_tokens=2048，不按输入长度截断题目。

```bash
LOG="$DATA_ROOT/logs/StepCluster/$RUN_NAME-pilot-$(date +%Y%m%d-%H%M%S).log"
nohup setsid bash -c '
  set -euo pipefail
  for spec in train:600 validation:200 test:200; do
    split=${spec%:*}; count=${spec#*:}
    LIMIT=$count bash scripts/run.sh generate gsm8k "$split"
    bash scripts/run.sh extract gsm8k "$split"
  done
  bash scripts/run.sh tsne gsm8k test
  bash scripts/run.sh steps gsm8k train
  bash scripts/run.sh distances gsm8k test
  bash scripts/run.sh heldout gsm8k
' > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG.pid"
echo "PID/PGID $PID"
echo "$LOG"
```

若任意划分仅有一类正确性标签，heldout 会报错。扩大 pilot，而不应尝试多个随机种子后挑有利结果。提取不符合 Step/#### 格式及截断样本时会在 `extraction.json` 记录排除，检查保留率和不同对错组保留率。

正式全量使用另一个名字，防止 pilot 的配置签名发生变化。

```bash
export RUN_NAME=qwen25-full-v1
LOG="$DATA_ROOT/logs/StepCluster/$RUN_NAME-full-$(date +%Y%m%d-%H%M%S).log"
nohup setsid bash -c '
  set -euo pipefail
  for split in train validation test; do
    LIMIT=0 bash scripts/run.sh generate gsm8k "$split"
    bash scripts/run.sh extract gsm8k "$split"
  done
  bash scripts/run.sh tsne gsm8k test
  bash scripts/run.sh steps gsm8k train
  bash scripts/run.sh distances gsm8k test
  bash scripts/run.sh heldout gsm8k
' > "$LOG" 2>&1 < /dev/null &
echo "PID/PGID $!"
echo "$LOG"
```

## 7. 作者正确性预测与轨迹 steering

作者正确性 predictor 在训练题语料内部再划分。以下复用其实现，heldout 是本项目额外的外部测试评估，两种结果分别报告。

```bash
bash scripts/background.sh predict gsm8k
# 多种特征或种子分别指定
FEATURE=hash_pca SEED=123 bash scripts/background.sh predict gsm8k
```

轨迹纠偏按固定基线步骤数 K 分组。先拟合、选择阈值，再在测试题同 K 子集干预。

```bash
export K=4
export PCA_DIM=128
bash scripts/run.sh ideal gsm8k
ALPHA=0.05 STEER_LIMIT=20 bash scripts/background.sh steer gsm8k
```

PCA 维数需要小于该 K 组正确训练边界的有效样本数。Pilot 数据不够时显式改 `PCA_DIM=16 RANK=8`，单独标记小规模验证。alpha 是试跑值，正式结果必须在 validation 上选取 alpha 后冻结。`ideal` 将 validation 传给作者 `--test-npz`，适配层修改阈值搜索以避免在测试题上选参数。最后测试的 `STEER_LIMIT` 可增加。

作者其他阶段保留通用入口。例如帮助查看 Prolong 和门控干预所需参数。

```bash
python scripts/pipeline.py author --script scripts/steering/intervene_all_steering.py -- --help
python scripts/pipeline.py author --script scripts/steering/error_aware/intervene_error_aware.py -- --help
python scripts/pipeline.py author --script scripts/steering/error_aware/intervene_text_injection.py -- --help
```

原生 Prolong 小规模测试。

```bash
python -u scripts/pipeline.py author --script scripts/steering/intervene_all_steering.py -- \
  --steering "$STEP_ROOT/$RUN_NAME/gsm8k/train/activations.npz" \
  --merged-dir "$STEP_ROOT/$RUN_NAME/gsm8k/test/merged" \
  --model "$MODEL_PATH" --mode PROLONG_LAST_N --n-layers 5 \
  --alpha -0.05 --num-questions 20 --max-new-tokens 2048 \
  --output-dir "$STEP_ROOT/$RUN_NAME/gsm8k/prolong-pilot"
```

该直接命令前台运行，如需断开 SSH，请用前文 `nohup setsid ... > 独立日志 2>&1 < /dev/null &` 包装。作者门控脚本固定构造 `hash_minus_last`，需要先训练匹配特征的 checkpoint。

```bash
FEATURE=hash_minus_last bash scripts/run.sh predict gsm8k
bash scripts/background.sh gated gsm8k
bash scripts/background.sh inject gsm8k
```

`inject` 默认 STEP，`INJECT_MODE=WAIT`、`HMM`、`DOUBLE_CHECK` 切换注入，`ALWAYS_` 前缀模式对应无门控对照。固定相同题目与预算，阈值在 validation 上确定，并报告纠正与破坏比例。该阶段保留作者逐 token 实现，先测 20 题的显存与耗时。

## 8. MATH-500 和已有完整 MATH

论文用 `HuggingFaceH4/MATH-500`，若之前没下载，只有它需要首次联网准备。

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 bash scripts/run.sh prepare math500
bash scripts/background.sh generate math500 test
# 确认生成完成后再执行
bash scripts/background.sh extract math500 test
bash scripts/background.sh tsne math500 test
bash scripts/background.sh steps math500 test
```

最后的 steps 是 MATH-500 内部步骤身份可分性实验，跨数据集迁移应使用作者 `apply_linear_probes.py`，见 `python scripts/pipeline.py author --script scripts/predictors/apply_linear_probes.py -- --help`。原作者 t-SNE 泛型入口可以直接画 MATH-500 的各层结构，属于原论文可视化的补充。

完整 MATH 使用你已导出的文件，不再下载。单独放 `math-full`，避免混淆 benchmark。

```bash
DATA_JSONL="$DATA_ROOT/datasets/hendrycks-math-v1/test.jsonl" bash scripts/run.sh prepare math-full
LIMIT=500 bash scripts/background.sh generate math-full test
# 完成后
bash scripts/background.sh extract math-full test
bash scripts/background.sh tsne math-full test
```

这 500 条是完整 MATH 的子集，并非官方 MATH-500。默认 math-full 只有 test，训练仍来自 GSM8K。直接运行 heldout math-full 会缺少 train/validation，这是预期的数据协议，需要另建跨域评估入口或显式使用 GSM8K 训练好的检测器。

## 9. 原论文三个模型

Qwen 完成链路后，下载或定位原模型到 `$DATA_ROOT/models/`，export MODEL_PATH 指向实际快照目录，并设置独立 RUN_NAME。Llama-3.1 的权限与 Llama-3.2 审核分别检查。R1-Distill-Llama-8B 也需要单独下载。

```bash
export MODEL_PATH=/mnt/data/wxh/wdl/models/你的原论文模型/实际快照目录
export RUN_NAME=llama31-instruct-v1
export LAYER=31
SMOKE_NAME=llama31-instruct-smoke bash scripts/background.sh smoke gsm8k
```

不同模型 smoke 需 `SMOKE_NAME=llama31-instruct-smoke`，否则默认的 qwen25-smoke 会拒绝混用配置。原论文 peak layer、训练 regime 和数据选择需按 README 逐项核对；Qwen 的数字应作为扩展结果报告。

## 10. 停止、恢复与取回结果

```bash
ps -u "$USER" -o pid,ppid,pgid,stat,etime,args --forest | grep -E 'StepCluster|scripts/pipeline.py|scripts/run.sh'
# 确认 PGID 属于本任务后，将数字替换为实际 PGID
kill -TERM -- -实际PGID
nvidia-smi
```

停止整个任务进程组，防止父 shell 继续进入下一阶段。不要使用 `killall python`。恢复 generation 使用相同名字、limit、prompt、模型和代码，已完整写入的题目自动跳过。extract 阶段中断后重新运行。分析/干预使用新输出目录，作者某些入口不提供完整断点续跑。

结果默认 `$DATA_ROOT/results/StepCluster/$RUN_NAME`。在本地 PowerShell 取回，网络较慢时优先图、JSON 表和必要的边界 NPZ。

```powershell
scp -r wxh@rtx6000.yangtan.top:/mnt/data/wxh/wdl/results/StepCluster/qwen25-pilot-v1 E:\Research\LLMCoT\methods\StepClusterResults
```
