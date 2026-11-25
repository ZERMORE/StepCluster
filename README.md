# StepCluster

复现 Sun 等《LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals》。作者公开实现保存在 `author/`。本工程适配服务器数据路径、离线模型、断点生成和边界激活提取。原始作者文件保持下载时内容，运行时补丁写入数据盘的 `runtime/`。

详细命令见 [服务器操作指引](SERVER_GUIDE.md)。

## 实验与对应入口

| 目标 | 入口 | 实现来源 |
|---|---|---|
| 步骤格式生成 | `generate` | 复用作者 `format_prompt`、答案提取和评分函数，独立的流式适配器 |
| 边界状态与 steering vectors | `extract` | 独立边界映射、单次 teacher-forcing + hooks；复用作者 trimmed mean；导出作者 NPZ schema |
| t-SNE，按步骤或对错显示 | `tsne`，或直接 author 入口 | 作者 `visualize_act_tsne.py` |
| 各层步骤 probe | `steps` | 作者 `train_stepwise_probes.py` |
| 早期、后期几何距离 | `distances` | 作者 `analyze_step_distances.py` |
| 原论文正确性预测 | `predict` | 作者 `train_predictors.py`，内部从训练语料再划分 |
| 外部保留测试集预测 | `heldout` | 本工程补充，PCA + logistic regression，独立 validation 选 C |
| 分步骤理想路径与阈值 | `ideal` | 作者实现，运行时将阈值选择改为独立 validation |
| 轨迹纠偏 | `steer` | 作者 `apply_trajectory_steering.py` |
| Prolong/Shorten、门控干预 | 通用 `author --script` | 作者 `scripts/steering/`，具体参数见指引 |

## 配置与复现边界

- 立即可运行的默认实验使用服务器已有 Qwen2.5-7B-Instruct，属于跨模型方法复现。原论文三个变体为 Llama-3.1-8B、Llama-3.1-8B-Instruct、DeepSeek-R1-Distill-Llama-8B。Llama-3.2 或 Qwen 都不能作为原论文同模型复现。
- 默认纯文本 `cot` 提示和 `####` 标记直接调用作者函数，greedy decoding。`PROMPT=chat` 是明确的额外变体，应另设 `RUN_NAME`。原 OOD 的 few-shot/chat 回答不能当作新固定 Step 协议下的生成结果。
- GSM8K 优先离线复用 HF 缓存。原始 JSONL 可手工指定。已导出的完整 MATH 可复用为 `math-full`，但 MATH-500 是指定的 500 题 benchmark，不能用完整 MATH 的前 500 条代替。
- `extract` 默认 `author-hidden` 兼容作者 `hidden_states[layer+1]`：最后一层使用 final norm 后的状态，其他层使用 block 输出。HF Llama/Qwen 的最后一个 hidden_states 实际经过 final norm，作者将其通称 resid_post。`STATE=resid-post` 则在所有层取真正 pre-final-norm block 输出。两者应放不同实验目录。
- 本工程通过 token offsets 严格对齐 Step/#### 起点，存其前一 token 激活。与作者部分基于字符串前缀长度的定位实现有细节差异。tokenizer decode/encode 不一致会报错，避免悄悄错位。
- t-SNE 使用作者原始入口及其随机种子，Qwen 用实际层数的通用绘图入口，避免原论文四幅图入口硬编码 31 层。不按 t-SNE 二维聚类拟合检测器。
- 每题生成 JSON 原子落盘，重启保持全部配置及代码相同即可跳过完整题目。提取阶段聚合 NPZ 只有全部完成后才保存，中断提取会重新运行提取阶段。
- 原生步骤 probe 按状态样本随机分割，同题不同步骤可能跨划分。原生正确性预测器主要使用训练语料内部划分，best layer 有探索性。请同时运行 heldout 做独立题目评估，禁止在 test 上选层、阈值、alpha。
- `ideal` 使用按原始轨迹总步骤 K 分组，这是离线分层协议，知道基线 K 不等于在线提前知道未来 K。作者 steering 用同 K 的基线子集，结果不应宣称覆盖所有题目。
- `ideal` 原脚本在训练集搜索阈值。适配层只修改运行时副本，用传入 `--test-npz` 的 validation 轨迹选择阈值。原脚本输出中部分字段仍写 TRAIN/TEST，此处 TEST 指 validation。数值口径因此与未改动作者脚本存在区别。
- 正确性、步骤身份、首次错误定位是不同任务。已有最终答案标签不提供首次错误位置。此工程保留作者 final-answer 标注协议，没有新增错误类型标注。

## 输出

默认 `$DATA_ROOT/results/StepCluster/$RUN_NAME/$TASK/`。

`train|validation|test/merged/` 保存输入 token、完整生成、答案和正确性；`activations.npz` 保存全部层的步骤边界及答案边界向量；`extraction.json` 保存剔除记录及来源。`tsne-*`、`step-probes-*`、`distances-*`、`predictors-*`、`heldout-*`、`ideal-*`、`steer-*` 各自保存图表或数值结果。HF 缓存仍在既有数据盘位置。常见 28 层、3584 维、每题约 5 个边界，float32 边界数据约 2 MB/题，具体随步数变化。不会保存全部 token 的全词表 logits。

## 来源

作者仓库 https://github.com/slhleosun/reasoning-trajectory 。下载记录见 `UPSTREAM.json`，逐文件 SHA256 见 `AUTHOR_FILES.json`。首次克隆失败后使用 GitHub codeload main 存档，保存 archive SHA256；没有把未知提交号写成固定 commit。

## 本地验证

运行 `python -B scripts/check.py`，或 `python -B scripts/check.py --full` 实际运行作者绘图及步骤 probe。本地验证采用合成小型轨迹，覆盖 marker 边界、模拟小模型生成/断点恢复/真实 hook 提取链路、作者 NPZ 兼容性、独立划分线性预测、官方 CLI 帮助。本机没有 Transformers 和模型权重，真实 Qwen/Llama 的生成、激活提取及干预需要在服务器执行 smoke 后验证。
