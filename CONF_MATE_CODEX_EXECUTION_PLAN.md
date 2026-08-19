# ConfMate：面向部分观察的置信度感知 VLM 配合部件匹配

## 0. 给 Codex 的执行目标

本项目不是直接复现一个完整的机器人插入系统，而是先实现其视觉感知前端：

> 在仿真环境中，机器人观察一个 peg 和多个候选 hole，使用视觉语言模型判断哪个 hole 能与 peg 配合，并在遮挡、视角变化和错误候选存在时输出匹配置信度。

项目暂时不依赖真实相机、机械臂或触觉传感器。所有输入来自 PyBullet 仿真相机；重点是视觉语言匹配、部分观察、多视角一致性和不确定性估计。

项目名称：`ConfMate`

建议仓库名称：`confmate-vlm-pybullet`

## 0.1 当前仓库状态与执行解释

当前仓库是一个空的 Git 工作区：

- 没有源码、README、依赖文件或测试；
- 当前没有 commit、远程地址或可供修改的上游代码；
- 本机已检测到 Python 3.13.5 和 RTX 3060 Laptop GPU，显存约 6GB；
- PyBullet、PyTorch、Transformers、OpenCV 和 PyYAML 尚未确认安装。

因此，本文档是**项目计划和约束说明**，不是要求 Codex 在空仓库中直接执行 Phase 1 的指令。必须先完成下面的 Bootstrap 阶段，导入上游代码并建立第一个可回退 checkpoint。

Codex 在任何写操作前都必须先报告：当前目录、Git 状态、Python 版本、CUDA/GPU 状态、磁盘空间和网络可用性，并等待用户确认 Bootstrap 计划。不能因为本文档中出现了命令示例就自动执行后续阶段。

## 1. 研究动机

Yajima 等人的 zero-shot peg insertion 工作使用 VLM 识别正确 mating hole，并利用多个视角与 Yes/No 响应概率提升匹配成功率。COG 工作则强调通过点级置信度和对应关系抑制不可重叠区域与错误匹配。

ConfMate 将这两个思想缩小为一个可在个人 GPU 上完成的仿真项目：

1. 生成具有不同几何形状和功能关系的 peg-hole 场景；
2. 从仿真相机获得多个视角；
3. 对候选 hole 进行视觉语言匹配；
4. 用多视角一致性和分数间隔估计置信度；
5. 在置信度不足时拒绝给出确定匹配；
6. 分析遮挡和视角变化如何影响结果。

不能把结果描述为“完成了真实机器人插入”。准确表述应是：

> A simulation-based study of confidence-aware VLM perception for zero-shot mating-part matching under partial observation.

## 2. 代码起点

第一阶段从这个仓库开始：

- [Nabil-Miri/vlm-robot-color-sorting](https://github.com/Nabil-Miri/vlm-robot-color-sorting)

它已经包含 PyBullet Panda、仿真相机、CLIP 图文匹配和 pick-and-place 流程，适合先验证环境和控制闭环。

可选参考：

- [rantaluca/MiniVLA-Nav](https://github.com/rantaluca/MiniVLA-Nav)：更简单的 PyBullet 移动机器人 + CLIP 导航；
- [huangwl18/VoxPoser](https://github.com/huangwl18/VoxPoser)：后续参考 LLM/VLM 生成操作规划，但需要 RLBench、PyRep/CoppeliaSim 和 API key，不作为第一阶段依赖；
- [YC-Che/COG](https://github.com/YC-Che/COG)：后续参考 confidence-aware correspondence，不要求现在完整重训。

CLIP 是第一阶段的视觉语言 baseline，不等同于可对图像进行长文本推理的聊天式 VLM。第二阶段再增加 Qwen2.5-VL、LLaVA 或其他可用的生成式 VLM adapter。

## 3. 硬约束

- 设备：NVIDIA RTX 3060 6GB；
- 当前没有可用的 RGB-D 相机；
- 主要语言：Python；
- 主要框架：PyBullet、PyTorch、Transformers、Pillow/OpenCV；
- 第一阶段不能依赖机械臂、触觉、ROS2 或真实传感器；
- 不允许把 API key 写入代码、日志或 Git；
- 不进行大规模 VLM 训练；
- 每个阶段必须能单独运行并产生可检查的输出；
- 原始上游代码必须保留，优先通过新模块扩展，不要大面积删除；
- 所有实验必须保存配置、随机种子、模型名称和结果 JSON。

## 4. 目标目录结构

最终逐步整理成如下结构：

```text
confmate-vlm-pybullet/
├── confmate/
│   ├── sim/              # PyBullet 场景、peg/hole、相机、多视角采样
│   ├── perception/       # crop、mask、CLIP/VLM adapter
│   ├── matching/         # candidate matching、yaw estimation、confidence
│   ├── evaluation/       # metrics、corruption、risk-coverage
│   └── visualization/    # frame、轨迹、匹配结果可视化
├── configs/
│   ├── baseline.yaml
│   └── confmate_demo.yaml
├── scripts/
│   ├── smoke_test.py
│   ├── run_baseline.py
│   ├── run_confmate.py
│   └── evaluate.py
├── tests/
├── outputs/              # 不提交大模型权重和大数据
├── README.md
├── AGENTS.md
└── requirements.txt
```

## 5. 分阶段执行计划

### Bootstrap：导入上游并建立可复现环境

目标：将空仓库变成一个可运行、可回退、可继续扩展的基线仓库。

#### Bootstrap-A：审计和远程锁定

- 确认当前工作区确实没有用户源码；
- 不删除现有用户文件，不使用破坏性 Git 命令；
- 添加上游远程：`https://github.com/Nabil-Miri/vlm-robot-color-sorting.git`；
- 读取上游默认分支和最新 commit，不要假设分支名；
- 记录仓库 URL、分支、commit SHA、获取日期到 `upstream.lock`；
- 保留上游的 `panda_vision_simulation.py`、`color_sorting_vlm.py`、`simple_pick_place_demo.py` 和 `requirements.txt`。

如果当前仓库已有空 `.git`，可以在确认工作区为空后将上游 commit 检出到当前仓库；不要重新初始化覆盖 Git 元数据，也不要把上游代码直接覆盖用户已有文件。

#### Bootstrap-B：环境策略

Python 3.13.5 先只作为审计结果记录，不把它直接视为兼容环境。优先准备 Python 3.10 或 3.11 的独立虚拟环境；如果上游确认支持 3.13，再记录实际验证结果。

依赖分层：

- `requirements-base.txt`：PyBullet、NumPy、Pillow、Matplotlib 等基础依赖；
- `requirements-clip.txt`：PyTorch、TorchVision、Transformers 或上游实际使用的 CLIP 依赖；
- `requirements-lock.txt`：成功运行后由当前环境导出的锁定版本；
- 不在第一阶段安装生成式 VLM、Detectron2、PyTorch3D 或其他大型可选依赖。

建立 `scripts/env_report.py`，输出：

- Python、pip、PyTorch 版本；
- `torch.cuda.is_available()`；
- GPU 名称、总显存和当前显存；
- PyBullet、Pillow、NumPy 等 import 状态；
- 操作系统和运行模式（Windows 原生或 WSL）。

Windows 原生环境优先用于 PyBullet GUI 验证；WSL2 只在确认图形转发或 headless 模式可用后使用。不要为了启动 GUI 修改系统级 CUDA 或删除现有环境。

#### Bootstrap-C：仓库规则和首个 checkpoint

创建：

- `README.md`：上游来源、环境创建、运行命令和已知限制；
- `AGENTS.md`：目录规范、测试命令、禁止事项和完成定义；
- `.gitignore`：`.venv/`、缓存、模型权重、生成数据、`outputs/`、视频和日志；
- `upstream.lock`：上游 commit 锁定信息；
- `scripts/env_report.py`；
- `requirements-base.txt`、`requirements-clip.txt`。

完成环境记录后创建第一个 commit：

```text
bootstrap: import upstream demo and record reproducible environment
```

验收标准：

- 当前仓库有至少一个 commit；
- `upstream.lock` 中有明确 SHA；
- 有独立 Python 环境和环境报告；
- 原始上游脚本仍然可见；
- 没有下载大型生成式 VLM 权重；
- `git status`、README 和复现命令清晰可读。

### Phase 1：运行已导入的上游 demo

目标：确认 PyBullet、PyTorch、CLIP 和仿真相机均可工作。

要求：

- 先运行上游的 `simple_pick_place_demo.py`；
- 再运行 `color_sorting_vlm.py`；
- 保存一张仿真相机图像、一段运行日志和一张 pick-and-place 截图；
- 记录 Python、PyTorch、CUDA、GPU 显存和系统环境；
- 不修改上游逻辑，除非是为了解决明确的兼容性问题。

验收标准：

- 可以启动 PyBullet；
- 可以读取仿真相机 RGB 图像；
- CLIP 对至少三个候选物体返回相似度分数；
- Panda 可以执行一次抓取和放置；
- 有明确的复现命令。

### Phase 2：建立 ConfMate 基础架构

目标：把上游 demo 从单文件实验整理为可扩展代码，但保持行为不变。

任务：

- 将仿真、感知、匹配和控制拆成独立模块；
- 增加 YAML 配置；
- 增加 `--headless`、`--seed`、`--num-objects` 和 `--save-video` 参数；
- 添加 smoke test；
- 每次运行输出 `run_config.json` 和 `summary.json`；
- 生成一张包含原始 frame、候选框、选中目标和置信度的可视化图。

验收标准：

```bash
python scripts/smoke_test.py
python scripts/run_baseline.py --config configs/baseline.yaml --headless
```

两条命令都应在干净环境下完成，并返回非零数量的候选及可解析 JSON。

### Phase 3：生成 peg-hole 仿真场景

目标：将颜色分类改为功能配合部件匹配。

先实现简单而可控的形状，不要一开始追求工业连接器真实建模：

- 圆柱 peg / 圆孔；
- 矩形 peg / 矩形孔；
- 带缺口的 keyed peg / keyed hole；
- 十字形、L 形和非对称多边形；
- 尺寸相近但不能配合的 distractor holes。

每个场景保存 ground truth：

```json
{
  "peg_id": "peg_003",
  "target_hole_id": "hole_007",
  "target_yaw_deg": 90.0,
  "candidate_hole_ids": ["hole_004", "hole_007", "hole_011"],
  "occlusion_ratio": 0.25,
  "view_ids": ["top", "oblique"]
}
```

第一版只要求 matching，不要求真实插入动力学。确认感知正确后，再增加一个简单的 geometric alignment action 作为演示。

#### 数据划分与观察模式

必须在生成数据时固定随机种子，并按形状实例而不是按图片随机划分：

- `train`：用于开发和 prompt 调整；
- `val`：只用于选择置信度阈值和校准参数；
- `test`：只在最终报告时使用；
- 至少保留一个 shape-family 或 geometry-instance-disjoint test split。

必须区分以下观察模式：

1. `oracle_crop`：使用仿真 ground-truth mask/crop，仅用于验证匹配器上限；
2. `render_mask_assisted`：使用渲染 mask 产生候选 crop，必须在报告中标记为仿真辅助；
3. `rgb_only`：只给匹配器 RGB 图像或 RGB crop，不能提供 object id、ground-truth mask 或目标候选标签。

如果使用 PyBullet 的 segmentation buffer，输出中必须记录 `observation_mode`，不能把 oracle crop 的结果写成开放世界检测结果。

### Phase 4：CLIP matching baseline

目标：建立可量化的 baseline。

对每个 peg-hole 候选对，主 baseline 必须明确为 image-image matching：

1. 从仿真相机获得 peg crop 和 hole crop；
2. 用同一个 CLIP image encoder 得到 `e_peg` 和 `e_hole`；
3. 对两个 embedding 做 L2 normalization；
4. 定义 `s(peg, hole) = e_peg · e_hole`；
5. 对同一 peg 的候选 hole 按分数排序；
6. 输出每个候选的原始分数、Top-1、Top-3 和 Top-1/Top-2 margin。

多视角 baseline 先使用简单平均：

```text
s_multi(peg, hole) = mean_v cosine(e_peg_v, e_hole_v)
```

不要把 image-text score 与 image-image score 混合成一个没有定义的分数。image-text 只能作为独立 ablation，例如比较固定模板 `a compatible mating hole`、`an incompatible hole` 的分类效果，并记录完整 prompt、模型和归一化方式。CLIP 的相似度不是功能配合关系的证明，只能作为视觉相似度 baseline。

必须实现的 baseline：

- random；
- geometric contour/Chamfer；
- single-view CLIP；
- multi-view averaged CLIP。

### 评测定义（在进入 VLM 前固定）

#### Matching label

每个 episode 只有一个 ground-truth target hole；如果设计多个物理等价 hole，必须显式保存 `valid_target_hole_ids`，不能在评测时临时修改标签。

#### Yaw error

圆柱、圆孔、圆形和其他旋转对称部件的 yaw 记为 `undefined`，不参与普通 yaw MAE。对于具有 `n` 重旋转对称性的形状，使用等价类误差：

```text
d_yaw = min_k angular_distance(pred_yaw, gt_yaw + 360*k/n)
```

只有 keyed 或非对称部件才报告普通 yaw MAE。最终表格必须同时报告 matching accuracy 和“有定义 yaw 的样本数量”。

#### 结果文件

每个 episode 至少写入：

```json
{
  "split": "test",
  "seed": 17,
  "observation_mode": "rgb_only",
  "model": "clip-model-name",
  "candidate_scores": {},
  "predicted_hole_id": "hole_007",
  "ground_truth_hole_id": "hole_007",
  "matching_correct": true,
  "yaw_error_deg": null,
  "confidence": 0.12
}
```

### Phase 5：生成式 VLM adapter

目标：让系统支持 Yes/No matching，而不是只依赖 embedding 相似度。

定义统一接口：

```python
class VLMMatcher:
    def match(self, peg_images, hole_images, prompt) -> dict:
        """Return label, score, raw_text, and optional token probabilities."""
```

输出格式固定为：

```json
{
  "label": "YES",
  "score": 0.78,
  "raw_text": "YES",
  "model": "model-name",
  "view_scores": {
    "top": 0.81,
    "oblique": 0.75
  }
}
```

实现顺序：

1. 先用 fake/mock adapter 测试 pipeline；
2. 再接入一个实际可用的 VLM；
3. 强制模型输出 `YES` 或 `NO`；
4. 记录原始输出，但不能让自由文本破坏评测；
5. 不把模型权重提交到 Git。

RTX 3060 上优先尝试小型量化模型；如果本地模型安装或显存不稳定，先保留 CLIP baseline，并把 VLM adapter 做成可选依赖。

### Phase 6：置信度与部分观察

目标：形成 ConfMate 的主要研究内容。

至少实现三种 confidence：

- top-1 与 top-2 分数差；
- 多视角预测的一致性；
- 遮挡/视角扰动下的预测稳定性。

置信度阈值不能直接在 test split 上调。必须在 `val` split 上选择阈值，并在固定 test split 上只进行一次最终报告。每个阈值都要记录 coverage、selective accuracy 和拒答数量。

实现 abstention：

```text
if confidence < threshold:
    return "uncertain"
else:
    return best_candidate
```

对以下条件分别实验：

- 无遮挡、轻度遮挡、中度遮挡、重度遮挡；
- 单视角、双视角、三视角；
- 候选数量 3、5、8；
- 几何相似 distractor 和完全不同 distractor。

指标：

- Top-1 accuracy；
- Top-3 accuracy；
- yaw MAE；
- confidence calibration / ECE；
- accuracy-coverage curve；
- 在允许拒答时的 selective accuracy。

至少报告以下对照：

- 不拒答的 Top-1 accuracy；
- 固定 coverage 下的 selective accuracy；
- random rejection；
- margin rejection；
- multi-view consistency rejection。

如果使用 ECE，必须说明 bin 数量、confidence 定义和是否经过 temperature scaling。temperature scaling 只能在 validation split 上拟合。

### Phase 7：实验报告与展示

最终至少生成：

- 一张系统结构图；
- 一张仿真场景图；
- 一张成功匹配图；
- 一张失败和拒答案例图；
- 一张不同遮挡率下的准确率曲线；
- 一张 risk-coverage 曲线；
- 一个 30–60 秒 GIF 或 MP4；
- 一个可复现 README；
- 一个 `results/summary.csv`。

不要在没有实验数据时填写虚假的准确率。README 中应明确区分：

- reproduced baseline；
- proposed extension；
- preliminary result；
- known limitation。

## 6. Codex 使用方式

### 推荐：CLI 作为主工作环境

适合原因：

- Codex 能直接看到本地仓库、虚拟环境、GPU 和运行日志；
- 可以边改代码边运行 PyBullet；
- 方便查看 diff、执行测试、保存 Git checkpoint；
- 适合连续多轮调试，尤其是 CUDA、窗口、PyBullet 和显存问题。

推荐流程：

```bash
cd <当前空仓库>
codex
```

进入 Codex 后先发送：

```text
请先只做只读审计，不要修改代码，不要安装依赖。
确认当前目录、Git 状态、Python 版本、CUDA/GPU 状态、磁盘空间和网络状态。
当前仓库可能是空的 .git；不要假设上游代码已经存在。
读取 CONF_MATE_CODEX_EXECUTION_PLAN.md，并只评估 Bootstrap 阶段。
最后报告：
1. 当前工作区是否为空；
2. 上游仓库的默认分支和最新 commit；
3. Python 3.13 是否适合作为运行环境；
4. RTX 3060 6GB 的风险；
5. 导入上游和创建首个 checkpoint 的具体步骤。
等待我确认后再写入或安装任何内容。
```

确认后，每次只交给 Codex 一个小任务，例如：

```text
现在只完成 Bootstrap 阶段。
导入上游代码并锁定具体 commit，创建 Python 3.10/3.11 的独立环境，
添加 README、AGENTS.md、.gitignore、upstream.lock 和环境报告脚本。
保留原始上游脚本，不进行 ConfMate 重构，不安装生成式 VLM。
运行环境报告并尝试最小依赖检查。
完成后报告修改文件、上游 SHA、环境问题和测试结果，等待下一步确认。
```

Bootstrap 完成并确认后，再发送：

```text
现在只完成 Phase 1 的上游 demo 验证。
保持原始 demo 行为不变，不添加生成式 VLM。
完成后运行 smoke test 和原始 demo，报告修改文件、测试命令和结果。
不要删除原始代码。
```

每个阶段结束后让 Codex：

```text
请检查当前 diff，运行相关测试，补充缺失测试，并列出仍未验证的风险。
不要擅自进入下一阶段。
```

### 网页 Codex 适合做什么

网页版本适合：

- 先阅读项目和论文；
- 设计系统架构；
- 讨论实验假设；
- 审查 Codex CLI 产生的 diff；
- 分析错误日志和截图；
- 让 Codex 帮你写 README、实验报告和套磁信中的项目描述。

不建议把整个项目长期只放在网页里，因为网页环境不一定能稳定看到你本地的 GPU、Python 虚拟环境、PyBullet 窗口和运行状态。

## 8. Bootstrap 和第一阶段完成定义

Bootstrap 完成后，必须满足：

- 上游仓库已经导入当前工作区；
- `upstream.lock` 中有准确的 commit SHA；
- 当前仓库至少有一个 checkpoint commit；
- 有 Python 3.10/3.11 环境，或有明确的 3.13 兼容性验证记录；
- 有环境报告；
- 没有安装不必要的大型生成式 VLM；
- 原始上游脚本仍可定位。

在开始 peg-hole 改造之前，Phase 1 还必须满足：

- 原始仓库 demo 可以运行；
- 仿真相机图片可以保存；
- CLIP 可以对至少三个候选返回分数；
- Panda 可以完成一次 pick-and-place；
- 有 `smoke_test.py`；
- 有 `AGENTS.md`；
- Git 中有 Bootstrap 和上游 demo 两个可回退 checkpoint；
- README 写清楚运行命令和已知问题。

如果 Bootstrap 或 Phase 1 在半天内无法稳定运行，先修复环境和上游 baseline，不要进入 peg-hole 或生成式 VLM 阶段。

## 9. 项目最终定位

只有完成 Phase 7 的实验后，才可以在 CV 中写成：

> Developed a PyBullet simulation pipeline for confidence-aware multi-view VLM matching of unseen mating parts under partial observation, with CLIP and generative VLM baselines, selective prediction, and robustness evaluation under occlusion.

在只完成 Bootstrap 和 Phase 1 时，只能写成：

> Implemented a PyBullet vision-language pick-and-place baseline using CLIP and simulated camera observations.
