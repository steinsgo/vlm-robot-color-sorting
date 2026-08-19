# ConfMate Phase 1–7 阶段总结

更新时间：2026-08-19  
当前状态：Phase 1、Phase 2、Phase 3、Phase 4、Phase 5、Phase 6、Phase 7 已完成。

## 1. 总体结论

本仓库已经从上游颜色分类 demo 发展为一个可复现实验起点：

1. Phase 1 验证了 PyBullet、Panda、仿真相机和 CLIP 环境可以稳定运行。
2. Phase 2 将 baseline 运行流程拆成配置、仿真、感知、匹配、控制和产物模块，并增加 headless/seed/对象数量/视频参数。
3. Phase 3 生成了带 ground truth 的 peg-hole 仿真数据集，并明确区分三种观察模式和 shape-family-disjoint 数据划分。
4. Phase 4 在固定 test split 上运行了 random、Chamfer、single-view CLIP 和 multi-view CLIP matching baseline。
5. Phase 5 建立了统一 VLMMatcher 接口、离线 mock adapter 和可选的 BLIP VQA adapter，并强制输出 YES/NO。
6. Phase 6 增加了 margin、多视角一致性、扰动稳定性、val 阈值选择、abstention、ECE 和 risk-coverage 评测。
7. Phase 7 生成了实验结构图、场景/匹配样例、遮挡曲线、risk-coverage 曲线、36 秒 GIF 和 `summary.csv`。

Phase 7 的图表和展示已生成；结果仍应按 preliminary baseline 解释，不能替代更大规模真实遮挡实验。

## 2. 运行环境与网络处理

运行环境：

| 项目 | 状态 |
| --- | --- |
| 操作系统 | Windows 11 |
| Python | `.venv` 中 Python 3.9.21 |
| PyTorch | 1.12.1+cu116，CUDA 可用 |
| GPU | NVIDIA RTX 3060 Laptop，6 GB |
| PyBullet | 3.25 |
| Transformers | 4.25.1 |
| Matplotlib | 3.9.4 |
| CLIP | `openai/clip-vit-base-patch32` |

代理 `127.0.0.1:10808` 可用。Git 默认 Schannel TLS 后端曾返回 `SEC_E_NO_CREDENTIALS`，使用 OpenSSL 后端访问 GitHub 正常；没有修改 Git 配置。Hugging Face CDN 下载过程中出现过不完整响应，模型已通过代理断点续传并缓存到 `.hf-cache-temp`。

为避免中文用户目录触发 Hugging Face 路径问题，模型缓存使用仓库内 ASCII 路径。临时 Conda 配置和包缓存已经清理；`.venv`、`.hf-cache-temp`、`runs/` 和 `datasets/` 已加入 `.gitignore`。

headless 运行使用 `MPLBACKEND=Agg`，因此不会弹出 Matplotlib 窗口；PyBullet 的 DIRECT 渲染仍然可用。

## 3. Phase 1：上游 demo 验证

Phase 1 没有改动原始 demo 的逻辑，验证了：

- PyBullet 可以启动并加载 Panda；
- 仿真相机可以捕获 RGB 图像；
- CLIP 可以对多个候选物体计算分数；
- Panda 可以完成 pick-and-place；
- 原始颜色分类流程可以完成自动排序。

验证命令：

```text
python simple_pick_place_demo.py
python color_sorting_vlm.py
```

结果：

| Demo | 结果 |
| --- | --- |
| `simple_pick_place_demo.py` | 退出码 0，完整完成抓取和放置 |
| `color_sorting_vlm.py` | 退出码 0，CLIP 加载成功，5/5 个物体排序成功 |

## 4. Phase 2：ConfMate 基础架构

### 4.1 模块结构

Phase 2 新增 `confmate/` 模块：

- `config.py`：YAML 配置和 CLI 覆盖；
- `simulation.py`：PyBullet 生命周期和场景边界；
- `perception.py`：相机观察、分割框和候选 crop；
- `matching.py`：CLIP 候选匹配接口；
- `control.py`：Panda pick-and-place 控制接口；
- `artifacts.py`：JSON、可视化图和 GIF 产物；
- `runner.py`：单次 baseline episode 编排；
- `cli.py`：统一命令行入口。

底层仍复用原始 `panda_vision_simulation.py` 的仿真和 CLIP 实现，Phase 2 主要建立了可替换的模块边界，保持原始行为兼容。

### 4.2 运行方式

```text
python scripts/smoke_test.py
python scripts/run_baseline.py --config configs/baseline.yaml --headless
```

支持的主要参数：

```text
--headless
--seed 17
--num-objects 5
--save-video
--output-dir runs
```

每次 baseline 运行生成：

- `run_config.json`：实际配置和 CLI 覆盖后的参数；
- `summary.json`：候选分数、选中目标、置信度和动作结果；
- `visualization.png`：相机帧、候选框、选中目标和 CLIP 分数；
- `episode.gif`：使用 `--save-video` 时生成。

默认 headless baseline 的验证结果：5 个候选中选中 `red_sphere`，CLIP 置信度约 `0.8946`，pick-and-place 成功。

## 5. Phase 3：peg-hole 仿真数据集

### 5.1 几何和场景

生成器位于 [confmate/peg_hole.py](confmate/peg_hole.py)，支持以下 shape family：

- cylinder；
- rectangle；
- keyed；
- cross；
- L；
- asymmetric。

每个场景包含一个 peg、一个目标 hole 和两个 distractor holes。第一个 distractor 是尺寸相近但不能配合的同族形状，另一个是异形 distractor。Phase 3 只生成 matching 数据，不模拟真实插入动力学。

### 5.2 Ground truth

每个 episode 保存 `ground_truth.json`，包含：

```json
{
  "peg_id": "peg_000",
  "target_hole_id": "hole_000",
  "target_yaw_deg": 30.0,
  "candidate_hole_ids": ["hole_000", "hole_001", "hole_002"],
  "occlusion_ratio": 0.0,
  "view_ids": ["top", "oblique"]
}
```

此外保存 `shape_family`、`geometry_instance_id`、`yaw_symmetry_order`、`valid_target_hole_ids`、`split` 和 `observation_modes` 等实验字段。圆柱的 yaw 记录为 `null`；当前 Phase 3 v1 没有加入实际遮挡，因此 `occlusion_ratio` 为 0.0。

### 5.3 数据划分

配置文件：[configs/peg_hole.yaml](configs/peg_hole.yaml)

固定 seed 为 17，默认生成 12 个 episode：

| Split | Shape family | Episode 数量 |
| --- | --- | ---: |
| train | cylinder、rectangle、keyed、cross | 8 |
| val | L | 2 |
| test | asymmetric | 2 |

test 的 shape family 与 train/val 不重叠，属于 shape-family-disjoint split。划分按照几何实例生成，不按照图片随机切分。

### 5.4 观察模式

每个 episode、每个视角都分别生成：

| 模式 | 输入 | 约束 |
| --- | --- | --- |
| `oracle_crop` | 已知仿真对象 mask 的 peg/hole crop | 只用于验证匹配器上限 |
| `render_mask_assisted` | PyBullet segmentation buffer 生成的候选 crop | 必须标记为仿真辅助 |
| `rgb_only` | 原始 RGB 图像 | 不提供 object ID、ground-truth mask 或目标标签 |

每个模式的 `metadata.json` 都记录了 `observation_mode` 和数据来源，避免把仿真辅助结果误写成开放世界 RGB 检测结果。

### 5.5 运行和产物

```text
python scripts/generate_peg_hole_dataset.py --config configs/peg_hole.yaml
```

完整数据集由 [configs/peg_hole.yaml](configs/peg_hole.yaml) 生成，生成后的 `datasets/peg_hole_v1/manifest.json` 被 `.gitignore` 排除，不提交大体积数据文件。

当前验证结果：12 个 episode、361 个文件、约 16.8 MB；每个 episode 都有 top/oblique 视角、ground truth、segmentation 和三种观察模式输入。

## 6. Phase 4：peg-hole image-image matching

Phase 4 新增：

- [confmate/phase4.py](confmate/phase4.py)：matching、Chamfer、CLIP image encoder 和聚合指标；
- [scripts/evaluate_matching.py](scripts/evaluate_matching.py)：评测入口；
- [configs/phase4.yaml](configs/phase4.yaml)：固定 test split 的评测配置。

CLIP baseline 使用同一个 image encoder，对 peg 和 hole image 分别提取 embedding，进行 L2 normalization，再计算点积 cosine similarity。没有把 image-text score 混入 image-image score。

固定 test split 的结果：

| Observation mode | Method | Matching accuracy | Top-3 | Mean margin |
| --- | --- | ---: | ---: | ---: |
| `oracle_crop` | random | 50.0% | 100.0% | 0.128019 |
| `oracle_crop` | Chamfer | 100.0% | 100.0% | 2.332893 |
| `oracle_crop` | single-view CLIP | 100.0% | 100.0% | 0.030059 |
| `oracle_crop` | multi-view CLIP | 100.0% | 100.0% | 0.023508 |
| `render_mask_assisted` | random | 50.0% | 100.0% | 0.128019 |
| `render_mask_assisted` | Chamfer | 100.0% | 100.0% | 3.285298 |
| `render_mask_assisted` | single-view CLIP | 100.0% | 100.0% | 0.035038 |
| `render_mask_assisted` | multi-view CLIP | 100.0% | 100.0% | 0.024479 |

当前 test split 每个 episode 只有 3 个候选，因此 Top-3 在 scored 结果中必然为 100%，不能解读为强指标。`rgb_only` 的 8 个组合被明确标记为 skipped，因为 Phase 3 的 rgb_only 输入只有整帧 RGB，没有候选 crop；当前系统尚未提供 RGB-only candidate detector。

每个 episode 的 JSON 结果、`results.jsonl` 和聚合指标位于本地 `evaluations/phase4/`；该目录被 `.gitignore` 排除，可用上面的命令重建。

运行命令：

```text
python scripts/evaluate_matching.py --config configs/phase4.yaml
```

Phase 4 不估计 yaw，因此所有结果的 `yaw_error_deg` 为 `null`，`yaw_defined_count` 为 0；这部分不能当作 yaw 性能报告。

## 7. Phase 5：生成式 VLM adapter

Phase 5 新增：

- [confmate/vlm.py](confmate/vlm.py)：统一 `VLMMatcher` 接口、`MockVLMMatcher` 和 `BlipVQAMatcher`；
- [scripts/evaluate_vlm.py](scripts/evaluate_vlm.py)：逐候选 YES/NO 评测入口；
- [configs/phase5.yaml](configs/phase5.yaml)：mock pipeline 配置。

统一接口为：

```python
matcher.match(peg_images, hole_images, prompt) -> {
    "label": "YES" or "NO",
    "score": float,
    "raw_text": str,
    "model": str,
    "view_scores": dict,
}
```

mock pipeline 已在 test split 的 `oracle_crop` 和 `render_mask_assisted` 上运行；实际 adapter 使用 `Salesforce/blip-vqa-base`，Transformers 升级到 4.40.2 后从本地缓存成功加载，并完成双视角 test 验证。

实际 BLIP 结果：

| Observation mode | Scored episodes | Matching accuracy | 观察 |
| --- | ---: | ---: | --- |
| `oracle_crop` | 2 | 100% | 所有候选多回答 NO，分数平局，不能视为有效性能 |
| `render_mask_assisted` | 2 | 100% | 部分视角回答 YES，但候选仍有大量平局 |

BLIP 的 YES/NO 分数目前是强制标签 proxy：YES=0.75、NO=0.25，不是 token probability，也没有校准。原始生成文本保存在每个候选结果中，模型自由文本不会直接破坏评测标签。`rgb_only` 因没有候选 crop 而被明确 skipped。

实际模型权重保存在 `.hf-cache-temp`，没有写入 Git。通过代理下载时出现 CDN `IncompleteRead`，已使用 Range 分块续传完成本地模型缓存；这属于环境/网络处理，不是仓库权重。

结果目录：本地 `evaluations/phase5/`；该目录被 `.gitignore` 排除，可用上面的命令重建。公开仓库中的最终展示结果见 [results/README.md](results/README.md)。

运行 mock：

```text
python scripts/evaluate_vlm.py --adapter mock --split test
```

运行实际 adapter（模型已缓存时）：

```text
python scripts/evaluate_vlm.py --adapter blip --model-name Salesforce/blip-vqa-base --split test
```

## 8. Phase 6：置信度与部分观察

Phase 6 新增：

- [confmate/phase6.py](confmate/phase6.py)：三种 confidence、扰动、候选池扩展、ECE、abstention 和 risk-coverage 计算；
- [scripts/evaluate_confidence.py](scripts/evaluate_confidence.py)：严格按照 val 选阈值、test 一次性报告的评测入口；
- [configs/phase6.yaml](configs/phase6.yaml)：遮挡、视角和候选数量条件网格。

运行命令：

```text
python scripts/evaluate_confidence.py --config configs/phase6.yaml
```

评测条件包括：

- `none/light/moderate/heavy` 四档确定性矩形遮挡；
- 单视角、双视角和 `synthetic_flip` 合成三视角；
- 3、5、8 个候选；5/8 候选从其他 episode 借用候选 crop，并记录 `similar` / `different` distractor 类型；
- `margin`、`multi_view_consistency` 和 `stability` 三种 confidence。

完整运行得到 val/test 各 240 个样本。固定 test coverage 为 0.8 时：

| Rejection policy | Selective top-1 accuracy | Coverage |
| --- | ---: | ---: |
| random rejection | 67.17% ± 1.52% | 80% |
| margin rejection | 73.44% | 80% |
| multi-view consistency rejection | 68.23% | 80% |
| stability rejection | 72.92% | 80% |

这些数字是 Chamfer baseline 在当前小型合成数据集上的结果，不是 BLIP 的 generative VLM 性能。ECE 使用 10 个 bin，没有做 temperature scaling；yaw 仍未估计，因此 `yaw_mae_deg` 为 `null`。遮挡和三视角条件是评测增强，其中 `synthetic_flip` 不是新增的 PyBullet 相机视角。

详细 `aggregate.json` 和 `results.jsonl` 位于本地 `evaluations/phase6/`；该目录被 `.gitignore` 排除。公开仓库中提交的是 Phase 7 生成的 [results/summary.csv](results/summary.csv) 和图表。

## 9. Phase 7：实验报告与展示

Phase 7 使用固定的 Phase 6 test 结果生成以下交付物：

- [results/system_architecture.png](results/system_architecture.png)：系统结构图；
- [results/simulation_scene.png](results/simulation_scene.png)：真实 Phase 3 PyBullet 场景与 crop；
- [results/success_match.png](results/success_match.png)：Chamfer baseline 成功匹配样例；
- [results/failure_rejection.png](results/failure_rejection.png)：轻度遮挡下的错误预测与 abstention 样例；
- [results/accuracy_vs_occlusion.png](results/accuracy_vs_occlusion.png)：不同遮挡条件下的 Top-1/Top-3 曲线；
- [results/risk_coverage.png](results/risk_coverage.png)：三种 confidence policy 的 risk-coverage 曲线；
- [results/phase7_demo.gif](results/phase7_demo.gif)：36 秒可复现展示 GIF；
- [results/summary.csv](results/summary.csv)：364 行条件与 fixed-coverage 汇总；
- [results/README.md](results/README.md)：重现实验和结果范围说明。

生成命令：

```text
python scripts/generate_report.py --phase6-json evaluations/phase6/aggregate.json --dataset datasets/peg_hole_v1 --phase4-dir evaluations/phase4 --output-dir results --duration-seconds 36 --fps 5
```

报告明确区分：Phase 4/6 的 Chamfer 是 reproduced/preliminary baseline；Phase 6 的 confidence-aware rejection 是 proposed extension；三视角、5/8 候选和遮挡条件是可复现增强；BLIP forced-label 分数没有被用于校准结论。

## 10. 验证记录

已通过：

```text
python -m compileall -q confmate scripts
python scripts/smoke_test.py
python scripts/run_baseline.py --config configs/baseline.yaml --headless
python scripts/generate_peg_hole_dataset.py --config configs/peg_hole.yaml
python scripts/evaluate_matching.py --config configs/phase4.yaml
python scripts/evaluate_vlm.py --adapter mock --split test
python scripts/evaluate_confidence.py --config configs/phase6.yaml
python scripts/generate_report.py --phase6-json evaluations/phase6/aggregate.json --dataset datasets/peg_hole_v1 --phase4-dir evaluations/phase4 --output-dir results --duration-seconds 36 --fps 5
```

参数覆盖也已验证：`--seed 23 --num-objects 3 --save-video` 能生成 3 个候选、正确记录配置并写出 `episode.gif`。

## 11. 当前限制和下一步边界

- `rgb_only` 尚未接入 RGB-only candidate detector，因此 Phase 4 只记录 skip，不报告该模式的准确率。
- 当前 BLIP VQA adapter 已接通但在 synthetic peg-hole 上出现全 NO/平局行为，不能把当前 100% accuracy 当作有效 VLM 性能。
- Phase 6 的 confidence 作用在 Chamfer baseline；BLIP 的 forced-label proxy 仍未校准，不能替代真实 token probability。
- 当前遮挡是 crop-level 确定性矩形 corruption，尚未重新渲染带遮挡物的 PyBullet 场景。
- 当前没有真实插入动力学，只保留了后续 geometric alignment action 的扩展位置。
- `oracle_crop` 和 `render_mask_assisted` 都是仿真辅助输入，不能作为开放世界 RGB 检测结果报告。

因此，当前准确表述是：已经完成可复现的 Phase 1 baseline、Phase 2 模块化运行架构、Phase 3 peg-hole 数据生成、Phase 4 image-image matching baseline、Phase 5 VLM adapter、Phase 6 confidence-aware evaluation 和 Phase 7 preliminary report artifacts；仍需更大规模数据、真实遮挡物重渲染、RGB-only detector 与真实 yaw/插入动力学才能扩展研究结论。
