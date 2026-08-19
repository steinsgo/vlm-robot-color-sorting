# ConfMate Phase 1–7 阶段总结

更新时间：2026-08-19  
当前定位：可复现的 PyBullet peg–hole matching prototype；结果属于 preliminary evaluation，不是零样本插入动力学或校准 VLM 性能声明。

## 1. 总体结论

仓库已经从上游颜色分类 demo 扩展为一条可运行的 ConfMate 实验流水线：

1. Phase 1 验证原始 PyBullet、Panda、仿真相机和 CLIP demo 可以运行。
2. Phase 2 将 baseline 拆成配置、仿真、感知、匹配、控制和产物模块，并支持 headless、seed、候选数量和视频参数。
3. Phase 3 生成 shape-family-disjoint 的 peg–hole 仿真数据与三种观察模式。
4. Phase 3.5 修复了目标固定为 `hole_000`、候选顺序固定、测试 episode 过少和没有几何 fit 标签等公平性问题。
5. Phase 4 在 100 个独立 test episode 上重跑 random、Chamfer、single-view CLIP 和 multi-view CLIP。
6. Phase 5 提供 mock 与 BLIP VQA adapter；BLIP 的 YES/NO 分数明确标记为 forced-label proxy。
7. Phase 6 在 Chamfer baseline 上执行遮挡、候选池扩展、置信度、abstention、ECE 和 risk–coverage 评测。
8. Phase 7 生成报告图表、CSV 和 GIF，并从实际结果动态选择示例 episode。
9. Phase 8 为每个候选运行 collision-enabled peg 的插入代理试验，验证 analytic fit 与物理 proxy 的一致性。

最安全的项目表述是：

> Built an end-to-end PyBullet prototype for multi-view peg–hole candidate matching, including CLIP and geometric baselines, a generative VLM adapter, controlled partial-observation evaluation, confidence-based abstention, and reproducible report artifacts. Current results are preliminary and use simulation-assisted crops and synthetic corruptions.

不能表述为：已完成真实插入动力学、RGB-only open-world detection、BLIP 有效推理 100%、或 calibrated VLM confidence。

## 2. 运行环境与网络

当前环境快照见 [reports/env_report.json](reports/env_report.json)：

| 项目 | 版本/状态 |
| --- | --- |
| OS | Windows 10/11 runtime，AMD64 |
| Python | 3.9.21 |
| PyBullet | 3.2.5 |
| PyTorch | 1.12.1+cu116，CUDA 11.6，可用 RTX 3060 Laptop GPU |
| Transformers | 4.40.2 |
| CLIP | `openai/clip-vit-base-patch32` |
| BLIP | `Salesforce/blip-vqa-base` |

Hugging Face 和 GitHub 网络访问需要使用本机代理 `127.0.0.1:10808`。PowerShell 中可使用：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:10808"
$env:HTTPS_PROXY = "http://127.0.0.1:10808"
$env:ALL_PROXY = "http://127.0.0.1:10808"
```

环境报告不会写入 token、代理 URL 或模型缓存内容。依赖入口为 [requirements.txt](requirements.txt)，分层文件为 `requirements-base.txt`、`requirements-vlm.txt`，当前直接依赖快照为 `requirements-lock.txt`。

## 3. Phase 1：上游 demo 验证

原始 demo 的逻辑没有被 ConfMate 修改。已验证 PyBullet 可加载 Panda、仿真相机可捕获 RGB、CLIP 可为候选物体计算分数，且 pick-and-place 与颜色排序流程可运行。

```text
python simple_pick_place_demo.py
python color_sorting_vlm.py
```

## 4. Phase 2：模块化基础架构

`confmate/` 提供配置、仿真、感知、CLIP matching、控制、artifact 和 runner 边界；主要入口为：

```text
python scripts/smoke_test.py
python scripts/run_baseline.py --config configs/baseline.yaml --headless
```

回归验证还包括：

```text
python -m compileall confmate scripts
python -m unittest discover -s tests
```

## 5. Phase 3/3.5：数据、随机化与几何标签

当前配置生成 `datasets/peg_hole_v2`：

| Split | Shape family | Episode 数量 |
| --- | --- | ---: |
| train | cylinder、rectangle、keyed、cross | 16 |
| val | L | 20 |
| test | asymmetric | 100 |

test family 与 train/val 不重叠。每个 scene 使用独立的 deterministic seed。Phase 3.5 的公平性约束为：

- `target_hole_id` 在每个 episode 的候选 ID 中随机采样；
- 候选 ID 的顺序和板面位置按 episode 打乱；
- 平分时返回 `uncertain`，不按候选 ID 或列表首项默认预测；
- 每个 episode 保存 `candidate_fit`，记录 family、variant、yaw 和 analytic clearance；
- fit 标签是 ordered-part footprint clearance，不是碰撞接触、插入成功或动力学模拟。

v2 数据核验结果：100 个 test episode 中目标 ID 计数为 `hole_000=27`、`hole_001=32`、`hole_002=41`；出现 6 种候选顺序和 6 种位置分配；几何标签的可配合候选数量违规为 0。

```powershell
python scripts/generate_peg_hole_dataset.py --config configs/peg_hole.yaml
```

数据和 evaluation cache 被 `.gitignore` 排除；可通过配置重新生成，不提交大体积图片和模型。

观察模式含义：

| 模式 | 说明 |
| --- | --- |
| `oracle_crop` | 使用生成时的 ground-truth object mask，属于上限验证输入 |
| `render_mask_assisted` | 使用 PyBullet segmentation buffer，属于仿真辅助输入 |
| `rgb_only` | 只有整帧 RGB；当前没有 candidate detector，因此后续匹配结果会标记为 skipped |

## 6. Phase 4：matching baseline

```powershell
python scripts/evaluate_matching.py --config configs/phase4.yaml
```

v2 test split 的主要结果如下；区间为 episode-level 95% Wilson CI：

| Observation mode | Method | Top-1 | 95% CI |
| --- | --- | ---: | --- |
| `oracle_crop` | random | 35% | [26.4%, 44.7%] |
| `oracle_crop` | Chamfer | 93% | [86.3%, 96.6%] |
| `oracle_crop` | single-view CLIP | 99% | [94.6%, 99.8%] |
| `oracle_crop` | multi-view CLIP | 92% | [85.0%, 95.9%] |
| `render_mask_assisted` | random | 35% | [26.4%, 44.7%] |
| `render_mask_assisted` | Chamfer | 92% | [85.0%, 95.9%] |
| `render_mask_assisted` | single-view CLIP | 94% | [87.5%, 97.2%] |
| `render_mask_assisted` | multi-view CLIP | 73% | [63.6%, 80.7%] |

每个 native episode 只有 3 个候选，故 Top-3 不能作为强指标。CLIP 这里是 image–image embedding matching，不是 image–text 分类；Phase 4 不估计 yaw。`rgb_only` 由于没有候选 crop 会被明确跳过。

## 7. Phase 5：generative VLM adapter

统一接口为 `VLMMatcher.match(peg_images, hole_images, prompt)`。运行命令：

```powershell
python scripts/evaluate_vlm.py --config configs/phase5.yaml --adapter mock
python scripts/evaluate_vlm.py --config configs/phase5.yaml --adapter blip --output-dir evaluations/phase5/blip
```

v2 test 结果：

| Adapter / mode | Top-1 | 95% CI | Uncertain rate |
| --- | ---: | --- | ---: |
| mock / `oracle_crop` | 98% | [93.0%, 99.4%] | 0% |
| mock / `render_mask_assisted` | 97% | [91.5%, 99.0%] | 0% |
| BLIP / `oracle_crop` | 1% | [0.2%, 5.4%] | 80% |
| BLIP / `render_mask_assisted` | 2% | [0.6%, 7.0%] | 77% |

BLIP 的 `YES=0.75`、`NO=0.25` 是 forced-label proxy，不是 token probability，也没有校准。由于平分现在会 abstain，旧版本可能出现的“全 NO 仍然 100%”数据泄漏已被消除。原始生成文本和每个候选的 parse 状态仍保存在本地 evaluation JSON 中。

## 8. Phase 6：confidence、abstention 与 partial observation

```powershell
python scripts/evaluate_confidence.py --config configs/phase6.yaml
```

Phase 6 只对 Chamfer baseline 的分数做 confidence study；测试网格包含 100 个 episode，合计 12,000 条 condition rows。条件包括 none/light/moderate/heavy crop-level occlusion、single/two/synthetic third view 和 3/5/8 candidate pool。5/8 candidate 条件借用了其他 episode 的 crops，并记录 distractor 类型。

在固定 80% coverage 的 test 比较中：

| Policy | Selective top-1 | Coverage |
| --- | ---: | ---: |
| random rejection | 59.04% ± 0.20% | 80% |
| margin | 64.94% | 80% |
| multi-view consistency | 60.89% | 80% |
| stability | 64.61% | 80% |

CI、uncertainty rate、ECE、risk–coverage 和每个 condition 的 episode-level 汇总写入 `evaluations/phase6/aggregate.json`，详细行写入 `results.jsonl`。`synthetic_flip` 是确定性镜像增强，不是第三个 PyBullet 相机；遮挡是 crop-level 矩形 corruption，不是新渲染的遮挡物。

## 9. Phase 7：报告产物

```powershell
python scripts/generate_report.py `
  --phase6-json evaluations/phase6/aggregate.json `
  --phase6-results evaluations/phase6/results.jsonl `
  --dataset datasets/peg_hole_v2 `
  --phase4-dir evaluations/phase4 `
  --output-dir results `
  --duration-seconds 36 `
  --fps 5
```

已生成 [results/README.md](results/README.md)、`system_architecture.png`、`simulation_scene.png`、`success_match.png`、`failure_rejection.png`、`accuracy_vs_occlusion.png`、`risk_coverage.png`、36 秒 `phase7_demo.gif` 和包含 CI/uncertainty 字段的 `summary.csv`。示例 episode 由实际 Phase 4 结果动态选择，不再假设固定 episode ID。

## 10. 可复现与工程文件

- [AGENTS.md](AGENTS.md)：仓库范围、声明边界和运行顺序。
- [upstream.lock](upstream.lock)：上游仓库、base commit 和 fork provenance。
- [reports/env_report.json](reports/env_report.json)：Python、包、CUDA、GPU 和 Git 环境快照。
- [tests/test_phase35_fairness.py](tests/test_phase35_fairness.py)：平分 abstention、几何 fit 和 100 episode 配置回归测试。
- [.github/workflows/ci.yml](.github/workflows/ci.yml)：compile、单元测试和 smoke test。

## 11. Phase 8：collision-proxy insertion validation

```powershell
python scripts/generate_peg_hole_dataset.py --config configs/peg_hole.yaml --output-dir datasets/peg_hole_v2_phase8
python scripts/evaluate_insertion.py --config configs/phase8.yaml
```

Phase 8 对 100 个 test episode 的 300 个候选分别执行 PyBullet drop/insertion trial。每个 opening 由 perimeter wall ring 和 collision floor 构成，记录 wall contact、最终高度、横向位移和 settling 状态。

| 指标 | 结果 | 95% Wilson CI |
| --- | ---: | --- |
| target insertion success | 100% | [96.3%, 100.0%] |
| distractor rejection | 100% | [98.1%, 100.0%] |
| all distractors rejected per episode | 100% | [96.3%, 100.0%] |
| analytic/physical fit agreement | 100% | [98.7%, 100.0%] |

结果位于本地 `evaluations/phase8/`。Phase 8 使用的是 physics-assisted collision proxy，不是完整的机械臂接近、接触力控制、摩擦辨识或真实插入动力学实验；因此这些结果只能说明当前 analytic fit 标签与该代理的一致性。

## 12. 当前边界

目前准确的研究定位是“simulation-assisted, preliminary matching pipeline”。以下结论仍不能从本仓库实验推出：

- 真实接触/碰撞下的插入成功率或零样本 mating dynamics；
- 开放世界 RGB-only candidate detection；
- BLIP 或其他 generative VLM 的校准 token confidence；
- 真实遮挡物重渲染、真实 yaw 估计和大规模跨形状泛化。

下一步若要提升为更强研究结果，应加入带 collision geometry 的插入/接触验证、真正的 RGB detector、更多独立几何实例、多随机遮挡物重渲染，并对 VLM confidence 做 token-level calibration。
