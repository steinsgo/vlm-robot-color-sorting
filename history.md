# ConfMate 修改与实验历史

更新时间：2026-08-19

## GitHub 发布整理

- README 状态更新为 Phase 1–8，并补充 Phase 8 collision-proxy 的定位说明。
- 增加 GitHub Actions CI 徽章，并确认安装命令指向个人 fork。

## CI 依赖修复

- GitHub Actions 原先只安装 `requirements-base.txt`，但 `scripts/smoke_test.py` 导入的 `SimulationBackend` 还需要 PyTorch 和 Transformers。
- 已将 `.github/workflows/ci.yml` 改为安装统一入口 `requirements.txt`，覆盖基础仿真依赖和 VLM 依赖。
- 该修复只需推送到个人 fork，不需要合并回原仓库。

## 本次专家评估修复

本次修改针对 [专家评估.md](专家评估.md) 指出的实验公平性、评测有效性和复现工程问题完成了 Phase 3.5 修复。

随后开始 Phase 8，加入 collision-enabled peg、perimeter-wall/floor opening proxy、插入轨迹和明确的 wall-contact/settling 成功判据。

### 数据与标签

- 将 peg–hole 数据集升级为 `datasets/peg_hole_v2`。
- 测试集从 2 个 episode 扩展为 100 个独立 episode；完整数据集为 train 16、val 20、test 100，共 136 个 episode。
- 每个 episode 随机化目标孔 ID、候选 ID 顺序和板面位置，消除 `hole_000` 固定目标和候选顺序泄漏。
- 新增 `candidate_fit` 解析几何标签，记录 family、variant、yaw 和 footprint clearance；该标签不代表真实插入动力学。
- v2 核验：目标 ID 计数为 `hole_000=27`、`hole_001=32`、`hole_002=41`；出现 6 种候选顺序和 6 种位置分配；fit 标签违规数为 0。

### 匹配与置信度

- Phase 4、Phase 6 和 Phase 5 VLM evaluator 统一采用 tie-aware ranking：top-1/top-2 平分时返回 `uncertain`，不再按候选 ID 默认选择第一个。
- Phase 4 和 Phase 5 aggregate 增加 episode-level 95% Wilson CI、uncertainty count/rate。
- Phase 6 condition summary 增加 episode-level CI、uncertainty 和可复现的 `evaluation_seed`。
- 明确 BLIP VQA 的 `YES=0.75`、`NO=0.25` 是 forced-label proxy，不是 token probability 或 calibrated confidence。

### 复现工程与文档

- 新增 `tests/`、`.github/workflows/ci.yml`、`AGENTS.md`、`upstream.lock`、`scripts/env_report.py`。
- 新增分层依赖 `requirements-base.txt`、`requirements-vlm.txt` 和直接依赖快照 `requirements-lock.txt`。
- 更新 README、Phase 1–7 总结和 Phase 7 报告，移除 v1、2-episode 和 BLIP 100% 等过时表述。
- Phase 7 示例不再硬编码 episode ID，并将 CI/uncertainty 写入 `results/summary.csv`。

## 重新实验结果

### Phase 4 test matching

| Observation mode | Method | Top-1 accuracy | 95% Wilson CI |
| --- | --- | ---: | --- |
| `oracle_crop` | random | 35% | [26.4%, 44.7%] |
| `oracle_crop` | Chamfer | 93% | [86.3%, 96.6%] |
| `oracle_crop` | single-view CLIP | 99% | [94.6%, 99.8%] |
| `oracle_crop` | multi-view CLIP | 92% | [85.0%, 95.9%] |
| `render_mask_assisted` | random | 35% | [26.4%, 44.7%] |
| `render_mask_assisted` | Chamfer | 92% | [85.0%, 95.9%] |
| `render_mask_assisted` | single-view CLIP | 94% | [87.5%, 97.2%] |
| `render_mask_assisted` | multi-view CLIP | 73% | [63.6%, 80.7%] |

`rgb_only` 仍因没有 candidate detector 而标记为 skipped。native episode 只有 3 个候选，Top-3 不作为强指标。

### Phase 5 VLM adapter

| Adapter / mode | Top-1 accuracy | 95% Wilson CI | Uncertain rate |
| --- | ---: | --- | ---: |
| mock / `oracle_crop` | 98% | [93.0%, 99.4%] | 0% |
| mock / `render_mask_assisted` | 97% | [91.5%, 99.0%] | 0% |
| BLIP / `oracle_crop` | 1% | [0.2%, 5.4%] | 80% |
| BLIP / `render_mask_assisted` | 2% | [0.6%, 7.0%] | 77% |

BLIP 结果说明 adapter 已接通，但当前 synthetic peg–hole 结果不能作为有效 VLM reasoning 或 calibration 证据。

### Phase 6 confidence / abstention

Phase 6 使用 Chamfer baseline，test 网格共 12,000 条 condition rows，覆盖 100 个 episode。固定 80% coverage 的结果：

| Policy | Selective top-1 accuracy | Coverage |
| --- | ---: | ---: |
| random rejection | 59.04% ± 0.20% | 80% |
| margin | 64.94% | 80% |
| multi-view consistency | 60.89% | 80% |
| stability | 64.61% | 80% |

Phase 6 的 confidence 不是 BLIP confidence；`synthetic_flip` 是镜像增强，遮挡是 crop-level rectangular corruption。

### Phase 7 artifacts

已重新生成：

- `results/system_architecture.png`
- `results/simulation_scene.png`
- `results/success_match.png`
- `results/failure_rejection.png`
- `results/accuracy_vs_occlusion.png`
- `results/risk_coverage.png`
- `results/phase7_demo.gif`（36 秒）
- `results/summary.csv`（364 条 condition/fixed-coverage 汇总）

### Phase 8 collision-proxy insertion validation

命令：

```text
python scripts/generate_peg_hole_dataset.py --config configs/peg_hole.yaml --output-dir datasets/peg_hole_v2_phase8
python scripts/evaluate_insertion.py --config configs/phase8.yaml
```

在 100 个 test episode 上完成 300 次候选试验：

| 指标 | 结果 | 95% Wilson CI |
| --- | ---: | --- |
| target insertion success | 100% | [96.3%, 100.0%] |
| distractor rejection | 100% | [98.1%, 100.0%] |
| episode-level all-distractors rejection | 100% | [96.3%, 100.0%] |
| analytic fit / physical proxy agreement | 100% | [98.7%, 100.0%] |

结果文件为 `evaluations/phase8/aggregate.json` 和 `results.jsonl`。这是一个 perimeter wall ring + floor 的 collision proxy；它证明了当前 analytic fit 标签与该物理代理一致，不等价于真实机械臂插入成功率。

## 验证记录

以下检查均通过：

```text
python -m compileall confmate scripts
python -m unittest discover -s tests
python scripts/smoke_test.py
python scripts/generate_peg_hole_dataset.py --config configs/peg_hole.yaml
python scripts/evaluate_matching.py --config configs/phase4.yaml
python scripts/evaluate_vlm.py --config configs/phase5.yaml --adapter mock
python scripts/evaluate_vlm.py --config configs/phase5.yaml --adapter blip --output-dir evaluations/phase5/blip
python scripts/evaluate_confidence.py --config configs/phase6.yaml
python scripts/generate_report.py --phase6-json evaluations/phase6/aggregate.json --phase6-results evaluations/phase6/results.jsonl --dataset datasets/peg_hole_v2 --phase4-dir evaluations/phase4 --output-dir results
```

原始 CLIP/BLIP 模型下载需要本机代理 `127.0.0.1:10808`；数据集、evaluation cache 和模型权重均未纳入 Git。

## 当前研究边界

当前项目应定位为 simulation-assisted preliminary matching pipeline。尚未完成真实碰撞/接触插入、RGB-only detection、真实遮挡物重渲染、yaw 估计或 token-level VLM calibration。后续若扩展研究结论，应优先加入这些实验。
