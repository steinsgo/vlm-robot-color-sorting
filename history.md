# ConfMate 修改与实验历史

更新时间：2026-08-21

## 2026-08-21 — Phase 9 Panda peg-hole GUI demo

- 新增 `confmate/phase9.py`、`scripts/run_phase9_demo.py` 和 `configs/phase9.yaml`。
- 复用原作者 Panda 的 URDF、末端 IK、夹爪关节和移动流程，创建独立的 ConfMate peg-hole GUI 场景。
- 新增 tie-aware confidence gate：高置信度才执行抓取/插入，平局、低 margin 或低 confidence 时 abstain。
- 新增 `oracle` 控制演示、Chamfer 和 CLIP 匹配模式；oracle 结果明确标记为 debug-only，不冒充 VLM 结果。
- Phase 9 headless oracle 验证：`grasped=true`、`inserted=true`、`wall_contact_count=0`。
- Phase 9 CLIP 验证（seed 17，演示阈值 `min_confidence=0.35`）：CLIP 选中真实 `hole_001`，confidence `0.3831`、margin `0.1059`，随后 `grasped=true`、`inserted=true`、`wall_contact_count=0`。

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

## 2026-08-21 — Phase 9 四对象 mission 升级

- 将 Phase 9 从单 peg/多候选 hole 演示升级为四对象顺序任务：圆柱、球体、正方体、长方体。
- 增加固定 seed 的伪随机 peg 槽位、yaw、hole ID 和执行顺序；hole 位置固定，每轮只保留尚未完成的 hole，完成后移除对应候选。
- 保留 Chamfer、CLIP 和 oracle 三种匹配路径；confidence/margin gate 接受后才执行 Panda 抓取与插入。
- Mission pass 条件改为四个 peg 全部正确抓取、对应插入、通过最终位姿检查，且无 perimeter-wall contact。
- 增加明确的 `clearance=0.004 m` 配置、逐对象日志、mission-level `summary.json` 和 `phase9_mission.gif`。
- 为圆柱和球体增加低矮环形 hole visual/collision proxy；为小物体增加显式记录的 guided release pose alignment，避免 IK/夹爪误差导致释放时偏移。
- oracle headless 验证：四个对象全部成功，四个对象 wall contact 均为 0；该结果仍是 simulation prototype，不是 contact-rich insertion 或硬件成功率。
- 同 seed baseline 记录：Chamfer 在第 3 个对象选错候选并正确失败；CLIP 第 1 个对象 confidence `0.2520`、margin `0.0031`，按默认 gate abstain。视觉 baseline 的失败/拒绝被保留在 `summary.json`，没有被当作 mission success。
- 修复 Phase 9 GUI：动态 peg 现在直接使用 visual shape，删除了源位置残留的静态副本；每个动作显式记录 wall/floor collision proxy contacts。流程不再因单步失败而跳过剩余对象，四个对象都会被评估，只有 4/4 成功才 `MISSION_PASS=true`。

## 2026-08-22 — Phase 9 几何与抓取判定修复

- 将第四个形状从锥体替换为球体；默认任务固定为圆柱、球体、正方体、长方体。
- 圆柱和球体 hole 改为低矮环形几何，内径按对应 peg 半径加 clearance，方块/长方体 hole 保留放大后的低矮盒壁；四个 hole 坐标固定，ID 和配对仍可打乱。
- 修正环壁碰撞盒的径向/切向轴向，消除环壁宽度错误侵入开口造成的球体/圆柱假碰撞。
- 抓取判定收紧为两根夹指 link 9/10 均接触、固定约束有效且 peg 实际抬升；失败时跳过插入并记录 `grasp_not_confirmed`。
- 新增环形 hole、固定位置、动态 peg visual body 和双夹指抓取的回归测试；16 项 unittest、smoke test、compileall，以及 GUI/headless oracle 四物体运行均通过，`MISSION_PASS=true`、`COMPLETED=4/4`、四步 wall contact 均为 0。

## 2026-08-22 — Phase 9 物理释放与姿态保持修复

- 移除释放后的 `resetBasePositionAndOrientation` 路径；peg 从 hole rim 上方作为动态刚体自由下落，success 必须同时满足开口几何包含、对应 hole floor contact、稳定速度/角速度和零最终 wall contact。
- 抓取后记录 peg 相对末端的刚体变换；Panda 在空中以不超过 15° 的分段 yaw 步进完成 hole 对齐，解决直接大角度 IK 跳转造成的 cuboid 姿态偏差。
- 提高固定约束强度；搬运阶段暂时关闭已确认 grasp 的机器人-peg 内部碰撞，释放前张开夹爪后恢复；完成的 peg 仅关闭机器人回程碰撞，保留 peg-hole 碰撞和可视状态。
- 修复多物体连续任务的 home 归位：为腕部关节设置速度上限并至少等待 360 个仿真步，避免上一轮腕部姿态导致下一次 IK 走错分支。
- 最终 headless oracle 验证：`MISSION_PASS=true`、`COMPLETED=4/4`；四步 `teleport_used=false`、`physics_release_used=true`，且均为 `inside_hole_opening=true`、`floor_supported=true`、`wall_contact_count=0`。

## 2026-08-22 — Phase 9 真实夹取闭环修复

- 抓取不再只依据“link 9/10 出现过接触”判定；新增 `grasp_validation`，要求双夹指接触、两侧侧向接触、接触法向相反、夹指间隙在合理范围、peg 位于两指之间、没有双指完全闭合，并且 peg 保持在可支撑高度。
- 发现长方体在夹爪闭合前的低位下降阶段会被指尖碰歪；按形状使用预抓取高度，正方体/长方体采用 `0.025 m`，圆柱/球体采用 `0.015 m`，避免把预接触扰动误认为成功抓取。
- 多对象回程增加高位 Cartesian waypoint，防止机械臂从上一 hole 返回 home 时碰到尚未抓取的 peg；长方体 hole 增加显式 `0.003 m` 未缩放容差，但最终仍要求真实开口包含、floor contact、稳定和零 wall contact。
- 新增抓取回归断言：每个成功对象都必须有有效 `grasp_validation`、固定约束和超过 `0.035 m` 的实际抬升；同时保留 `teleport_used=false` 和物理释放检查。
- 最新 headless oracle 验证：`MISSION_PASS=true`、`COMPLETED=4/4`，四个对象均为双侧真实接触抓取，随后完成空中姿态调整和物理释放；结果写入 `runs/phase9_grasp_final_headless4/phase9_seed17/summary.json`（运行产物不纳入 Git）。
- 最终 GUI oracle 验证：窗口模式下四个对象均完成机械臂抓取/空中调整/物理释放，`MISSION_PASS=true`、`COMPLETED=4/4`、总 wall contact 为 0；结果写入 `runs/phase9_grasp_final_gui/phase9_seed17/summary.json`（运行产物不纳入 Git）。

## 2026-08-22 — Phase 9 运输夹指保持修复

- 修复抓取后夹指继续向 `0.0` 收缩、穿过 peg 的问题：抓取确认时保存 link 9/10 的实测关节开度，固定约束运输期间持续保持该开度。
- 保留抓取瞬间的真实双指接触验证；运输阶段为避免 finger contacts 与 hand-peg 固定约束互相施力，继续关闭内部碰撞，释放前张开夹爪并恢复碰撞。运输日志新增 `grasp_hold_joint_positions`、`transport_finger_joint_positions` 和 `transport_grasp_contact_count`。
- 最后下降阶段加入中间高度 waypoint，避免保持夹指开度后长方体释放时发生 IK 分支跳变。
- 最新 headless oracle：`MISSION_PASS=true`、`COMPLETED=4/4`；运输前后夹指关节变化不超过 `0.000002 rad`，四步 wall contact 均为 0。结果写入 `runs/phase9_gripper_hold_fix4/phase9_seed17/summary.json`（运行产物不纳入 Git）。
