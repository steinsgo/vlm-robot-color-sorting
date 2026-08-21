# Phase 9：四对象置信度门控 Panda 插入任务

Phase 9 现在是独立于原始 Color Sorting Demo 的四对象 peg-hole mission：

- 四个 peg 采用圆柱、球体、正方体、长方体；
- peg 位置/yaw 和 hole yaw 使用固定 seed 的伪随机序列，hole 位置固定在四个板面槽位；目标 hole yaw 不同于 peg 时由 Panda 在空中分段对齐；
- 每个 peg 有一个对应 hole，hole ID、peg/hole 配对和候选顺序打乱，但 hole 几何位置固定；
- 每轮只对尚未完成的 hole 进行两视角 `Chamfer`、`CLIP` 或显式 `oracle` 匹配；
- confidence/margin gate 接受后才启动 Panda 抓取和插入；
- 四个 peg 全部正确插入且通过最终姿态/碰撞代理检查，才记录 `mission_pass=true`。

默认配置位于 `configs/phase9.yaml`。运行控制路径（不下载 CLIP 模型）：

```bash
python scripts/run_phase9_demo.py --headless --matcher oracle --no-sleep --no-video
```

实际 image-image CLIP 路径：

```bash
python scripts/run_phase9_demo.py --matcher clip --keep-open
```

也可以显式改变四种形状，但必须仍然传入四个不重复的 Phase 9 形状：

```bash
python scripts/run_phase9_demo.py --shapes cylinder,sphere,square,cuboid --matcher clip
```

## 默认验证结果

在 seed `17`、`clearance=0.004 m`、DIRECT、oracle 条件下，四个对象均完成：

| Peg | Shape | Result | Wall contacts |
| --- | --- | --- | ---: |
| `peg_000` | cylinder | success | 0 |
| `peg_001` | sphere | success | 0 |
| `peg_002` | square | success | 0 |
| `peg_003` | cuboid | success | 0 |

结果写入 `runs/phase9/phase9_seed17*/summary.json`，其中包含每个对象的候选分数、confidence、margin、选择结果、抓取结果、最终位姿和 mission-level pass/fail。运行 GIF（开启 `save_video` 时）写入同一目录。

每个 episode 都会尝试处理四个 peg；即使某一步 abstain 或插入失败，后续对象也会被记录并继续评估。只有四个对象全部成功才是 `mission_pass=true`。命令行会额外打印 `MISSION_PASS`、`COMPLETED` 和每个 peg 的 `STEPS`，避免把进程正常结束误认为任务成功。

同一 seed 的视觉 baseline 结果会按任务定义写入 summary：错误匹配、abstain 和 wall contact 都保留为失败/拒绝结果，不会被包装成 mission success。

## 规格与限制

`clearance` 是生成几何时的每侧线性间隙，默认配置为 `4 mm`；由于场景使用统一 `GUI_SCALE=0.60`，summary 中同时记录了实际 collision proxy 的有效间隙（普通形状约 `2.4 mm`，长方体额外增加 `3 mm` 的未缩放 hole 容差）。这是用于 PyBullet 数值稳定性的明确配置参数，不是“零间隙”接触动力学。

当前 hole 是低矮的 ring/box socket collision proxy。抓取确认后，固定 grasp constraint 保持 peg 相对夹爪的位姿；若 hole yaw 不同，Panda 在安全高度分段旋转 peg，再补偿夹爪到 peg 的刚体偏移。释放时不再调用 base-pose teleport，而是让动态 peg 从 hole rim 上方自由下落，由 ring/box walls 和 hole floor 决定最终状态。因此结果仍是 simulation prototype 和 physics-assisted fit validation，不是完整的 contact-rich insertion controller，也不是硬件成功率。

圆柱和球体的 hole 是低矮环形标记：内半径为对应 peg 半径加配置 clearance，外圈只保留窄环宽；方块和长方体继续使用放大的低矮盒状墙。四个 hole 的板面坐标固定，只有 hole ID、peg/hole 配对和执行顺序打乱。所有 hole 都由静态环壁/盒壁和底板组成 collision proxy，peg 则使用动态 collision body。

抓取结果只有在两根 Panda 夹指（link 9/10）都与 peg 接触、两侧都有侧向接触、接触法向相反、夹指间隙非零且 peg 位于两指之间时才会进入固定 grasp constraint；随后 peg 还必须实际抬升超过阈值才记为 `grasped=true`。因此“只有 link ID、夹爪合拢但没有夹住物体”不会被视为成功。窄长方体使用更高的预抓取接触带，避免开放夹指在下降时先撞翻上缘；这只是避免预接触扰动，最终仍由 PyBullet 接触点和抬升验证决定。多对象任务在返回 home 前先经过安全高度 waypoint，避免回程碰到尚未抓取的 peg。抓取确认后记录两个夹指的实测关节开度，并在固定 grasp constraint 搬运期间保持该开度，避免夹指继续向零位收缩穿过 peg。为避免 finger contact solver 与 hand-peg 固定约束互相推挤，搬运期间仍暂时关闭机器人-peg内部碰撞；真实夹指碰撞用于抓取确认，释放前张开夹爪并恢复碰撞。最后下降到 rim 前增加一个中间高度 waypoint，避免 IK 分支跳变造成 peg 横向偏移。已完成的 peg 保留可视和 peg-hole 碰撞，但关闭机器人回程碰撞以免阻塞下一个抓取。每个动作结果还记录 `grasp_contact_link_indices`、`grasp_finger_link_indices`、`grasp_validation`、`grasp_hold_joint_positions`、`transport_finger_joint_positions`、`air_rotation_steps`、`physics_release_used`、`teleport_used`、`inside_hole_opening`、`floor_supported`、`stable`、`wall_contact_count` 和 `floor_contact_count`。Phase 9 的最新修复将 peg 的可视 shape 与动态 body 正确分离，避免源位置残留一个静态副本；同时即使某个对象 abstain/失败，剩余三个对象也会继续评估，不再静默显示为 skipped。

`oracle` 使用对应关系和解析 fit label，只用于验证机器人动作链路，不能报告为 VLM accuracy。`clip` 是当前真正参与候选选择的 image-image baseline；Phase 9 的 confidence 仍未经过校准。
