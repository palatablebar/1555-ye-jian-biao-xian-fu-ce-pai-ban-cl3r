# 夜间标线复测排班数据与算法

导入自备批次或复核估计时阅读。先核对下面的字段和时间窗。查看隔离行原因，再核对估计和派发结果。

## 单据

| 字段 | 用途 | 单位 | 有效区间 |
|---|---|---|---|
| `retro_cd` | 当前反光系数 | mcd | 70–520 |
| `age_days` | 涂覆后天数 | 天 | 4–450 |
| `wet_pct` | 潮湿面积 | % | 0–92 |
| `traffic_k` | 日车辆 | 千辆 | 1–75 |
| `beads_g` | 玻璃珠投放量 | g/m² | 165–435 |
| `surface_c` | 路表温度 | ℃ | 2–37 |

`key` 是单号，`cohort` 是作业区，文本先做 NFKC 归一化。`day` 为测量工日，`received` 为到站工日，`revision` 为正整数版次。同版记录出现多个不同测量结果时隔离该单全部版次。缺测可空置，但每条单至少包含四项观测。

`outcome` 与 `result_day` 成对填写。实绩单位为衰减分。 `observed` 省略时为 `true`，只在留架右截尾场景允许 `false`。

### 原始行业单据

允许使用 `form` 对象代替平铺量测。保留单号、版次、工日和标签等顶层字段。表中的换算为 `(单据值 − 原点偏移) / 倍率`。每行内部校验完成后，与平铺数据使用同一分窗、建模和派单过程。

| 单据组 | 原始字段 | 用途 | 倍率 | 原点偏移 |
|---|---|---|---:|---:|
| 夜测仪 | `reflectance_tenth` | 逆反射十分值 | 10 | 0 |
| 夜测仪 | `laid_hours` | 涂覆后小时 | 24 | 0 |
| 涂覆回执 | `wet_basis_points` | 潮湿面积万分数 | 100 | 0 |
| 涂覆回执 | `daily_vehicle_count` | 日车流量 | 1000 | 0 |
| 路线巡查 | `beads_mg_per_m2` | 玻璃珠每平米毫克 | 1000 | 0 |
| 路线巡查 | `pavement_deci_c` | 路面温度十分度 | 10 | 0 |

示例见 `sample_data/lane_samples_form.json`。混合使用两种量测表示，或单据内读数互相矛盾时，保留隔离原因。

## 时序与训练

拟合 `[0,40)`、校准 `[43,62)`、回放 `[65,85)`、当班从 `88` 日起。前三个窗口的入站和结果都须早于窗口终点。到站日晚于截止日的单据在入站隔离。拟合期冻结补值中位数、均值、尺度及训练包络。

| 特征 | 语义 | 公式 |
|---|---|---|
| `retention` | 龄期反光余量 | `retro_cd/(1+age_days/140)` |
| `wet_loss` | 车流潮湿协变 | `wet_pct*traffic_k/100` |
| `grain_coverage` | 湿面等效玻璃珠 | `beads_g/(1+wet_pct/70)` |
| `traffic_exposure` | 累计车流暴露 | `age_days*traffic_k/1000` |
| `thermal_mismatch` | 温度偏离 | `abs(surface_c-20)` |

### RBF 局部反光衰减回归

带宽由历史点最近邻的平方距离中位数估算。以同线路近邻优先并做指数权重归一化，在距离指数内先减最近距离。

本工作台的独立样本校准方式为湿度调整衰减误差。风险校准无法确认正负样本时转人工判读。连续估计使用独立校准残差计算双侧区间，误差尺度随作业条件变化。分布监测对训练和当班量测比较分区累积分布差。未知作业区、两项缺测或两项明显超训练包络的单据需要人工判读。

## 分配便携复测仪

| 优先级 | 原因码 | 条件 | 处置依据 |
|---:|---|---|---|
| 1 | `WET_LOW_LIGHT` | `wet_pct >= 75 and retro_cd <= 140` | 潮湿低反光，现场复测 |
| 2 | `LONG_SERVICE` | `age_days >= 300 and traffic_k >= 40` | 高暴露路段优先复测 |
| 3 | `BEAD_SUPPLY` | `beads_g < 210 and retro_cd < 230` | 核对投放班组记录 |

第一条为暂停规则。其余命中与模型阈值 `25` 共同决定是否列入候选。单据所需工位数按 `1 + int(traffic_k >= 38)` 计算。均衡策略对分区限额，集中复核策略允许把额度集中投入高风险区。未分派原因在导出的处置记录中保留。签发前核对每张当班单和工位余额。

### 标线干面对照单

同路段、近龄期且留有完整实测值的干面点可以做对照。潮湿测点先保留读数，低反光、少投珠和车流高的复测安排分别记录。 在派发环节展开现场安排，或从关联单据回查实际读数。

## 联动养护签发

联动养护单使用同一路段 `segment`（或 `segment_id`、`road_segment`、`cohort`）的养护前、养护后成熟实绩学习衰减降幅。成对记录可直接给 `before_outcome`/`after_outcome`（或 `reduction`），也可用相同 `pair_id`/`link_id` 加 `phase: before|after` 的两条事件表示。只有 `result_day <= as_of`、`received <= as_of` 且未截尾的实绩可进入当时回放；晚到结果会被截止日过滤，不能改写旧建议。

校准样本必须来自未参与拟合的路段；与拟合路段重号的校准记录会被剔除。没有未见过的联动校准样本时，方案状态为 `CALIBRATION_SUSPENDED`，单据为 `暂停签发 / MISSING_LINKED_CALIBRATION`。

签发策略固定为：

1. 实测湿滑并已暂停（`WET_LOW_LIGHT` 等湿滑硬规则）的单保持 `暂停`，不转成养护动作。
2. 少占道动作的衰减区间上界能确认 `upper <= target_decay`（默认 `25`）时，优先签发少占道动作。
3. 没有少占道达标动作时，才允许封路复铺；本班 `closure_capacity=1`，在候选路段中按区间上界最低、安全余量最大选择一段。
4. 没有可信区间确认达标的动作转 `人工判读 / NO_QUALIFIED_TREATMENT`；封路名额不足的其他路段转人工。

结果位于 `maintenance_advice`，其 `orders`/`signed_orders` 中可查 `养护动作`、`action`、`interval`、`decay_interval`、`预计衰减区间` 和降幅区间。`GET /api/linebeads/maintenance-orders/<order_id>` 可按签发单号回查。`/api/linebeads/run` 也接受顶层 `maintenance`、`maintenance_records`、`calibration`、`maintenance_candidates` 等自备字段。
