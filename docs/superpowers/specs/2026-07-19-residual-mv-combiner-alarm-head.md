# 残差多变量融合报警头(Residual MV-Combiner Alarm Head)设计

**日期:** 2026-07-19
**目标产线:** XMEAS10(Purge Rate)/ IDV13 / mag=25(温和故障),horizon = 5 步(15 min)
**关联:** 承接 `2026-07-16-alarm-aware-gate-design.md`(凸 softmax alarm gate)与 partial5 微调实验;本设计是对「凸组合天花板」的一次结构性突破。

---

## 1. 动机:凸组合的真实天花板

- 现有 gate 是 softmax-over-experts(w≥0, Σw=1)⇒ 输出恒 ∈ [minᵢ ŷᵢ, maxᵢ ŷᵢ]。
- 推论:若四个专家**共同低估**了 purge rate 的上冲(mag=25 慢坡最典型:斜坡平缓,四个 TSFM 大概率都够不到限值),全体不穿越 ⇒ 凸 gate 也**永不穿越**。recall 被死锁在「专家报警集合的并集」内。
- 凸组合能消掉**相互独立的方差型误差**(这是集成学习的意义),但**消不掉四专家共享的系统性偏差**。
- **partial5 实测佐证**:h=5 时 diff-P5 单专家(recall 0.737)反超凸 gate(0.658)——凸 gate 在此设定下无增益。
- **结论**:要真正超越专家,必须**放开凸约束**,让组合器能输出专家包络之外的值,以补上共享低估。这不是「超过最好专家」的口号,而是「纠正四专家共犯的系统偏差」这一具体机制。

## 2. 目标与验收线(可证伪)

**主目标**:在 recall / false prognosis rate(FAR)/ mean lead time 三个指标上**超越最强单专家 diff-P5**。

**验收线**:扩展集 **Run18-27** 上,新头在 **FAR ≤ 5%** 内 **recall 严格超过 diff-P5**,且 **mean lead time 不低于 diff-P5**。

**Sanity check**:真穿越点邻域的 Δ 应**以正为主**(在补低估);若 Δ ≈ 白噪声 ⇒ 判定过拟合、未学到真偏差。

**回退**:达不到验收线 ⇒ 结论「共享偏差不可稳定迁移」,交付回退到 diff-P5,如实报告,不粉饰。

## 3. 架构

### 3.1 残差锚点

```
corrected_t(h) = ŷ_diff-P5,t(h) + Δ_t(h),   h = 1..H,  H = 5
```

- 锚 = diff-P5(h=5 下最强单专家)。头只学「相对 diff 的修正」Δ。
- `λ_reg·‖Δ‖²` 正则把默认行为拉回「就等于 diff-P5」⇒ 无真信号时不动 ⇒ 保护 FAR、抗过拟合。
- Δ **无约束**(可正可负、可使 corrected > maxᵢ ŷᵢ)⇒ **跨出凸包**,这是相对凸 gate 的本质区别。

### 3.2 融合头:小 MLP(1 隐层,多变量)

每个预测原点 t 产出整条 Δ 向量(H 维)。输入拼接:

| 输入块 | 内容 | 维度 |
|---|---|---|
| ① 专家预测 | 4 专家对 t+1..t+H 的预测 | 4×H = 20 |
| ② 多变量上下文 | `[target + 5 协变量]` 最近 **10** 步历史(**lookback = 10,默认值,可 CV 调**),逐通道共享轻线性编码器 `Linear(10→3)` 抽 level/slope(或直接展平 6×10=60) | 6×3 = 18 |
| ③ 报警定位标量 | 锚到限值距离 (limit − ŷ_diff,t(h)) | H = 5 |
| ④ 报警定位标量 | 专家分歧 (maxᵢ − minᵢ) | H = 5 |

- MLP:`concat(~48) → hidden(≈32) → Δ(H)`。含上下文编码器,总参数 **~2–3k**。
- **lookback 只取 10 的理由**:头是残差纠正器,不重新预报,只需判断「此刻四专家是否在系统性低估」——最有信息量的是**近期斜率/水平 + 协变量是否刚异动**,10 步(30 min)足够;更短 ⇒ 更少参数 ⇒ 更抗过拟合(贴 Run11-17 数据有限)。lookback 列为 CV 可调项,若 lead time 不足再放长。
- **非线性的作用**:表达「上游协变量已异动 **且** 逼近限值时才顶上去」这类**条件修正**,把修正**定位**到该发力处(保护 FAR、真正吃到协变量互相关)——纯线性头做不到(只能均匀抬升,FAR 定位弱)。

**为何不用 DLinear**:标准 `models/DLinear.py` 是**通道独立**的(`Linear(seq_len→pred_len)` 逐通道广播,无任何跨通道混合),即便 MS 模式也只是切出 target 通道、且 target 仅由自身历史预测。本任务本质需要「多变量→单变量」的通道混合(融合 4 条专家预测 + 多变量上下文窗),DLinear 结构上做不到,故排除。

## 4. 训练目标(alarm-aware,可微)

```
L = MSE(corrected, GT)_{1..H}
  + λ_far  · softFAR(corrected, limit)      # clean 窗口误穿越惩罚
  − λ_lead · softLead(corrected, limit)     # true-alarm 窗口提前穿越奖励
  + λ_reg  · ‖Δ‖²                            # ★防 FAR 爆、抗过拟合的命门项(新增)
```

- `softFAR` / `softLead` 复用 `fuse_gate_alarm.py` 的软穿越 surrogate(`sigmoid((corrected − limit)/τ_a)`)。
- 训练目标直接朝三指标优化,**不是只压 MSE**——因为 mag=25 上纯 MSE 训练会让模型**削平尖峰、系统性低估上冲**,反而掉 recall(见 `mag25_gate_conservative_finding`:上一版 gate MSE 训练变保守,diff 权重 0.44→0.28、recall 下降)。

## 5. 数据 / 无泄漏 / 选参 / 评估

- **专家来源**:partial5(`loss_pred_len=5`)raw & diff(已训练)+ Time-MoE、Sundial zero-shot。
- **无泄漏训练**:融合头**只训练在 Run11-17(gate25)** 的专家预测——这些 run 专家从未训练过,故无泄漏。具体 split / arg 接线**镜像 `fuse_gate_alarm.py`**(实现阶段读代码固定)。
- **选参**:grouped-CV(按 run 留一 / 分折)选 λ_far、λ_lead、λ_reg、hidden、τ_a、k;复用 `select_alarmgate_cv` 那套基础设施。
- **评估**:Run9-10(test)+ Run18-27(扩展),mag=25 限,clean 窗口,`eval_steps=5`;结果并入 `_partial5` 对照表,与 diff-P5 / 凸 gate **直接可比**。

## 6. 组件与文件(零风险,只新增)

- 新增 `scripts/adaptation/foundation_experts/fuse_residual_alarm.py`:头模块 + 训练/推理;复用 fuse_gate_alarm 的数据加载、软穿越 loss、限值处理(import 或复制,**不改原文件**)。
- 新增 driver 脚本(scratchpad 或 `scripts/adaptation/xmeas10/`):串「训练头 → 推理 → batch_metrics(新 summary/figure suffix)」。
- (可选)新增或扩展 CV selector。
- **不改任何已验证脚本**;±3σ 与 mag=100 限固定;batch_metrics 复用,仅换 suffix。

## 7. 非目标 / YAGNI

- 不做 h=15(先在 h=5 验证;换 horizon 仅改参数)。
- 不引入时序大模型 / transformer 头(Run11-17 数据量下过拟合风险高)。
- DLinear 头已排除(见 3.2)。
- 协变量已进入上下文分支;不再单独追求更复杂的协变量结构。

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 共享偏差不可迁移(核心科学风险) | 第 2 节验收线证伪 + 回退 diff-P5 |
| FAR 爆(自由输出乱造穿越) | `λ_reg·‖Δ‖²` + 报警定位标量 + grouped-CV 选 λ_far |
| 过拟合(Run11-17 有限) | 小头(~2–3k 参数)+ 正则 + 留 run CV + Δ 白噪声 sanity |
| lead time 因 horizon 短而偏低 | 上下文含协变量(上游更早显故障)争取提前量;不足则后续做协变量增强 |

## 9. 测试策略

- **单元**:Δ=0 时 corrected 精确等于 diff-P5;softFAR/softLead 与 fuse_gate_alarm 在已知输入上数值一致;断言训练从不触及 Run9-10/18-27(无泄漏守卫)。
- **集成**:端到端跑出 summary 行,输出形状正确;固定随机种子可复现。
- **科学验收**:第 2 节验收线 + sanity check。
