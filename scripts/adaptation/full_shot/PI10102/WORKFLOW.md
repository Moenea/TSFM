# PI10102 过渡过程预测 —— 完整训练/推理/评估流程

> 适用范围：`scripts/adaptation/full_shot/PI10102/` 下的 Timer-XL 4 个微调脚本 + `baselines/` 下的 8 个 baseline 训练脚本 + `infer_baselines_all.sh` + `batch_metrics_zjsh.sh` + `figures/PI10102/plot_fault_prognosis.ipynb`。
>
> 一句话概括：**所有模型在 ZJSH 2103 装置 PI10102 这一关键过程变量的 24 段过渡过程数据上训练/推理；最终论文级 MSE 指标统一切到「前 15 步、Timer-XL 可覆盖的对齐窗口」上比较**。

---

## 0. 数据与划分

### 数据来源
- 根目录：`/home/aicode/sherwin/dataset/ZJSH/`
- 过渡过程子目录：`2103ts/`，由 `dataset/ZJSH/code/preprocessing/export_transitions_to_csv.py` 导出。
- 文件格式：`PI10102_transition_<idx>_c<from>_to_c<to>_len<L>.csv`，共 24 段 (000–023)。
- 每段 CSV 包含 16 个变量列；**`PI10102` 始终是最后一列**（导出脚本保证），是预测 target；**完全是过渡（非稳态）数据**。

### Split 文件
唯一权威：`/home/aicode/sherwin/TSFM/setting/ZJSH_PI10102ts.yaml`

按 transition id 时序划分 17 / 3 / 4：
- **train**：000–016（涉及簇 c1..c13 之间的过渡）
- **val**：017（c7→c0）、018（c0→c5）、019（c5→c17）
- **test**：020（c17→c5, len=904）、021（c5→c16, len=3625）、022（c16→c14, len=1475）、023（c14→c15, len=1134）

val/test 故意挑训练里没出现过的稳态簇（c0/c5/c14..c17），属 out-of-sample 检验。**Timer-XL 微调和所有 baseline 共用同一份 split 文件**，所以训练/验证/测试样本来源完全一致。

### 报警限值
- `setting/limits_zjsh.csv` 给 PI10102 设：HH/H/L/LL = 260 / 160（用于报警判定）。
- 该限值只在 `batch_metrics.py` 的报警相关指标里使用，模型训练 loss 不感知。

---

## 1. Timer-XL 微调（4 个脚本）

文件：`timer_xl_zjsh_pi10102ts_{ms,s,ms_partial15,s_partial15}.sh`

### 共同超参（全部 4 个变体一致）
| 项 | 值 |
|---|---|
| pretrain ckpt | `/home/aicode/sherwin/TSFM/checkpoint.pth` |
| token_len | 96 |
| token_num | 8 |
| seq_len | 8 × 96 = **768** |
| pred_len（输入/输出 token）| 96 |
| test_pred_len | 96 |
| e_layers | 8 |
| d_model | 1024 |
| d_ff | 2048 |
| n_heads | 8 |
| batch_size | 32 |
| learning_rate | 5e-6 |
| train_epochs | 10 |
| patience | 3 |
| scheduler | cosine, tmax=10 |
| flags | `--use_norm --valid_last --adaptation` |

> 768 这个上下文长度：最短的 transition 长度是 904（test 中的 020），保证每段至少有 904 − 768 − 96 + 1 = 41 个滑窗。

### 4 个变体差异

| 变体 | 入口 | features | 额外 flag | loss 范围 |
|---|---|---|---|---|
| `*_ms.sh` | `run.py` | `MS` | `--covariate --last_token` | 完整 96 步 |
| `*_s.sh` | `run.py` | `S` | （无）| 完整 96 步 |
| `*_ms_partial15.sh` | `run_partial.py` | `MS` | `--covariate --last_token --loss_pred_len 15` | **每个 token 只取前 15 步** |
| `*_s_partial15.sh` | `run_partial.py` | `S` | `--loss_pred_len 15` | 每个 token 只取前 15 步 |

- **MS 模式**：16 变量都进 encoder，但 `--covariate --last_token` 让监督只作用在最后一列（PI10102），输入 `[B,768,16]` → loss 在 `[B, 96]` 上。
- **S 模式**：只读 PI10102 这一列，`[B,768,1]` → 自回归预测自己。
- **partial15**：走 `run_partial.py` → `Exp_Forecast_Partial`。它**只覆写 `train()` 与 `vali()`**：把 `[B, N*P]` reshape 成 `[B, N, P]` 后取 `[:, :, :15]` 再算 MSE 监督；**没覆写 `test()`**。所以 partial15 的 ckpt 是用前 15 步监督训出来的，但 inference 仍输出完整 96 步并保存。

### Train/Test 阶段输出
- ckpt 落到 `checkpoints/<setting>/checkpoint.pth`，setting 字符串由 `run.py`/`run_partial.py` 拼接（包含 model_id, model, data, sl, it, ot, lr, bt, wd, el, dm, dff, nh, cos, des, iter）。
- 训练完即调 `exp.test(setting)`：
  - 推理时 `exp_forecast.py:307-319` 做 autoregressive 多 token 滚动（虽然这里 test_pred_len = output_token_len，所以只走 1 步）。
  - 反归一化后取最后一列（如 covariate 模式）。
  - 保存到 `results/<setting>/`：
    - `pred.npy` 形状 `(N=3686, 96)` (MS) 或 `(N, 96, 1)` (S)，**完整 96 步**
    - `true.npy` 同上
    - `metrics.npy` = `[mae, mse, rmse, mape, mspe]`，**在完整 96 步上算的**

### 实际生成的 result 目录（test 设定 sl768/it96/ot96/lr5e-06/bt32/wd0/el8/dm1024/dff2048/nh8/cosTrue）
- `forecast_ZJSH_PI10102TS_MS_full_shot_timer_xl_..._test_0`
- `forecast_ZJSH_PI10102TS_S_full_shot_timer_xl_..._test_0`
- `forecast_ZJSH_PI10102TS_MS_partial15_timer_xl_..._test_0`
- `forecast_ZJSH_PI10102TS_S_partial15_timer_xl_..._test_0`
- 另外还有 zero-shot（未微调，直接拿预训练 ckpt 推理）：`zeroshot_ZJSH_PI10102TS_{S,MS}/`

---

## 2. Baseline 训练 + 推理（`baselines/` 子目录）

### 共同环境（`_common.sh`）
| 项 | 值 |
|---|---|
| ROOT_PATH | `/home/aicode/sherwin/dataset/ZJSH/` |
| SPLIT_FILE | `/home/aicode/sherwin/TSFM/setting/ZJSH_PI10102ts.yaml`（与 Timer-XL 同） |
| SEQ_LEN | 30 |
| LABEL_LEN | 15 |
| PRED_LEN | **15** |
| PATCH_LEN | 10 |
| DROPOUT | 0.2 |
| BATCH_SIZE | 128 |
| TRAIN_EPOCHS | 50 |
| PATIENCE | 5 |
| LR | 1e-3 |
| ENC_IN/DEC_IN/C_OUT (MS) | 16 / 16 / 1 |
| ENC_IN/DEC_IN/C_OUT (S) | 1 / 1 / 1 |

> **关键差异**：baseline 用 sl=30/pred=15 的短窗口，从零训练（无预训练）；Timer-XL 用 sl=768/pred=96 + adaptation。这是后续指标对齐的根本原因。

### 8 个模型（每个 MS + S 两份脚本）
- CNNLSTM (`d_model=230, d_ff=1408, e_layers=2`)
- DiPCALSTM (`472, 2048, 2`)
- LSTMGRU (`316, 2048, 1`)
- STAConvBiLSTM (`268, 2048, 1`)
- TCNTransformer (`128, 512, 2`，额外 `--d_layers 1 --activation gelu`，**启用 tail-aware loss**：α=2.0 β=0.003，two-sided，τ_high=0.98 τ_low=0.85)
- TimeXer (`248, 992, 1`) —— **只有 MS 脚本**，S 模式跳过（需要 exogenous）
- GTProger (`248, 992, 1`，启用 tail-aware loss) —— 只有 MS
- GTProgerV13 (`248, 992, 1`) —— 只有 MS

每个 `<MODEL>_<MODE>.sh` 内部走的还是 `run.py --task_name long_term_forecast --is_training 1`，但模型走 `BASELINE_MODELS` 分支（5-arg forward `(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)`）。`_baseline_forward` 把 `batch_x[:, -label_len:]` 拼上 zeros 当 `x_dec`。MS 模式 `c_out=1` 时 `_pad_baseline_outputs` 会把 (B, P, 1) 在通道维零填到 (B, P, 16) 让下游 covariate 切片能正确取最后一列。

### 推理脚本：`infer_baselines_all.sh`
- 复用训练时的所有超参以保证 setting 字符串与 `checkpoints/<setting>/checkpoint.pth` 路径匹配。
- 跳过逻辑：
  - `SKIP_MODELS="..."` 环境变量整模型跳过
  - S 模式且模型是 TimeXer / GTProger / GTProgerV13 时自动跳过
  - 找不到 ckpt 时直接跳过并打印
- 含 tail_aware 的模型（TCNTransformer, GTProger）会在推理时也加 `--use_tail_aware_loss --tail_alpha 2.0 --tail_beta 0.003 --tail_mode two_sided --alarm_threshold_high 0.98 --alarm_threshold_low 0.85`（只是为构造完全一致的 model 路径，不影响推理输出）。
- 每个 baseline 输出落到 `results/long_term_forecast_ZJSH_PI10102TS_<MODEL>_<MODE>_..._<DES>_0/`：
  - `pred.npy` 形状 `(N=6962, 15)` —— **完整 15 步**
  - `true.npy` 同上
  - `metrics.npy` —— 在 15 步上算的

> N=6962 vs Timer-XL 的 3686：因为 baseline 的 seq_len 只要 30，每段过渡可以滑出更多起点。

---

## 3. metrics.npy ≠ metrics_C.json（极易踩的坑）

每个 result 目录里有两个指标文件，**含义完全不同**：

| 文件 | 由谁写 | Timer-XL horizon | baseline horizon | 跨模型可比？ |
|---|---|---|---|---|
| `metrics.npy` | `exp_forecast.test()` line 358 (`metric(preds, trues)`) | **完整 96 步** | 15 步 | **不可比**，仅供训练日志参考 |
| `metrics_C.json` | `utils/batch_metrics.py` | **前 15 步** | 15 步 | **可比**，论文用 |

⚠️ 写论文/做条形图前请只用 `metrics_C.json`。`metrics.npy` 只能用来 sanity check 训练是否收敛。

---

## 4. 论文级评估：`batch_metrics_zjsh.sh`

### 入口
```bash
cd /home/aicode/sherwin/TSFM
bash scripts/adaptation/full_shot/PI10102/batch_metrics_zjsh.sh
# 内部即：python -u ./utils/batch_metrics.py --config ./setting/batch_metrics_zjsh_pi10102ts.yaml
```

### 配置 `setting/batch_metrics_zjsh_pi10102ts.yaml` 关键字段
```yaml
params:
  target: "PI10102"
  limit_csv_path: "setting/limits_zjsh.csv"      # HH/H/L/LL = 260/160
  data_root: "/home/aicode/sherwin/dataset/ZJSH"
  results_root: "./results"
  split_file: "setting/ZJSH_PI10102ts.yaml"
  alarm_quality_rmse_factor: 0.01                 # 报警质量门槛 = alarm_band * 0.01
  eval_steps: 15                                  # ★ 把所有模型的 pred 切到前 15 步
  input_clean_steps: 30                           # 输入侧"干净"判定步数
  align_eval_to: { seq_len: 768, pred_len: 96 }   # ★ 把 baseline 窗口对齐到 Timer-XL 可覆盖区域

model_dirs:
  # 6 个 Timer-XL 变体（zero-shot S/MS, full-shot S/MS, partial15 S/MS）→ seq_len=768, pred_len=96
  # 11 个启用 baseline（DiPCALSTM 已注释掉）→ seq_len=30, pred_len=15
  # 注意：TimeXer / GTProger / GTProgerV13 只有 MS 版

test:                                              # 与 ZJSH_PI10102ts.yaml 的 test 块一致
  - 2103ts/PI10102_transition_020_c17_to_c5_len904.csv
  - 2103ts/PI10102_transition_021_c5_to_c16_len3625.csv
  - 2103ts/PI10102_transition_022_c16_to_c14_len1475.csv
  - 2103ts/PI10102_transition_023_c14_to_c15_len1134.csv
```

### `utils/batch_metrics.py` 处理流程
1. 读 4 个 test CSV 的 PI10102 列，**按顺序拼接**为一条 `true_series`（窗口起点不跨文件边界）。
2. 对每个 model_dirs 条目：
   1. 读 `pred.npy` / `true.npy`，三维则取最后一列（target）。
   2. `build_window_starts(sl, pl, file_lengths)` 用模型自己的 sl/pl 重建每行 pred 对应的原始时间索引。
   3. **窗口对齐**：`align_window_mask` 只保留满足 `window_start ∈ [file_off + 768, file_off + L − 96]` 的窗口 → 把 baseline 早起的一大堆窗口（sl=30 时起点更靠前）裁掉，使所有模型在**同一组绝对时间索引**上比较。
   4. **horizon 截断**（line 314-317）：`pred_t = pred_t[:, :15]`，`true_t = true_t[:, :15]`。**这一步把 Timer-XL 的 96 步切到前 15 步**，与 baseline 自然对齐。
   5. 计算 per-window 指标：`patch_se = (pred − true)²`，`patch_mse = mean(axis=1)`，类似得到 `patch_rmse / patch_mae / patch_mape`。
   6. 报警判定：`pred_alarm_patch = any((pred_t > 260) | (pred_t < 160), axis=1)`，true 同理。`half_start = eval_steps // 2 = 7`，再算后 8 步内的报警布尔值（`true_alarm_last5`，名字虽叫 last5 实际由 eval_steps 控制）。
   7. **Quality gate**：`pred_quality_ok = patch_rmse <= alarm_band * 0.01`，把误差大的预测报警筛掉得到带 `_qf` 后缀的版本。
   8. 输出 dict 含 mse/rmse/mae/mape × {true_alarm_patch, no_true_alarm_patch, pred_alarm_patch, no_pred_alarm_patch} 共 16 个指标，加上 ratio_pred_in_true / ratio_pred_in_no_true、mean_lead_time_patch、mean_prognosis_error 等共约 30+ 字段，分别带 `_clean` `_qf` `_clean_qf` 4 类条件版本。
   9. 写到 `results/<model_result_dir>/metrics_C.json`。
3. 全部模型跑完后调 `plot_metrics()` / `plot_radar()` 把跨模型对比图保存到 `figures/PI10102/`：
   - `mse_true_alarm_patch.png`、`mse_no_true_alarm_patch.png`、`mse_pred_alarm_patch.png`、`mse_no_pred_alarm_patch.png`
   - `ratio_pred_in_true_alarm_patches{,_clean,_qf,_clean_qf}.png`
   - `ratio_pred_in_no_true_alarm_patches{,_clean,_qf,_clean_qf}.png`
   - `mean_lead_time_patch{,_clean,_qf,_clean_qf}.png`
   - `mean_prognosis_error{,_clean,_qf,_clean_qf}.png`
   - `summary_radar_C.png`

> **特别提醒**：figures 文件夹里 `mse_*_alarm_patch.png` 与雷达图全部是 **15 步 horizon** 上的 MSE，**不是 96 步**。

---

## 5. 局部时序图：`figures/PI10102/plot_fault_prognosis.ipynb`

这是一个独立的 Jupyter notebook，用来出**逐窗口 / 逐时间点**的可视化（论文 figure-quality 的过渡过程对比图）。

### Notebook 数据加载逻辑
1. 读同一份 `setting/batch_metrics_zjsh_pi10102ts.yaml` 拿到 `eval_steps=15`、`align_eval_to={768, 96}`、`test` 文件列表、各 model 的 result_dir 与各自 sl/pl。
2. 拼接 `true_series`（4 个 transition 的 PI10102 列首尾相接）。
3. `build_window_starts` 重建每行 pred 的起点；`align_window_mask` 过滤到 Timer-XL 可覆盖区域。
4. **`_load_npy` 关键一行**：`return arr[keep][:, :eval_steps]` —— **磁盘上的 96 步 Timer-XL 预测被截到前 15 步**，baseline 本身就是 15 步。所有 `pred_*` / `plot_true` 进入绘图函数前形状都是 `(N_aligned, 15)`。
5. 加载的模型清单（行内注释里的 15 个）：
   - Timer-XL: 6 个（full-shot S/MS, zero-shot S/MS, partial15 S/MS）
   - Baselines (S): CNNLSTM, LSTMGRU, STAConvBiLSTM, TCNTransformer
   - Baselines (MS): 上面 4 个 + TimeXer, GTProger, GTProgerV13
   - DiPCALSTM 在 notebook 中也被注释掉（与 yaml 一致）

### 主要绘图函数 / 输出文件
- `fault_window_plot_datetime` —— 单个 window 的 (历史 30 + 预测 15) 时序图，输出 `a_fault_window_*.png`、`a_fault_window_file{0..3}_*.png`
- `plot_range_comparison` —— 在指定 patch 索引区间 `[start_t, end_t]` 内逐时刻取每个窗口预测的最后一步重建一条连续轨迹（`reconstruct_signal` 用 `j = pred_len - 1`），输出 `a_range_021_*.png`
- `concatenated_test_set.png` —— 4 段 test 拼接后的原始观测；标出文件边界、每段的 min/max/mean/std/p05/p95
- `a_small_multiples_021.png`、`a_dual_panel_021.png`、`a_envelope_021.png`、`a_error_heatmap_021.png`、`a_rolling_mae_021.png` —— 不同形式的多模型对比图（其中后几个的 cell 在 notebook 里被注释掉了，需要时手动启用）
- 报警限值 `ALARM_HLINES`、报警起始点都基于 PI10102=260/160 派生

> notebook 自身不重新计算 MSE，所有显示的误差/重建都是基于 `_load_npy` 截过的 15 步切片，跟 `metrics_C.json` 同源。

---

## 6. 端到端完整跑通顺序

```bash
cd /home/aicode/sherwin/TSFM

# A. Timer-XL 4 个变体（每个脚本自己 train + test → 写 results/forecast_ZJSH_PI10102TS_*）
bash scripts/adaptation/full_shot/PI10102/timer_xl_zjsh_pi10102ts_ms.sh
bash scripts/adaptation/full_shot/PI10102/timer_xl_zjsh_pi10102ts_s.sh
bash scripts/adaptation/full_shot/PI10102/timer_xl_zjsh_pi10102ts_ms_partial15.sh
bash scripts/adaptation/full_shot/PI10102/timer_xl_zjsh_pi10102ts_s_partial15.sh
# 另外 zero-shot 在别处脚本生成（results/zeroshot_ZJSH_PI10102TS_{S,MS}/）

# B. 8 个 baseline × {MS, S}（按需 source CUDA_VISIBLE_DEVICES 改卡）
for f in scripts/adaptation/full_shot/PI10102/baselines/*_ms.sh \
         scripts/adaptation/full_shot/PI10102/baselines/*_s.sh; do
  bash "$f"
done
# 单独训练完成后可对全部模型做一次推理重写 pred/true
bash scripts/adaptation/full_shot/PI10102/baselines/infer_baselines_all.sh

# C. 跨模型统一对齐 + 报警相关指标 + 条形/雷达图
bash scripts/adaptation/full_shot/PI10102/batch_metrics_zjsh.sh
# → 写 results/<...>/metrics_C.json
# → 写 figures/PI10102/{mse_*_alarm_patch,summary_radar_C,ratio_*,mean_lead_time*,mean_prognosis_error*}.png

# D. 局部 / 单窗口可视化（手动跑 notebook）
jupyter nbconvert --to notebook --execute figures/PI10102/plot_fault_prognosis.ipynb
# 或直接 jupyter lab 打开交互式
# → 写 figures/PI10102/{a_*.png,concatenated_test_set.png}
```

---

## 7. 关键原则速查表

| 问题 | 答案 |
|---|---|
| Timer-XL 训练数据是什么？ | `2103ts/transition_000..016`（17 段过渡），共用 `ZJSH_PI10102ts.yaml` |
| Baseline 训练数据是什么？ | 与 Timer-XL **完全相同**的 17 段 |
| 谁负责切到 15 步比较？ | `utils/batch_metrics.py` line 314-317 + notebook `_load_npy()` |
| `metrics.npy` 是多长 horizon？ | Timer-XL 96，baseline 15，**不可直接比** |
| `metrics_C.json` 是多长 horizon？ | 都是 15，可比 |
| `figures/PI10102/*.png` 用的是 15 还是 96？ | **15** |
| partial15 推理输出多少步？ | 96（`Exp_Forecast_Partial` 不覆写 `test()`） |
| baseline 的 N=6962 vs Timer-XL N=3686 为什么不同？ | seq_len 差异（30 vs 768）导致每段可滑窗起点数不同；`align_window_mask` 在评估时把它们对齐到同一时间区域 |
| 报警限值在哪里？ | `setting/limits_zjsh.csv`，PI10102 HH/H/L/LL = 260 / 160 |
| TimeXer/GTProger/GTProgerV13 为什么没有 S 版？ | cross-attention 路径要求 ≥1 个 exogenous 变量，S 模式（仅 PI10102 自身）会 crash |
| DiPCALSTM 为什么不在最终对比里？ | 在 yaml 与 notebook 中都被注释掉（训练脚本仍在，可手动启用） |

---

## 8. 容易出错 / 反直觉的点

1. **`metrics.npy` vs `metrics_C.json` 差异**：见 §3，已踩坑。
2. **partial15 不影响推理输出**：用 partial15 是为了让训练 loss 集中在前 15 步（论文 horizon），但保存的 pred 仍是 96 步；要看"前 15 步效果"必须靠 `batch_metrics.py` 的 `eval_steps` 截断或 notebook `_load_npy` 截断，**不是看 `metrics.npy`**。
3. **窗口数对不齐**：跨模型对比绝不能直接 `pred_a vs pred_b`，必须先做 `align_window_mask`，否则 Timer-XL 与 baseline 不在同一时间点上。
4. **target 必须是 CSV 最后一列**：`MultivariateDatasetYAMLSplit` 与 covariate 切片硬编码取 `[:, -1]`；导出脚本已保证 PI10102 在最后。
5. **`half_start` 的命名**：`utils/batch_metrics.py` 里 `true_alarm_last5 = np.any(true_patch_alarm[:, half_start:], axis=1)` 这个变量名叫 last5 是历史遗留；当 `eval_steps=15` 时 `half_start = 15//2 = 7`，实际是后 8 步。
6. **TCNTransformer / GTProger 训练时启用 tail_aware loss**：超参 α=2.0, β=0.003, τ_high=0.98, τ_low=0.85 是 two-sided 模式，对**接近报警限值**的样本加权。其他 baseline 用普通 MSE。
7. **inference 阶段拼 setting 字符串若与训练不一致就找不到 ckpt**：`infer_baselines_all.sh` 里所有超参（包括 cosine=False, wd=0, nh=8）都必须与训练完全一致；新增超参时记得同步两边。

---

*这份文档配合 scripts/adaptation/full_shot/PI10102/ 内的脚本与 setting/、utils/batch_metrics.py、figures/PI10102/plot_fault_prognosis.ipynb 一起阅读。如果未来流程变更（例如改 eval_steps、增加 baseline、换 split），请同步更新本文件 §0–§5。*
