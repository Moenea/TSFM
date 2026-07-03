# TEP IDV13 Gate-T2 workflow

This experiment is isolated from the ZJSH and PCA101A pipelines.

- Data: `/home/aicode/sherwin/dataset/TEP/csv`
- Train: Run1-Run7
- Gate training / validation: Run8
- Final test: Run9-Run10
- Fault: IDV13, injected at 30 h (sample 600, sampling interval 0.05 h)
- Target: `XMEAS07 Reactor Pressure`
- Context: 96 samples (4.8 h) = a single Timer-XL input token
- Forecast: 96 samples; Gate-T2 fuses the first 15 samples (45 min)
- Experts: raw Timer-XL-S and DIFF-to-ABS Timer-XL-S

The target was selected using only Run1-Run7. It has the strongest median
standardized early response among XMEAS channels when comparing 20-30 h with
35-45 h.

Run everything:

```bash
cd /home/aicode/sherwin/TSFM
bash scripts/adaptation/full_shot/TEP_IDV13/run_all.sh
```

Use `EPOCHS`, `BATCH_SIZE`, and `GPU_PHYSICAL` to override defaults.

Outputs:

- raw and DIFF checkpoints under `checkpoints/`
- base predictions under `results/forecast_TEP_IDV13_*`
- Gate-T2 predictions under `results/ensemble_Gate-T2-TEP-IDV13-XMEAS07-S_test_0`
- final report at `results/TEP_IDV13_XMEAS07_Summary/metrics.json`
