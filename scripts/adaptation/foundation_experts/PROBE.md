# Foundation-model environment + API probe (Task 1)

Probed 2026-06-30 on GPU 0 (the only idle GPU; GPU 1 is occupied by another
user's training jobs). **Always run model code with `CUDA_VISIBLE_DEVICES=0`.**
The first probe attempt ran on CPU under heavy machine load and hung; the GPU-0
path is fast.

## Environment decision

- **FM_PY = `/home/aicode/miniconda3/envs/tsfm/bin/python`** — the existing
  `tsfm` env. **No `transformers` bump needed**: both Time-MoE and Sundial load
  under transformers 4.40.1 via `trust_remote_code=True`. The `tsfm` env is NOT
  mutated. No isolated clone env was created.
- Device for all adapters: `cuda:0` (~14.7 GB free, idle). Never use GPU 1.

## Per-model findings

| model | interpreter | load call | inference call | output shape | training interface | decision |
|---|---|---|---|---|---|---|
| **Time-MoE** | FM_PY (cuda:0) | `AutoModelForCausalLM.from_pretrained("Maple728/TimeMoE-50M", trust_remote_code=True)` | `model.generate(ctx, max_new_tokens=15)` | `(B, 96+15)=(B,111)` → slice `[:, -15:]` | **YES** — `forward(input_ids, labels=..., loss_masks=..., max_horizon_length=...)` returns `MoeCausalLMOutputWithPast` with `.loss` | **FINE-TUNE** via `forward(input_ids=ctx, labels=future).loss` |
| **Sundial** | FM_PY (cuda:0) | `AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True)` | `model.generate(ctx, max_new_tokens=15, num_samples=20).mean(dim=1)` | pending weight download (see below) | **LIKELY YES** — repo ships `flow_loss.py` (flow-matching loss); confirm `forward(...).loss` once weights land | **FINE-TUNE** (fallback: zero-shot) |
| **MOIRAI** | (uni2ts) | `MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")` + `MoiraiForecast(...)` | predictive mean of samples | n/a | requires `uni2ts` (NOT installed) | **DEFER to Task 7** — default zero-shot / may drop |

### Time-MoE (confirmed, decisive)
- 113,352,192 params. Loads fully cached.
- `forward` signature: `(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, loss_masks, use_cache, output_attentions, output_hidden_states, return_dict, max_horizon_length)`.
- `forward(input_ids=ctx, labels=ctx)` returns `MoeCausalLMOutputWithPast` with a non-None `.loss`. **Task 5 should fine-tune via `forward(labels=...).loss`, NOT the generate-MSE fallback.** Per-window instance-normalize the context and the future target the same way before computing the loss.
- `generate(ctx, max_new_tokens=15)` returns the full sequence `(B, 111)`; the forecast is `out[:, -15:]`. Output is float32 in the (normalized) input space → de-normalize with the per-window mean/std.

### Sundial (weights downloading)
- Code files cached: `configuration_sundial.py`, `modeling_sundial.py`, `flow_loss.py`, `ts_generation_mixin.py`. **`flow_loss.py` indicates a flow-matching training loss is implemented** → fine-tuning is plausible; confirm the exact `forward(...).loss` entry point once weights are present.
- `model.safetensors` is **513 MB**, served via HF's **Xet CDN**, which stalled (0 bytes after 5 min). Re-downloading with **`HF_HUB_DISABLE_XET=1`** (classic path) in the background. Re-run the Sundial probe (generate shape, forward/loss) after the download finishes, and update this table.
- Predict recipe: `generate(ctx, max_new_tokens=15, num_samples=20).mean(dim=1)` then slice `[:, -15:]` and de-normalize.

### MOIRAI / uni2ts (deferred)
- `uni2ts` is not installed. It pulls `torch`/`lightning`/`gluonts`, which could upgrade torch and **break the `tsfm` Timer-XL pipeline**. Per the isolation rule, if pursued it must go in a cloned env (`tsfm_fm`), not `tsfm`.
- Decision deferred to Task 7: attempt the isolated install there; if it is heavy or conflicting, MOIRAI runs **zero-shot** (or is dropped from the gate), recorded in its `meta.json`. The gate and the other four experts do not depend on MOIRAI.

## Note on process

This probe was completed by the controller (not a subagent) after the first
implementer subagent orphaned a CPU-bound probe that hung for ~11 min. Tasks
2-4 (expert_io, gate, evaluate) need none of these models and proceed
immediately; the Sundial/MOIRAI rows are finalized before Tasks 6-7.
