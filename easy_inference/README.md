# Easy Inference

Use this folder for the simplest Surya flow:
1. choose date window,
2. download only required hourly files,
3. run rollout inference,
4. save one `prediction.nc`.

## Quick start

```bash
source .venv/bin/activate
bash easy_inference/run_easy_inference.sh
```

Non-interactive defaults:

```bash
bash easy_inference/run_easy_inference.sh --no-prompt
```

## Config

Edit `easy_inference/config_easy.yaml`.

- Normal users: edit only the top `user:` section.
- Advanced users: optional changes in `advanced:`.

Default and override behavior:

```bash
# Uses easy_inference/config_easy.yaml by default.
python easy_inference/run_easy_inference.py

# Optional: use a different YAML file.
python easy_inference/run_easy_inference.py --config-path /path/to/custom_easy.yaml
```

### Debug mode

Set in `advanced:`:
- `debug_mode: true`
- optional `debug_log_path: "path/to/inference_debug.txt"` (default is `<user.output_dir>/inference_debug.txt`)

When enabled, the text log contains stage timings and per-step diagnostics with line number + UTC timestamp:
- input file read / transform timing
- GT file read timing
- per-step forward / CPU-copy / inverse-transform / write timing
- per-step memory stats (`CUDA` peak/allocated/reserved when available)

## Metrics Notebook

Use `easy_inference/compare_prediction_groundtruth.ipynb` to compare `prediction.nc` vs GT and compute:
- overall metrics (`MSE`, `RMSE`, `MAE`, `bias`, `max_abs_error`)
- per-channel metrics
- per-step metrics
- visual prediction vs ground-truth plots
